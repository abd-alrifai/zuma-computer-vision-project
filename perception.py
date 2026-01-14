import cv2
import numpy as np


class ZumaPerception:
    """
    Perception Layer for Zuma (NO ML).

    Output schema:
    perception_output = {
      "frog": {"x": int, "y": int},
      "frog_ball": {"x": int, "y": int, "r": int, "color": str} or None,
      "path": {
          "points": [(x1,y1), (x2,y2), ...],
          "end_point": (xe, ye)
      },
      "balls": [
          {
            "id": int,
            "x": int,
            "y": int,
            "r": int,
            "color": str,
            "order": int,
            "distance_to_end": float,
            "danger": bool
          },
          ...
      ]
    }
    """

    def __init__(
        self,
        danger_distance_px: float = 220.0,

        # Hough circle params (path balls)
        hough_dp: float = 1.2,
        hough_minDist: int = 18,
        hough_param1: int = 140,
        hough_param2: int = 18,
        hough_minRadius: int = 14,
        hough_maxRadius: int = 22,

        # Path extraction params (kept for fallback)
        path_dark_thresh: int = 145,
        path_morph_ksize: int = 5,
        min_track_area_ratio: float = 0.01,
        track_band_dilate: int = 13,

        # Cache update frequency
        path_update_every: int = 12,
        frog_update_every: int = 12,

        # Speed/robustness
        max_path_points: int = 2500,

        # ROI crop ratios to avoid browser/UI noise
        roi_left: float = 0.03,
        roi_right: float = 0.03,
        roi_top: float = 0.16,
        roi_bottom: float = 0.10,

        # Canny params for path edges
        canny_t1: int = 45,
        canny_t2: int = 140,

        # NEW: frog ball detection params (pure perception)
        frog_ball_roi_size: int = 180,
        frog_ball_min_r: int = 12,
        frog_ball_max_r: int = 26,
        frog_ball_min_dist_from_frog: int = 16,
        frog_ball_max_dist_from_frog: int = 95,
        frog_ball_update_every: int = 1,   # detect every frame (cheap ROI)
        frog_ball_keepalive_frames: int = 6,  # if missed, keep last for a few frames
    ):
        # ROI (left, top, width, height) in SCREEN coordinates
        self._roi = None
        self.danger_distance_px = float(danger_distance_px)

        self.hough_dp = float(hough_dp)
        self.hough_minDist = int(hough_minDist)
        self.hough_param1 = int(hough_param1)
        self.hough_param2 = int(hough_param2)
        self.hough_minRadius = int(hough_minRadius)
        self.hough_maxRadius = int(hough_maxRadius)

        self.path_dark_thresh = int(path_dark_thresh)
        self.path_morph_ksize = int(path_morph_ksize)
        self.min_track_area_ratio = float(min_track_area_ratio)
        self.track_band_dilate = int(track_band_dilate)

        self.path_update_every = int(path_update_every)
        self.frog_update_every = int(frog_update_every)
        self.max_path_points = int(max_path_points)

        self.roi_left = float(roi_left)
        self.roi_right = float(roi_right)
        self.roi_top = float(roi_top)
        self.roi_bottom = float(roi_bottom)

        self.canny_t1 = int(canny_t1)
        self.canny_t2 = int(canny_t2)

        # frog ball params
        self.frog_ball_roi_size = int(frog_ball_roi_size)
        self.frog_ball_min_r = int(frog_ball_min_r)
        self.frog_ball_max_r = int(frog_ball_max_r)
        self.frog_ball_min_dist_from_frog = int(frog_ball_min_dist_from_frog)
        self.frog_ball_max_dist_from_frog = int(frog_ball_max_dist_from_frog)
        self.frog_ball_update_every = int(frog_ball_update_every)
        self.frog_ball_keepalive_frames = int(frog_ball_keepalive_frames)

        # Cached state
        self._frame_id = 0

        self._cached_path_points = None
        self._cached_end_point = None
        self._cached_cumlen = None

        self._cached_frog = None

        self._cached_track_mask = None
        self._cached_track_band = None

        # NEW: cached frog ball
        self._cached_frog_ball = None
        self._frog_ball_last_seen = -10**9

        # -------------------------
        # Direction lock (temporal, NO ML)
        # -------------------------
        self.direction_warmup_frames = 3
        self.track_match_max_dist = 35
        self.direction_lock_min_votes = 4

        self._direction_locked = False
        self._path_should_be_reversed = False

        self._dir_vote_accum = 0
        self._dir_vote_count = 0
        self._dir_frames_seen = 0

        self._prev_ball_tracks = {}
        self._next_track_id = 0
    
    # ------------------------- Public API -------------------------
    def process_frame(self, frame_bgr: np.ndarray, debug: bool = False):
        self._frame_id += 1
        h, w = frame_bgr.shape[:2]

        # 1) Frog (cached)
        if (self._cached_frog is None) or (self._frame_id % self.frog_update_every == 1):
            frog_xy = self._detect_frog_center(frame_bgr)
            if frog_xy is None:
                frog_xy = (w // 2, h // 2)
            self._cached_frog = frog_xy
        frog_xy = self._cached_frog

        # 2) Path (cached)
        if (self._cached_path_points is None) or (self._frame_id % self.path_update_every == 1):
            path_points, end_point, cumlen = self._extract_path_ordered(frame_bgr, frog_xy, debug=debug)
            if path_points is not None and len(path_points) >= 10:
                if self._direction_locked and self._path_should_be_reversed:
                    path_points = list(reversed(path_points))
                    cumlen = self._cumulative_lengths(path_points)
                    end_point = path_points[-1]

                self._cached_path_points = path_points
                self._cached_end_point = end_point
                self._cached_cumlen = cumlen

        path_points = self._cached_path_points if self._cached_path_points is not None else []
        end_point = self._cached_end_point if self._cached_end_point is not None else (0, 0)
        cumlen = self._cached_cumlen

        # 3) Balls on path
        balls_basic = self._detect_balls(frame_bgr)

        # 3.5) Direction lock from ball motion (no control logic, just perception direction consistency)
        if path_points and (cumlen is not None) and balls_basic and (len(cumlen) == len(path_points)):
            path_arr = np.array(path_points, dtype=np.int32)

            curr_dets = []
            for b in balls_basic:
                bx, by = int(b["x"]), int(b["y"])
                idx = self._nearest_path_index(path_arr, bx, by)
                curr_dets.append({"x": bx, "y": by, "color": b["color"], "idx": idx})

            matches, unmatched_curr = self._match_tracks(self._prev_ball_tracks, curr_dets)

            if not self._direction_locked:
                self._accumulate_direction_votes(matches)

                if (self._dir_frames_seen >= self.direction_warmup_frames and
                        self._dir_vote_count >= self.direction_lock_min_votes):

                    if self._dir_vote_accum < 0:
                        self._path_should_be_reversed = True
                        if self._cached_path_points:
                            self._cached_path_points = list(reversed(self._cached_path_points))
                            self._cached_cumlen = self._cumulative_lengths(self._cached_path_points)
                            self._cached_end_point = self._cached_path_points[-1]
                    else:
                        self._path_should_be_reversed = False

                    self._direction_locked = True

            new_tracks = {}
            for tid, curr in matches:
                new_tracks[tid] = {"x": curr["x"], "y": curr["y"], "color": curr["color"], "idx": curr["idx"]}

            for c in unmatched_curr:
                tid = self._next_track_id
                self._next_track_id += 1
                new_tracks[tid] = {"x": c["x"], "y": c["y"], "color": c["color"], "idx": c["idx"]}

            self._prev_ball_tracks = new_tracks

        # 4) Assign order + distance_to_end + danger
        balls_struct = []
        if path_points and (cumlen is not None) and (len(cumlen) == len(path_points)):
            path_arr = np.array(path_points, dtype=np.int32)

            for i, b in enumerate(balls_basic):
                bx, by = int(b["x"]), int(b["y"])
                idx = self._nearest_path_index(path_arr, bx, by)

                dist_to_end = float(cumlen[-1] - cumlen[idx])
                danger = (dist_to_end <= self.danger_distance_px)

                balls_struct.append({
                    "id": i,
                    "x": int(bx),
                    "y": int(by),
                    "r": int(b["r"]),
                    "color": str(b["color"]),
                    "order": int(idx),
                    "distance_to_end": dist_to_end,
                    "danger": bool(danger),
                })

            balls_struct.sort(key=lambda d: d["order"])
            for new_id, d in enumerate(balls_struct):
                d["id"] = new_id
        else:
            for i, b in enumerate(balls_basic):
                balls_struct.append({
                    "id": i,
                    "x": int(b["x"]),
                    "y": int(b["y"]),
                    "r": int(b["r"]),
                    "color": str(b["color"]),
                    "order": -1,
                    "distance_to_end": float("inf"),
                    "danger": False,
                })

        # 5) Frog ball (CRITICAL) - independent entity (not on path)
        frog_ball = None
        if (self._frame_id % self.frog_ball_update_every == 0) or (self._cached_frog_ball is None):
            fb = self._detect_frog_ball(frame_bgr, frog_xy)
            if fb is not None:
                self._cached_frog_ball = fb
                self._frog_ball_last_seen = self._frame_id

        # keepalive (pure perception temporal smoothing; no actions)
        if self._cached_frog_ball is not None:
            if (self._frame_id - self._frog_ball_last_seen) <= self.frog_ball_keepalive_frames:
                frog_ball = dict(self._cached_frog_ball)
            else:
                self._cached_frog_ball = None

        perception_output = {
            "frog": {"x": int(frog_xy[0]), "y": int(frog_xy[1])},
            "frog_ball": frog_ball,  # <-- NEW
            "path": {
                "points": [(int(x), int(y)) for (x, y) in path_points],
                "end_point": (int(end_point[0]), int(end_point[1])),
            },
            "balls": balls_struct
        }

        if debug:
            vis = self._draw_debug(frame_bgr, perception_output)
            return perception_output, vis

        return perception_output

    # -------------------------
    # Direction lock helpers
    # -------------------------
    def _match_tracks(self, prev_tracks, curr_dets):
        matches = []
        used_curr = set()

        prev_items = list(prev_tracks.items())
        for tid, p in prev_items:
            best_j = None
            best_d2 = None

            for j, c in enumerate(curr_dets):
                if j in used_curr:
                    continue
                if c["color"] != p["color"]:
                    continue

                dx = c["x"] - p["x"]
                dy = c["y"] - p["y"]
                d2 = dx * dx + dy * dy

                if d2 <= self.track_match_max_dist * self.track_match_max_dist:
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best_j = j

            if best_j is not None:
                used_curr.add(best_j)
                matches.append((tid, curr_dets[best_j]))

        unmatched_curr = [c for j, c in enumerate(curr_dets) if j not in used_curr]
        return matches, unmatched_curr

    def _accumulate_direction_votes(self, matches):
        frame_vote = 0
        frame_votes = 0

        for tid, curr in matches:
            prev = self._prev_ball_tracks.get(tid)
            if prev is None:
                continue
            delta = int(curr["idx"]) - int(prev["idx"])

            if delta > 0:
                frame_vote += 1
                frame_votes += 1
            elif delta < 0:
                frame_vote -= 1
                frame_votes += 1

        self._dir_frames_seen += 1
        if frame_votes == 0:
            return

        self._dir_vote_accum += frame_vote
        self._dir_vote_count += frame_votes

    # ------------------------- Frog Detection -------------------------
    def _detect_frog_center(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]

        cx0, cy0 = w // 2, h // 2
        rw, rh = int(0.55 * w), int(0.55 * h)
        x0 = max(0, cx0 - rw // 2)
        y0 = max(0, cy0 - rh // 2)
        x1 = min(w, cx0 + rw // 2)
        y1 = min(h, cy0 + rh // 2)

        roi = frame_bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower = np.array([35, 60, 50], dtype=np.uint8)
        upper = np.array([85, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        k = max(3, (min(h, w) // 200) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        best = None
        best_score = -1e18
        rcx, rcy = (x1 - x0) // 2, (y1 - y0) // 2

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 120:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            d2 = (cx - rcx) ** 2 + (cy - rcy) ** 2
            score = float(area) - 0.0015 * float(d2)
            if score > best_score:
                best_score = score
                best = (cx, cy)

        if best is None:
            return None

        return (int(best[0] + x0), int(best[1] + y0))

    # ------------------------- Ball Color Helpers -------------------------
    def _mean_hsv_in_circle(self, hsv, x, y, r_small=4):
        h_img, w_img = hsv.shape[:2]
        x0 = max(0, x - r_small)
        x1 = min(w_img, x + r_small + 1)
        y0 = max(0, y - r_small)
        y1 = min(h_img, y + r_small + 1)
        patch = hsv[y0:y1, x0:x1]
        mean = patch.reshape(-1, 3).mean(axis=0)
        return int(mean[0]), int(mean[1]), int(mean[2])

    def _mean_hsv_on_mask(self, hsv, mask_01):
        # mask_01: 0/1 uint8
        ys, xs = np.where(mask_01 > 0)
        if len(xs) < 10:
            return None
        vals = hsv[ys, xs].astype(np.float32)
        m = vals.mean(axis=0)
        return int(m[0]), int(m[1]), int(m[2])

    def _classify_color_from_hsv(self, h, s, v):
        if s < 60 or v < 60:
            return "unknown"
        if (h <= 10 or h >= 170):
            return "red"
        if 18 <= h <= 38:
            return "yellow"
        if 40 <= h <= 85:
            return "green"
        if 90 <= h <= 130:
            return "blue"
        if 135 <= h <= 165:
            return "purple"
        return "unknown"

    def _classify_color_hsv_mean(self, hsv, x, y):
        h, s, v = self._mean_hsv_in_circle(hsv, x, y, r_small=4)
        return self._classify_color_from_hsv(h, s, v)

    # ------------------------- Ball Detection (Path Balls) -------------------------
    def _suppress_duplicates(self, circles, min_dist=12):
        kept = []
        for x, y, r in circles:
            ok = True
            for kx, ky, kr in kept:
                dx = kx - x
                dy = ky - y
                if dx * dx + dy * dy < min_dist * min_dist:
                    ok = False
                    break
            if ok:
                kept.append((x, y, r))
        return kept

    def _is_on_path_darkness(self, gray_blur, x, y, r):
        # Kept as fallback only (less relied on now)
        h, w = gray_blur.shape
        r_outer = r + 6
        x0 = max(0, x - r_outer)
        x1 = min(w, x + r_outer)
        y0 = max(0, y - r_outer)
        y1 = min(h, y + r_outer)
        patch = gray_blur[y0:y1, x0:x1]
        if patch.size == 0:
            return False
        mean_intensity = float(np.mean(patch))
        return mean_intensity < 185  # relaxed (was 170)

    def _detect_balls(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=self.hough_minDist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.hough_minRadius,
            maxRadius=self.hough_maxRadius
        )

        if circles is None:
            return []

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        circles = np.round(circles[0]).astype(int)
        circles = [(int(x), int(y), int(r)) for (x, y, r) in circles]
        circles = self._suppress_duplicates(circles, min_dist=int(self.hough_minDist * 0.8))

        h_img, w_img = frame_bgr.shape[:2]
        balls = []

        track_band = self._cached_track_band

        for (x, y, r) in circles:
            if not (0 <= x < w_img and 0 <= y < h_img):
                continue

            # Must be on track band if available (strong geometric prior)
            if track_band is not None and track_band[y, x] == 0:
                continue

            # Better color stability: sample a slightly larger patch
            color = self._classify_color_hsv_mean(hsv, x, y)
            if color == "unknown":
                continue

            # If we DO have track_band, do not over-filter by darkness.
            # If no track_band (rare), use relaxed darkness fallback.
            if track_band is None:
                if not self._is_on_path_darkness(gray_blur, x, y, r):
                    continue

            balls.append({"x": x, "y": y, "r": r, "color": color})

        return balls

    # ------------------------- Frog Ball Detection (CRITICAL) -------------------------
    def _hsv_ball_candidate_mask(self, hsv_roi):
        """
        Broad HSV masks for Zuma ball colors (traditional).
        Returns 0/255 mask.
        """
        # Basic quality gate: require some saturation/value to avoid gray noise
        s = hsv_roi[:, :, 1]
        v = hsv_roi[:, :, 2]
        sv_gate = ((s >= 70) & (v >= 60)).astype(np.uint8) * 255

        # Red wraps around hue (0..10) U (170..179)
        m_red1 = cv2.inRange(hsv_roi, (0, 70, 60), (10, 255, 255))
        m_red2 = cv2.inRange(hsv_roi, (170, 70, 60), (179, 255, 255))
        m_red = cv2.bitwise_or(m_red1, m_red2)

        m_yel = cv2.inRange(hsv_roi, (16, 70, 60), (40, 255, 255))
        m_grn = cv2.inRange(hsv_roi, (40, 70, 50), (90, 255, 255))
        m_blu = cv2.inRange(hsv_roi, (90, 70, 50), (135, 255, 255))
        m_pur = cv2.inRange(hsv_roi, (135, 70, 50), (170, 255, 255))

        m = m_red
        m = cv2.bitwise_or(m, m_yel)
        m = cv2.bitwise_or(m, m_grn)
        m = cv2.bitwise_or(m, m_blu)
        m = cv2.bitwise_or(m, m_pur)

        m = cv2.bitwise_and(m, sv_gate)

        # cleanup
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        return m

    def _detect_frog_ball(self, frame_bgr: np.ndarray, frog_xy):
        """
        Detect the current ball in frog mouth as an independent entity.
        - Partial visibility supported (semi-circle blobs).
        - NOT on path band.
        - Traditional CV only (HSV segmentation + contours + minEnclosingCircle).
        """
        h, w = frame_bgr.shape[:2]
        fx, fy = int(frog_xy[0]), int(frog_xy[1])

        half = self.frog_ball_roi_size // 2
        x0 = max(0, fx - half)
        y0 = max(0, fy - half)
        x1 = min(w, fx + half)
        y1 = min(h, fy + half)

        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Candidate colors mask
        cand = self._hsv_ball_candidate_mask(hsv)

        # Exclude frog body: green mask dilated
        frog_green = cv2.inRange(hsv, (35, 50, 40), (90, 255, 255))
        frog_green = cv2.dilate(frog_green, np.ones((9, 9), np.uint8), iterations=1)
        cand = cv2.bitwise_and(cand, cv2.bitwise_not(frog_green))

        # Exclude path band if available (frog ball is not on path)
        if self._cached_track_band is not None:
            band_roi = self._cached_track_band[y0:y1, x0:x1]
            cand = cv2.bitwise_and(cand, cv2.bitwise_not(band_roi))

        # Find blobs
        cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        best = None
        best_score = -1e18

        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < 80:
                continue

            (cx, cy), r = cv2.minEnclosingCircle(c)
            r = float(r)
            if r < self.frog_ball_min_r or r > self.frog_ball_max_r:
                continue

            cx_full = float(cx + x0)
            cy_full = float(cy + y0)

            dx = cx_full - fx
            dy = cy_full - fy
            dist = (dx * dx + dy * dy) ** 0.5

            # Must be near frog but not exactly at center
            if dist < self.frog_ball_min_dist_from_frog or dist > self.frog_ball_max_dist_from_frog:
                continue

            # fill ratio: partial ball still OK (semi-circle gives ~0.5)
            fill = area / (np.pi * r * r + 1e-6)

            # color confidence from mean HSV on contour mask
            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 1, -1)
            mhsv = self._mean_hsv_on_mask(hsv, mask)
            if mhsv is None:
                continue
            hh, ss, vv = mhsv
            color = self._classify_color_from_hsv(hh, ss, vv)
            if color == "unknown":
                continue

            # Score: prefer reasonable fill, higher saturation/value, and mid-distance
            # (still perception-only heuristic)
            dist_score = -abs(dist - 55.0) * 0.15
            score = (fill * 120.0) + (ss * 0.08) + (vv * 0.03) + dist_score

            if score > best_score:
                best_score = score
                best = (int(round(cx_full)), int(round(cy_full)), int(round(r)), color)

        if best is None:
            return None

        bx, by, br, bc = best
        return {"x": int(bx), "y": int(by), "r": int(br), "color": str(bc)}

    # ------------------------- Path Extraction (robust edges + ROI) -------------------------
    def _get_game_roi(self, h, w):
        x0 = int(self.roi_left * w)
        x1 = int((1.0 - self.roi_right) * w)
        y0 = int(self.roi_top * h)
        y1 = int((1.0 - self.roi_bottom) * h)
        x0 = max(0, min(w - 1, x0))
        x1 = max(1, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(1, min(h, y1))
        return x0, y0, x1, y1

    def _extract_path_ordered(self, frame_bgr: np.ndarray, frog_xy, debug=False):
        h, w = frame_bgr.shape[:2]

        x0, y0, x1, y1 = self._get_game_roi(h, w)
        roi = frame_bgr[y0:y1, x0:x1]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 1.2)

        # 1) Edges on ROI
        edges = cv2.Canny(gray_blur, self.canny_t1, self.canny_t2)

        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

        # 2) Pick best component
        track_mask_roi = self._pick_best_path_component(edges)

        # fallback to old threshold if edges failed
        if track_mask_roi is None:
            _, bw = cv2.threshold(gray_blur, self.path_dark_thresh, 255, cv2.THRESH_BINARY_INV)
            bw = self._clean_mask(bw, self.path_morph_ksize)
            track_mask_roi = self._pick_best_path_component(bw)

        if track_mask_roi is None:
            self._cached_track_mask = None
            self._cached_track_band = None
            return None, None, None

        # 3) Expand to full image coords
        track_mask = np.zeros((h, w), dtype=np.uint8)
        track_mask[y0:y1, x0:x1] = track_mask_roi

        # band for balls filtering
        d = int(self.track_band_dilate)
        d = max(3, d if d % 2 == 1 else d + 1)
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        track_band = cv2.dilate(track_mask, dil_kernel, iterations=1)

        self._cached_track_mask = track_mask
        self._cached_track_band = track_band
        self._cached_track_mask = track_mask
        self._cached_track_band = track_band

        # (اختياري) سد فجوات صغيرة على الماسك الكامل — مو ضروري للدجكسترا لأنه شغال على ROI
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        # ---- NEW (Replace skeleton with Dijkstra centerline on ROI mask) ----
        fx, fy = int(frog_xy[0]), int(frog_xy[1])
        frog_xy_roi = (fx - x0, fy - y0)

        points_roi = self._trace_track_dijkstra_centerline(track_mask_roi, frog_xy_roi)
        if points_roi is None or len(points_roi) < 30:
            return None, None, None

        # Debug windows (safe)
        if debug:
            cv2.imshow("track_mask_roi", track_mask_roi)
            cv2.imshow("track_mask", track_mask)

            dbg = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            for (px, py) in points_roi:
                if 0 <= py < dbg.shape[0] and 0 <= px < dbg.shape[1]:
                    dbg[py, px] = 255
            cv2.imshow("centerline_dijkstra", dbg)
            cv2.waitKey(1)

        # convert ROI points to full image coords
        points = [(int(px + x0), int(py + y0)) for (px, py) in points_roi]


        points = self._smooth_polyline(points, win=5)

        if len(points) > self.max_path_points:
            step = max(1, len(points) // self.max_path_points)
            points = points[::step]

        cumlen = self._cumulative_lengths(points)
        end_point = points[-1]
        return points, end_point, cumlen

    def _clean_mask(self, bw, ksize):
        k = int(ksize)
        k = max(3, k if k % 2 == 1 else k + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
        return bw

    def _pick_best_path_component(self, bw_or_edges_roi: np.ndarray):
        mask = (bw_or_edges_roi > 0).astype(np.uint8) * 255
        if cv2.countNonZero(mask) < 50:
            return None

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return None

        h, w = mask.shape[:2]
        best_label = None
        best_score = -1e18

        for lbl in range(1, num):
            x, y, ww, hh, area = stats[lbl]
            if area < 400:
                continue

            touches = 0
            if x <= 1: touches += 1
            if y <= 1: touches += 1
            if x + ww >= w - 2: touches += 1
            if y + hh >= h - 2: touches += 1

            comp = (labels == lbl).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            per = float(cv2.arcLength(cnts[0], True))

            score = per + 0.15 * float(area) - 1200.0 * float(touches)

            if score > best_score:
                best_score = score
                best_label = lbl

        if best_label is None:
            return None

        out = (labels == best_label).astype(np.uint8) * 255
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
        return out

    # ------------------------- Skeleton + Trace (IMPROVED) -------------------------
    def _morph_skeleton(self, binary_mask: np.ndarray):
        """
        Stable skeletonization without wiping 1px skeleton.
        Returns 0/255 uint8 skeleton.
        """
        img = (binary_mask > 0).astype(np.uint8) * 255

        # --- Prefer thinning (opencv-contrib) ---
        try:
            skel = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            skel = (skel > 0).astype(np.uint8) * 255

            # IMPORTANT: do NOT MORPH_OPEN here (it deletes 1px lines)
            # Optional: remove only single isolated pixels (safe)
            nb = cv2.filter2D((skel > 0).astype(np.uint8), -1, np.ones((3, 3), np.uint8))
            skel[((skel > 0) & (nb <= 2))] = 0  # keep lines/endpoints, drop lonely dots

            return skel
        except Exception:
            pass

        # --- Fallback: morphological skeleton (keep it, but also NO MORPH_OPEN) ---
        skel = np.zeros_like(img, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        work = img.copy()
        max_iters = 12000  # safer
        it = 0

        while True:
            it += 1
            if it > max_iters:
                break

            eroded = cv2.erode(work, element)
            opened = cv2.dilate(eroded, element)
            temp = cv2.subtract(work, opened)
            skel = cv2.bitwise_or(skel, temp)
            work = eroded

            if cv2.countNonZero(work) == 0:
                break

        # IMPORTANT: do NOT MORPH_OPEN here
        nb = cv2.filter2D((skel > 0).astype(np.uint8), -1, np.ones((3, 3), np.uint8))
        skel[((skel > 0) & (nb <= 2))] = 0

        return skel



    def _trace_skeleton_longest_polyline(self, skel_255: np.ndarray, frog_xy):
        skel = (skel_255 > 0).astype(np.uint8)
        ys, xs = np.where(skel > 0)
        if len(xs) < 60:
            return None

        skel_set = set(zip(xs.tolist(), ys.tolist()))
        endpoints = []
        for (x, y) in skel_set:
            n = self._count_8_neighbors(skel_set, x, y)
            if n == 1:
                endpoints.append((x, y))

        # If no clear endpoints (loop-ish), fallback to farthest pair heuristic
        if len(endpoints) < 2:
            pts = np.array(list(skel_set), dtype=np.int32)
            if len(pts) < 2:
                return None
            a = pts[0]
            d = np.sum((pts - a) ** 2, axis=1)
            b = pts[int(np.argmax(d))]
            d2 = np.sum((pts - b) ** 2, axis=1)
            c = pts[int(np.argmax(d2))]
            endpoints = [(int(b[0]), int(b[1])), (int(c[0]), int(c[1]))]

        # Choose a diameter-like longest path using 2 BFS on skeleton graph (8-neighbor)
        # Start BFS from an endpoint closest to frog (good anchor)
        fx, fy = frog_xy
        e0 = min(endpoints, key=lambda p: (p[0] - fx) ** 2 + (p[1] - fy) ** 2)

        far1, _ = self._bfs_farthest(skel_set, e0)
        far2, parent = self._bfs_farthest(skel_set, far1, return_parent=True)

        if far1 is None or far2 is None or parent is None:
            return None

        path = self._reconstruct_path(parent, far1, far2)
        if path is None or len(path) < 10:
            return None

        # Orient path: start near frog, end far (hole candidate)
        if ((path[0][0] - fx) ** 2 + (path[0][1] - fy) ** 2) > ((path[-1][0] - fx) ** 2 + (path[-1][1] - fy) ** 2):
            path = list(reversed(path))

        return [(int(x), int(y)) for (x, y) in path]

    def _bfs_farthest(self, skel_set, start, return_parent=False):
        from collections import deque
        q = deque([start])
        dist = {start: 0}
        parent = {start: None}

        far = start
        far_d = 0

        while q:
            u = q.popleft()
            du = dist[u]
            if du > far_d:
                far_d = du
                far = u
            for v in self._neighbors_8(u[0], u[1]):
                if v in skel_set and v not in dist:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)

        if return_parent:
            return far, parent
        return far, None

    def _reconstruct_path(self, parent, start, end):
        # parent from BFS rooted at start; reconstruct end->start
        if end not in parent:
            return None
        cur = end
        out = []
        while cur is not None:
            out.append(cur)
            if cur == start:
                break
            cur = parent.get(cur)
        if not out or out[-1] != start:
            return None
        out.reverse()
        return out

    def _smooth_polyline(self, pts, win=5):
        if pts is None or len(pts) < win or win < 3:
            return pts
        win = int(win)
        if win % 2 == 0:
            win += 1
        k = win // 2
        out = []
        for i in range(len(pts)):
            a = max(0, i - k)
            b = min(len(pts), i + k + 1)
            xs = [p[0] for p in pts[a:b]]
            ys = [p[1] for p in pts[a:b]]
            out.append((int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))))
        # remove consecutive duplicates
        cleaned = [out[0]]
        for p in out[1:]:
            if p != cleaned[-1]:
                cleaned.append(p)
        return cleaned

    def _neighbors_8(self, x, y):
        return [
            (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
            (x - 1, y),                 (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)
        ]

    def _count_8_neighbors(self, skel_set, x, y):
        c = 0
        for p in self._neighbors_8(x, y):
            if p in skel_set:
                c += 1
        return c

    def _cumulative_lengths(self, points):
        n = len(points)
        cum = np.zeros((n,), dtype=np.float32)
        total = 0.0
        for i in range(1, n):
            dx = float(points[i][0] - points[i - 1][0])
            dy = float(points[i][1] - points[i - 1][1])
            total += (dx * dx + dy * dy) ** 0.5
            cum[i] = total
        return cum

    def _nearest_path_index(self, path_arr: np.ndarray, x: int, y: int):
        dx = path_arr[:, 0].astype(np.int32) - int(x)
        dy = path_arr[:, 1].astype(np.int32) - int(y)
        d2 = dx * dx + dy * dy
        return int(np.argmin(d2))

    # ------------------------- Debug Drawing -------------------------
    def _draw_debug(self, frame_bgr, perception_output):
        vis = frame_bgr.copy()

        pts = perception_output["path"]["points"]
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                cv2.line(vis, pts[i - 1], pts[i], (255, 255, 255), 1)

        ex, ey = perception_output["path"]["end_point"]
        cv2.circle(vis, (ex, ey), 6, (0, 0, 255), -1)

        fx = perception_output["frog"]["x"]
        fy = perception_output["frog"]["y"]
        cv2.circle(vis, (fx, fy), 8, (0, 255, 0), 2)

        # NEW: frog ball draw
        fb = perception_output.get("frog_ball", None)
        if fb is not None:
            cv2.circle(vis, (int(fb["x"]), int(fb["y"])), int(fb["r"]), (255, 255, 0), 2)
            cv2.putText(
                vis,
                f'frog_ball: {fb["color"]}',
                (int(fb["x"]) - 60, int(fb["y"]) - int(fb["r"]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        if self._direction_locked:
            txt = "DIR: LOCKED (reversed)" if self._path_should_be_reversed else "DIR: LOCKED"
        else:
            txt = f"DIR: WARMUP v={self._dir_vote_accum} c={self._dir_vote_count}"
        cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        for b in perception_output["balls"]:
            x, y, r = b["x"], b["y"], b["r"]
            danger = b["danger"]
            cv2.circle(vis, (x, y), r, (0, 0, 255) if danger else (0, 255, 0), 2)
            dist = b["distance_to_end"]
            dist_txt = "inf" if not np.isfinite(dist) else str(int(dist))
            cv2.putText(
                vis,
                f'{b["color"]} o={b["order"]} d={dist_txt}',
                (x - 50, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return vis
        # ------------------------- Windows + MSS Stream -------------------------
    def _enable_dpi_awareness_windows(self):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    def find_game_window(self, title_keywords=("zuma",), exact_title=None, bring_to_front=True):
        import win32gui
        import win32con

        matches = []

        def enum_handler(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd) or ""
            if not title.strip():
                return

            t = title.lower()
            if exact_title is not None:
                if t == exact_title.lower():
                    matches.append((hwnd, title))
            else:
                for kw in title_keywords:
                    if kw.lower() in t:
                        matches.append((hwnd, title))
                        break

        win32gui.EnumWindows(enum_handler, None)
        if not matches:
            return None

        hwnd = matches[0][0]

        if bring_to_front:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass

        return hwnd

    def get_window_bbox(self, hwnd, client_only=True):
        import win32gui

        if hwnd is None or not win32gui.IsWindow(hwnd):
            return None

        try:
            if client_only:
                left, top, right, bottom = win32gui.GetClientRect(hwnd)
                (x0, y0) = win32gui.ClientToScreen(hwnd, (left, top))
                (x1, y1) = win32gui.ClientToScreen(hwnd, (right, bottom))
            else:
                x0, y0, x1, y1 = win32gui.GetWindowRect(hwnd)

            width = int(x1 - x0)
            height = int(y1 - y0)
            if width <= 0 or height <= 0:
                return None

            return int(x0), int(y0), int(width), int(height)
        except Exception:
            return None

    def run_game_stream_win32_mss(
        self,
        title_keywords=("zuma",),
        exact_title=None,
        client_only=True,
        show_debug=True,
        max_fps=60,
        auto_retry=True
    ):
        import time
        import mss

        self._enable_dpi_awareness_windows()

        sct = mss.mss()
        hwnd = None
        bbox = None

        while True:
            if hwnd is None or bbox is None:
                hwnd = self.find_game_window(title_keywords=title_keywords, exact_title=exact_title, bring_to_front=True)
                if hwnd is None:
                    if not auto_retry:
                        raise RuntimeError("Zuma window not found. Make sure the game is open.")
                    print("Waiting for Zuma window...")
                    time.sleep(0.5)
                    continue

                bbox = self.get_window_bbox(hwnd, client_only=client_only)
                if bbox is None:
                    hwnd = None
                    time.sleep(0.2)
                    continue

            left, top, width, height = bbox
            monitor = {"left": left, "top": top, "width": width, "height": height}

            img = np.array(sct.grab(monitor))  # BGRA
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if show_debug:
                output, vis = self.process_frame(frame_bgr, debug=True)
                cv2.imshow("Perception Debug (Zuma Window)", vis)
            else:
                output = self.process_frame(frame_bgr, debug=False)

            yield output

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            if max_fps and max_fps > 0:
                time.sleep(1.0 / float(max_fps))

            # refresh bbox occasionally (window moved/resized)
            if self._frame_id % 30 == 0:
                new_bbox = self.get_window_bbox(hwnd, client_only=client_only)
                if new_bbox is None:
                    hwnd, bbox = None, None
                else:
                    bbox = new_bbox

        cv2.destroyAllWindows()
    
    def _nearest_mask_point(self, mask01: np.ndarray, x: int, y: int):
        """Find nearest ON-pixel to (x,y) in a binary mask01 (0/1)."""
        h, w = mask01.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        if mask01[y, x] > 0:
            return (x, y)

        # search outward (small radius is enough)
        max_r = 80
        for r in range(1, max_r + 1):
            x0, x1 = max(0, x - r), min(w - 1, x + r)
            y0, y1 = max(0, y - r), min(h - 1, y + r)

            # top & bottom rows
            for xx in range(x0, x1 + 1):
                if mask01[y0, xx] > 0: return (xx, y0)
                if mask01[y1, xx] > 0: return (xx, y1)

            # left & right cols
            for yy in range(y0, y1 + 1):
                if mask01[yy, x0] > 0: return (x0, yy)
                if mask01[yy, x1] > 0: return (x1, yy)

        return None


    def _dijkstra_farthest(self, passable01: np.ndarray, cost_map: np.ndarray, start, step: int = 2, return_parent=False):
        """
        Dijkstra on a downsampled grid (step pixels).
        passable01: 0/1
        cost_map: float32, lower is better (we will add cost)
        start: (x,y) in full-res ROI coords.
        """
        import heapq

        h, w = passable01.shape[:2]
        sx, sy = start

        # snap start to grid
        sx = int((sx // step) * step)
        sy = int((sy // step) * step)
        sx = int(np.clip(sx, 0, w - 1))
        sy = int(np.clip(sy, 0, h - 1))

        if passable01[sy, sx] == 0:
            ns = self._nearest_mask_point(passable01, sx, sy)
            if ns is None:
                return None, None
            sx, sy = ns
            sx = int((sx // step) * step)
            sy = int((sy // step) * step)

        INF = 1e18
        dist = {}
        parent = {} if return_parent else None

        heap = [(0.0, (sx, sy))]
        dist[(sx, sy)] = 0.0
        if return_parent:
            parent[(sx, sy)] = None

        far_node = (sx, sy)
        far_cost = 0.0

        neigh = [(-step, 0), (step, 0), (0, -step), (0, step),
                (-step, -step), (step, -step), (-step, step), (step, step)]

        while heap:
            d, (x, y) = heapq.heappop(heap)
            if d != dist.get((x, y), INF):
                continue

            if d > far_cost:
                far_cost = d
                far_node = (x, y)

            for dx, dy in neigh:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                if passable01[ny, nx] == 0:
                    continue

                # movement length
                base = 1.0 if (dx == 0 or dy == 0) else 1.4142

                # cost prefers center: cost_map is smaller at center (we'll add it)
                c = float(cost_map[ny, nx])

                nd = d + base + c
                if nd < dist.get((nx, ny), INF):
                    dist[(nx, ny)] = nd
                    if return_parent:
                        parent[(nx, ny)] = (x, y)
                    heapq.heappush(heap, (nd, (nx, ny)))

        return far_node, parent


    def _reconstruct_parent_path(self, parent, start, end):
        if parent is None or end not in parent:
            return None
        cur = end
        out = []
        while cur is not None:
            out.append(cur)
            if cur == start:
                break
            cur = parent.get(cur)
        if not out or out[-1] != start:
            return None
        out.reverse()
        return out


    def _trace_track_dijkstra_centerline(self, track_mask_roi_255: np.ndarray, frog_xy_roi):
        """
        Main replacement for skeleton: get a single ordered centerline using Dijkstra.
        track_mask_roi_255: 0/255
        frog_xy_roi: frog coords in ROI space (fx-x0, fy-y0)
        Returns list of (x,y) in ROI coords.
        """
        mask01 = (track_mask_roi_255 > 0).astype(np.uint8)
        if cv2.countNonZero(mask01) < 2000:
            return None

        # close tiny gaps (important)
        mask01 = cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

        # distance transform: bigger in center
        dt = cv2.distanceTransform(mask01, cv2.DIST_L2, 5).astype(np.float32)

        # build cost map: smaller at center, large near edges
        # (add small epsilon to avoid div0)
        eps = 1e-3
        inv = 1.0 / (dt + eps)

        # normalize to reasonable range
        inv = inv / (inv.mean() + 1e-6)
        cost_map = inv.astype(np.float32)

        fx, fy = int(frog_xy_roi[0]), int(frog_xy_roi[1])
        start = self._nearest_mask_point(mask01, fx, fy)
        if start is None:
            return None

        # 2-pass "diameter" on weighted graph (Dijkstra)
        step = 2  # 2px grid is fast & accurate enough
        far1, _ = self._dijkstra_farthest(mask01, cost_map, start, step=step, return_parent=False)
        if far1 is None:
            return None

        far2, parent = self._dijkstra_farthest(mask01, cost_map, far1, step=step, return_parent=True)
        if far2 is None or parent is None:
            return None

        path = self._reconstruct_parent_path(parent, far1, far2)
        if path is None or len(path) < 30:
            return None

        # orient path: start near frog
        if (path[0][0] - fx) ** 2 + (path[0][1] - fy) ** 2 > (path[-1][0] - fx) ** 2 + (path[-1][1] - fy) ** 2:
            path = list(reversed(path))

        return [(int(x), int(y)) for (x, y) in path]


    def select_roi_from_screen(self, monitor_index: int = 1):
        """
        Capture a screenshot of the monitor and allow manual ROI selection via cv2.selectROI.
        Returns ROI in SCREEN coordinates: (left, top, width, height).
        """
        import mss

        with mss.mss() as sct:
            monitors = sct.monitors
            if monitor_index < 1 or monitor_index >= len(monitors):
                monitor_index = 1
            mon = monitors[monitor_index]
            img = np.array(sct.grab(mon))  # BGRA
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        win = "Select ROI - Drag then ENTER | ESC to cancel"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        rect = cv2.selectROI(win, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win)

        x, y, w, h = [int(v) for v in rect]
        if w <= 0 or h <= 0:
            raise RuntimeError("ROI selection canceled/invalid. Please try again.")

        left = int(mon["left"] + x)
        top  = int(mon["top"] + y)
        self._roi = (left, top, w, h)
        return self._roi


    def grab_roi_frame(self):
        """Grab ROI frame (BGR) using MSS."""
        import mss

        if self._roi is None:
            raise RuntimeError("ROI not set. Call select_roi_from_screen() first.")
        left, top, w, h = self._roi
        with mss.mss() as sct:
            mon = {"left": left, "top": top, "width": w, "height": h}
            img = np.array(sct.grab(mon))  # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


    def run_game_stream_roi(
        self,
        monitor_index: int = 1,
        show_debug: bool = True,
        max_fps: int = 60,
        auto_reselect_key: bool = True,
    ):
        """
        Cross-platform stream using manual ROI selection.
        Keys (when show_debug=True):
        - q or ESC: quit
        - r: reselect ROI
        """
        import time

        if self._roi is None:
            self.select_roi_from_screen(monitor_index=monitor_index)

        last_time = 0.0
        while True:
            if max_fps and max_fps > 0:
                now = time.time()
                min_dt = 1.0 / float(max_fps)
                if now - last_time < min_dt:
                    time.sleep(max(0.0, min_dt - (now - last_time)))
                last_time = time.time()

            frame_bgr = self.grab_roi_frame()

            if show_debug:
                output, vis = self.process_frame(frame_bgr, debug=True)
                cv2.imshow("Perception Debug (ROI)", vis)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    break
                if auto_reselect_key and key == ord("r"):
                    self.select_roi_from_screen(monitor_index=monitor_index)
            else:
                output = self.process_frame(frame_bgr, debug=False)

            yield output

        cv2.destroyAllWindows()




if __name__ == "__main__":
    perceiver = ZumaPerception(
        danger_distance_px=220.0,

        # Ball detection (Hough)
        hough_dp=1.2,
        hough_minDist=18,
        hough_param1=140,
        hough_param2=16,
        hough_minRadius=10,
        hough_maxRadius=28,

        # Path fallback threshold params
        path_dark_thresh=145,
        path_morph_ksize=5,
        track_band_dilate=21,

        # Cache update frequency
        path_update_every=10,
        frog_update_every=10,

        # ROI crop (عدّلها حسب المتصفح/النافذة عندك)
        roi_top=0.16,
        roi_bottom=0.10,
        roi_left=0.03,
        roi_right=0.03,

        # Canny (مسار أوضح)
        canny_t1=45,
        canny_t2=140,
    )
    for out in perceiver.run_game_stream_roi(
        monitor_index=1,
        show_debug=True,
        max_fps=60,
    ):
        if perceiver._frame_id % 30 == 0:
            print(out)

    # for out in perceiver.run_game_stream_win32_mss(
    #     title_keywords=("zuma",),
    #     exact_title=None,      # إذا بدك عنوان نافذة محدد اكتب هنا
    #     client_only=True,
    #     show_debug=True,
    #     max_fps=60,
    #     auto_retry=True
    # ):
    #     # اطبع كل 30 فريم
    #     if perceiver._frame_id % 30 == 0:
    #         print(out)

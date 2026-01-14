import time
import cv2

from perception import ZumaPerception
from world_model import build_world_state, worldstate_to_dict

# اختياري: إذا بدك تربط مع decision أيضاً
try:
    from decision import choose_best_action
except Exception:
    choose_best_action = None


def main(
    monitor_index: int = 1,
    max_fps: int = 60,
    print_every: int = 30,
    show_debug: bool = True,
    use_decision: bool = True,
):
    """
    Pipeline:
      Frame -> Perception -> WorldModel -> (Decision optional)

    - show_debug=True: يعرض نافذة واحدة فيها (المسار + الكرات + frog_ball) + (قرار التصويب إن فعّلته)
    - اضغط:
        q أو ESC للخروج
        r لإعادة اختيار ROI
    """

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

        # ROI crop داخل الفريم (لتجاهل واجهة المتصفح)
        roi_top=0.16,
        roi_bottom=0.10,
        roi_left=0.03,
        roi_right=0.03,

        # Canny للمسار
        canny_t1=45,
        canny_t2=140,
    )

    # 1) اختيار ROI مرة واحدة
    perceiver.select_roi_from_screen(monitor_index=monitor_index)

    last_time = 0.0
    while True:
        # تحكم FPS
        if max_fps and max_fps > 0:
            now = time.time()
            min_dt = 1.0 / float(max_fps)
            if now - last_time < min_dt:
                time.sleep(max(0.0, min_dt - (now - last_time)))
            last_time = time.time()

        # 2) Grab frame
        frame_bgr = perceiver.grab_roi_frame()

        # 3) Perception
        if show_debug:
            perception_out, vis = perceiver.process_frame(frame_bgr, debug=True)
        else:
            perception_out = perceiver.process_frame(frame_bgr, debug=False)
            vis = None

        # 4) جهّز shooter_color من frog_ball (هذه أهم وصلة للـ world_model/decision)
        frog_ball = perception_out.get("frog_ball", None)
        shooter_color = None
        if isinstance(frog_ball, dict):
            shooter_color = frog_ball.get("color", None)

        # 5) World Model
        world = build_world_state(
            frame_index=perceiver._frame_id,
            perception_or_polyline=perception_out,
            shooter_color_override=shooter_color,   # <<<<<< وصلة مهمة
            next_color_override=None,
        )

        # اطبع world_state كل N فريم
        if print_every and (perceiver._frame_id % int(print_every) == 0):
            print(worldstate_to_dict(world))

        # 6) Decision (اختياري)
        decision_res = None
        if use_decision and (choose_best_action is not None):
            decision_res = choose_best_action(world)

            # Overlay على نافذة الديبغ
            if show_debug and vis is not None:
                if decision_res is not None and decision_res.feasible:
                    fx, fy = int(world.frog_x), int(world.frog_y)
                    tx, ty = int(round(decision_res.target_x)), int(round(decision_res.target_y))
                    cv2.line(vis, (fx, fy), (tx, ty), (0, 255, 255), 2)
                    cv2.circle(vis, (tx, ty), 6, (0, 255, 255), -1)
                    cv2.putText(
                        vis,
                        f"DECISION: idx={decision_res.insert_index} val={decision_res.value:.2f}",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                else:
                    cv2.putText(
                        vis,
                        "DECISION: None (missing shooter/path/balls?)",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

        # 7) عرض
        if show_debug and vis is not None:
            cv2.imshow("Perception + WorldModel (+Decision)", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("r"):
                perceiver.select_roi_from_screen(monitor_index=monitor_index)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(
        monitor_index=1,
        max_fps=60,
        print_every=30,
        show_debug=True,
        use_decision=True,   # غيّرها False إذا بدك فقط world_model
    )

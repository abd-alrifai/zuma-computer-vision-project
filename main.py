# main.py
import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from perception import ZumaPerception
from world_model import build_world_state, worldstate_to_dict, reset_world


def _polyline_length(points) -> float:
    if not points or len(points) < 2:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    d = pts[1:] - pts[:-1]
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def _overlay_text(img, lines, x=10, y=22, dy=18):
    for i, t in enumerate(lines):
        cv2.putText(
            img,
            t,
            (x, y + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_world_ball_ids(img, ws, limit: int = 40):
    """
    Draw ball_id and s from world_model near each detected ball.
    """
    for b in ws.balls[:limit]:
        cv2.putText(
            img,
            f"{b.ball_id}:{b.s:.0f}",
            (int(b.x) + 6, int(b.y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    """
    Pipeline جزئي (Perception -> WorldModel) على mac/any OS:

    1) افتح لعبة Zuma في المتصفح
    2) شغّل هذا الملف
    3) ستظهر صورة للشاشة وتحدد ROI يدويًا (اسحب مستطيل على نافذة اللعبة ثم ENTER)
    4) ستظهر نافذة Debug فيها الدوائر حول الكرات والمسار والضفدع
       - اضغط r لإعادة اختيار ROI
       - اضغط q أو ESC للخروج

    ملاحظة:
    - إسقاط الكرات وحساب s يتم هنا من world_model وليس من perception.
    """
    perceiver = ZumaPerception(
        danger_distance_px=220.0,
        path_dark_thresh=110,
        path_morph_ksize=5,
        path_update_every=10,
        frog_update_every=10,
        track_band_dilate=21,     # أوسع
        hough_minRadius=10,       # أوسع
        hough_maxRadius=28,       # أوسع
        hough_param2=16,          # أقل = يلقط أكثر (قد يزيد false positives)
    )
    perceiver.debug_path = True


    reset_world()

    frame_index = 0
    last_print = 0
    PRINT_EVERY = 30  # غيّرها إلى 0 إذا لا تريد طباعة

    for _out in perceiver.run_game_stream_roi(
        monitor_index=1,
        show_debug=True,
        max_fps=60,
        print_every=0,  # نخلي طباعة perceiver نفسها مطفأة
    ):
        # _out هو perception_output dict
        # world_model هو من يحسب s عبر الإسقاط على المسار
        ws = build_world_state(frame_index, _out)
        ws_dict: Dict[str, Any] = worldstate_to_dict(ws)

        # -------- Diagnostics from PERCEPTION (path points + length فقط للفحص)
        path = (_out.get("path") or {})
        pts = path.get("points") or []
        try:
            pts_f = [(float(p[0]), float(p[1])) for p in pts]
        except Exception:
            pts_f = []
        p_len = _polyline_length(pts_f)

        # -------- Terminal print (اختياري)
        if PRINT_EVERY and (frame_index - last_print) >= PRINT_EVERY:
            last_print = frame_index
            print(ws_dict)
            if ws.num_balls > 0:
                s_list = [b.s for b in ws.balls]
                monotonic = all(s_list[i] <= s_list[i + 1] for i in range(len(s_list) - 1))
                print(f"  s_monotonic={monotonic}  s_range=({min(s_list):.1f}..{max(s_list):.1f})  path_len_px={p_len:.1f}")
            else:
                print(f"  (no balls) path_len_px={p_len:.1f}")

        # -------- Overlay على نافذة debug (ملاحظة: نافذة debug تُدار داخل perceiver)
        # نحن لا نملك handle مباشر للصورة التي يعرضها perceiver.
        # لذلك نضيف overlay عبر نافذة إضافية خفيفة من عندنا.
        # (هذه الطريقة لا تغيّر debug الأصلي لكنها تُظهر world_model بوضوح)

        # اصنع لوحة صغيرة لعرض النصوص فقط (HUD)
        hud = np.zeros((180, 520, 3), dtype=np.uint8)
        lines = [
            f"frame: {ws_dict.get('frame_index')}",
            f"balls(world): {ws_dict.get('num_balls')}  median_r: {ws_dict.get('median_radius'):.1f}",
            f"spacing(world): {ws_dict.get('median_spacing'):.2f}  vel: {ws_dict.get('chain_velocity'):.3f}",
            f"path_pts(world): {ws_dict.get('path_num_points')}  path_len(perception): {p_len:.1f}px",
            f"shooter(world): {ws_dict.get('shooter_color')}  next: {ws_dict.get('next_color')}",
            "Tip: if path_pts(world)=2 => perception path failed (fallback)",
        ]
        _overlay_text(hud, lines)

        # نعرض الـ HUD
        cv2.imshow("WorldModel HUD", hud)

        # التحكم بالمفاتيح (q/ESC للخروج)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        frame_index += 1

    cv2.destroyAllWindows()
    print("Stopped.")

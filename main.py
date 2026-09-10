"""Air Sketch Studio: draw, choose colors, save, and crop with hand gestures."""

import os
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from gestures import HandDetector


OUTPUT_DIRECTORY = "saved_sketches"
BRUSH_THICKNESS = 8
ACTION_COOLDOWN_SECONDS = 1.0
WINDOW_NAME = "Air Sketch Studio"


def save_image(image: np.ndarray, prefix: str) -> str:
    """Save an image with a timestamp and return its path."""
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    path = os.path.join(OUTPUT_DIRECTORY, f"{prefix}_{int(time.time())}.png")
    cv2.imwrite(path, image)
    return path


def overlay_canvas(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Blend only painted pixels so an empty canvas leaves the camera visible."""
    painted = np.any(canvas != 0, axis=2)
    blended = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    return np.where(painted[:, :, None], blended, frame)


def draw_corner_box(
    frame: np.ndarray, left_tip: Tuple[int, int], right_tip: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Draw and return the normalized crop rectangle from two fingertips."""
    min_x = min(left_tip[0], right_tip[0])
    min_y = min(left_tip[1], right_tip[1])
    max_x = max(left_tip[0], right_tip[0])
    max_y = max(left_tip[1], right_tip[1])

    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 255, 255), 2)
    for point in (left_tip, right_tip):
        cv2.circle(frame, point, 10, (0, 255, 255), -1)
    cv2.putText(
        frame,
        "CROP READY",
        (min_x, max(25, min_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    return min_x, min_y, max_x, max_y


def main() -> None:
    """Run the real-time Air Sketch Studio webcam application."""
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and index 0.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    detector = HandDetector()
    canvas: Optional[np.ndarray] = None
    previous_point: Optional[Tuple[int, int]] = None
    last_action_time: Dict[str, float] = {}
    brush_color = (255, 0, 0)
    color_names = {1: ("BLUE", (255, 0, 0)), 2: ("GREEN", (0, 255, 0)), 3: ("RED", (0, 0, 255)), 4: ("YELLOW", (0, 255, 255))}

    def action_allowed(action: str) -> bool:
        now = time.monotonic()
        if now - last_action_time.get(action, 0.0) < ACTION_COOLDOWN_SECONDS:
            return False
        last_action_time[action] = now
        return True

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            if canvas is None or canvas.shape != frame.shape:
                canvas = np.zeros_like(frame)

            results = detector.find_hands(frame, draw=True)
            hands: Dict[str, Tuple[list, Tuple[int, int]]] = {}
            if results.hand_landmarks and results.handedness:
                for landmarks, handedness in zip(
                    results.hand_landmarks, results.handedness
                ):
                    label = handedness[0].category_name
                    details = detector.get_hand_details(landmarks, label)
                    tip = detector.get_index_tip(landmarks, width, height)
                    hands[label] = (details, tip)

            right_hand = hands.get("Right")
            left_hand = hands.get("Left")
            crop_box = None

            if left_hand:
                left_fingers, _ = left_hand
                finger_count = sum(left_fingers)
                if finger_count in color_names:
                    color_name, brush_color = color_names[finger_count]
                    cv2.putText(frame, f"COLOR: {color_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, brush_color, 2)

            if right_hand:
                right_fingers, right_tip = right_hand
                if right_fingers == [0, 1, 0, 0, 0]:
                    if previous_point is not None:
                        cv2.line(canvas, previous_point, right_tip, brush_color, BRUSH_THICKNESS, cv2.LINE_AA)
                    previous_point = right_tip
                else:
                    previous_point = None

                if right_fingers == [1, 0, 0, 0, 0] and action_allowed("save"):
                    save_image(canvas, "sketch")
                if right_fingers == [1, 1, 1, 1, 1] and action_allowed("clear"):
                    canvas[:] = 0

            if left_hand and right_hand:
                left_fingers, left_tip = left_hand
                right_fingers, right_tip = right_hand
                if left_fingers[1] and right_fingers[1]:
                    crop_box = draw_corner_box(frame, left_tip, right_tip)
                    if left_fingers[0] and right_fingers[0] and action_allowed("crop"):
                        min_x, min_y, max_x, max_y = crop_box
                        if max_x > min_x and max_y > min_y:
                            save_image(canvas[min_y:max_y, min_x:max_x], "crop")

            display = overlay_canvas(frame, canvas)
            cv2.putText(display, "Index: draw | Thumb-up: save | Open palm: clear | Q: quit", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

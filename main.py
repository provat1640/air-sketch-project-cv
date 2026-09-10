"""Air Sketch Studio: draw, choose colors, save, and crop with hand gestures."""

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from gestures import HandDetector


OUTPUT_DIRECTORY = "saved_sketches"
BRUSH_THICKNESS = 8
ACTION_COOLDOWN_SECONDS = 1.0
WINDOW_NAME = "Air Sketch Studio"
TOOLBAR_HEIGHT = 112
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_READ_RETRIES = 30
Color = Tuple[int, int, int]
Rect = Tuple[int, int, int, int]


def point_in_rect(point: Tuple[int, int], rect: Rect) -> bool:
    """Return whether a fingertip is inside a toolbar cell."""
    x, y = point
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom


def is_pinching(fingers: list) -> bool:
    """Use thumb plus index with other fingers down as a deliberate click."""
    return fingers[0] == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0


def draw_toolbar(
    frame: np.ndarray,
    selected_tool: str,
    selected_color: Color,
    brush_size: int,
    hovered_cell: Optional[str],
) -> Dict[str, Rect]:
    """Draw a compact finger-grid toolbar and return its clickable cells."""
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, TOOLBAR_HEIGHT), (24, 28, 36), -1)
    cv2.putText(frame, "AIR SKETCH  |  LEFT PINCH TO SELECT", (18, 27), cv2.FONT_HERSHEY_DUPLEX, 0.58, (235, 240, 245), 1)
    cv2.putText(frame, "RIGHT INDEX DRAW  |  THUMB SAVE  |  OPEN PALM CLEAR", (18, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (155, 170, 185), 1)

    cells: Dict[str, Rect] = {}
    palette = [("blue", "BLUE", (255, 0, 0)), ("green", "GREEN", (0, 255, 0)), ("red", "RED", (0, 0, 255)), ("yellow", "YELLOW", (0, 255, 255)), ("eraser", "ERASER", (175, 180, 185))]
    cell_width = 94
    start_x = max(255, width - cell_width * len(palette) - 18)
    for index, (key, label, color) in enumerate(palette):
        left = start_x + index * cell_width
        rect = (left + 3, 8, left + cell_width - 5, 50)
        cells[key] = rect
        active = key == selected_tool or color == selected_color
        border = (255, 255, 255) if active or key == hovered_cell else (75, 85, 98)
        cv2.rectangle(frame, rect[:2], rect[2:], color, -1)
        cv2.rectangle(frame, rect[:2], rect[2:], border, 2 if active or key == hovered_cell else 1)
        cv2.putText(frame, label, (left + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (15, 20, 25), 1)

    for index, (key, size, label) in enumerate((("thin", 5, "THIN"), ("medium", 10, "MED"), ("thick", 18, "THICK"))):
        left = 18 + index * 82
        rect = (left, 65, left + 72, 101)
        cells[key] = rect
        border = (255, 255, 255) if size == brush_size or key == hovered_cell else (75, 85, 98)
        cv2.rectangle(frame, rect[:2], rect[2:], (48, 56, 68), -1)
        cv2.rectangle(frame, rect[:2], rect[2:], border, 2 if size == brush_size or key == hovered_cell else 1)
        cv2.circle(frame, (left + 17, 83), max(2, size // 3), selected_color, -1)
        cv2.putText(frame, label, (left + 29, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 225, 230), 1)

    for index, (key, label) in enumerate((("undo", "UNDO"), ("redo", "REDO"), ("save", "SAVE"), ("clear", "CLEAR"))):
        left = width - 272 + index * 66
        rect = (left, 65, left + 58, 101)
        cells[key] = rect
        border = (255, 255, 255) if key == hovered_cell else (75, 85, 98)
        cv2.rectangle(frame, rect[:2], rect[2:], (48, 56, 68), -1)
        cv2.rectangle(frame, rect[:2], rect[2:], border, 2 if key == hovered_cell else 1)
        cv2.putText(frame, label, (left + 5, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 225, 230), 1)
    return cells


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


def open_camera() -> cv2.VideoCapture:
    """Open a responsive Windows webcam, preferring DirectShow."""
    backends = (cv2.CAP_DSHOW, cv2.CAP_ANY) if os.name == "nt" else (cv2.CAP_ANY,)
    for backend in backends:
        camera = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not camera.isOpened():
            camera.release()
            continue
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera
    raise RuntimeError(
        "Could not open a webcam. Close other camera apps and check Windows camera permissions."
    )


def main() -> None:
    """Run the real-time Air Sketch Studio webcam application."""
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    camera = open_camera()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    detector = HandDetector()
    canvas: Optional[np.ndarray] = None
    previous_point: Optional[Tuple[int, int]] = None
    history: List[np.ndarray] = []
    history_index = -1
    last_action_time: Dict[str, float] = {}
    brush_color = (255, 0, 0)
    selected_tool = "blue"
    brush_size = BRUSH_THICKNESS
    hovered_cell: Optional[str] = None
    color_names = {1: ("BLUE", (255, 0, 0)), 2: ("GREEN", (0, 255, 0)), 3: ("RED", (0, 0, 255)), 4: ("YELLOW", (0, 255, 255))}

    def action_allowed(action: str) -> bool:
        now = time.monotonic()
        if now - last_action_time.get(action, 0.0) < ACTION_COOLDOWN_SECONDS:
            return False
        last_action_time[action] = now
        return True

    def record_canvas() -> None:
        nonlocal history, history_index
        assert canvas is not None
        history = history[: history_index + 1]
        history.append(canvas.copy())
        history_index = len(history) - 1

    def restore_canvas(index: int) -> None:
        nonlocal history_index
        assert canvas is not None
        if history and 0 <= index < len(history):
            canvas[:] = history[index]
            history_index = index

    failed_reads = 0
    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                failed_reads += 1
                if failed_reads < CAMERA_READ_RETRIES:
                    continue
                raise RuntimeError(
                    "The webcam opened but returned no frames. Check camera permissions, privacy covers, and other camera apps."
                )
            failed_reads = 0

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            if canvas is None or canvas.shape != frame.shape:
                canvas = np.zeros_like(frame)
                history = [canvas.copy()]
                history_index = 0

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

            toolbar_cells = draw_toolbar(frame, selected_tool, brush_color, brush_size, hovered_cell)
            right_hand = hands.get("Right")
            left_hand = hands.get("Left")
            crop_box = None
            hovered_cell = None

            if left_hand:
                left_fingers, left_tip = left_hand
                for cell, rect in toolbar_cells.items():
                    if point_in_rect(left_tip, rect):
                        hovered_cell = cell
                        cv2.circle(frame, left_tip, 12, (255, 255, 255), 2)
                        if is_pinching(left_fingers) and action_allowed(f"select_{cell}"):
                            if cell in ("blue", "green", "red", "yellow"):
                                brush_color = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255)}[cell]
                                selected_tool = cell
                            elif cell == "eraser":
                                selected_tool = cell
                            elif cell in ("thin", "medium", "thick"):
                                brush_size = {"thin": 5, "medium": 10, "thick": 18}[cell]
                            elif cell == "clear":
                                canvas[:] = 0
                                record_canvas()
                            elif cell == "undo":
                                restore_canvas(history_index - 1)
                            elif cell == "redo":
                                restore_canvas(history_index + 1)
                            elif cell == "save":
                                save_image(canvas, "sketch")

                finger_count = sum(left_fingers)
                if finger_count in color_names and not hovered_cell:
                    color_name, brush_color = color_names[finger_count]
                    selected_tool = color_name.lower()
                    cv2.putText(frame, f"COLOR: {color_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, brush_color, 2)

            if right_hand:
                right_fingers, right_tip = right_hand
                if right_fingers == [0, 1, 0, 0, 0]:
                    if previous_point is not None and right_tip[1] > TOOLBAR_HEIGHT:
                        color = (0, 0, 0) if selected_tool == "eraser" else brush_color
                        thickness = brush_size * 2 if selected_tool == "eraser" else brush_size
                        cv2.line(canvas, previous_point, right_tip, color, thickness, cv2.LINE_AA)
                    previous_point = right_tip
                else:
                    if previous_point is not None:
                        record_canvas()
                    previous_point = None

                if right_fingers == [1, 0, 0, 0, 0] and action_allowed("save"):
                    save_image(canvas, "sketch")
                if right_fingers == [1, 1, 1, 1, 1] and action_allowed("clear"):
                    canvas[:] = 0
                    record_canvas()
                if right_fingers == [0, 1, 1, 0, 0] and action_allowed("undo"):
                    restore_canvas(history_index - 1)

            if left_hand and right_hand:
                left_fingers, left_tip = left_hand
                right_fingers, right_tip = right_hand
                if left_fingers[1] and right_fingers[1] and left_tip[1] > TOOLBAR_HEIGHT and right_tip[1] > TOOLBAR_HEIGHT:
                    crop_box = draw_corner_box(frame, left_tip, right_tip)
                    if left_fingers[0] and right_fingers[0] and action_allowed("crop"):
                        min_x, min_y, max_x, max_y = crop_box
                        if max_x > min_x and max_y > min_y:
                            save_image(canvas[min_y:max_y, min_x:max_x], "crop")

            display = overlay_canvas(frame, canvas)
            cv2.putText(display, "Index: draw | Left pinch: grid | Two fingers: undo | Q: quit", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
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

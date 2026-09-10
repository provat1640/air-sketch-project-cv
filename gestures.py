"""MediaPipe Tasks hand tracking and simple finger-state detection."""

from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    """Wrap MediaPipe Tasks and expose landmarks in a small application API."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        model_path: str | None = None,
    ) -> None:
        resolved_model_path = Path(model_path or Path(__file__).parent / "assets" / "hand_landmarker.task")
        if not resolved_model_path.is_file():
            raise FileNotFoundError(f"Hand landmarker model not found: {resolved_model_path}")

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def find_hands(self, frame, draw: bool = True):
        """Process a BGR frame and return MediaPipe Tasks hand results."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._timestamp_ms += 1
        results = self.hands.detect_for_video(image, self._timestamp_ms)

        if draw:
            for hand_landmarks in results.hand_landmarks:
                points = [
                    (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                    for landmark in hand_landmarks
                ]
                for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    cv2.line(frame, points[connection.start], points[connection.end], (0, 255, 0), 2)
                for point in points:
                    cv2.circle(frame, point, 3, (0, 0, 255), -1)
        return results

    def get_hand_details(self, hand_landmarks, hand_label: str) -> List[int]:
        """Return extended state for thumb, index, middle, ring, and pinky."""
        landmarks = hand_landmarks
        fingers = [0, 0, 0, 0, 0]

        # In MediaPipe's image coordinates, each thumb points outward on a
        # different side of the hand, so the X comparison depends on handedness.
        if hand_label.lower() == "right":
            fingers[0] = int(landmarks[4].x < landmarks[3].x)
        else:
            fingers[0] = int(landmarks[4].x > landmarks[3].x)

        finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
        for position, (tip, pip) in enumerate(finger_pairs, start=1):
            fingers[position] = int(landmarks[tip].y < landmarks[pip].y)

        return fingers

    def get_index_tip(
        self, hand_landmarks, width: int, height: int
    ) -> Tuple[int, int]:
        """Convert landmark 8, the index fingertip, into pixel coordinates."""
        index_tip = hand_landmarks[8]
        x = max(0, min(width - 1, int(index_tip.x * width)))
        y = max(0, min(height - 1, int(index_tip.y * height)))
        return x, y

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()

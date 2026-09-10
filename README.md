# Air Sketch Studio

Air Sketch Studio is a real-time computer-vision drawing application. Use hand gestures in front of a webcam to draw, change brush colors, save sketches, clear the canvas, and crop selected areas.

## Setup

1. Create and activate a Python virtual environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   python main.py
   ```

The webcam and the included `assets/hand_landmarker.task` model are required. Press `Q` to quit.

## Gestures

- Right index finger: draw
- Right thumb up: save the sketch
- Right open palm: clear the canvas
- Left hand with 1-4 fingers: choose blue, green, red, or yellow
- Both index fingers: select a crop rectangle
- Both thumbs and index fingers: save the selected crop
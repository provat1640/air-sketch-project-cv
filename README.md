# Air Sketch Studio

Air Sketch Studio is a real-time computer-vision drawing application. Use hand gestures in front of a webcam to draw, change brush colors, choose brush sizes, erase, undo, redo, save sketches, clear the canvas, and crop selected areas.

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

The webcam and the included `assets/hand_landmarker.task` model are required. The app requests a 1280x720, 30 FPS stream with a one-frame buffer for low-latency gesture tracking. Press `Q` to quit.

## Gestures

### Finger-grid toolbar

The toolbar is displayed across the top of the camera view. Move the left index fingertip over a cell and pinch with the left thumb to select it.

- Color cells: blue, green, red, or yellow
- `ERASER`: draw with black to remove marks
- `THIN`, `MED`, `THICK`: change brush size
- `UNDO`, `REDO`, `SAVE`, `CLEAR`: manage the current sketch

### Gestures

- Right index finger: draw
- Right thumb up: save the sketch
- Right open palm: clear the canvas
- Right index and middle fingers: undo
- Left hand with 1-4 fingers: quick-select blue, green, red, or yellow
- Both index fingers: select a crop rectangle
- Both thumbs and index fingers: save the selected crop
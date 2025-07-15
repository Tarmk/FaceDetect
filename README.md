# FaceDetect - Student Attention Detection System

A real-time computer vision system that uses MediaPipe to detect if a student is paying attention by analyzing their face and eye movements through a camera feed.

## Features

- **Real-time Face Detection**: Uses MediaPipe Face Mesh for precise facial landmark detection
- **Eye Tracking**: Tracks eye movements and calculates Eye Aspect Ratio (EAR) to determine if eyes are open
- **Head Pose Estimation**: Analyzes head orientation to determine if student is looking at the camera
- **Attention Scoring**: Provides a continuous attention score with smoothing over multiple frames
- **Visual Feedback**: 
  - Blue lines and arrows from eyes showing gaze direction
  - Green/Red bounding box around face indicating attention status
  - Real-time attention score and metrics display

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd FaceDetect
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **System is ready to run** - no additional model downloads required

## Usage

1. **Run the attention detector**:
   ```bash
   python attention_detector.py
   ```

2. **Camera will open** showing your face with:
   - Blue lines around your eyes with direction arrows
   - Green box around your face if you're paying attention
   - Red box around your face if you're not paying attention
   - Real-time metrics including attention score, EAR values, and head pose

3. **Press 'q' to quit** the application

## How It Works

### Attention Detection Algorithm

The system uses multiple factors to determine if a student is paying attention:

1. **Eye Aspect Ratio (EAR)**: Measures if eyes are open
   - Calculates the ratio of vertical to horizontal eye distances
   - EAR > 0.25 indicates eyes are open

2. **Head Pose Estimation**: Determines if head is facing forward
   - Analyzes facial landmark positions to calculate yaw and pitch
   - Head should be within certain thresholds for forward-facing position

3. **Attention Scoring**: Combines multiple factors:
   - Eyes open + head forward = 1.0 (full attention)
   - Eyes open + slight head turn = 0.7 (partial attention)
   - Eyes open only = 0.4 (minimal attention)
   - Eyes closed = 0.0 (no attention)

4. **Smoothing**: Uses rolling average over 30 frames to reduce noise

### Visual Indicators

- **Blue Eye Lines**: Show eye contours and gaze direction
- **Green Face Box**: Student is paying attention (score > 0.3)
- **Red Face Box**: Student is not paying attention (score ≤ 0.3)
- **Yellow Text**: Real-time metrics and scores

## Technical Details

- **MediaPipe Face Mesh**: 468 facial landmarks for precise face tracking
- **OpenCV**: Computer vision operations and camera interface
- **Real-time Processing**: Optimized for live video feed with minimal dependencies

## Requirements

- Python 3.8+
- Webcam or camera device
- Good lighting conditions for optimal face detection
- CPU: Modern multi-core processor recommended
- RAM: 4GB+ recommended

## Troubleshooting

- **Camera not opening**: Check if camera is being used by another application
- **Low FPS**: Reduce video resolution or use a more powerful computer
- **Inaccurate detection**: Ensure good lighting and face the camera directly
- **Installation issues**: Make sure all dependencies are installed correctly

## Future Enhancements

- Multiple student detection in classroom settings
- Integration with learning management systems
- Attention analytics and reporting
- Mobile app support
- Cloud-based processing

## License

This project is open source and available under the MIT License.
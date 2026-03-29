# Multi-Camera Bowling Setup Instructions

## Hardware Setup

### Recommended Cameras:
1. **Lane Camera** (Board/ball tracking): Logitech BRIO 4K or Sony FDR-X3000R
   - **Option A (Preferred)**: Mount 12-15 feet above lane center, pointing straight down
   - **Option B (Tripod)**: High tripod (8-12 feet) positioned 15-25 feet from foul line, angled down at lane
   - FOV: Cover full lane width (42 inches) and 40-60 feet length
   - **Tripod Setup Notes**: Use perspective correction in software; ensure camera angle provides clear view of lane markings

2. **Side Camera** (Swing analysis): GoPro HERO 12 Silver
   - Mount at bowler waist height, 10-15 feet from foul line
   - FOV: Capture full swing motion and ball release

3. **Behind Camera** (Approach tracking): GoPro HERO 12 Silver or IP Camera
   - Mount behind bowler, 15-20 feet back
   - FOV: Capture approach steps and stance

### Logitech BRIO 4K Tripod Mounting

The Logitech BRIO 4K is a **USB webcam** (not WiFi/network camera), so it requires direct USB connection to your computer:

#### USB Connection Requirements:
- **USB 3.0 cable** (included with camera, 6-10 feet recommended)
- **USB 3.0 port** on computer (blue-colored ports)
- **USB extension cable** if needed for distance (up to 50 feet with active extension)
- **Powered USB hub** if using multiple USB devices

#### Finding USB Camera Index:
Run the detection script to find your BRIO's device index:
```bash
python detect_usb_cameras.py
```
This will show available cameras and let you test them. Choose between:
- **Visual test**: Shows live video (may have key detection issues on Windows)
- **Simple test**: Console-based testing (more reliable)

**Note**: On Windows, OpenCV windows sometimes don't respond to key presses. If 'q' doesn't quit the visual test, use ESC key or switch to simple test mode.

#### Testing BRIO 4K with Application:
Once you know the camera index, test it with the full application processing:
```bash
python test_brio_single.py
```
This will test your BRIO 4K camera with perspective correction and application-style processing.

#### Testing Two-Camera Configurations:
When you have multiple cameras, test different setups:
```bash
python test_two_cameras.py
```
This will help you test BRIO + GoPro in different configurations (lane+behind, lane+side) and determine the best connection method for your GoPro.

#### GoPro Connection Troubleshooting:
If your GoPro isn't connecting, run the troubleshooting script:
```bash
python troubleshoot_gopro.py
```
This will test USB and RTSP connections and provide specific setup instructions for your GoPro model.

#### GoPro Webcam Setup Guide:
For detailed model-specific instructions:
```bash
python gopro_webcam_guide.py
```
Select your GoPro model to get exact setup steps.

#### Alternative GoPro Methods:
If webcam mode doesn't work:
```bash
python gopro_alternatives.py
```
Try card reader mode or alternative connection methods.

#### Cable Management for Tripod Setup:
- Use cable ties to secure USB cable along tripod legs
- Consider wireless USB extenders for longer distances
- Protect cable from foot traffic in bowling alley
- Test cable routing to ensure camera positioning flexibility

#### Computer Proximity:
- BRIO must be within USB cable distance of computer (~10-50 feet)
- Position computer near lane area or use extension solutions
- Consider laptop placement for optimal cable routing

#### Mounting Steps:
1. **Attach adapter to tripod**: Screw the adapter onto your tripod's 1/4"-20 thread
2. **Secure BRIO to adapter**: Use the BRIO's built-in monitor clip mount
3. **Position camera**: Adjust tripod height (8-12 feet) and angle camera downward
4. **Secure connections**: Ensure USB cable is safely routed and protected

```
Tripod Mounting Setup:
┌─────────────────┐
│   Logitech BRIO │
│       4K        │
└─────────────────┘
         │
    Monitor Clip
         │
┌─────────────────┐
│ Webcam Adapter  │
│  (1/4"-20 thread)│
└─────────────────┘
         │
    1/4"-20 Thread
         │
┌─────────────────┐
│     Tripod      │
│   (8-12 feet)   │
└─────────────────┘
```

#### Alternative Mounting Options:
- **Ball head tripod mount** for easier angle adjustment
- **Extension arm** if you need to position camera away from tripod center
- **Counterweight** for stability with longer extension arms

#### Stability Tips:
- Use a heavy-duty tripod (supports 10+ lbs)
- Add sandbags to tripod legs for wind resistance
- Secure USB cable with cable ties to prevent strain

### Recommended Tripod Specifications:
- **Height**: 8-12 feet adjustable
- **Load capacity**: 10+ pounds
- **Head type**: Ball head for easy angle adjustment
- **Leg type**: Quick-release levers
- **Cost**: $50-150 for suitable models

Search for: "professional camera tripod 8-12 feet" or "video tripod with ball head"

### Testing Your Mount:
1. Mount camera and check stability by gently shaking tripod
2. Verify camera angle provides clear lane view
3. Test USB connection and video feed
4. Run calibration script to ensure proper perspective correction

### Network Setup:
- Use a Gigabit Ethernet switch (PoE preferred)
- Assign static IPs to cameras: 192.168.1.101, 102, 103
- **BRIO 4K uses USB**: Connect directly to computer via USB 3.0
- Ensure cameras support RTSP streaming
- Test network bandwidth (each 1080p/30fps stream needs ~5-8 Mbps)

## Tripod Camera Setup (Alternative to Overhead Mount)

If mounting above the lane isn't feasible, a high tripod setup will work but requires additional software considerations:

### Positioning Guidelines:
- **Height**: 8-12 feet tall tripod
- **Distance**: 15-25 feet from the foul line
- **Angle**: Point camera down at 30-45 degree angle toward lane center
- **Position**: Center the camera on the lane's longitudinal axis

### Camera Requirements for Angled View:
- **Resolution**: Minimum 4K for detail at distance
- **Lens**: Wide-angle lens (FOV 90-120°) to capture full lane width
- **Focus**: Manual focus recommended for consistent lane sharpness
- **Stability**: Use a heavy-duty tripod with sandbags for wind resistance

### Software Compensation:
Since the angled view introduces perspective distortion, you'll need computer vision techniques:

1. **Camera Calibration**: Use the provided `lane_calibration.py` script to calibrate your camera
2. **Perspective Transform**: Apply homography to correct lane to rectangular view
3. **Reference Points**: Use lane markings, foul line, and pin deck as calibration points
4. **AI Training**: Train your models with angled-view data for better accuracy

### Calibration Process:
```bash
python lane_calibration.py
```
Follow the prompts to:
1. Capture a calibration frame
2. Click on lane reference points
3. Test the perspective correction

### Expected Performance:
- **Board Detection**: 85-95% accuracy (vs 95-98% with overhead)
- **Ball Tracking**: Good for trajectory but may need perspective correction
- **Processing**: Additional CPU/GPU load for perspective correction

### Synchronization:
- Enable NTP on all cameras pointing to same time server
- For hardware sync: Use Arduino with IR LEDs to trigger exposure
- Software sync: Timestamp frames and align by detected events

## Software Setup

### Install Dependencies:
```bash
pip install opencv-python numpy inference-sdk
```

### Camera Configuration:
1. Connect cameras to network
2. Access web interface to enable RTSP
3. Note stream URLs (usually rtsp://ip:554/live or similar)
4. Update CAMERA_URLS in multi_camera_capture.py

### Testing:
1. Test single camera first:
```python
import cv2
cap = cv2.VideoCapture('rtsp://192.168.1.101:554/stream')
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

2. Run multi-camera script:
```bash
python multi_camera_capture.py
```

## Integration with AI Processing

To integrate with your Roboflow workflow, modify the `process_frame_with_ai` method:

```python
def process_frame_with_ai(self, frame, camera_name, timestamp):
    # Convert frame to base64 for Roboflow
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Run inference (adapt from your existing code)
    # Note: For real-time, you might need to batch or use different endpoint

    return {
        'timestamp': timestamp,
        'frame': frame,
        'camera': camera_name,
        'detections': inference_results  # Your AI detections
    }
```

## Challenges & Solutions

- **Latency**: Use low-latency cameras and optimize network
- **Bandwidth**: Compress streams or use edge processing
- **Sync Accuracy**: Calibrate with known reference points
- **Processing Power**: Use GPU acceleration for AI inference

## Cost Breakdown
- Cameras: $400-1000
- Ethernet switch: $50-200
- Mounts/Tripods: $100-300
- Total: $550-1500

Start with 2 cameras if budget is a concern, then add the third.</content>
<parameter name="filePath">c:\Development\Bowling-MCP\backend\MULTI_CAMERA_SETUP.md
import cv2
import numpy as np

# Simple motion-based bowler event detection for behind view
# Video: C:/video/behind/bowler/GX010016.MP4

def detect_bowler_events(video_path, min_area=5000, display=True):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    frame_count = 0
    event_state = 'waiting'  # waiting, entered, paused, walking, thrown
    pause_frames = 0
    walk_frames = 0
    throw_detected = False
    ball_detected_frames = []
    events = []
    max_display_height = 900
    bowler_bbox = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        fgmask = fgbg.apply(frame)
        # Clean up mask
        fgmask = cv2.medianBlur(fgmask, 7)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = sum(cv2.contourArea(c) for c in contours)
        # Find bowler bounding box (largest contour)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x_b, y_b, w_b, h_b = cv2.boundingRect(largest)
            bowler_bbox = (x_b, y_b, w_b, h_b)
        # State machine
        if event_state == 'waiting' and motion > min_area:
            event_state = 'entered'
            events.append((frame_count, 'entered'))
            print(f'Bowler entered at frame {frame_count}')
        elif event_state == 'entered':
            if motion < min_area:
                pause_frames += 1
                if pause_frames > 10:
                    event_state = 'paused'
                    events.append((frame_count, 'paused'))
                    print(f'Bowler paused at frame {frame_count}')
            else:
                pause_frames = 0
        elif event_state == 'paused':
            if motion > min_area:
                walk_frames += 1
                if walk_frames > 5:
                    event_state = 'walking'
                    events.append((frame_count, 'walking'))
                    print(f'Bowler started walking at frame {frame_count}')
            else:
                walk_frames = 0
        elif event_state == 'walking':
            # Heuristic: sudden drop in motion = throw
            if motion < min_area:
                event_state = 'thrown'
                events.append((frame_count, 'thrown'))
                print(f'Bowler threw ball at frame {frame_count}')
                throw_detected = True
        # Ball detection only between 'paused' and 'thrown', and only to the right of bowler
        ball_circle = None
        ball_edges_visible = False
        if event_state in ['paused', 'walking'] and bowler_bbox is not None:
            x_b, y_b, w_b, h_b = bowler_bbox
            roi = frame[y_b:y_b+h_b, x_b+w_b:]
            if roi.shape[1] > 0 and roi.shape[0] > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred = cv2.medianBlur(gray, 11)
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60, param1=100, param2=30, minRadius=20, maxRadius=80)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    ball_circle = max(circles[0, :], key=lambda c: c[2])
                    x, y, r = ball_circle
                    # Map back to full frame coordinates
                    x += x_b + w_b
                    y += y_b
                    ball_detected_frames.append(frame_count)
                    # Check if ball edges are visible outside hand contour
                    hand_mask = fgmask.copy()
                    ball_left = x - r
                    ball_right = x + r
                    hand_left = None
                    hand_right = None
                    for cnt in contours:
                        xh, yh, wh, hh = cv2.boundingRect(cnt)
                        if yh < y < yh + hh:
                            if hand_left is None or xh < hand_left:
                                hand_left = xh
                            if hand_right is None or xh + wh > hand_right:
                                hand_right = xh + wh
                    if hand_left is not None and hand_right is not None:
                        if ball_left < hand_left and ball_right > hand_right:
                            ball_edges_visible = True
                    # Draw ball
                    cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                    # Draw hand bounding box
                    if hand_left is not None and hand_right is not None:
                        cv2.rectangle(frame, (hand_left, y - 40), (hand_right, y + 40), (255, 0, 0), 2)
                    cv2.putText(frame, f'Ball edges visible: {ball_edges_visible}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255) if ball_edges_visible else (0,0,255), 2)
                    if ball_edges_visible:
                        print(f'Ball edges visible at frame {frame_count}')
        if display:
            disp = frame.copy()
            h, w = disp.shape[:2]
            if h > max_display_height:
                scale = max_display_height / h
                disp = cv2.resize(disp, (int(w * scale), int(h * scale)))
            cv2.putText(disp, f'State: {event_state}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if bowler_bbox is not None:
                x_b, y_b, w_b, h_b = bowler_bbox
                cv2.rectangle(disp, (x_b, y_b), (x_b + w_b, y_b + h_b), (0,255,0), 2)
            cv2.imshow('Bowler Event Detection', disp)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        if throw_detected:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Frames where ball detected:', ball_detected_frames)
    return events

if __name__ == "__main__":
    video_path = "C:/video/behind/bowler/GX010016.MP4"
    detect_bowler_events(video_path)

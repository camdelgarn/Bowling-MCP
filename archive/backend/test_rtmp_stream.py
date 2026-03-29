#!/usr/bin/env python3
"""
Test RTMP Stream from GoPro
"""

import cv2
import time

def test_rtmp_stream(rtmp_url):
    """Test RTMP stream connection"""
    print(f"Testing RTMP stream: {rtmp_url}")

    cap = cv2.VideoCapture(rtmp_url)

    if not cap.isOpened():
        print("❌ Cannot open RTMP stream")
        print("Make sure:")
        print("1. RTMP server is running")
        print("2. GoPro is streaming to the correct URL")
        print("3. Firewall allows RTMP traffic (port 1935)")
        return False

    print("✓ RTMP stream opened successfully")

    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")

    # Test frame capture
    frame_count = 0
    start_time = time.time()

    window_name = 'GoPro RTMP Stream Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nTesting stream... Press 'q' to stop")

    while True:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Lost stream connection")
            break

        # Test for 30 seconds
        if time.time() - start_time > 30:
            break

    cap.release()
    cv2.destroyWindow(window_name)

    elapsed = time.time() - start_time
    print(f"\nTest completed:")
    print(f"  Frames received: {frame_count}")
    print(f"  Duration: {elapsed:.1f} seconds")
    print(f"  FPS: {frame_count/elapsed:.1f}")

    return frame_count > 0

if __name__ == "__main__":
    # Test common RTMP URLs
    urls_to_test = [
        "rtmp://localhost/live/stream",
        "rtmp://127.0.0.1/live/stream"
    ]

    for url in urls_to_test:
        if test_rtmp_stream(url):
            print(f"\n✓ Working RTMP URL: {url}")
            break
    else:
        print("\n❌ No RTMP streams working")
        print("Check GoPro streaming configuration")
#!/usr/bin/env python3
"""
RTMP Stream Tester
Tests if RTMP server is receiving and processing streams correctly
"""

import cv2
import time
import sys
import requests
import json
from datetime import datetime

class RTMPStreamTester:
    def __init__(self):
        self.rtmp_url = "rtmp://localhost/live/stream"
        self.hls_url = "http://localhost:8080/hls/stream.m3u8"
        self.stats_url = "http://localhost:8080/stat"

    def check_server_status(self):
        """Check if RTMP server is running"""
        try:
            response = requests.get("http://localhost:8080/", timeout=5)
            if response.status_code == 200:
                print("✓ RTMP server HTTP interface is responding")
                return True
        except requests.exceptions.RequestException:
            print("❌ RTMP server HTTP interface not responding")
            return False

    def check_rtmp_stats(self):
        """Check RTMP server statistics"""
        try:
            response = requests.get(self.stats_url, timeout=5)
            if response.status_code == 200:
                print("✓ RTMP stats page accessible")
                # Parse the XML stats (basic check)
                if "nginx_rtmp_version" in response.text:
                    print("✓ RTMP module loaded")
                if "live" in response.text:
                    print("✓ Live application configured")
                return True
        except requests.exceptions.RequestException:
            print("❌ Cannot access RTMP stats")
            return False

    def test_rtmp_stream_opencv(self):
        """Test RTMP stream using OpenCV"""
        print(f"\nTesting RTMP stream: {self.rtmp_url}")
        print("This will attempt to connect to the stream for 10 seconds...")

        cap = cv2.VideoCapture(self.rtmp_url)

        if not cap.isOpened():
            print("❌ Failed to open RTMP stream")
            print("Possible issues:")
            print("  - No device is streaming to this URL")
            print("  - GoPro not configured correctly")
            print("  - Network/firewall issues")
            return False

        print("✓ Successfully connected to RTMP stream")

        # Try to read a few frames
        frame_count = 0
        start_time = time.time()

        while time.time() - start_time < 10:  # Test for 10 seconds
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                if frame_count == 1:
                    height, width = frame.shape[:2]
                    print(f"✓ Receiving frames: {width}x{height}")
            else:
                print("⚠ No frame received, but stream is open")
                break

            time.sleep(0.1)  # Small delay

        cap.release()

        if frame_count > 0:
            fps = frame_count / (time.time() - start_time)
            print(f"✓ Stream test successful: {frame_count} frames received (~{fps:.1f} FPS)")
            return True
        else:
            print("❌ No frames received from stream")
            return False

    def test_hls_stream(self):
        """Test HLS stream availability"""
        try:
            response = requests.get(self.hls_url, timeout=5)
            if response.status_code == 200:
                print("✓ HLS playlist accessible")
                if "#EXTM3U" in response.text:
                    print("✓ Valid HLS playlist format")
                    return True
            else:
                print(f"❌ HLS playlist not accessible (HTTP {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot access HLS stream: {e}")

        return False

    def check_hls_segments(self):
        """Check if HLS segments are being created"""
        try:
            # Get the playlist
            response = requests.get(self.hls_url, timeout=5)
            if response.status_code == 200:
                lines = response.text.split('\n')
                segment_files = [line for line in lines if line.endswith('.ts')]

                if segment_files:
                    print(f"✓ HLS segments found: {len(segment_files)} segments")
                    # Try to access the first segment
                    first_segment = segment_files[0]
                    segment_url = f"http://localhost:8080/hls/{first_segment}"
                    seg_response = requests.head(segment_url, timeout=5)
                    if seg_response.status_code == 200:
                        print("✓ HLS segments are accessible")
                        return True
                    else:
                        print("❌ HLS segments not accessible")
                else:
                    print("⚠ No HLS segments found yet (stream may just be starting)")
        except Exception as e:
            print(f"❌ Error checking HLS segments: {e}")

        return False

    def run_full_test(self):
        """Run complete stream test"""
        print("RTMP Stream Connection Test")
        print("=" * 40)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Test 1: Server status
        if not self.check_server_status():
            print("\n❌ RTMP server not running. Start it with:")
            print("   python rtmp_server_manager.py start")
            return False

        # Test 2: RTMP stats
        self.check_rtmp_stats()

        # Test 3: RTMP stream
        stream_working = self.test_rtmp_stream_opencv()

        # Test 4: HLS stream
        hls_working = self.test_hls_stream()
        if hls_working:
            self.check_hls_segments()

        print("\n" + "=" * 40)
        if stream_working:
            print("🎉 SUCCESS: RTMP stream is working correctly!")
            print("Your GoPro/camera is successfully streaming to the server.")
        else:
            print("❌ FAILURE: RTMP stream is not receiving data.")
            print("\nTroubleshooting steps:")
            print("1. Check that your GoPro/camera is configured to stream to:")
            print(f"   {self.rtmp_url}")
            print("2. Verify the device is connected to the same network")
            print("3. Check GoPro RTMP settings in the app")
            print("4. Try restarting the RTMP server")
            print("5. Check firewall settings")

        return stream_working

def main():
    tester = RTMPStreamTester()

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test - just check if stream is available
        success = tester.test_rtmp_stream_opencv()
        sys.exit(0 if success else 1)
    else:
        # Full diagnostic test
        success = tester.run_full_test()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
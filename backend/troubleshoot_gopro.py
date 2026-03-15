#!/usr/bin/env python3
"""
GoPro Camera Troubleshooting Script
Helps diagnose and fix GoPro connection issues
"""

import cv2
import subprocess
import sys
import time

def test_usb_camera(index, name="Camera"):
    """Test a specific USB camera index"""
    print(f"\nTesting {name} at USB index {index}...")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"❌ Failed to open {name} at index {index}")
        return False, None

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"✓ {name} opened successfully!")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")

    # Test frame capture
    ret, frame = cap.read()
    if ret:
        print(f"  Frame capture: ✓ SUCCESS")
        # Save test frame
        filename = f'gopro_test_index_{index}.jpg'
        cv2.imwrite(filename, frame)
        print(f"  Test frame saved: {filename}")
    else:
        print(f"  Frame capture: ❌ FAILED")

    cap.release()
    return ret, (width, height, fps)

def test_rtsp_stream(url, name="RTSP Stream"):
    """Test RTSP stream connection"""
    print(f"\nTesting {name}: {url}")

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"❌ Failed to open RTSP stream: {url}")
        return False, None

    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("✓ RTSP stream opened successfully!")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")

    # Test frame capture
    ret, frame = cap.read()
    if ret:
        print("  Frame capture: ✓ SUCCESS")
        # Save test frame
        filename = f'gopro_rtsp_test.jpg'
        cv2.imwrite(filename, frame)
        print(f"  Test frame saved: {filename}")
    else:
        print("  Frame capture: ❌ FAILED")
    cap.release()
    return ret, (width, height, fps)

def scan_usb_cameras():
    """Scan for all available USB cameras"""
    print("Scanning for USB cameras...")
    print("=" * 40)

    found_cameras = []

    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            camera_info = {
                'index': i,
                'width': width,
                'height': height,
                'fps': fps,
                'name': f"Camera {i}"
            }

            # Try to identify BRIO vs GoPro based on resolution
            if width >= 3000:  # BRIO 4K
                camera_info['name'] = f"BRIO 4K (index {i})"
                camera_info['type'] = 'brio'
            else:
                camera_info['name'] = f"GoPro? (index {i})"
                camera_info['type'] = 'gopro'

            found_cameras.append(camera_info)
            cap.release()

    if not found_cameras:
        print("❌ No USB cameras found!")
        return []

    print(f"Found {len(found_cameras)} camera(s):")
    for cam in found_cameras:
        print(f"  {cam['name']}: {cam['width']}x{cam['height']} @ {cam['fps']} FPS")

    return found_cameras

def test_gopro_connection_methods():
    """Test different GoPro connection methods"""
    print("\nGoPro Connection Troubleshooting")
    print("=" * 50)

    # Method 1: USB Webcam Mode
    print("\n1. Testing USB Webcam Mode")
    print("-" * 30)
    print("Make sure GoPro is:")
    print("  - Connected via USB cable")
    print("  - Set to 'Webcam' or 'USB' mode")
    print("  - Powered on")

    usb_cameras = scan_usb_cameras()

    # Identify potential GoPro
    gopro_candidates = [cam for cam in usb_cameras if cam['type'] == 'gopro']

    if gopro_candidates:
        print(f"\nFound {len(gopro_candidates)} potential GoPro camera(s)")
        for cam in gopro_candidates:
            success, _ = test_usb_camera(cam['index'], f"GoPro (USB {cam['index']})")
            if success:
                print(f"✓ GoPro working at USB index {cam['index']}")
                return 'usb', cam['index']
    else:
        print("No GoPro detected via USB")

    # Method 2: RTSP Stream
    print("\n2. Testing RTSP Stream Mode")
    print("-" * 30)
    print("Make sure GoPro is:")
    print("  - Connected to WiFi network")
    print("  - RTSP streaming enabled")
    print("  - You know the IP address")

    # Common RTSP URLs to try
    rtsp_urls = [
        "rtsp://192.168.1.104:554/stream",
        "rtsp://192.168.1.105:554/live",
        "rtsp://10.5.5.9:554/live",  # GoPro default
        "rtsp://10.5.5.9:8554/live"  # Alternative port
    ]

    custom_url = input("Enter custom RTSP URL (or press Enter to skip): ").strip()
    if custom_url:
        rtsp_urls.insert(0, custom_url)

    for url in rtsp_urls:
        success, _ = test_rtsp_stream(url, f"GoPro RTSP ({url})")
        if success:
            print(f"✓ GoPro RTSP working: {url}")
            return 'rtsp', url

    print("❌ No working GoPro connection found")

    # Provide troubleshooting tips
    print("\n" + "=" * 60)
    print("GoPro Troubleshooting Tips")
    print("=" * 60)

    print("\nFor USB Connection:")
    print("1. Connect GoPro to computer with USB cable")
    print("2. Power on GoPro")
    print("3. On GoPro: Settings > Connections > USB > Webcam")
    print("4. Wait 10-15 seconds for driver installation")
    print("5. Run this script again")

    print("\nFor RTSP/WiFi Connection:")
    print("1. On GoPro: Settings > Connections > WiFi > On")
    print("2. Connect GoPro to your WiFi network")
    print("3. Find GoPro IP: Use GoPro app or check router")
    print("4. Enable RTSP: Some GoPros need specific apps/mods")
    print("5. Test URL in VLC: Media > Open Network Stream")

    print("\nAlternative: Use GoPro as USB webcam (most reliable)")
    print("- This works with most GoPro models")
    print("- No WiFi setup required")
    print("- Direct USB connection")

    return None, None

def main():
    print("GoPro Camera Troubleshooting Tool")
    print("Diagnose and fix GoPro connection issues")
    print("=" * 50)

    # First verify BRIO is working
    print("\nStep 1: Verify BRIO 4K is working")
    brio_index = int(input("What USB index is your BRIO 4K at? (usually 0-2): "))

    success, brio_info = test_usb_camera(brio_index, "BRIO 4K")
    if not success:
        print("❌ BRIO 4K not working. Fix BRIO first, then test GoPro.")
        print("Run: python detect_usb_cameras.py")
        return

    print(f"✓ BRIO 4K confirmed at index {brio_index}")

    # Now test GoPro
    print("\nStep 2: Test GoPro connection")
    connection_type, connection_param = test_gopro_connection_methods()

    if connection_type:
        print("\n✓ GoPro connection successful!")
        print(f"  Type: {connection_type.upper()}")
        print(f"  Connection: {connection_param}")

        print("\nUpdate your camera configuration:")
        if connection_type == 'usb':
            print(f"  'gopro': {{'type': 'usb', 'source': {connection_param}}}")
        else:
            print(f"  'gopro': {{'type': 'rtsp', 'source': '{connection_param}'}}")

        print("\nReady to test two-camera setup!")
        print("Run: python quick_two_camera_test.py")
    else:
        print("\n❌ Could not establish GoPro connection")
        print("Follow the troubleshooting tips above")

if __name__ == "__main__":
    main()
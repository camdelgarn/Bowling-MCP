#!/usr/bin/env python3
"""
Alternative GoPro Setup: Card Reader Mode
If webcam mode doesn't work, try this method
"""

import cv2
import os
import time
import subprocess

def setup_gopro_card_reader():
    """Guide for setting up GoPro in card reader mode"""
    print("GoPro Card Reader Mode Setup")
    print("=" * 40)
    print()
    print("This method works when webcam mode fails.")
    print("GoPro appears as a storage device, not a camera.")
    print()

    print("GoPro Setup Steps:")
    print("1. Power on GoPro")
    print("2. Go to Settings > Connections > USB")
    print("3. Select 'Storage' or 'MTP' (NOT Webcam)")
    print("4. Connect USB cable to computer")
    print("5. GoPro should appear as a drive (like E: or F:)")
    print()

    # Check for GoPro drive
    print("Checking for GoPro drive...")
    drives = []
    for letter in 'DEFGHIJKLMNOPQRSTUVWXYZ':
        drive = f"{letter}:"
        if os.path.exists(drive):
            try:
                # Check if it's a removable drive (likely GoPro)
                import ctypes
                drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive)
                if drive_type == 2:  # DRIVE_REMOVABLE
                    drives.append(drive)
                    print(f"Found removable drive: {drive}")
            except:
                pass

    if not drives:
        print("No removable drives found. Make sure GoPro is connected in Storage mode.")
        return None

    print(f"\nFound {len(drives)} removable drive(s)")
    if len(drives) == 1:
        gopro_drive = drives[0]
    else:
        print("Select your GoPro drive:")
        for i, drive in enumerate(drives):
            print(f"{i+1}. {drive}")
        try:
            choice = int(input("Enter drive number: ")) - 1
            gopro_drive = drives[choice]
        except:
            print("Invalid choice")
            return None

    print(f"\nUsing GoPro drive: {gopro_drive}")

    # Check for DCIM folder (standard GoPro structure)
    dcim_path = os.path.join(gopro_drive, "DCIM")
    if os.path.exists(dcim_path):
        print("✓ Found DCIM folder (GoPro confirmed)")
    else:
        print("⚠ DCIM folder not found. Is this the correct drive?")
        return None

    print("\nCard Reader Mode Limitations:")
    print("- Cannot stream live video")
    print("- Must record video to SD card first")
    print("- Then copy files for processing")
    print("- Not suitable for real-time analysis")
    print()

    print("Alternative: Use GoPro app for live streaming")
    print("1. Install GoPro app on phone")
    print("2. Connect GoPro to phone WiFi")
    print("3. Stream to computer via app")
    print()

    return gopro_drive

def test_gopro_webcam_fallback():
    """Fallback test if webcam mode should work but doesn't"""
    print("\nWebcam Mode Fallback Test")
    print("=" * 30)

    print("Trying alternative webcam detection methods...")

    # Try different backend APIs
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY)
    ]

    for backend_name, backend in backends:
        print(f"\nTesting {backend_name} backend:")
        for i in range(5):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  ✓ Camera {i}: {width}x{height} ({backend_name})")
                if width < 3000:  # Likely GoPro
                    print(f"    → This might be your GoPro at index {i}!")
                    cap.release()
                    return i, backend_name
            cap.release()

    print("No GoPro detected with alternative methods")
    return None, None

def main():
    print("GoPro Alternative Setup Methods")
    print("=" * 40)

    print("Choose setup method:")
    print("1. Card Reader mode (if webcam fails)")
    print("2. Webcam fallback test")
    print("3. Show webcam setup guide")

    choice = input("Enter choice (1-3): ").strip()

    if choice == '1':
        setup_gopro_card_reader()
    elif choice == '2':
        index, backend = test_gopro_webcam_fallback()
        if index is not None:
            print(f"\n✓ GoPro found at index {index} using {backend} backend")
            print("Update your camera config:")
            print(f"  'gopro': {{'type': 'usb', 'source': {index}}}")
        else:
            print("\n❌ GoPro not found")
    elif choice == '3':
        try:
            subprocess.run([__import__('sys').executable, "gopro_webcam_guide.py"])
        except:
            print("Run: python gopro_webcam_guide.py")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
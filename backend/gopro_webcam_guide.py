#!/usr/bin/env python3
"""
GoPro Webcam Mode Setup Guide
Step-by-step instructions for different GoPro models
"""

def show_gopro_setup_guide():
    print("GoPro Webcam Mode Setup Guide")
    print("=" * 50)
    print()

    print("Select your GoPro model:")
    print("1. HERO 7 Silver/Gold/Black")
    print("2. HERO 8 Black")
    print("3. HERO 9 Black")
    print("4. HERO 10 Black")
    print("5. HERO 11 Black")
    print("6. HERO 12 Black")
    print("7. MAX")
    print("8. General troubleshooting")

    while True:
        try:
            choice = int(input("\nEnter your GoPro model (1-8): "))
            if 1 <= choice <= 8:
                break
            else:
                print("Invalid choice. Enter 1-8.")
        except ValueError:
            print("Invalid input. Enter a number.")

    print("\n" + "=" * 60)

    if choice == 1:  # HERO 7
        print("GoPro HERO 7 Webcam Setup:")
        print("-" * 30)
        print("1. Power on your HERO 7")
        print("2. Swipe down on touchscreen > Preferences")
        print("3. Scroll to 'Connections' > USB")
        print("4. Select 'Webcam'")
        print("5. Connect USB cable to computer")
        print("6. Wait 10-15 seconds for driver installation")
        print("7. GoPro screen should show 'Webcam Connected'")

    elif choice == 2:  # HERO 8
        print("GoPro HERO 8 Webcam Setup:")
        print("-" * 30)
        print("1. Power on your HERO 8")
        print("2. Swipe down > Preferences")
        print("3. Go to 'Connections'")
        print("4. Select 'USB' > 'Webcam'")
        print("5. Connect USB-C cable to computer")
        print("6. Wait for 'Webcam' message on screen")

    elif choice == 3:  # HERO 9
        print("GoPro HERO 9 Webcam Setup:")
        print("-" * 30)
        print("1. Power on your HERO 9")
        print("2. Swipe down > Preferences")
        print("3. Go to 'Connections' > 'USB'")
        print("4. Select 'Webcam'")
        print("5. Connect USB-C cable")
        print("6. Screen shows 'Webcam Mode'")

    elif choice == 4:  # HERO 10
        print("GoPro HERO 10 Webcam Setup:")
        print("-" * 30)
        print("1. Power on your HERO 10")
        print("2. Swipe down > Preferences")
        print("3. Go to 'Connections' > 'USB'")
        print("4. Select 'Webcam'")
        print("5. Connect USB-C cable")
        print("6. Wait for confirmation")

    elif choice == 5:  # HERO 11
        print("GoPro HERO 11 Webcam Setup:")
        print("-" * 30)
        print("1. Power on your HERO 11")
        print("2. Swipe down > Preferences")
        print("3. Go to 'Connections' > 'USB'")
        print("4. Select 'Webcam'")
        print("5. Connect USB-C cable")
        print("6. Enable webcam mode")

    elif choice == 6:  # HERO 12
        print("GoPro HERO 12 Webcam Setup:")
        print("-" * 30)
        print("1. Power on your HERO 12")
        print("2. Swipe down > Preferences")
        print("3. Go to 'Connections' > 'USB'")
        print("4. Select 'Webcam'")
        print("5. Connect USB-C cable")
        print("6. Confirm webcam activation")

    elif choice == 7:  # MAX
        print("GoPro MAX Webcam Setup:")
        print("-" * 30)
        print("1. Power on MAX")
        print("2. Open lens cover")
        print("3. Swipe down > Preferences > Connections")
        print("4. Select 'USB' > 'Webcam'")
        print("5. Connect USB-C cable")
        print("6. Wait for webcam mode")

    elif choice == 8:  # General troubleshooting
        print("General GoPro Webcam Troubleshooting:")
        print("-" * 40)
        print("1. Ensure GoPro is charged (battery > 20%)")
        print("2. Use original GoPro USB cable")
        print("3. Try different USB ports on computer")
        print("4. Restart both GoPro and computer")
        print("5. Update GoPro firmware via app")
        print("6. Try 'Storage' mode first, then switch to 'Webcam'")
        print("7. Check Windows Device Manager for driver issues")

    print("\n" + "=" * 60)
    print("After Setup - Test Connection:")
    print("-" * 30)
    print("1. Run: python troubleshoot_gopro.py")
    print("2. Or run: python detect_usb_cameras.py")
    print("3. Look for your GoPro in the camera list")
    print("4. Note the USB index number")

    print("\nIf webcam mode doesn't work:")
    print("- Try using GoPro as 'Card Reader' instead")
    print("- Use GoPro app for WiFi streaming")
    print("- Consider RTSP streaming (advanced)")

    print("\n" + "=" * 60)
    input("Press Enter to run camera detection...")

    # Run camera detection
    import subprocess
    try:
        subprocess.run([sys.executable, "detect_usb_cameras.py"])
    except FileNotFoundError:
        print("Run: python detect_usb_cameras.py")

if __name__ == "__main__":
    import sys
    show_gopro_setup_guide()
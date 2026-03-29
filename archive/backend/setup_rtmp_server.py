#!/usr/bin/env python3
"""
RTMP Server Setup for GoPro Streaming
Sets up a local RTMP server to receive streams from GoPro
"""

import os
import subprocess
import sys
import time
import socket

def check_nginx_rtmp():
    """Check if nginx with RTMP module is installed"""
    try:
        result = subprocess.run(['nginx', '-v'], capture_output=True, text=True)
        if 'nginx' in result.stderr.lower():
            print("✓ Nginx found")
            return True
    except FileNotFoundError:
        pass

    print("❌ Nginx with RTMP module not found")
    return False

def create_nginx_config():
    """Create nginx configuration for RTMP streaming"""
    config_content = '''# Nginx RTMP Configuration for GoPro Streaming
worker_processes 1;

events {
    worker_connections 1024;
}

rtmp {
    server {
        listen 1935;
        chunk_size 4096;

        application live {
            live on;
            record off;

            # Allow publishing from GoPro
            allow publish all;
            allow play all;
        }

        # HLS streaming (optional)
        application hls {
            live on;
            hls on;
            hls_path /tmp/hls;
            hls_fragment 3;
            hls_playlist_length 60;
        }
    }
}

http {
    server {
        listen 8080;

        location /hls {
            types {
                application/vnd.apple.mpegurl m3u8;
                video/mp2t ts;
            }
            root /tmp;
            add_header Cache-Control no-cache;
        }

        location /stat {
            rtmp_stat all;
            rtmp_stat_stylesheet stat.xsl;
        }

        location /stat.xsl {
            root /usr/local/nginx/html;
        }
    }
}
'''

    config_path = 'nginx_rtmp.conf'
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✓ Created nginx config: {config_path}")
    return config_path

def start_rtmp_server(config_path):
    """Start nginx RTMP server"""
    try:
        print("Starting RTMP server...")
        # Kill any existing nginx processes
        subprocess.run(['taskkill', '/F', '/IM', 'nginx.exe'], capture_output=True)

        # Start nginx with our config
        result = subprocess.run(['nginx', '-c', config_path], capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ RTMP server started successfully")
            print("  RTMP URL: rtmp://localhost/live/stream")
            print("  HTTP Preview: http://localhost:8080/stat")
            return True
        else:
            print(f"❌ Failed to start RTMP server: {result.stderr}")
            return False

    except FileNotFoundError:
        print("❌ Nginx executable not found in PATH")
        print("Please install nginx with RTMP module")
        return False

def test_rtmp_connection():
    """Test RTMP server connectivity"""
    print("\nTesting RTMP server...")

    # Test if port 1935 is open
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        result = sock.connect_ex(('localhost', 1935))
        if result == 0:
            print("✓ RTMP port 1935 is open")
            return True
        else:
            print("❌ RTMP port 1935 is not accessible")
            return False
    except:
        print("❌ Cannot connect to RTMP port")
        return False
    finally:
        sock.close()

def show_gopro_rtmp_setup():
    """Show GoPro RTMP setup instructions"""
    print("\n" + "=" * 60)
    print("GoPro RTMP Streaming Setup")
    print("=" * 60)

    print("Configure your GoPro for RTMP streaming:")
    print()

    print("Method 1: Using GoPro Labs (Recommended for HERO 7+)")
    print("-" * 50)
    print("1. Install 'GoPro Labs' on your phone")
    print("2. Enable RTMP in Labs settings")
    print("3. Set RTMP URL: rtmp://YOUR_COMPUTER_IP/live/stream")
    print("4. Set Stream Key: stream (or any name)")
    print("5. Start streaming from GoPro")
    print()

    print("Method 2: Using GoPro App + RTMP")
    print("-" * 50)
    print("1. Connect GoPro to WiFi")
    print("2. Use GoPro app on phone")
    print("3. Enable 'Live Streaming' in app")
    print("4. Configure RTMP settings in app")
    print()

    print("Method 3: Custom Firmware/Mods")
    print("-" * 50)
    print("Some GoPro models need custom firmware for RTMP")
    print("Check GoPro forums or use alternative streaming apps")
    print()

    # Get local IP address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"Your computer IP address: {local_ip}")
        print(f"RTMP URL for GoPro: rtmp://{local_ip}/live/stream")
    except:
        print("Could not determine local IP address")
        print("RTMP URL: rtmp://YOUR_LOCAL_IP/live/stream")

def create_rtmp_test_script():
    """Create a test script for RTMP streaming"""
    test_script = '''#!/usr/bin/env python3
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

    print("\\nTesting stream... Press 'q' to stop")

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
    print(f"\\nTest completed:")
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
            print(f"\\n✓ Working RTMP URL: {url}")
            break
    else:
        print("\\n❌ No RTMP streams working")
        print("Check GoPro streaming configuration")
'''

    with open('test_rtmp_stream.py', 'w') as f:
        f.write(test_script)

    print("✓ Created RTMP test script: test_rtmp_stream.py")

def main():
    print("GoPro RTMP Streaming Setup")
    print("=" * 40)

    # Check for nginx
    if not check_nginx_rtmp():
        print("\\nInstalling nginx with RTMP module...")
        print("Please download and install nginx with RTMP from:")
        print("https://github.com/illuspas/nginx-rtmp-win32")
        print("\\nOr use a pre-built Windows binary")
        print("Extract to a folder and add to PATH")
        return

    # Create config
    config_path = create_nginx_config()

    # Start server
    if start_rtmp_server(config_path):
        # Test connection
        if test_rtmp_connection():
            print("\\n✓ RTMP server is ready!")

            # Show GoPro setup
            show_gopro_rtmp_setup()

            # Create test script
            create_rtmp_test_script()

            print("\\n" + "=" * 60)
            print("Next Steps:")
            print("1. Configure GoPro for RTMP streaming")
            print("2. Start streaming from GoPro")
            print("3. Test stream: python test_rtmp_stream.py")
            print("4. Update your camera config:")
            print("   'gopro': {'type': 'rtsp', 'source': 'rtmp://localhost/live/stream'}")
        else:
            print("❌ RTMP server started but not accessible")
    else:
        print("❌ Failed to start RTMP server")

if __name__ == "__main__":
    main()
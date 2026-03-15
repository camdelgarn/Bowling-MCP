#!/usr/bin/env python3
"""
RTMP Server Manager
Manages nginx RTMP server as a separate service
"""

import subprocess
import sys
import os
import time
import socket
import signal

class RTMPServerManager:
    def __init__(self, config_file='nginx_rtmp.conf'):
        self.nginx_dir = os.path.join(os.path.dirname(__file__), '..', 'rtmp')
        self.nginx_exe = os.path.join(self.nginx_dir, 'nginx.exe')
        self.config_file = os.path.join(self.nginx_dir, 'nginx_rtmp.conf')

    def check_nginx_installed(self):
        """Check if nginx is installed and accessible"""
        if os.path.exists(self.nginx_exe):
            print("✓ Nginx RTMP found at", self.nginx_dir)
            return True
        else:
            print("❌ Nginx RTMP not found")
            print("Please run setup_rtmp_server.bat to install nginx")
            return False

    def is_server_running(self):
        """Check if RTMP server is running"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 1935))
            sock.close()
            return result == 0
        except:
            return False

    def start_server(self):
        """Start RTMP server"""
        if not os.path.exists(self.config_file):
            print(f"❌ Config file not found: {self.config_file}")
            return False

        if self.is_server_running():
            print("✓ RTMP server already running")
            return True

        try:
            print("Starting RTMP server...")
            # Run nginx from its directory
            self.nginx_process = subprocess.Popen(
                [self.nginx_exe, '-c', 'nginx_rtmp.conf'],
                cwd=self.nginx_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait a bit for server to start
            time.sleep(2)

            if self.is_server_running():
                print("✓ RTMP server started successfully")
                print("  RTMP URL: rtmp://localhost/live/stream")
                print("  Web Stats: http://localhost:8080/stat")
                print("  HLS Stream: http://localhost:8080/hls/stream.m3u8")
                return True
            else:
                print("❌ Server failed to start")
                # Get error output
                if self.nginx_process:
                    stdout, stderr = self.nginx_process.communicate(timeout=5)
                    if stderr:
                        print(f"Error: {stderr.decode()}")
                return False

        except FileNotFoundError:
            print("❌ nginx executable not found")
            return False
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False

    def stop_server(self):
        """Stop RTMP server"""
        if self.nginx_process:
            try:
                self.nginx_process.terminate()
                self.nginx_process.wait(timeout=5)
                print("✓ RTMP server stopped")
                return True
            except subprocess.TimeoutExpired:
                self.nginx_process.kill()
                print("✓ RTMP server force stopped")
                return True
        else:
            # Try to kill nginx processes
            try:
                subprocess.run(['taskkill', '/F', '/IM', 'nginx.exe'],
                             capture_output=True)
                print("✓ RTMP server stopped (via taskkill)")
                return True
            except:
                print("❌ Could not stop RTMP server")
                return False

    def restart_server(self):
        """Restart RTMP server"""
        print("Restarting RTMP server...")
        self.stop_server()
        time.sleep(1)
        return self.start_server()

    def show_status(self):
        """Show server status"""
        running = self.is_server_running()
        status = "RUNNING" if running else "STOPPED"
        print(f"RTMP Server Status: {status}")

        if running:
            print("  RTMP URL: rtmp://localhost/live/stream")
            print("  Web Stats: http://localhost:8080/stat")
            print("  HLS Stream: http://localhost:8080/hls/stream.m3u8")

            # Get local IP
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                print(f"  Network URL: rtmp://{local_ip}/live/stream")
            except:
                pass

def main():
    manager = RTMPServerManager()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'start':
            if not manager.check_nginx_installed():
                sys.exit(1)
            success = manager.start_server()
            sys.exit(0 if success else 1)

        elif command == 'stop':
            success = manager.stop_server()
            sys.exit(0 if success else 1)

        elif command == 'restart':
            success = manager.restart_server()
            sys.exit(0 if success else 1)

        elif command == 'status':
            manager.show_status()
            sys.exit(0)

    # Interactive mode
    print("RTMP Server Manager")
    print("=" * 30)

    if not manager.check_nginx_installed():
        print("Please install nginx first")
        sys.exit(1)

    while True:
        print("\nCommands:")
        print("1. Start server")
        print("2. Stop server")
        print("3. Restart server")
        print("4. Show status")
        print("5. Exit")

        choice = input("Enter choice (1-5): ").strip()

        if choice == '1':
            manager.start_server()
        elif choice == '2':
            manager.stop_server()
        elif choice == '3':
            manager.restart_server()
        elif choice == '4':
            manager.show_status()
        elif choice == '5':
            if manager.is_server_running():
                print("Stopping server...")
                manager.stop_server()
            print("Goodbye!")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
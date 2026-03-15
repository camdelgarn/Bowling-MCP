#!/usr/bin/env python3
"""
Quick RTMP Stream Monitor
Monitors RTMP server stats and checks for active streams
"""

import requests
import time
import xml.etree.ElementTree as ET
from datetime import datetime

def check_rtmp_stats():
    """Check RTMP server statistics"""
    try:
        response = requests.get("http://localhost:8080/stat", timeout=5)
        if response.status_code == 200:
            return response.text
    except:
        return None

def parse_stats(xml_content):
    """Parse RTMP stats XML"""
    try:
        root = ET.fromstring(xml_content)

        # Get server info
        uptime = root.find('.//uptime')
        naccepted = root.find('.//naccepted')

        # Get live application stats
        live_app = root.find('.//application[name="live"]')
        if live_app is not None:
            live_clients = live_app.find('.//nclients')
            live_clients_count = int(live_clients.text) if live_clients is not None else 0
        else:
            live_clients_count = 0

        # Get HLS application stats
        hls_app = root.find('.//application[name="hls"]')
        if hls_app is not None:
            hls_clients = hls_app.find('.//nclients')
            hls_clients_count = int(hls_clients.text) if hls_clients is not None else 0
        else:
            hls_clients_count = 0

        return {
            'uptime': int(uptime.text) if uptime is not None else 0,
            'connections': int(naccepted.text) if naccepted is not None else 0,
            'live_clients': live_clients_count,
            'hls_clients': hls_clients_count
        }
    except:
        return None

def format_uptime(seconds):
    """Format uptime in human readable format"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    print("RTMP Stream Monitor")
    print("===================")
    print("Monitoring RTMP server for active streams...")
    print("Press Ctrl+C to stop")
    print()

    last_live_clients = 0
    last_hls_clients = 0

    try:
        while True:
            stats_xml = check_rtmp_stats()
            if stats_xml:
                stats = parse_stats(stats_xml)
                if stats:
                    current_time = datetime.now().strftime("%H:%M:%S")

                    # Check for changes
                    live_changed = stats['live_clients'] != last_live_clients
                    hls_changed = stats['hls_clients'] != last_hls_clients

                    status_line = f"[{current_time}] Uptime: {format_uptime(stats['uptime'])}, "
                    status_line += f"Connections: {stats['connections']}, "
                    status_line += f"Live: {stats['live_clients']}, "
                    status_line += f"HLS: {stats['hls_clients']}"

                    if live_changed or hls_changed:
                        print(f"🔄 {status_line}")
                        if stats['live_clients'] > 0:
                            print("  🎥 RTMP STREAM ACTIVE! Device is streaming.")
                        elif last_live_clients > 0:
                            print("  ⏹️  RTMP stream stopped.")
                    else:
                        print(f"   {status_line}", end='\r')

                    last_live_clients = stats['live_clients']
                    last_hls_clients = stats['hls_clients']
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Error parsing stats")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Cannot connect to RTMP server")

            time.sleep(2)  # Check every 2 seconds

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if last_live_clients > 0:
            print("✅ Stream was active when monitoring stopped.")
        else:
            print("ℹ️  No active streams detected during monitoring.")

if __name__ == "__main__":
    main()
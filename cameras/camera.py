# camera.py

"""
Generic camera and video stream handling module.
Implement base classes and utilities for camera operations here.
"""

class CameraBase:
    def connect(self):
        """Connect to the camera (generic)."""
        pass

    def start_stream(self):
        """Start video stream (generic)."""
        pass

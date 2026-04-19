# camera.py

"""
Generic camera and video stream handling module.
Implement base classes and utilities for camera operations here.
"""


class CameraBase:
    def connect(self):
        """Connect to the camera (generic)."""
        raise NotImplementedError("Subclasses must implement connect().")

    def start_stream(self):
        """Start video stream (generic)."""
        raise NotImplementedError("Subclasses must implement start_stream().")

#!/usr/bin/env python3
"""Analyze a bowling image for objects and print descriptions."""
import sys
import os
import json

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from app import processing

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_image.py <image_path>")
        return

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        return

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    result = processing.analyze_bowling_image(img_path, api_key)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
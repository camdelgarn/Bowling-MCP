import cv2
import base64
import numpy as np
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoFileSource, StreamConfig, VideoMetadata

client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key="dgLzGdzo9QuYLRxslB1f"
)

source = VideoFileSource("C:\\Development\\Bowling-MCP\\video\\behind\\20260123_145715.mp4", realtime_processing=False)  # Buffer and process all frames

VIDEO_OUTPUT = "visualization"
DATA_OUTPUT = "predictions"

config = StreamConfig(
    stream_output=[],
    data_output=["visualization","predictions"],
    requested_plan="webrtc-gpu-medium",
    requested_region="us"
)

session = client.webrtc.stream(
    source=source,
    workflow="find-people-bowling-balls-arts-and-racks",
    workspace="grassdanbowling",
    image_input="image",
    config=config
)

frames = []

@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    timestamp_ms = metadata.pts * metadata.time_base * 1000
    img = cv2.imdecode(np.frombuffer(base64.b64decode(data[VIDEO_OUTPUT]["value"]), np.uint8), cv2.IMREAD_COLOR)
    frames.append((timestamp_ms, metadata.frame_id, img))
    print(f"Processed frame {metadata.frame_id}")
    # print(f"Frame {metadata.frame_id} predictions: {data[DATA_OUTPUT]}")

session.run()

# Stitch frames into output video
frames.sort(key=lambda x: x[1])
fps = (len(frames) - 1) / ((frames[-1][0] - frames[0][0]) / 1000)
h, w = frames[0][2].shape[:2]
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for _, _, frame in frames:
    out.write(frame)
out.release()
print(f"Done! {len(frames)} frames at {fps:.1f} FPS -> output.mp4")

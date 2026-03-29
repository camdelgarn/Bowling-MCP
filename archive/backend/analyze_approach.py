"""Analyze bowler tracking data to find meaningful start/end of approach."""
import json
import numpy as np

with open(r"C:\Development\Bowling-MCP\backend\bowler_tracking.json") as f:
    data = json.load(f)

frames = data["all_frames"]
print(f"Total tracked: {len(frames)} frames")
print(f"\n{'Time':>6s} {'Frame':>5s} {'FootX':>6s} {'FootY':>6s} {'Board':>6s}")
print("-" * 35)
for f in frames:
    print(f"{f['time']:6.2f}s {f['frame']:5d} {f['foot_x']:6d} {f['foot_y']:6d} {f['board']:6.1f}")

boards = np.array([f["board"] for f in frames])
times = np.array([f["time"] for f in frames])

# Find the approach: look for sustained leftward movement (decreasing boards)
# The approach starts when board number starts consistently decreasing
# and ends when it stabilizes (the slide)

# Compute board velocity (boards/sec)
if len(boards) > 3:
    dt = np.diff(times)
    db = np.diff(boards)
    vel = db / np.where(dt > 0, dt, 0.033)
    
    # Find the fastest leftward movement (most negative velocity)
    # Smooth velocity
    window = min(5, len(vel))
    smooth_vel = np.convolve(vel, np.ones(window)/window, mode='same')
    
    # Approach phase: sustained negative velocity (moving left)
    # Start: first frame where smoothed velocity < -5 boards/sec
    # End: first frame after peak movement where velocity returns near 0
    
    moving_left = np.where(smooth_vel < -5)[0]
    if len(moving_left) > 0:
        approach_start_idx = moving_left[0]
        # End: find where board stops changing significantly after the movement
        peak_movement_idx = np.argmin(smooth_vel)
        # After peak, find where velocity stabilizes
        post_peak = smooth_vel[peak_movement_idx:]
        stable = np.where(np.abs(post_peak) < 3)[0]
        if len(stable) > 0:
            approach_end_idx = peak_movement_idx + stable[0]
        else:
            approach_end_idx = len(frames) - 1
        
        start = frames[approach_start_idx]
        end = frames[min(approach_end_idx, len(frames)-1)]
        
        # Also find the final standing position (average of last 10 frames)
        settled_boards = boards[-min(10, len(boards)):]
        settled_board = np.mean(settled_boards)
        
        print(f"\n{'='*55}")
        print(f" BOWLER APPROACH ANALYSIS")
        print(f"{'='*55}")
        print(f"\n Approach START:")
        print(f"   Time: {start['time']:.2f}s (frame {start['frame']})")
        print(f"   Foot: x={start['foot_x']}, y={start['foot_y']}")
        print(f"   Board: {start['board']:.1f}")
        print(f"\n Approach END (slide/release):")
        print(f"   Time: {end['time']:.2f}s (frame {end['frame']})")
        print(f"   Foot: x={end['foot_x']}, y={end['foot_y']}")
        print(f"   Board: {end['board']:.1f}")
        print(f"\n Settled position (last frames):")
        print(f"   Board: {settled_board:.1f}")
        print(f"\n Duration: {end['time'] - start['time']:.2f}s")
        print(f" Drift: {start['board']:.1f} -> {end['board']:.1f} "
              f"({end['board'] - start['board']:+.1f} boards)")
        
        # Update the JSON
        data["start"] = {"time": start["time"], "frame": start["frame"],
                        "foot_x": start["foot_x"], "foot_y": start["foot_y"],
                        "board": start["board"]}
        data["end"] = {"time": end["time"], "frame": end["frame"],
                      "foot_x": end["foot_x"], "foot_y": end["foot_y"],
                      "board": end["board"]}
        data["duration_s"] = round(end["time"] - start["time"], 2)
        data["board_drift"] = round(end["board"] - start["board"], 1)
        data["settled_board"] = round(float(settled_board), 1)
        
        with open(r"C:\Development\Bowling-MCP\backend\bowler_tracking.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n Updated bowler_tracking.json")
    else:
        print("\nNo clear leftward approach detected")

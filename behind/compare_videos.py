"""
Compare all 4 videos from behind/ to determine which lens gives
the best view for lane detection. Extract an empty frame from each
and analyze resolution, brightness, contrast, and lane visibility.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

VIDEO_DIR = '../video/behind/'
videos = ['1.MP4', '2.MP4', '3.MP4', '4.MP4']

fig, axes = plt.subplots(4, 3, figsize=(30, 24))

for i, vname in enumerate(videos):
    path = VIDEO_DIR + vname
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n=== {vname} ===")
    print(f"  Resolution: {w}x{h}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f}s")
    
    # Try to find an empty frame near the end (last 20% of video)
    # Sample several frames and pick one with lowest variance in the approach area
    best_frame = None
    best_score = float('inf')
    best_frame_idx = 0
    
    sample_indices = np.linspace(int(total_frames * 0.7), total_frames - 5, 20, dtype=int)
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Check center region for motion/emptiness
        # Lower half should be approach (uniform if empty)
        region = gray[h//2:, w//3:2*w//3]
        score = np.std(region)
        if score < best_score:
            best_score = score
            best_frame = frame.copy()
            best_frame_idx = idx
    
    cap.release()
    
    if best_frame is None:
        print(f"  ERROR: Could not read any frames")
        continue
    
    print(f"  Best empty frame: #{best_frame_idx} (std={best_score:.1f})")
    
    gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
    
    # Row variance analysis to find foul line
    center_x = w // 2
    strip_width = w // 3
    strip = gray[:, center_x - strip_width//2 : center_x + strip_width//2]
    row_var = np.var(strip.astype(np.float64), axis=1)
    smoothed_var = gaussian_filter1d(row_var, sigma=20)
    
    # Find foul line: sharp drop in variance
    foul_line_y = None
    for y in range(h // 4, 3 * h // 4):
        if smoothed_var[y] > 200 and y + 100 < h:
            # Check if variance drops significantly in next 100 rows
            if smoothed_var[min(y + 100, h-1)] < 100 and smoothed_var[y] / max(smoothed_var[min(y + 100, h-1)], 1) > 5:
                foul_line_y = y
                break
    
    # If not found with strict criteria, try looser
    if foul_line_y is None:
        max_var_y = np.argmax(smoothed_var[h//4:3*h//4]) + h//4
        for y in range(max_var_y, 3*h//4):
            if smoothed_var[y] < 100:
                foul_line_y = y
                break
    
    print(f"  Foul line estimate: y={foul_line_y}")
    
    # Intensity profile at mid-lane (y slightly above foul line)
    if foul_line_y:
        profile_y = foul_line_y - 50
    else:
        profile_y = h // 3
    
    row_data = gray[profile_y, :].astype(np.float64)
    smoothed_row = gaussian_filter1d(row_data, sigma=30)
    
    # Mean brightness
    mean_bright = np.mean(gray)
    lane_bright = np.mean(gray[:foul_line_y if foul_line_y else h//2, :]) if foul_line_y else np.mean(gray[:h//2, :])
    approach_bright = np.mean(gray[foul_line_y:, :]) if foul_line_y else np.mean(gray[h//2:, :])
    
    print(f"  Mean brightness: overall={mean_bright:.0f}, lane={lane_bright:.0f}, approach={approach_bright:.0f}")
    
    # Lane visibility: intensity contrast at profile_y
    gradient = np.abs(np.gradient(smoothed_row))
    max_gradient = np.max(gradient)
    print(f"  Max gradient at y={profile_y}: {max_gradient:.1f}")
    
    # Detect lane edges at profile_y
    grad = np.gradient(smoothed_row)
    # Find prominent peaks
    from scipy.signal import find_peaks
    pos_peaks, _ = find_peaks(grad, height=0.5, distance=50)
    neg_peaks, _ = find_peaks(-grad, height=0.5, distance=50)
    
    print(f"  Positive gradient peaks (dark→bright): {pos_peaks[:10]}")
    print(f"  Negative gradient peaks (bright→dark): {neg_peaks[:10]}")
    
    # --- Plots ---
    # Column 1: Frame overview
    ax = axes[i, 0]
    frame_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    if foul_line_y:
        ax.axhline(y=foul_line_y, color='red', linewidth=2, label=f'Foul line y={foul_line_y}')
    ax.set_title(f'{vname} - {w}x{h} - Frame #{best_frame_idx}')
    ax.legend(loc='upper right', fontsize=8)
    
    # Column 2: Row variance
    ax = axes[i, 1]
    ax.plot(smoothed_var, range(h), 'b-', linewidth=0.5)
    if foul_line_y:
        ax.axhline(y=foul_line_y, color='red', linewidth=2)
    ax.set_xlabel('Row variance')
    ax.set_ylabel('y')
    ax.set_title(f'{vname} - Row variance')
    ax.invert_yaxis()
    ax.set_xlim(0, min(3000, np.max(smoothed_var) * 1.1))
    
    # Column 3: Intensity profile at profile_y
    ax = axes[i, 2]
    ax.plot(range(w), smoothed_row, 'b-', linewidth=0.5)
    ax.scatter(pos_peaks, smoothed_row[pos_peaks], c='green', s=20, zorder=5, label='Dark→Bright')
    ax.scatter(neg_peaks, smoothed_row[neg_peaks], c='red', s=20, zorder=5, label='Bright→Dark')
    ax.set_xlabel('x')
    ax.set_ylabel('Intensity')
    ax.set_title(f'{vname} - Intensity at y={profile_y}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('debug_compare_videos.png', dpi=120)
print("\nSaved debug_compare_videos.png")

"""
Compare all 4 videos - improved foul line detection and lane edge analysis.
Uses the proven variance-drop method from detect_foul_line3.py.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

VIDEO_DIR = '../video/behind/'
videos = ['1.MP4', '2.MP4', '3.MP4', '4.MP4']

fig, axes = plt.subplots(4, 4, figsize=(36, 24))

results = []

for i, vname in enumerate(videos):
    path = VIDEO_DIR + vname
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n{'='*60}")
    print(f"=== {vname} === {w}x{h}, {total_frames} frames, {fps:.1f} fps")
    print(f"{'='*60}")
    
    # Find empty frame near end
    best_frame = None
    best_score = float('inf')
    best_idx = 0
    
    for idx in np.linspace(int(total_frames * 0.7), total_frames - 5, 30, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Approach region (lower half, center third)
        region = gray[h*2//3:, w//3:2*w//3]
        score = np.std(region)
        if score < best_score:
            best_score = score
            best_frame = frame.copy()
            best_idx = idx
    
    cap.release()
    
    if best_frame is None:
        print("  ERROR: Could not read frames")
        continue
    
    print(f"  Empty frame: #{best_idx} (approach std={best_score:.1f})")
    gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
    
    # === VARIANCE ANALYSIS (proven method) ===
    # Use center strip for variance analysis
    cx = w // 2
    strip_half = 300
    strip = gray[:, cx-strip_half:cx+strip_half].astype(np.float64)
    row_var = np.var(strip, axis=1)
    smoothed_var = gaussian_filter1d(row_var, sigma=20)
    
    # Foul line: find where variance drops below threshold
    # Look for the transition from high (>200) to low (<100)
    var_threshold = 80
    foul_line_y = None
    
    # First find peak variance region
    peak_var = np.max(smoothed_var[h//6:h//2])
    peak_var_y = np.argmax(smoothed_var[h//6:h//2]) + h//6
    
    # Then scan downward from peak to find where it drops to <20% of peak
    drop_threshold = max(peak_var * 0.15, 50)
    for y in range(peak_var_y, 3*h//4):
        if smoothed_var[y] < drop_threshold:
            foul_line_y = y
            break
    
    print(f"  Peak variance: {peak_var:.0f} at y={peak_var_y}")
    print(f"  Foul line (variance drop): y={foul_line_y}")
    
    if foul_line_y:
        lane_pct = foul_line_y / h * 100
        approach_pct = (h - foul_line_y) / h * 100
        print(f"  Lane: {lane_pct:.0f}% of frame, Approach: {approach_pct:.0f}% of frame")
    
    # === LANE EDGE DETECTION ===
    # At the foul line area, find lane edges via smoothed gradient
    if foul_line_y:
        profile_y = foul_line_y - 30
    else:
        profile_y = h // 3
    
    row_data = gray[profile_y, :].astype(np.float64)
    smoothed_row = gaussian_filter1d(row_data, sigma=40)
    grad = np.gradient(smoothed_row)
    
    # Find the most prominent lane (brightest wide region)
    # Threshold: above mean brightness
    mean_bright = np.mean(smoothed_row)
    bright_mask = smoothed_row > mean_bright
    
    # Find contiguous bright regions
    regions = []
    in_region = False
    start = 0
    for x in range(w):
        if bright_mask[x] and not in_region:
            in_region = True
            start = x
        elif not bright_mask[x] and in_region:
            in_region = False
            regions.append((start, x, x - start, np.mean(smoothed_row[start:x])))
    if in_region:
        regions.append((start, w, w - start, np.mean(smoothed_row[start:w])))
    
    # Sort by width
    regions.sort(key=lambda r: r[2], reverse=True)
    
    print(f"\n  Bright regions at y={profile_y} (top 5 by width):")
    for j, (rs, re, rw, rb) in enumerate(regions[:5]):
        print(f"    #{j+1}: x={rs}-{re}, width={rw}, mean brightness={rb:.0f}")
    
    # For lane detection: find the specific gradient peaks
    # Look for strong positive (left edge) and negative (right edge) gradients
    pos_peaks, pos_props = find_peaks(grad, height=0.3, distance=50, prominence=0.2)
    neg_peaks, neg_props = find_peaks(-grad, height=0.3, distance=50, prominence=0.2)
    
    print(f"\n  Strong gradient edges at y={profile_y}:")
    print(f"    Dark→Bright edges: {pos_peaks}")
    print(f"    Bright→Dark edges: {neg_peaks}")
    
    # Try to identify gutter channels (narrow dark strips flanking the lane)
    # At foul line, gutters should be visible as dark strips
    if foul_line_y:
        # Check multiple y levels above foul line
        for check_y in [foul_line_y - 20, foul_line_y - 50, foul_line_y - 100]:
            if check_y < 0:
                continue
            row_check = gaussian_filter1d(gray[check_y, :].astype(np.float64), sigma=15)
            grad_check = np.gradient(row_check)
            
            # Find pairs: positive peak followed by negative peak within 200px
            pp, _ = find_peaks(grad_check, height=0.5, distance=30)
            np_arr, _ = find_peaks(-grad_check, height=0.5, distance=30)
            
            # Lane = bright region between a pos and neg peak
            pairs = []
            for p in pp:
                for n in np_arr:
                    if n > p and 100 < (n - p) < 800:
                        lane_brightness = np.mean(row_check[p:n])
                        if lane_brightness > mean_bright * 0.8:
                            pairs.append((p, n, n-p, lane_brightness))
            
            if pairs:
                pairs.sort(key=lambda x: x[3], reverse=True)
                print(f"\n    Detected lanes at y={check_y}:")
                for p in pairs[:3]:
                    print(f"      x={p[0]}-{p[1]}, width={p[2]}, brightness={p[3]:.0f}")
    
    # === OVERALL QUALITY METRICS ===
    lane_area = gray[:foul_line_y, :] if foul_line_y else gray[:h//3, :]
    approach_area = gray[foul_line_y:, :] if foul_line_y else gray[2*h//3:, :]
    
    lane_mean = np.mean(lane_area)
    lane_std = np.std(lane_area)
    approach_mean = np.mean(approach_area)
    approach_std = np.std(approach_area)
    contrast = approach_mean - lane_mean
    
    # Edge sharpness: max gradient magnitude at foul line
    if foul_line_y:
        edge_sharpness = np.max(np.abs(grad))
    else:
        edge_sharpness = 0
    
    print(f"\n  Quality metrics:")
    print(f"    Lane brightness: {lane_mean:.0f} ± {lane_std:.0f}")
    print(f"    Approach brightness: {approach_mean:.0f} ± {approach_std:.0f}")
    print(f"    Contrast (approach - lane): {contrast:.0f}")
    print(f"    Edge sharpness: {edge_sharpness:.1f}")
    
    results.append({
        'name': vname,
        'foul_line_y': foul_line_y,
        'lane_mean': lane_mean,
        'lane_std': lane_std,
        'approach_mean': approach_mean,
        'contrast': contrast,
        'edge_sharpness': edge_sharpness,
        'peak_var': peak_var,
        'best_idx': best_idx,
    })
    
    # --- Plots ---
    # Col 1: Frame
    ax = axes[i, 0]
    frame_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    if foul_line_y:
        ax.axhline(y=foul_line_y, color='red', linewidth=2)
    ax.set_title(f'{vname} - Frame #{best_idx}', fontsize=12)
    ax.set_ylabel(vname, fontsize=14, fontweight='bold')
    
    # Col 2: Zoomed center showing lane/approach transition
    ax = axes[i, 1]
    if foul_line_y:
        y1 = max(0, foul_line_y - 300)
        y2 = min(h, foul_line_y + 300)
        x1, x2 = w//4, 3*w//4
        crop = best_frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ax.imshow(crop_rgb)
        ax.axhline(y=foul_line_y - y1, color='red', linewidth=2, label='Foul line')
        ax.set_title(f'Foul line area (y={foul_line_y})')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No foul line detected', transform=ax.transAxes, ha='center')
        ax.set_title('Foul line area')
    
    # Col 3: Variance profile
    ax = axes[i, 2]
    ax.plot(smoothed_var, range(h), 'b-', linewidth=0.8)
    if foul_line_y:
        ax.axhline(y=foul_line_y, color='red', linewidth=2, label=f'Foul y={foul_line_y}')
    ax.set_xlabel('Row variance')
    ax.set_title('Row Variance Profile')
    ax.invert_yaxis()
    ax.legend()
    
    # Col 4: Intensity profile at foul line
    ax = axes[i, 3]
    ax.plot(range(w), smoothed_row, 'b-', linewidth=0.5)
    ax.scatter(pos_peaks, smoothed_row[pos_peaks], c='green', s=30, zorder=5, label='L edge')
    ax.scatter(neg_peaks, smoothed_row[neg_peaks], c='red', s=30, zorder=5, label='R edge')
    ax.set_xlabel('x')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Intensity at y={profile_y}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('debug_compare_all.png', dpi=120)
print("\n\nSaved debug_compare_all.png")

# Summary
print("\n" + "="*60)
print("SUMMARY - Video Comparison")
print("="*60)
print(f"{'Video':>6} | {'Foul Y':>7} | {'Lane%':>6} | {'Lane μ':>7} | {'Appr μ':>7} | {'Contrast':>8} | {'Edge':>6} | {'PeakVar':>8}")
print("-" * 75)
for r in results:
    fy = r['foul_line_y']
    lane_pct = f"{fy/2988*100:.0f}%" if fy else "N/A"
    print(f"{r['name']:>6} | {str(fy):>7} | {lane_pct:>6} | {r['lane_mean']:>7.0f} | {r['approach_mean']:>7.0f} | {r['contrast']:>8.0f} | {r['edge_sharpness']:>6.1f} | {r['peak_var']:>8.0f}")

# Recommendation
best = max(results, key=lambda r: r['contrast'] + r['edge_sharpness'] * 10 + (r['lane_mean'] if r['lane_mean'] > 60 else 0))
print(f"\n→ Best video for lane detection: {best['name']}")
print(f"  Reason: highest combined contrast ({best['contrast']:.0f}), edge sharpness ({best['edge_sharpness']:.1f}), lane brightness ({best['lane_mean']:.0f})")

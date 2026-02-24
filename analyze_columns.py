"""
Analyze the vertical height profile of each glasses image column.
Frame-front columns = tall (full frame height).
Temple arm columns = short (thin arm, ~10-20px).
"""
from PIL import Image
import numpy as np
import os

base = r"C:\Users\laci\projects\hackathon\virtual-eyewear-tryon\static\glasses"
images = ["rayban_clubmaster.png", "rayban_wayfarer.png", "rayban_erika.png"]

results = []

for name in images:
    path = os.path.join(base, name)
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]
    W, H = img.width, img.height

    col_heights = []
    for x in range(W):
        col = alpha[:, x]
        opaque = np.where(col > 20)[0]
        if len(opaque) == 0:
            col_heights.append(0)
        else:
            extent = int(opaque[-1]) - int(opaque[0]) + 1
            col_heights.append(extent)
    col_heights = np.array(col_heights)
    max_h = int(np.max(col_heights))

    # Threshold: 30% of max height → frame front vs temple
    thr = max_h * 0.30
    frame_cols = np.where(col_heights > thr)[0]
    left = int(frame_cols[0]) if len(frame_cols) > 0 else 0
    right = int(frame_cols[-1]) + 1 if len(frame_cols) > 0 else W

    # Sample heights at various x positions for the report
    samples = []
    step = max(1, W // 30)
    for x in range(0, W, step):
        samples.append((x, int(col_heights[x])))

    results.append({
        "name": name,
        "size": (W, H),
        "max_h": max_h,
        "thr": thr,
        "frame_left": left,
        "frame_right": right,
        "frame_width": right - left,
        "total_width": W,
        "frame_frac": (right - left) / W,
        "samples": samples,
    })

out_lines = []
for r in results:
    out_lines.append(f"\n{r['name']}  size={r['size']}  max_col_h={r['max_h']}px")
    out_lines.append(f"  threshold={r['thr']:.0f}px")
    out_lines.append(f"  frame_front: x=[{r['frame_left']}, {r['frame_right']})  width={r['frame_width']}px  ({r['frame_frac']*100:.0f}% of image)")
    out_lines.append(f"  height profile (sampled every ~{r['size'][0]//30}px):")
    for x, h in r["samples"]:
        bar = "#" * int(h / r["max_h"] * 30)
        out_lines.append(f"    x={x:4d}  h={h:4d}  {bar}")

output = "\n".join(out_lines)
print(output)

out_path = r"C:\Users\laci\projects\hackathon\virtual-eyewear-tryon\column_analysis_out.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(output)
print(f"\nSaved to {out_path}")

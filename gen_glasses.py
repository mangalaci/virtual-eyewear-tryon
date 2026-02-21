"""
Generate glasses PNG images with transparent backgrounds and transparent lens areas.
Uses alpha channel manipulation for clean lens cutouts.
"""
from PIL import Image, ImageDraw, ImageFilter
import math

W, H = 1200, 400
SCALE = 4
SW, SH = W * SCALE, H * SCALE


def punch_hole(img, draw_fn):
    """
    Draw a shape onto the ALPHA channel directly (making those pixels transparent).
    draw_fn receives an ImageDraw on a grayscale mask; draw white to punch holes.
    """
    mask = Image.new("L", (SW, SH), 0)
    md = ImageDraw.Draw(mask)
    draw_fn(md)
    # Where mask is white (255), set alpha to 0
    r, g, b, a = img.split()
    a = Image.composite(Image.new("L", (SW, SH), 0), a, mask)
    return Image.merge("RGBA", (r, g, b, a))


def add_shape(img, draw_fn, color):
    """Draw a filled shape onto the image."""
    layer = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    draw_fn(ld, color)
    return Image.alpha_composite(img, layer)


def rounded_rect(d, x0, y0, x1, y1, r, **kwargs):
    d.rounded_rectangle([x0, y0, x1, y1], radius=r, **kwargs)


def ellipse_pts(cx, cy, rx, ry, n=64):
    return [(cx + rx * math.cos(2 * math.pi * i / n),
             cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def save(img, path):
    # Slight blur to smooth jagged edges from supersampling
    final = img.resize((W, H), Image.LANCZOS)
    final.save(path, "PNG")
    size = __import__('os').path.getsize(path)
    print(f"Saved {path}  ({W}x{H}, {size//1024} KB)")


# ─── CLUBMASTER ───────────────────────────────────────────────────────────────
def gen_clubmaster(path):
    img = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))

    s = SCALE
    cx, cy = SW // 2, SH // 2

    NEAR_BLACK = (18, 18, 18, 255)
    GOLD = (180, 145, 55, 255)

    gap = 36 * s
    lw, lh = 255 * s, 175 * s    # lens width/height
    brow_extra = 50 * s           # brow bar extends above lens top
    frame = 18 * s                # frame thickness
    t_len = 280 * s               # temple length
    t_h = 14 * s                  # temple height

    lx0, lx1 = cx - gap // 2 - lw, cx - gap // 2
    rx0, rx1 = cx + gap // 2, cx + gap // 2 + lw
    top_y = cy - lh // 2
    bot_y = cy + lh // 2
    lcx, rcx = (lx0 + lx1) // 2, (rx0 + rx1) // 2

    # --- Brow bars (tall rounded rectangles) ---
    def draw_brow(d, col):
        r = 26 * s
        d.rounded_rectangle([lx0, top_y - brow_extra, lx1, top_y + 20 * s], radius=r, fill=col)
        d.rounded_rectangle([rx0, top_y - brow_extra, rx1, top_y + 20 * s], radius=r, fill=col)
    img = add_shape(img, draw_brow, NEAR_BLACK)

    # --- Bottom frame rings (outer) ---
    def draw_outer(d, col):
        d.ellipse([lx0, top_y, lx1, bot_y], fill=col)
        d.ellipse([rx0, top_y, rx1, bot_y], fill=col)
    img = add_shape(img, draw_outer, NEAR_BLACK)

    # --- Punch lens holes ---
    def hole(md):
        md.ellipse([lx0 + frame, top_y + frame, lx1 - frame, bot_y - frame], fill=255)
        md.ellipse([rx0 + frame, top_y + frame, rx1 - frame, bot_y - frame], fill=255)
    img = punch_hole(img, hole)

    # --- Gold bridge ---
    def draw_bridge(d, col):
        bw = gap + 4 * s
        bh = 20 * s
        by = cy - 8 * s
        d.rounded_rectangle([cx - bw // 2, by - bh // 2, cx + bw // 2, by + bh // 2],
                             radius=8 * s, fill=col)
    img = add_shape(img, draw_bridge, GOLD)

    # --- Temples ---
    def draw_temples(d, col):
        ty = cy - 5 * s
        # Left
        d.polygon([
            (lx0, ty - t_h), (lx0 - t_len, ty + t_h * 2),
            (lx0 - t_len, ty + t_h * 3), (lx0, ty + t_h),
        ], fill=col)
        # Right
        d.polygon([
            (rx1, ty - t_h), (rx1 + t_len, ty + t_h * 2),
            (rx1 + t_len, ty + t_h * 3), (rx1, ty + t_h),
        ], fill=col)
    img = add_shape(img, draw_temples, NEAR_BLACK)

    save(img, path)


# ─── WAYFARER ─────────────────────────────────────────────────────────────────
def gen_wayfarer(path):
    img = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))

    s = SCALE
    cx, cy = SW // 2, SH // 2

    NEAR_BLACK = (18, 18, 18, 255)

    gap = 38 * s
    lw, lh = 255 * s, 185 * s
    taper = 22 * s      # top wider than bottom
    frame = 22 * s
    t_len = 275 * s
    t_h = 16 * s

    lx0, lx1 = cx - gap // 2 - lw, cx - gap // 2
    rx0, rx1 = cx + gap // 2, cx + gap // 2 + lw
    top_y = cy - lh // 2
    bot_y = cy + lh // 2

    def trap(x0, y0, x1, y1, tap):
        return [(x0 - tap, y0), (x1 + tap, y0), (x1, y1), (x0, y1)]

    def draw_outer(d, col):
        d.polygon(trap(lx0, top_y, lx1, bot_y, taper), fill=col)
        d.polygon(trap(rx0, top_y, rx1, bot_y, taper), fill=col)
    img = add_shape(img, draw_outer, NEAR_BLACK)

    # Lens holes
    def hole(md):
        md.polygon(trap(lx0 + frame, top_y + frame, lx1 - frame, bot_y - frame, taper - frame), fill=255)
        md.polygon(trap(rx0 + frame, top_y + frame, rx1 - frame, bot_y - frame, taper - frame), fill=255)
    img = punch_hole(img, hole)

    # Bridge
    def draw_bridge(d, col):
        d.ellipse([cx - gap * 1.5, cy - 16 * s, cx + gap * 1.5, cy + 16 * s], fill=col)
    img = add_shape(img, draw_bridge, NEAR_BLACK)

    # Temples (angled up from top-outer corner)
    def draw_temples(d, col):
        ty = top_y + 5 * s
        d.polygon([
            (lx0 - taper, ty), (lx0 - taper - t_len, ty - t_h * 2),
            (lx0 - taper - t_len, ty + t_h), (lx0 - taper, ty + t_h * 2),
        ], fill=col)
        d.polygon([
            (rx1 + taper, ty), (rx1 + taper + t_len, ty - t_h * 2),
            (rx1 + taper + t_len, ty + t_h), (rx1 + taper, ty + t_h * 2),
        ], fill=col)
    img = add_shape(img, draw_temples, NEAR_BLACK)

    save(img, path)


# ─── ERIKA (cat-eye) ──────────────────────────────────────────────────────────
def gen_erika(path):
    img = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))

    s = SCALE
    cx, cy = SW // 2, SH // 2

    AMBER = (140, 75, 15, 255)

    gap = 34 * s
    lw, lh = 260 * s, 180 * s
    frame = 22 * s
    t_len = 260 * s
    t_h = 15 * s
    lift = 0.30   # how much the outer-top is lifted

    lx0, lx1 = cx - gap // 2 - lw, cx - gap // 2
    rx0, rx1 = cx + gap // 2, cx + gap // 2 + lw
    top_y = cy - lh // 2
    bot_y = cy + lh // 2
    lcx = (lx0 + lx1) // 2
    rcx = (rx0 + rx1) // 2

    def cat_pts(ox, oy, rx_r, ry_r, flip=False):
        """Cat-eye shape: outer-top corner pulled up."""
        n = 64
        pts = []
        for i in range(n):
            t = 2 * math.pi * i / n
            ex = rx_r * math.cos(t)
            ey = ry_r * math.sin(t)
            # "outer" side: right-outer for right lens, left-outer for left lens
            outer_frac = (-ex / rx_r) if flip else (ex / rx_r)
            if outer_frac > 0 and ey < 0:
                ey -= lift * ry_r * outer_frac * abs(ey / ry_r)
            pts.append((ox + ex, oy + ey))
        return pts

    lpts = cat_pts(lcx, cy, lw // 2, lh // 2, flip=True)
    rpts = cat_pts(rcx, cy, lw // 2, lh // 2, flip=False)

    def draw_outer(d, col):
        d.polygon(lpts, fill=col)
        d.polygon(rpts, fill=col)
    img = add_shape(img, draw_outer, AMBER)

    def inner_cat(ox, oy, rx_r, ry_r, flip):
        return cat_pts(ox, oy, rx_r - frame, ry_r - frame, flip)

    def hole(md):
        md.polygon(inner_cat(lcx, cy, lw // 2, lh // 2, flip=True), fill=255)
        md.polygon(inner_cat(rcx, cy, lw // 2, lh // 2, flip=False), fill=255)
    img = punch_hole(img, hole)

    # Bridge
    def draw_bridge(d, col):
        d.ellipse([cx - gap * 1.3, cy - 12 * s, cx + gap * 1.3, cy + 12 * s], fill=col)
    img = add_shape(img, draw_bridge, AMBER)

    # Temples (sweep upward – cat-eye style)
    def draw_temples(d, col):
        # Find topmost outer point of each lens
        tl_x = min(x for x, y in lpts)
        tl_y = min(y for x, y in lpts if x < lcx - lw // 4)
        tr_x = max(x for x, y in rpts)
        tr_y = min(y for x, y in rpts if x > rcx + lw // 4)

        d.polygon([
            (tl_x, tl_y), (tl_x - t_len, tl_y - t_h * 4),
            (tl_x - t_len, tl_y - t_h * 4 + t_h * 2), (tl_x, tl_y + t_h),
        ], fill=col)
        d.polygon([
            (tr_x, tr_y), (tr_x + t_len, tr_y - t_h * 4),
            (tr_x + t_len, tr_y - t_h * 4 + t_h * 2), (tr_x, tr_y + t_h),
        ], fill=col)
    img = add_shape(img, draw_temples, AMBER)

    save(img, path)


if __name__ == "__main__":
    base = "C:/Users/laci/projects/hackathon/virtual-eyewear-tryon/static/glasses"
    gen_clubmaster(f"{base}/rayban_clubmaster.png")
    gen_wayfarer(f"{base}/rayban_wayfarer.png")
    gen_erika(f"{base}/rayban_erika.png")
    print("All done!")

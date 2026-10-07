"""Draws the website's app icons into web/icons/ (run: py -3.14 tools/make_icons.py, needs Pillow).

The icon is a score ring with a sparkle in it, on the app's accent blue. It is not a face. Everything is drawn at a
bigger size and shrunk, which gives smooth edges.
"""
import math
import os

from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "icons")
TOP, BOTTOM = (38, 140, 226), (0, 82, 160)  # the accent #0067c0, a little lighter at the top and darker at the bottom
WHITE = (255, 255, 255)
BIG = 2048  # drawing size


def gradient(size):
    # Vertical gradient from TOP to BOTTOM
    column = Image.new("RGB", (1, size))
    column.putdata([tuple(round(a + (b - a) * y / (size - 1)) for a, b in zip(TOP, BOTTOM)) for y in range(size)])
    return column.resize((size, size))


def sparkle(draw, cx, cy, radius, colour):
    # A four-point star: four curved spikes from the middle
    points = []
    for n in range(64):
        angle = 2 * math.pi * n / 64
        # |cos|^3 and |sin|^3 pull the sides in, so the corners are sharp
        x, y = math.cos(angle), math.sin(angle)
        points.append((cx + radius * math.copysign(abs(x) ** 3, x), cy + radius * math.copysign(abs(y) ** 3, y)))
    draw.polygon(points, fill=colour)


def glyph(size, scale):
    # White ring (open at the top left, like a score gauge) and a sparkle, on a transparent square. scale is how much
    # of the square the ring may fill
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c = size / 2
    r = size * scale / 2
    stroke = size * scale * 0.115
    box = (c - r, c - r, c + r, c + r)
    # Faint full track, then the bright arc from 12 o'clock round to about 9 o'clock
    d.ellipse(box, outline=(255, 255, 255, 70), width=round(stroke))
    start, end = -90, 190
    d.arc(box, start, end, fill=WHITE, width=round(stroke))
    for angle in (start, end):
        # Round caps
        x, y = c + (r - stroke / 2) * math.cos(math.radians(angle)), c + (r - stroke / 2) * math.sin(math.radians(angle))
        d.ellipse((x - stroke / 2, y - stroke / 2, x + stroke / 2, y + stroke / 2), fill=WHITE)
    sparkle(d, c, c, r * 0.46, WHITE)
    return layer


def rounded_mask(size, radius):
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius, fill=255)
    return mask


def icon(size, maskable=False, rounded=True):
    # maskable: full-bleed background and a smaller glyph, so any mask the phone applies leaves the ring intact
    big = gradient(BIG).convert("RGBA")
    big.alpha_composite(glyph(BIG, 0.5 if maskable else 0.62))
    if rounded and not maskable:
        big.putalpha(rounded_mask(BIG, round(BIG * 0.225)))
    return big.resize((size, size), Image.LANCZOS)


def main():
    os.makedirs(OUT, exist_ok=True)
    icon(512).save(os.path.join(OUT, "icon-512.png"))
    icon(192).save(os.path.join(OUT, "icon-192.png"))
    icon(512, maskable=True).save(os.path.join(OUT, "icon-maskable-512.png"))
    # iOS rounds the corners itself and doesn't like transparency
    icon(180, rounded=False).convert("RGB").save(os.path.join(OUT, "apple-touch-icon.png"))
    icon(32).save(os.path.join(OUT, "favicon-32.png"))
    print("Icons written to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()

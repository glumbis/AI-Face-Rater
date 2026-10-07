# Overlay style

What the desktop app draws on a face (`overlay.py`), written so the website can match it with canvas 2D. All sizes
are shares of **W = the face width**: the distance between landmarks 127 and 356 (the temples). `L` is the line width.

    L = max(1.2 px, 0.004 * W)

Canvas notes: use `lineCap = lineJoin = "round"`. Landmark coordinates are in picture pixels, so scale them by the
picture's display scale first and then scale `W` (and so `L`) the same way. The desktop version draws at 4x (result)
or 2x (preview) and shrinks the picture again; canvas 2D already anti-aliases, so just draw normally.

## Colours

| Name | Value | Used for |
| --- | --- | --- |
| line | `#FFFFFF` at 78% | all contour lines |
| line, turned head (preview only) | `#FFCD8C` at 78% | the lines while the preview says "turn a little" |
| halo | `#141414` at 34% | a soft dark glow behind everything, so white shows on pale skin |
| accent | `#60CDFF` at 95% | the dots at the key points |
| accent, turned head (preview only) | `#F5A028` at 95% | the dots while the head is turned |
| uneven skin | `#F87171`, up to 50% | the tint on uneven skin (result only) |

Layering, bottom to top: halo, then the white lines, then the dot rings, then the accent dots.

## Contours (thin lines, no dots on them)

MediaPipe Face Landmarker landmark numbers. Join the points in order with a smooth curve (draw through the
points, or use Chaikin corner cutting 2 rounds, 1 in the preview). "Strength" multiplies the line's alpha.

| Contour | Landmarks, in order | Closed | Strength | Width |
| --- | --- | --- | --- | --- |
| Face outline | 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109 | yes | 0.55 | 0.9 L |
| Right brow (viewer's right) | the middle of each pair: (276,300), (283,293), (282,334), (295,296), (285,336) | no | 1 | 1.5 L |
| Left brow | (46,70), (53,63), (52,105), (65,66), (55,107) | no | 1 | 1.5 L |
| Right eye | 263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249 | yes | 1 | L |
| Left eye | 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7 | yes | 1 | L |
| Outer lips | 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146 | yes | 1 | L |
| Inner lips | 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95 | yes | 0.5 | L |
| Nose bridge | 168, 6, 197, 195, 5, 4 | no | 0.7 | L |
| Nose base | 48, 64, 98, 97, 2, 326, 327, 294, 278 | no | 0.7 | L |

The brows are a single line along the middle of the brow (the average of the two landmarks in each pair), not an
outline. Pupils, cheeks and the forehead have no lines.

## Halo

The same paths as the lines, drawn first in `#141414` with width `3.4 L`, alpha 34% at full strength, and blurred
a little (`ctx.filter = "blur(" + 0.4 * L + "px)"`, or `shadowBlur`). The halo is the same strength on every
contour; it is a glow, not an outline. For the cheek brackets the width is `1.1 L * 3.4`.

## Key points (6 accent dots)

Outer eye corners 33 and 263, nose tip 4, mouth corners 61 and 291, chin 152. Each is three circles:

1. halo: radius `1.35 L + 1.6 L`, `#141414` at 34%
2. ring: radius `1.35 L + 0.7 L`, white at 78%
3. dot: radius `1.35 L`, accent colour at 95%

## Cheeks (result only)

Each cheek is a square of side `0.2 W` centred on the middle of cheek landmarks (left: 234, 58, 98, 144; right:
454, 288, 327, 373). It is not drawn as a box but as four rounded corner brackets: an arm of `0.07 W` along each
side, joined by a quarter circle of radius `0.03 W` at the corner. Line width `1.1 L`, white, drawn with the
same halo, everything at 70% of the normal alpha. A cheek that could not be judged (hidden, beard, grayscale) gets
no brackets.

## Uneven skin (result only)

The spots the skin check finds, as a soft translucent red tint and never as hard pixels: blur the spot mask
with a Gaussian of sigma `side / 20` (side = the cheek square), multiply by 1.6 and clamp to 0..1, fade it to zero
over `0.12 * side` towards the edges of the square, and mix `#F87171` in with at most 50% alpha. On the website, a
radial gradient or blurred circles at the spot positions in `#F87171` (about 45% alpha) gives the same look.

## Preview and result differences

| | Result picture | Live preview |
| --- | --- | --- |
| Contours, halo, accent dots | yes | yes |
| Cheek brackets, uneven-skin tint | yes | no |
| Smoothing | 2 rounds | 1 round |
| Drawing scale | 4x then shrunk | 2x then shrunk |
| Head turned or tilted (a tip) | not drawn (the photo is refused instead) | lines `#FFCD8C` and dots `#F5A028` |

The preview is mirrored like a mirror (the app flips the frame before drawing, so nothing in the style is
left/right-specific). The preview detects on a 320 px copy and the landmarks are scaled back up to the frame.

## Legend

One row under the score bars, three small swatches on a dark chip (`#444444`) so the white lines can be seen on a
light window: "Face lines" (a short curve with an accent dot), "Cheeks" (four small corner brackets) and "Uneven
skin" (a soft red blob).

"""Height quality for optional FB reach, independent of detection."""
import math


def body_height_signal(box, confidence, width, height):
    """Return height and quality: 0 invalid, 1 full height, 2 lower bound.

    Side clipping does not truncate height. Vertical clipping can justify
    retreat from an oversized person, never approach.
    """
    if not all(math.isfinite(float(v)) for v in [*box, confidence, width, height]) or width <= 0 or height <= 0:
        return 0., 0
    x1,y1,x2,y2 = box
    if x2 <= x1 or y2 <= y1 or confidence < .6:
        return 0., 0
    measured = max(0., min(height,y2)-max(0.,y1))/height
    if measured <= 0: return 0., 0
    return measured, (2 if y1 <= .02*height or y2 >= .98*height else 1)

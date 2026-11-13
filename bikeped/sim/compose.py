"""Layouts that arrange camera views and the top-down panel into one frame.

All images are BGR ``uint8`` arrays and sizes are ``(width, height)`` in
pixels. Every layout returns a frame of exactly the requested size, so the
composite video keeps one resolution whatever the views look like. Views are
never stretched: each one is scaled to fit its cell and centred on the page
background.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Size = Tuple[int, int]
Cell = Tuple[int, int, int, int]

#: Layout names accepted by :func:`compose`.
LAYOUTS: Tuple[str, ...] = ("single", "split", "quad")

#: Page background of the project site (``#0a0b0d``) in BGR order.
BACKGROUND_BGR: Tuple[int, int, int] = (13, 11, 10)

#: Muted text colour of the project site (``#9b9ca3``) in BGR order.
MUTED_BGR: Tuple[int, int, int] = (163, 156, 155)

#: Outer margin and gutter between cells, as a fraction of the frame height.
GUTTER_FRACTION = 0.015

#: Smallest share of the usable width that ``split`` leaves to the camera view.
_MIN_VIEW_SHARE = 0.4


def _as_bgr(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as a three-channel BGR ``uint8`` array."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_BGRA2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"expected a grayscale, BGR or BGRA image, got shape {arr.shape}")


def _checked_size(size: Size) -> Size:
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be positive, got {size!r}")
    return width, height


def _gutter(height: int) -> int:
    return max(2, int(round(GUTTER_FRACTION * height)))


def _even(value: int) -> int:
    return value + (value % 2)


def letterbox(
    image: np.ndarray,
    size: Size,
    background: Tuple[int, int, int] = BACKGROUND_BGR,
) -> np.ndarray:
    """Fit ``image`` inside a frame of ``size`` without changing its aspect ratio.

    The image is scaled to the largest size that fits (area averaging when it
    shrinks, bicubic when it grows), centred, and the rest of the frame is
    filled with ``background`` (BGR). Grayscale and BGRA inputs are converted
    to BGR.
    """
    width, height = _checked_size(size)
    src = _as_bgr(image)
    src_h, src_w = src.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, min(width, int(round(src_w * scale))))
    new_h = max(1, min(height, int(round(src_h * scale))))
    if (new_w, new_h) != (src_w, src_h):
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        src = cv2.resize(src, (new_w, new_h), interpolation=interpolation)

    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = background
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = src
    return canvas


def _split_cells(size: Size) -> Tuple[Cell, Cell]:
    """Cells ``(x, y, w, h)`` of the camera view and of the square panel."""
    width, height = size
    gutter = _gutter(height)
    inner_h = height - 2 * gutter
    usable_w = width - 3 * gutter
    # On narrow frames the square panel shrinks so the camera view keeps a
    # usable share of the width; on 16:9 and wider it spans the inner height.
    side = max(1, min(inner_h, usable_w - int(_MIN_VIEW_SHARE * usable_w)))
    panel_x = width - gutter - side
    panel_y = gutter + (inner_h - side) // 2
    view = (gutter, gutter, panel_x - 2 * gutter, inner_h)
    return view, (panel_x, panel_y, side, side)


def _quad_cells(size: Size) -> List[Cell]:
    """The four cells ``(x, y, w, h)`` of the 2x2 grid in reading order."""
    width, height = size
    gutter = _gutter(height)
    cell_w = (width - 3 * gutter) // 2
    cell_h = (height - 3 * gutter) // 2
    xs = (gutter, width - gutter - cell_w)
    ys = (gutter, height - gutter - cell_h)
    return [(x, y, cell_w, cell_h) for y in ys for x in xs]


def _place(canvas: np.ndarray, image: np.ndarray, cell: Cell) -> None:
    x, y, w, h = cell
    if w > 0 and h > 0:
        canvas[y:y + h, x:x + w] = letterbox(image, (w, h))


def _draw_label(canvas: np.ndarray, text: str, cell: Cell) -> None:
    """Write ``text`` in the muted colour at the top-left corner of ``cell``."""
    if not text:
        return
    height = canvas.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, 0.5 * height / 1080.0)
    thickness = max(1, int(round(height / 1080.0)))
    pad = max(4, int(round(0.012 * height)))
    (_, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = (cell[0] + pad, cell[1] + pad + text_h)
    # A background-coloured outline keeps the label legible over bright footage.
    cv2.putText(canvas, text, origin, font, scale, BACKGROUND_BGR, thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, font, scale, MUTED_BGR, thickness, cv2.LINE_AA)


def fit_size(layout: str, view_size: Size, height: int = 1080) -> Size:
    """Frame size at which a view fills its cell of ``layout`` without bars.

    ``view_size`` is the ``(width, height)`` of the camera view, of which only
    the aspect ratio matters, and ``height`` the wanted frame height in
    pixels. The returned ``(width, height)`` is even in both dimensions, as
    H.264 with 4:2:0 chroma requires.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {', '.join(LAYOUTS)}")
    view_w, view_h = _checked_size(view_size)
    height = _even(int(height))
    aspect = view_w / view_h
    gutter = _gutter(height)
    if layout == "single":
        width = int(round(height * aspect))
    elif layout == "split":
        inner_h = height - 2 * gutter
        width = 3 * gutter + int(round(inner_h * aspect)) + inner_h
    else:
        cell_h = (height - 3 * gutter) // 2
        width = 3 * gutter + 2 * int(round(cell_h * aspect))
    return _even(width), height


def compose(
    layout: str,
    images: Sequence[np.ndarray],
    bev: Optional[np.ndarray] = None,
    *,
    size: Size = (1920, 1080),
    labels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Arrange camera views and the top-down panel into one frame.

    Parameters
    ----------
    layout:
        ``"single"`` shows the first view alone. ``"split"`` puts the first
        view on the left and ``bev`` in a square panel on the right that spans
        the inner height of the frame; without ``bev`` it is the same as
        ``"single"``. ``"quad"`` is a 2x2 grid with up to three views and
        ``bev`` in the last cell; without ``bev`` a fourth view takes that
        cell, and cells with nothing to show stay background.
    images:
        Camera views, primary view first. A single array counts as one view.
    bev:
        Top-down panel, usually from ``overlay.draw_bev``.
    size:
        ``(width, height)`` of the returned frame in pixels.
    labels:
        One short caption per view, drawn in the top-left corner of its cell.

    Returns
    -------
    numpy.ndarray
        BGR ``uint8`` frame of shape ``(height, width, 3)``. Margins and
        gutters are 1.5 percent of the frame height and use the page
        background colour.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {', '.join(LAYOUTS)}")
    if isinstance(images, np.ndarray) and images.ndim < 4:
        images = [images]
    if images is None or len(images) == 0:
        raise ValueError("compose needs at least one image")
    width, height = _checked_size(size)
    captions = list(labels) if labels is not None else []

    if layout == "single" or (layout == "split" and bev is None):
        frame = letterbox(images[0], (width, height))
        if captions:
            _draw_label(frame, captions[0], (0, 0, width, height))
        return frame

    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = BACKGROUND_BGR

    if layout == "split":
        view_cell, panel_cell = _split_cells((width, height))
        _place(frame, images[0], view_cell)
        _place(frame, bev, panel_cell)
        if captions:
            _draw_label(frame, captions[0], view_cell)
        return frame

    cells = _quad_cells((width, height))
    view_cells = cells[:3] if bev is not None else cells
    for i, (image, cell) in enumerate(zip(images, view_cells)):
        _place(frame, image, cell)
        if i < len(captions):
            _draw_label(frame, captions[i], cell)
    if bev is not None:
        _place(frame, bev, cells[3])
    return frame

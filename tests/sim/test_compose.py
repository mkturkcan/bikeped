"""Layouts keep the requested frame size and never distort a view."""

from __future__ import annotations

import numpy as np
import pytest

from bikeped.sim.compose import BACKGROUND_BGR, LAYOUTS, compose, fit_size, letterbox

WHITE = (255, 255, 255)
VIEW_COLOURS = ((40, 80, 200), (60, 180, 60), (200, 120, 40), (150, 40, 150))
BEV_COLOUR = (90, 90, 220)


def solid(width: int, height: int, colour) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = colour
    return image


def bounds(frame: np.ndarray, colour):
    """Inclusive ``(x1, y1, x2, y2)`` of the pixels that equal ``colour``."""
    mask = (frame == np.array(colour, dtype=np.uint8)).all(axis=2)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def is_background(pixel) -> bool:
    return tuple(int(v) for v in pixel) == BACKGROUND_BGR


@pytest.mark.parametrize("size", [(300, 300), (100, 100)])
def test_letterbox_preserves_aspect_and_centres(size):
    out = letterbox(solid(400, 200, WHITE), size)
    assert out.shape == (size[1], size[0], 3)
    assert out.dtype == np.uint8
    x1, y1, x2, y2 = bounds(out, WHITE)
    assert (x1, x2) == (0, size[0] - 1)
    assert y2 - y1 + 1 == size[1] // 2
    assert y1 == size[1] // 4
    assert is_background(out[0, 0]) and is_background(out[-1, -1])


def test_letterbox_pillarboxes_tall_images():
    out = letterbox(solid(100, 200, WHITE), (400, 200))
    assert bounds(out, WHITE) == (150, 0, 249, 199)


def test_letterbox_converts_gray_and_bgra():
    gray = np.full((50, 80), 255, dtype=np.uint8)
    bgra = np.full((50, 80, 4), 255, dtype=np.uint8)
    for image in (gray, bgra, gray[:, :, None]):
        out = letterbox(image, (160, 100))
        assert out.shape == (100, 160, 3)
        assert tuple(out[50, 80]) == WHITE


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("size", [(640, 360), (1280, 720), (500, 500)])
def test_layouts_return_requested_size(layout, size):
    views = [solid(160, 90, c) for c in VIEW_COLOURS[:3]]
    out = compose(layout, views, solid(120, 120, BEV_COLOUR), size=size)
    assert out.shape == (size[1], size[0], 3)
    assert out.dtype == np.uint8


def test_single_is_the_letterboxed_first_view():
    views = [solid(160, 120, VIEW_COLOURS[0]), solid(160, 90, VIEW_COLOURS[1])]
    out = compose("single", views, solid(100, 100, BEV_COLOUR), size=(640, 360))
    assert np.array_equal(out, letterbox(views[0], (640, 360)))


def test_split_has_square_panel_with_margins_and_gutter():
    width, height = 640, 360
    gutter = round(0.015 * height)
    out = compose(
        "split", [solid(160, 90, VIEW_COLOURS[0])], solid(200, 200, BEV_COLOUR),
        size=(width, height),
    )
    px1, py1, px2, py2 = bounds(out, BEV_COLOUR)
    side = height - 2 * gutter
    assert (px2 - px1 + 1, py2 - py1 + 1) == (side, side)
    assert (py1, px2) == (gutter, width - gutter - 1)

    vx1, vy1, vx2, vy2 = bounds(out, VIEW_COLOURS[0])
    assert vx1 == gutter
    assert px1 - vx2 - 1 == gutter
    assert abs((vx2 - vx1 + 1) / (vy2 - vy1 + 1) - 160 / 90) < 0.02

    for y, x in ((0, 0), (height - 1, width - 1), (height // 2, px1 - 1), (0, width // 2)):
        assert is_background(out[y, x])


def test_split_without_bev_falls_back_to_single():
    view = solid(160, 90, VIEW_COLOURS[0])
    assert np.array_equal(
        compose("split", [view], None, size=(640, 360)),
        compose("single", [view], size=(640, 360)),
    )


def test_quad_places_three_views_and_the_panel():
    width, height = 640, 360
    views = [solid(160, 90, c) for c in VIEW_COLOURS[:3]]
    out = compose("quad", views, solid(100, 100, BEV_COLOUR), size=(width, height))
    quarter = {
        VIEW_COLOURS[0]: (width // 4, height // 4),
        VIEW_COLOURS[1]: (3 * width // 4, height // 4),
        VIEW_COLOURS[2]: (width // 4, 3 * height // 4),
        BEV_COLOUR: (3 * width // 4, 3 * height // 4),
    }
    for colour, (x, y) in quarter.items():
        assert tuple(int(v) for v in out[y, x]) == colour
    assert is_background(out[height // 2, width // 2])
    assert is_background(out[0, 0])


def test_quad_leaves_missing_cells_as_background():
    out = compose("quad", [solid(160, 90, VIEW_COLOURS[0])], None, size=(640, 360))
    assert tuple(int(v) for v in out[90, 160]) == VIEW_COLOURS[0]
    for y, x in ((90, 480), (270, 160), (270, 480)):
        assert is_background(out[y, x])


def test_quad_without_bev_uses_the_last_cell_for_a_fourth_view():
    views = [solid(160, 90, c) for c in VIEW_COLOURS]
    out = compose("quad", views, None, size=(640, 360))
    assert tuple(int(v) for v in out[270, 480]) == VIEW_COLOURS[3]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_labels_are_drawn_inside_the_frame(layout):
    views = [solid(160, 90, (0, 0, 0)) for _ in range(3)]
    bev = solid(100, 100, (0, 0, 0))
    plain = compose(layout, views, bev, size=(960, 540))
    labelled = compose(layout, views, bev, size=(960, 540), labels=["pole", "fisheye", "chase"])
    assert labelled.shape == plain.shape
    assert not np.array_equal(plain, labelled)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_fit_size_fills_the_view_cell(layout):
    width, height = fit_size(layout, (1920, 1080), height=360)
    assert height == 360 and width % 2 == 0
    gutter = round(0.015 * height)
    cell_w, cell_h = {
        "single": (width, height),
        "split": (width - 3 * gutter - (height - 2 * gutter), height - 2 * gutter),
        "quad": ((width - 3 * gutter) // 2, (height - 3 * gutter) // 2),
    }[layout]
    out = compose(
        layout, [solid(192, 108, WHITE)], solid(100, 100, BEV_COLOUR), size=(width, height)
    )
    x1, y1, x2, y2 = bounds(out, WHITE)
    assert 0 <= cell_w - (x2 - x1 + 1) <= 2
    assert 0 <= cell_h - (y2 - y1 + 1) <= 2


def test_single_array_counts_as_one_view():
    view = solid(160, 90, VIEW_COLOURS[0])
    assert np.array_equal(compose("single", view, size=(640, 360)),
                          compose("single", [view], size=(640, 360)))


def test_fit_size_rejects_unknown_layout():
    with pytest.raises(ValueError, match="mosaic"):
        fit_size("mosaic", (1920, 1080))


def test_unknown_layout_and_empty_input_raise():
    view = solid(160, 90, WHITE)
    with pytest.raises(ValueError, match="mosaic"):
        compose("mosaic", [view])
    with pytest.raises(ValueError):
        compose("single", [])
    with pytest.raises(ValueError):
        compose("single", [view], size=(0, 360))

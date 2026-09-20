"""Unit tests for the core logic in wm_remover.py."""

from PIL import Image

from wm_remover import (
    MyImg,
    alpha_img_builder,
    build_ev_image,
    build_variance_image,
    color_img_builder,
    fill_gaps,
    original_image_builder,
    points_in_circle,
    search_for_point,
)


def test_myimg_initializes_to_zero():
    img = MyImg((2, 3))
    assert img.size == (2, 3)
    assert len(img.data) == 3
    assert img.color(1, 2) == (0, 0, 0)


def test_myimg_each_px_builder():
    img = MyImg((2, 2), each_px=lambda i, x, y: i + x + y)
    assert img.color(0, 0) == (0, 1, 2)
    assert img.color(1, 1) == (2, 3, 4)


def test_myimg_to_image():
    img = MyImg((2, 2), each_px=lambda i, x, y: 0)
    pil_img = img.to_Image()
    assert pil_img.size == (2, 2)
    assert pil_img.getpixel((0, 0)) == (0, 0, 0)


def test_points_in_circle_stays_within_radius():
    pts = list(points_in_circle((5, 5), 2, (0, 10), (0, 10)))
    assert (5, 5) in pts
    assert (6, 5) in pts
    assert (8, 5) not in pts
    for x, y in pts:
        assert (x - 5) ** 2 + (y - 5) ** 2 <= 4


def test_points_in_circle_respects_bounds():
    pts = list(points_in_circle((0, 0), 3, (0, 10), (0, 10)))
    assert pts
    assert all(x >= 0 and y >= 0 for x, y in pts)


def test_search_for_point_finds_first_match():
    def ignore_pt(i, x, y):
        return 1 if x >= 3 else 0

    assert search_for_point((0, 0), 0, (1, 0), ignore_pt, (0, 10), (0, 10)) == (3, 0)


def test_search_for_point_overshoot():
    def ignore_pt(i, x, y):
        return 1 if x >= 3 else 0

    # overshoot delays the return by one loop iteration but does not advance
    # the position, so the first matching point is still returned.
    result = search_for_point((0, 0), 0, (1, 0), ignore_pt, (0, 10), (0, 10), overshoot=1)
    assert result == (3, 0)


def test_search_for_point_returns_none_out_of_bounds():
    def ignore_pt(i, x, y):
        return 0

    assert search_for_point((0, 0), 0, (1, 0), ignore_pt, (0, 10), (0, 10)) is None


def test_alpha_img_builder_zero_variance():
    var_img = MyImg((1, 1), each_px=lambda i, x, y: 50)
    var_no_wm = MyImg((1, 1), each_px=lambda i, x, y: 0)
    builder = alpha_img_builder(var_img, var_no_wm)
    assert builder(0, 0, 0) == 0


def test_alpha_img_builder_ratio():
    var_img = MyImg((1, 1), each_px=lambda i, x, y: 25)
    var_no_wm = MyImg((1, 1), each_px=lambda i, x, y: 100)
    builder = alpha_img_builder(var_img, var_no_wm)
    # ratio = 0.25 -> sqrt = 0.5 -> round(255 * 0.5) = 128
    assert builder(0, 0, 0) == 128


def test_color_img_builder():
    ev_img = MyImg((1, 1), each_px=lambda i, x, y: 100)
    ev_no_wm = MyImg((1, 1), each_px=lambda i, x, y: 50)
    alpha_img = MyImg((1, 1), each_px=lambda i, x, y: 128)
    builder = color_img_builder(ev_img, ev_no_wm, alpha_img)
    # a = 128/255; round(100/a + 50 * (1 - 1/a)) = 150
    assert builder(0, 0, 0) == 150


def test_color_img_builder_zero_alpha():
    ev_img = MyImg((1, 1), each_px=lambda i, x, y: 100)
    ev_no_wm = MyImg((1, 1), each_px=lambda i, x, y: 50)
    alpha_img = MyImg((1, 1), each_px=lambda i, x, y: 0)
    builder = color_img_builder(ev_img, ev_no_wm, alpha_img)
    assert builder(0, 0, 0) == 0


def test_original_image_builder_reconstructs():
    target = MyImg((1, 1), each_px=lambda i, x, y: 100)
    wm_alpha = MyImg((1, 1), each_px=lambda i, x, y: 128)
    wm_color = MyImg((1, 1), each_px=lambda i, x, y: 40)
    builder = original_image_builder(target, wm_alpha, wm_color)
    # a = 128/255; I = (100 - a*40) / (1 - a) = 160
    assert builder(0, 0, 0) == 160


def test_original_image_builder_fully_watermarked():
    target = MyImg((1, 1), each_px=lambda i, x, y: 100)
    wm_alpha = MyImg((1, 1), each_px=lambda i, x, y: 255)
    wm_color = MyImg((1, 1), each_px=lambda i, x, y: 40)
    builder = original_image_builder(target, wm_alpha, wm_color)
    assert builder(0, 0, 0) == 0


def test_fill_gaps_fills_watermarked_region():
    base = MyImg((4, 4), each_px=lambda i, x, y: 100)

    def ignore_pt(i, x, y):
        return 1.0 if x < 2 else 0.0

    result = fill_gaps((4, 4), base, ignore_pt, search_radius=3)
    assert result.color(0, 0) == (100, 100, 100)
    assert result.color(2, 0) == (100, 100, 100)
    assert result.color(3, 0) == (100, 100, 100)


def test_build_ev_image(tmp_path):
    size = (2, 2)
    for name, color in [("a.png", (10, 20, 30)), ("b.png", (30, 20, 10))]:
        Image.new("RGB", size, color).save(tmp_path / name)

    ev = build_ev_image(size, str(tmp_path) + "/", ["a.png", "b.png"])
    assert ev.color(0, 0) == (20, 20, 20)


def test_build_variance_image(tmp_path):
    size = (2, 2)
    for name, color in [("a.png", (10, 20, 30)), ("b.png", (30, 40, 50))]:
        Image.new("RGB", size, color).save(tmp_path / name)

    ev = build_ev_image(size, str(tmp_path) + "/", ["a.png", "b.png"])
    var = build_variance_image(size, ev, str(tmp_path) + "/", ["a.png", "b.png"])
    # each channel has squared deviation 200 -> mapped to 255
    assert var.color(0, 0) == (255, 255, 255)
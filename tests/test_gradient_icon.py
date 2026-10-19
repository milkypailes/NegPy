from PyQt6.QtCore import QSize
from PyQt6.QtGui import QColor, QIcon

from negpy.desktop.view.styles.templates import Sweep, gradient_icon
from negpy.desktop.view.styles.theme import THEME


def _opaque(image) -> list[QColor]:
    """The glyph's pixels, left to right."""
    return [image.pixelColor(x, y) for x in range(image.width()) for y in range(image.height()) if image.pixelColor(x, y).alpha() > 200]


def test_the_glyph_runs_red_on_the_left_to_blue_on_the_right():
    pixels = _opaque(gradient_icon("fa5s.square").pixmap(QSize(32, 32)).toImage())
    left, right = pixels[0], pixels[-1]

    assert left.red() > max(left.green(), left.blue())
    assert right.blue() > max(right.red(), right.green())


def test_a_disabled_glyph_stays_flat_gray():
    image = gradient_icon("fa5s.square").pixmap(QSize(32, 32), QIcon.Mode.Disabled).toImage()

    pixels = _opaque(image)
    assert pixels and all(c.red() == c.green() == c.blue() for c in pixels)


def test_a_sweep_at_90_degrees_runs_top_to_bottom():
    sweep = Sweep((THEME.channel_red_text, THEME.channel_blue_text), angle=90)
    image = gradient_icon("fa5s.square", sweep).pixmap(QSize(32, 32)).toImage()
    column = [image.pixelColor(16, y) for y in range(32) if image.pixelColor(16, y).alpha() > 200]
    top, bottom = column[0], column[-1]

    assert top.red() > top.blue()
    assert bottom.blue() > bottom.red()


def test_a_checked_glyph_is_flat_on_accent():
    """A checked tool toggle sits on the accent fill, where a red stop would vanish."""
    image = gradient_icon("fa5s.square").pixmap(QSize(32, 32), QIcon.Mode.Normal, QIcon.State.On).toImage()

    assert {c.name() for c in _opaque(image)} == {QColor(THEME.text_on_accent).name()}

"""A resize draws the frame at once, before Qt paints the last bitmap stretched to the new size."""

from unittest.mock import MagicMock

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.canvas.gpu_widget import GPUCanvasWidget


def _resize(widget) -> None:
    QApplication.sendEvent(widget, QResizeEvent(QSize(600, 300), QSize(400, 300)))


def test_a_resize_with_a_frame_draws_at_once(qapp):
    widget = GPUCanvasWidget()
    widget.canvas.force_draw = MagicMock()
    widget.current_texture_view = object()

    _resize(widget)

    widget.canvas.force_draw.assert_called_once()


def test_a_resize_without_a_frame_draws_nothing(qapp):
    widget = GPUCanvasWidget()
    widget.canvas.force_draw = MagicMock()

    _resize(widget)

    widget.canvas.force_draw.assert_not_called()


def test_a_resize_during_the_draw_does_not_draw_again(qapp):
    widget = GPUCanvasWidget()
    widget.current_texture_view = object()
    widget.canvas.force_draw = MagicMock(side_effect=lambda: _resize(widget))

    _resize(widget)

    widget.canvas.force_draw.assert_called_once()

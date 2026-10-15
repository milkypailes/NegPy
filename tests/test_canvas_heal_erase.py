from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay


def _mouse_event(kind: QEvent.Type, pos: QPointF, buttons=Qt.MouseButton.LeftButton) -> QMouseEvent:
    return QMouseEvent(kind, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)


def _overlay() -> CanvasOverlay:
    overlay = CanvasOverlay(AppState())
    overlay._view_rect = QRectF(0, 0, 100, 100)
    overlay.set_tool_mode(ToolMode.HEAL_ERASE)
    return overlay


def test_drag_emits_the_deletion_rect() -> None:
    overlay = _overlay()
    emitted = []
    overlay.heal_erase_rect.connect(emitted.append)

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(10, 10)))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(60, 50), Qt.MouseButton.NoButton))

    assert len(emitted) == 1
    (c1, c2) = emitted[0]
    assert c1 == (0.1, 0.1) and c2 == (0.6, 0.5)
    assert overlay._erase_p1 is None and overlay._erase_p2 is None


def test_click_without_drag_emits_nothing() -> None:
    overlay = _overlay()
    emitted = []
    overlay.heal_erase_rect.connect(emitted.append)

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(10, 10)))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(12, 11), Qt.MouseButton.NoButton))

    assert emitted == []
    assert overlay._erase_p1 is None


def test_press_outside_the_frame_arms_nothing() -> None:
    overlay = _overlay()

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(150, 150)))

    assert overlay._erase_p1 is None


def test_leaving_the_tool_clears_the_rubber_band() -> None:
    overlay = _overlay()
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(10, 10)))
    assert overlay._erase_p1 is not None

    overlay.set_tool_mode(ToolMode.NONE)
    assert overlay._erase_p1 is None and overlay._erase_p2 is None

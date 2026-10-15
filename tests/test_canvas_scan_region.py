from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay


def _mouse_event(kind: QEvent.Type, pos: QPointF, buttons=Qt.MouseButton.LeftButton) -> QMouseEvent:
    return QMouseEvent(kind, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)


def _overlay() -> CanvasOverlay:
    overlay = CanvasOverlay(AppState())
    overlay._view_rect = QRectF(0, 0, 100, 100)
    overlay.set_tool_mode(ToolMode.SCAN_REGION)
    return overlay


def test_drag_emits_the_scan_region() -> None:
    overlay = _overlay()
    emitted = []
    overlay.scan_region_picked.connect(emitted.append)

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(10, 20)))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(60, 50), Qt.MouseButton.NoButton))

    assert len(emitted) == 1
    (c1, c2) = emitted[0]
    assert c1 == (0.1, 0.2) and c2 == (0.6, 0.5)
    assert overlay._scan_p1 is None and overlay._scan_p2 is None


def test_click_without_drag_emits_nothing() -> None:
    overlay = _overlay()
    emitted = []
    overlay.scan_region_picked.connect(emitted.append)

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(10, 10)))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(12, 11), Qt.MouseButton.NoButton))

    assert emitted == []
    assert overlay._scan_p1 is None


def test_leaving_the_tool_clears_the_rubber_band() -> None:
    overlay = _overlay()
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(10, 10)))
    assert overlay._scan_p1 is not None

    overlay.set_tool_mode(ToolMode.NONE)
    assert overlay._scan_p1 is None and overlay._scan_p2 is None


def test_escape_clears_a_half_drawn_region() -> None:
    overlay = _overlay()
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(10, 10)))

    assert overlay.cancel_in_progress() is True
    assert overlay._scan_p1 is None

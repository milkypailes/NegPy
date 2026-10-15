"""Full-res dust scan options: which dust polarity to find, over the whole image or a picked rectangle."""

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QDialog, QHBoxLayout, QLabel, QPushButton, QRadioButton, QVBoxLayout

from negpy.desktop.view.styles import templates
from negpy.desktop.view.styles.templates import hint_label, pin_dialog_default, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry

_POLARITIES = (
    ("print_bright", "Bright dust", "Bright spots on the print: dust that blocked the scanner light."),
    ("print_dark", "Dark dust", "Dark spots on the print: film the camera never exposed."),
    ("both", "Both", "Every detectable speck, either polarity."),
)


class DustScanDialog(QDialog):
    """Polarity and area for one full-resolution dust scan. After ``accept()``,
    ``polarity_value()`` names the dust to find and ``region_value()`` the rectangle
    to cover, or None for the whole image."""

    select_requested = pyqtSignal()

    def __init__(
        self,
        initial_polarity: str = "both",
        initial_region: Optional[tuple[float, float, float, float]] = None,
        parent=None,
        *,
        repo=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Full-Res Dust Scan")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(templates.section_subheader("FIND"))
        self._polarity_group = QButtonGroup(self)
        self._polarity_btns = {}
        for key, label, tip in _POLARITIES:
            btn = QRadioButton(label)
            btn.setToolTip(wrap_tooltip(tip))
            btn.toggled.connect(lambda _on, k=key: _on and self._refresh())
            layout.addWidget(btn)
            self._polarity_group.addButton(btn)
            self._polarity_btns[key] = btn
        if initial_polarity not in self._polarity_btns:
            initial_polarity = "both"

        layout.addWidget(templates.section_subheader("COVER"))
        self._scope_group = QButtonGroup(self)
        self._whole_btn = QRadioButton("Whole image")
        self._whole_btn.setToolTip(wrap_tooltip("Scan the full-resolution decode end to end, as before."))
        self._whole_btn.toggled.connect(lambda _on: _on and self._refresh())
        layout.addWidget(self._whole_btn)
        self._region_btn = QRadioButton("Rectangle")
        self._scope_group.addButton(self._whole_btn)
        self._scope_group.addButton(self._region_btn)
        self._region_btn.setToolTip(wrap_tooltip("Scan only the picked rectangle: faster, and nothing outside it is touched."))
        self._region_btn.toggled.connect(lambda _on: _on and self._refresh())
        layout.addWidget(self._region_btn)

        region_row = QHBoxLayout()
        self._select_btn = QPushButton("Select on Canvas…")
        self._select_btn.setToolTip(
            wrap_tooltip("Close this dialog and drag a rectangle over the photo; it reopens with the rectangle set.")
        )
        self._select_btn.clicked.connect(self._on_select)
        region_row.addWidget(self._select_btn)
        self._region_label = QLabel()
        self._region_label.setStyleSheet(f"color: {THEME.text_hint};")
        region_row.addWidget(self._region_label, 1)
        layout.addLayout(region_row)

        self._hint = hint_label()
        layout.addWidget(self._hint)

        footer = QHBoxLayout()
        footer.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        footer.addWidget(self._cancel_btn)
        self._scan_btn = QPushButton("Scan")
        self._scan_btn.clicked.connect(self.accept)
        footer.addWidget(self._scan_btn)
        pin_dialog_default(self._scan_btn, self._cancel_btn, scope=self)
        layout.addLayout(footer)

        self._region = initial_region
        # Checked states last: each toggled emission refreshes widgets below.
        self._polarity_btns[initial_polarity].setChecked(True)
        if initial_region is not None:
            self._region_btn.setChecked(True)
        else:
            self._whole_btn.setChecked(True)
        self._refresh()
        remember_dialog_geometry(self, repo, "dust_scan")

    def _on_select(self) -> None:
        self._region_btn.setChecked(True)
        self.select_requested.emit()

    def set_region(self, rect: Optional[tuple[float, float, float, float]]) -> None:
        self._region = rect
        if rect is not None:
            self._region_btn.setChecked(True)
        self._refresh()

    def _refresh(self) -> None:
        if self._region is None:
            self._region_label.setText("No rectangle picked yet.")
        else:
            x1, y1, x2, y2 = self._region
            pct = max(0.0, (x2 - x1) * (y2 - y1)) * 100.0
            self._region_label.setText(f"Rectangle: {pct:.0f}% of the frame.")
        needs_rect = self._region_btn.isChecked() and self._region is None
        self._scan_btn.setEnabled(not needs_rect)
        self._hint.setText("Pick a rectangle on the canvas first." if needs_rect else "")

    def polarity_value(self) -> str:
        for key, btn in self._polarity_btns.items():
            if btn.isChecked():
                return key
        return "both"

    def use_region(self) -> bool:
        return self._region_btn.isChecked() and self._region is not None

    def region_value(self) -> Optional[tuple[float, float, float, float]]:
        return self._region if self.use_region() else None

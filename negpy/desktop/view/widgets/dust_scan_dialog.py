"""Full-res dust scan options: which dust polarity to find."""

from PyQt6.QtWidgets import QButtonGroup, QDialog, QHBoxLayout, QLabel, QPushButton, QRadioButton, QVBoxLayout

from negpy.desktop.view.styles.templates import pane_header_qss, pin_dialog_default, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry

_POLARITIES = (
    ("print_bright", "Bright dust", "Bright spots on the print: dust that blocked the scanner light."),
    ("print_dark", "Dark dust", "Dark spots on the print: film the camera never exposed."),
    ("both", "Both", "Every detectable speck, either polarity."),
)


class DustScanDialog(QDialog):
    """Polarity for one full-resolution dust scan. The area is chosen before the dialog
    opens, so after ``accept()`` ``polarity_value()`` names the dust to find."""

    def __init__(
        self,
        initial_polarity: str = "both",
        parent=None,
        *,
        repo=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Full-Res Dust Scan")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(THEME.space_xl, THEME.space_xl, THEME.space_xl, THEME.space_xl)
        layout.setSpacing(THEME.space_md)

        find = QLabel("FIND")
        find.setStyleSheet(pane_header_qss())
        layout.addWidget(find)
        self._polarity_group = QButtonGroup(self)
        self._polarity_btns = {}
        for key, label, tip in _POLARITIES:
            btn = QRadioButton(label)
            btn.setToolTip(wrap_tooltip(tip))
            layout.addWidget(btn)
            self._polarity_group.addButton(btn)
            self._polarity_btns[key] = btn
        if initial_polarity not in self._polarity_btns:
            initial_polarity = "both"

        footer = QHBoxLayout()
        footer.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        footer.addWidget(self._cancel_btn)
        self._scan_btn = QPushButton("Scan")
        self._scan_btn.clicked.connect(self.accept)
        footer.addWidget(self._scan_btn)
        pin_dialog_default(self._scan_btn, self._cancel_btn, scope=self)
        layout.addSpacing(THEME.space_md)
        layout.addLayout(footer)

        self._polarity_btns[initial_polarity].setChecked(True)
        # Content-sized height: a restored geometry from a taller dialog would otherwise
        # leave a gap, and the Preferred-height FIND label would absorb it.
        self.setFixedHeight(self.sizeHint().height())
        remember_dialog_geometry(self, repo, "dust_scan")

    def polarity_value(self) -> str:
        for key, btn in self._polarity_btns.items():
            if btn.isChecked():
                return key
        return "both"

"""Full-res dust scan options: which dust polarity to find."""

from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from negpy.desktop.view.styles.templates import pane_header_qss, pin_dialog_default
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.choice_button import SegmentedChoice
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.kernel.system.i18n import tr

_POLARITIES = (
    ("print_bright", tr("Bright dust"), tr("Bright spots on the print: dust that blocked the scanner light.")),
    ("print_dark", tr("Dark dust"), tr("Dark spots on the print: film the camera never exposed.")),
    ("both", tr("Both"), tr("Every detectable speck, either polarity.")),
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
        self.setWindowTitle(tr("Full-Res Dust Scan"))
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(THEME.space_xl, THEME.space_xl, THEME.space_xl, THEME.space_xl)
        layout.setSpacing(THEME.space_md)

        find = QLabel(tr("FIND"))
        find.setStyleSheet(pane_header_qss())
        layout.addWidget(find)
        self._polarity = SegmentedChoice(
            tuple(("", label) for _key, label, _tip in _POLARITIES),
            tr("Which dust to find"),
            data=tuple(key for key, _label, _tip in _POLARITIES),
        )
        for i, (_key, _label, tip) in enumerate(_POLARITIES):
            self._polarity.set_choice_tooltip(i, tip)
        index = self._polarity.findData(initial_polarity)
        if index < 0:
            index = self._polarity.findData("both")
        if index >= 0:
            self._polarity.setCurrentIndex(index)
        layout.addWidget(self._polarity)

        footer = QHBoxLayout()
        footer.addStretch()
        self._cancel_btn = QPushButton(tr("Cancel"))
        self._cancel_btn.clicked.connect(self.reject)
        footer.addWidget(self._cancel_btn)
        self._scan_btn = QPushButton(tr("Scan"))
        self._scan_btn.clicked.connect(self.accept)
        footer.addWidget(self._scan_btn)
        pin_dialog_default(self._scan_btn, self._cancel_btn, scope=self)
        layout.addSpacing(THEME.space_md)
        layout.addLayout(footer)

        # Content-sized height: a restored geometry from a taller dialog would otherwise
        # leave a gap, and the Preferred-height FIND label would absorb it.
        self.setFixedHeight(self.sizeHint().height())
        remember_dialog_geometry(self, repo, "dust_scan")

    def polarity_value(self) -> str:
        return self._polarity.currentData()

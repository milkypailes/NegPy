from negpy.kernel.system.i18n import tr
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLineEdit, QVBoxLayout

from negpy.desktop.view.styles.templates import field_label, hint_label, pin_button_box, set_hint_kind, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.features.geometry.models import parse_custom_crop_ratio


class CropRatioDialog(QDialog):
    """One width:height entry for the Ratio picker's Custom choice. Accept stays disabled
    until the text is two positive numbers."""

    def __init__(self, current: str = "", parent=None, *, repo=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Custom Crop Ratio"))
        self.setMinimumWidth(320)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_xl, THEME.space_xl, THEME.space_xl, THEME.space_xl)
        root.setSpacing(THEME.space_lg)

        root.addWidget(field_label(tr("Aspect ratio")))
        self.ratio_edit = QLineEdit(parse_custom_crop_ratio(current) or "3:2")
        self.ratio_edit.setPlaceholderText("3:2")
        self.ratio_edit.setToolTip(wrap_tooltip(tr("Width:height of the crop, for example 3:2 or 16:9")))
        self.ratio_edit.selectAll()
        root.addWidget(self.ratio_edit)

        self.hint = hint_label()
        root.addWidget(self.hint)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        pin_button_box(buttons)
        self._buttons = buttons
        root.addWidget(buttons)

        self.ratio_edit.textChanged.connect(self._validate)
        self._validate(self.ratio_edit.text())
        self.ratio_edit.setFocus()
        remember_dialog_geometry(self, repo, "custom_crop_ratio")

    def _validate(self, text: str) -> None:
        valid = parse_custom_crop_ratio(text) is not None
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(valid)
        set_hint_kind(self.hint, "muted" if valid else "error")
        self.hint.setText(
            tr("Width:height, for example 3:2.") if valid else tr("Enter width:height with positive numbers, for example 3:2.")
        )

    def ratio(self) -> str:
        return parse_custom_crop_ratio(self.ratio_edit.text()) or ""

"""The crop ratio dropdown, shared by the Geometry and Crop cards.

Both cards write the same roll field, so a Custom ratio entered in one must display in
the other. The list is the standard shapes plus a Custom… action; a stored custom ratio
adds its own item until the ratio changes again.
"""

from negpy.kernel.system.i18n import tr
from PyQt6.QtWidgets import QComboBox

from negpy.desktop.view.styles.templates import add_choice_items
from negpy.domain.models import CROP_RATIO_CHOICES, canonical_crop_ratio

# Not a ratio: selecting it opens the Custom Crop Ratio dialog. Kept out of the item
# data space real ratios use.
CUSTOM_RATIO_ACTION = "__custom_crop_ratio__"

_STANDARD_RATIOS = frozenset(r.value for r in CROP_RATIO_CHOICES)


def fill_crop_ratio_combo(combo: QComboBox, current: str) -> None:
    """Build the list once, at panel construction."""
    add_choice_items(combo, [r.value for r in CROP_RATIO_CHOICES])
    combo.addItem(tr("Custom…"), CUSTOM_RATIO_ACTION)
    sync_crop_ratio_combo(combo, current)


def sync_crop_ratio_combo(combo: QComboBox, current: str) -> None:
    """Show *current*: one of the standard shapes, or a custom ratio's own item before
    Custom…. Signals must already be blocked when the list changes under the user."""
    value = canonical_crop_ratio(str(current))
    for i in reversed(range(combo.count())):
        data = combo.itemData(i)
        if data != CUSTOM_RATIO_ACTION and data not in _STANDARD_RATIOS:
            combo.removeItem(i)
    if value not in _STANDARD_RATIOS:
        combo.insertItem(combo.findData(CUSTOM_RATIO_ACTION), value, value)
    combo.setCurrentIndex(combo.findData(value))

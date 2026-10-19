from PyQt6.QtWidgets import QLabel

from negpy.desktop.view.widgets.dust_scan_dialog import DustScanDialog


def test_defaults_to_both_polarities(qapp) -> None:
    dialog = DustScanDialog()
    assert dialog.polarity_value() == "both"


def test_polarity_choice_reads_back(qapp) -> None:
    dialog = DustScanDialog(initial_polarity="print_dark")
    assert dialog.polarity_value() == "print_dark"


def test_unknown_polarity_falls_back_to_both(qapp) -> None:
    dialog = DustScanDialog(initial_polarity="sideways")
    assert dialog.polarity_value() == "both"


def test_switching_polarity_reads_back(qapp) -> None:
    dialog = DustScanDialog()
    dialog._polarity.setCurrentIndex(dialog._polarity.findData("print_bright"))
    assert dialog.polarity_value() == "print_bright"


def test_scan_is_ready_without_a_rectangle(qapp) -> None:
    dialog = DustScanDialog()
    assert dialog._scan_btn.isEnabled() is True


def test_the_dialog_has_no_region_picker(qapp) -> None:
    dialog = DustScanDialog()
    assert not hasattr(dialog, "_region_btn")
    assert not hasattr(dialog, "region_value")


def test_the_header_does_not_take_a_taller_dialogs_slack(qapp) -> None:
    """A stored geometry taller than the content must not stretch FIND over the choices."""
    dialog = DustScanDialog()
    dialog.resize(240, 320)
    dialog.show()
    qapp.processEvents()

    label = next(child for child in dialog.findChildren(QLabel) if child.text() == "FIND")
    assert label.height() <= label.sizeHint().height() + 2
    dialog.close()

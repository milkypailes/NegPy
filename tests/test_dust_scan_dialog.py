from negpy.desktop.view.widgets.dust_scan_dialog import DustScanDialog


def test_defaults_repeat_the_plain_scan(qapp) -> None:
    dialog = DustScanDialog()
    assert dialog.polarity_value() == "both"
    assert dialog.use_region() is False
    assert dialog.region_value() is None


def test_polarity_choice_reads_back(qapp) -> None:
    dialog = DustScanDialog(initial_polarity="print_dark")
    assert dialog.polarity_value() == "print_dark"


def test_region_arrives_preset_with_scope(qapp) -> None:
    dialog = DustScanDialog(initial_region=(0.1, 0.2, 0.4, 0.5))
    assert dialog.use_region() is True
    assert dialog.region_value() == (0.1, 0.2, 0.4, 0.5)


def test_scan_blocks_on_missing_rectangle(qapp) -> None:
    dialog = DustScanDialog()
    dialog._region_btn.setChecked(True)
    assert dialog.use_region() is False
    assert dialog.region_value() is None
    assert dialog._scan_btn.isEnabled() is False


def test_select_arms_the_rectangle_scope(qapp) -> None:
    dialog = DustScanDialog()
    emitted = []
    dialog.select_requested.connect(lambda: emitted.append(True))
    dialog._select_btn.click()
    assert emitted == [True]
    assert dialog._region_btn.isChecked()


def test_picked_rectangle_enables_scan(qapp) -> None:
    dialog = DustScanDialog()
    dialog.set_region((0.1, 0.2, 0.4, 0.5))
    assert dialog.use_region() is True
    assert dialog._scan_btn.isEnabled() is True

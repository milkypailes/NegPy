"""The Crop Ratio picker's Custom entry: parse, store, display and dialog validation."""

import json
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QDialog, QDialogButtonBox

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.desktop.view.widgets.crop_ratio_dialog import CropRatioDialog
from negpy.desktop.view.widgets.crop_ratio_picker import CUSTOM_RATIO_ACTION, fill_crop_ratio_combo, sync_crop_ratio_combo
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.logic import enforce_roi_aspect_ratio
from negpy.features.geometry.models import AspectRatio, GeometryConfig, parse_custom_crop_ratio
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3:2", "3:2"),
        (" 16 : 9 ", "16:9"),
        ("03:02", "3:2"),
        ("0.5:2", "0.5:2"),
        ("4:1.0", "4:1"),
        ("1:1.4142", "1:1.4142"),
    ],
)
def test_parse_accepts_and_normalizes_width_height(text, expected):
    assert parse_custom_crop_ratio(text) == expected


@pytest.mark.parametrize("text", ["", "3", "3:", ":2", "3:0", "0:1", "-3:2", "3:-2", "3:2:1", "abc", "3x2", "1e2:1", "3: 2px"])
def test_parse_rejects_non_positive_or_malformed(text):
    assert parse_custom_crop_ratio(text) is None


def test_geometry_config_keeps_a_valid_custom_ratio():
    assert GeometryConfig(autocrop_ratio="16:11").autocrop_ratio == "16:11"


def test_geometry_config_drops_an_unparseable_ratio_to_free():
    assert GeometryConfig(autocrop_ratio="13x17").autocrop_ratio == AspectRatio.FREE


def test_a_custom_ratio_survives_a_config_round_trip():
    config = WorkspaceConfig(geometry=GeometryConfig(autocrop_ratio="16:11"))
    reloaded = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(config.to_dict())))
    assert reloaded.geometry.autocrop_ratio == "16:11"


def test_enforce_roi_aspect_ratio_reads_a_custom_ratio():
    roi = enforce_roi_aspect_ratio((0, 300, 0, 300), 300, 300, "16:11")
    y1, y2, x1, x2 = roi
    assert (y2 - y1) / (x2 - x1) == pytest.approx(11 / 16, rel=0.02)


def test_picker_lists_the_standard_shapes_and_custom(qapp):
    combo = StepCombo()
    fill_crop_ratio_combo(combo, "4:3")
    assert combo.currentData() == "4:3"
    assert combo.findData(CUSTOM_RATIO_ACTION) >= 0


def test_picker_shows_a_custom_ratio_as_its_own_item(qapp):
    combo = StepCombo()
    fill_crop_ratio_combo(combo, "4:3")

    sync_crop_ratio_combo(combo, "16:11")
    assert combo.currentData() == "16:11"
    assert combo.currentText() == "16:11"

    sync_crop_ratio_combo(combo, "3:2")
    assert combo.currentData() == "3:2"
    assert combo.findData("16:11") < 0


def test_dialog_disables_accept_until_the_text_is_two_positive_numbers(qapp):
    dialog = CropRatioDialog("4:3")
    assert dialog.ratio() == "4:3"

    ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
    dialog.ratio_edit.setText("16:11")
    assert ok.isEnabled()

    dialog.ratio_edit.setText("16:0")
    assert not ok.isEnabled()
    assert dialog.ratio() == ""


def _panel():
    controller = MagicMock()
    controller.state = AppState()
    controller.state.config = DEFAULT_WORKSPACE_CONFIG
    return controller, ControlsPanel(controller)


def _select_custom(panel):
    combo = panel.geometry_sidebar.ratio_combo
    combo.setCurrentIndex(combo.findData(CUSTOM_RATIO_ACTION))


def test_selecting_custom_opens_the_dialog_and_sets_the_entered_ratio(qapp, monkeypatch):
    controller, panel = _panel()
    monkeypatch.setattr(CropRatioDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(CropRatioDialog, "ratio", lambda self: "16:11")

    _select_custom(panel)

    controller.set_crop_ratio.assert_called_once_with("16:11")


def test_cancelling_custom_restores_the_stored_ratio(qapp, monkeypatch):
    controller, panel = _panel()
    monkeypatch.setattr(CropRatioDialog, "exec", lambda self: QDialog.DialogCode.Rejected)

    _select_custom(panel)

    controller.set_crop_ratio.assert_not_called()
    assert panel.geometry_sidebar.ratio_combo.currentData() == "Free"

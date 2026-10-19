"""The crosstalk editor's mixing terms: one labeled slider row per term, wide enough for a measured matrix."""

import pytest

from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.crosstalk import CrosstalkProfiles


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(tmp_path / "_none"))


def test_each_term_is_a_named_row_and_a_wide_term_round_trips() -> None:
    from negpy.desktop.view.widgets.crosstalk_editor_dialog import CrosstalkEditorDialog

    wide = [1.0, -0.85, 0.1, 0.2, 1.0, -0.6, 0.0, 0.9, 1.0]
    CrosstalkProfiles.save("Measured Rig", wide)
    dlg = CrosstalkEditorDialog("Measured Rig", 1.0)
    labels = [s.label.text() for s in dlg.findChildren(CompactSlider) if s.label.text().startswith("From")]
    assert labels == ["From Green", "From Blue", "From Red", "From Blue", "From Red", "From Green"]
    assert dlg.working_matrix() == pytest.approx(wide)
    dlg._on_save()
    assert CrosstalkProfiles.get_matrix("Measured Rig") == pytest.approx(wide)

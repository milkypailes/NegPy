import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from negpy.desktop import file_manager
from negpy.desktop.controller import AppController
from negpy.desktop.view import file_actions
from negpy.services.assets.sidecar import sidecar_path_for


def _file(tmp_path, name="IMG_1.ARW") -> str:
    path = str(tmp_path / name)
    open(path, "wb").close()
    return path


def _qt_refuses(error: str):
    return lambda path: SimpleNamespace(moveToTrash=lambda: False, errorString=lambda: error)


def test_windows_never_trashes_on_a_drive_without_a_recycle_bin(tmp_path, monkeypatch):
    path = _file(tmp_path)
    qt_file = MagicMock()
    monkeypatch.setattr(file_manager.sys, "platform", "win32")
    monkeypatch.setattr(file_manager, "_windows_bin_takes", lambda p: False)
    monkeypatch.setattr(file_manager, "QFile", qt_file)
    assert file_manager.move_to_trash(path) == "the Recycle Bin would delete it for good"
    qt_file.assert_not_called()
    assert os.path.exists(path)


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_other_platforms_report_the_qt_error(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(file_manager.sys, "platform", platform)
    monkeypatch.setattr(file_manager, "QFile", _qt_refuses("Permission denied"))
    assert file_manager.move_to_trash(_file(tmp_path)) == "Permission denied"


def test_linux_names_a_drive_trash_owned_by_another_user(tmp_path, monkeypatch):
    (tmp_path / ".Trash-4242").mkdir()
    monkeypatch.setattr(file_manager.sys, "platform", "linux")
    monkeypatch.setattr(file_manager.os, "getuid", lambda: 4242, raising=False)
    monkeypatch.setattr(file_manager, "QStorageInfo", lambda p: SimpleNamespace(rootPath=lambda: str(tmp_path)))
    monkeypatch.setattr(file_manager, "QFile", _qt_refuses("No such file or directory"))
    error = file_manager.move_to_trash(_file(tmp_path))
    assert error == f"{tmp_path / '.Trash-4242'} belongs to another user. Mount its drive with uid=4242"


@pytest.mark.parametrize(
    "platform, expected",
    [("darwin", ["open", "-R", "{path}"]), ("win32", 'explorer /select,"{path}"')],
)
def test_show_in_folder_selects_the_file(tmp_path, monkeypatch, platform, expected):
    path = _file(tmp_path)
    calls = []
    monkeypatch.setattr(file_manager.sys, "platform", platform)
    monkeypatch.setattr(file_manager.subprocess, "Popen", lambda cmd: calls.append(cmd))
    assert file_manager.show_in_folder(path) is True
    norm = os.path.normpath(path) if platform == "win32" else path
    want = [a.format(path=norm) for a in expected] if isinstance(expected, list) else expected.format(path=norm)
    assert calls == [want]


def test_linux_falls_back_to_the_folder(tmp_path, monkeypatch):
    path = _file(tmp_path)
    opened = []
    monkeypatch.setattr(file_manager.sys, "platform", "linux")
    monkeypatch.setattr(file_manager, "_linux_show_items", lambda p: False)
    monkeypatch.setattr(file_manager.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    assert file_manager.show_in_folder(path) is True
    assert opened == [str(tmp_path)]


def test_show_in_folder_reports_a_missing_folder(tmp_path):
    assert file_manager.show_in_folder(str(tmp_path / "gone" / "IMG_1.ARW")) is False


def _controller(files, selected):
    state = SimpleNamespace(uploaded_files=files, selected_indices=selected, selected_file_idx=selected[0])
    return MagicMock(session=SimpleNamespace(state=state))


@pytest.mark.parametrize(
    "composite",
    [
        {"stitch_paths": ("b.ARW",)},
        {"hdr_paths": ("b.ARW",)},
        {"green_path": "g.ARW", "blue_path": "b.ARW"},
        {"half": 1},
        {"diptych": True},
    ],
)
def test_a_composite_in_the_selection_blocks_the_trash(monkeypatch, composite):
    warned, confirmed = [], []
    monkeypatch.setattr(file_actions, "warn_trash_composite", lambda parent: warned.append(True))
    monkeypatch.setattr(file_actions, "confirm_move_to_trash", lambda parent, n: confirmed.append(n) or True)
    ctrl = _controller([{"path": "a.ARW"}, {"path": "c.ARW", **composite}], [0, 1])
    file_actions.move_selected_to_trash(None, ctrl)
    assert warned and not confirmed
    ctrl.move_frames_to_trash.assert_not_called()


def test_plain_frames_go_to_the_trash_after_confirmation(monkeypatch):
    monkeypatch.setattr(file_actions, "confirm_move_to_trash", lambda parent, n: n == 2)
    ctrl = _controller([{"path": "a.ARW"}, {"path": "b.ARW"}], [0, 1])
    file_actions.move_selected_to_trash(None, ctrl)
    ctrl.move_frames_to_trash.assert_called_once_with([0, 1])


def test_trashing_moves_sidecars_and_unloads_only_what_moved(tmp_path, monkeypatch):
    ok, refused = _file(tmp_path, "ok.ARW"), _file(tmp_path, "refused.ARW")
    open(sidecar_path_for(ok), "w").close()
    trashed: list = []
    monkeypatch.setattr("negpy.desktop.controller.move_to_trash", lambda p: "refused" if p == refused else trashed.append(p) or "")
    ctrl = MagicMock()
    ctrl.state.uploaded_files = [{"path": ok}, {"path": refused}]
    AppController.move_frames_to_trash(ctrl, [0, 1])
    assert trashed == [ok, sidecar_path_for(ok)]
    ctrl.session.remove_files.assert_called_once_with([0])
    assert ctrl.set_status.call_args.kwargs["kind"] == "warning"


def test_a_sidecar_the_trash_refuses_is_reported(tmp_path, monkeypatch):
    path = _file(tmp_path)
    sidecar = sidecar_path_for(path)
    open(sidecar, "w").close()
    monkeypatch.setattr("negpy.desktop.controller.move_to_trash", lambda p: "refused" if p == sidecar else "")
    ctrl = MagicMock()
    ctrl.state.uploaded_files = [{"path": path}]
    AppController.move_frames_to_trash(ctrl, [0])
    ctrl.session.remove_files.assert_called_once_with([0])
    assert "1 file could not go to the Trash: refused" in ctrl.set_status.call_args.args[0]
    assert ctrl.set_status.call_args.kwargs["kind"] == "warning"

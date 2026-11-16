"""Relink a folder roll after its folder was renamed or moved on disk: only stored
locations change, while edits, marks and roll settings (all keyed by content hash) stay."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls
from negpy.services.assets.rehome import (
    RELINK_FOLDER_TAKEN,
    RELINK_NOT_A_FOLDER,
    RELINK_NOT_A_FOLDER_ROLL,
    relink_folder_roll,
)


def _repo(tmp_path) -> StorageRepository:
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    return repo


def _roll_dirs(tmp_path):
    old = tmp_path / "network" / "roll_a"
    new = tmp_path / "internal" / "roll_a"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "a1.NEF").write_bytes(b"1")
    (new / "a1.NEF").write_bytes(b"1")
    return str(old), str(new)


# --- rehome.relink_folder_roll ---------------------------------------------------


def test_relink_points_the_roll_at_the_new_folder(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    roll_id = rolls.recognize_folder(repo, old)

    assert relink_folder_roll(repo, roll_id, new) == (old, "")
    assert rolls.roll_for_id(repo, roll_id)["folder_path"] == new
    assert rolls.folder_roll_id_for_path(repo, new) == roll_id
    assert rolls.folder_roll_id_for_path(repo, old) is None


def test_relink_to_the_same_folder_is_a_noop_success(tmp_path):
    repo = _repo(tmp_path)
    old, _new = _roll_dirs(tmp_path)
    roll_id = rolls.recognize_folder(repo, old)

    assert relink_folder_roll(repo, roll_id, old) == (old, "")
    assert rolls.roll_for_id(repo, roll_id)["folder_path"] == old


def test_relink_refuses_a_folder_another_roll_points_at(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    roll_id = rolls.recognize_folder(repo, old)
    rolls.recognize_folder(repo, new)

    assert relink_folder_roll(repo, roll_id, new) == (None, RELINK_FOLDER_TAKEN)
    assert rolls.roll_for_id(repo, roll_id)["folder_path"] == old


def test_relink_refuses_a_virtual_roll_and_a_missing_folder(tmp_path):
    repo = _repo(tmp_path)
    old, _new = _roll_dirs(tmp_path)
    virtual = rolls.create_virtual_roll(repo, "Portra", [])
    folder_roll = rolls.recognize_folder(repo, old)

    assert relink_folder_roll(repo, virtual, old) == (None, RELINK_NOT_A_FOLDER_ROLL)
    assert relink_folder_roll(repo, folder_roll, str(tmp_path / "gone")) == (None, RELINK_NOT_A_FOLDER)
    assert relink_folder_roll(repo, "no-such-roll", old) == (None, RELINK_NOT_A_FOLDER_ROLL)


def test_relink_undismisses_the_new_folder(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    roll_id = rolls.recognize_folder(repo, old)
    doomed = rolls.recognize_folder(repo, new)
    rolls.delete_roll(repo, doomed)

    assert relink_folder_roll(repo, roll_id, new) == (old, "")
    assert rolls.folder_roll_id_for_path(repo, new) == roll_id


# --- controller.relink_roll -------------------------------------------------------


def _controller_double(repo, active_roll_id=None, uploaded_files=None):
    return SimpleNamespace(
        session=SimpleNamespace(repo=repo, rehome_folder_paths=MagicMock()),
        state=SimpleNamespace(active_roll_id=active_roll_id, uploaded_files=uploaded_files or []),
        open_roll=MagicMock(),
        invalidate_library_walk=MagicMock(),
    )


def test_relink_roll_rehomes_every_store(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    roll_id = rolls.recognize_folder(repo, old, name="roll_a")
    frame = os.path.join(old, "a1.NEF")
    repo.save_file_settings("ha", WorkspaceConfig(), file_path=frame)
    repo.save_file_mark("ha", "keeper", file_path=frame)
    repo.save_global_setting("library_roots", [old])
    fake = _controller_double(repo)

    assert AppController.relink_roll(fake, roll_id, new) == ""

    assert rolls.roll_for_id(repo, roll_id)["folder_path"] == new
    assert repo.path_for_file_hash("ha") == os.path.join(new, "a1.NEF")
    assert repo.load_file_marks_by_path() == {os.path.join(new, "a1.NEF"): "keeper"}
    assert repo.get_global_setting("library_roots") == [new]
    fake.open_roll.assert_called_once_with(roll_id)
    fake.invalidate_library_walk.assert_called_once()


def test_relink_roll_keeps_the_open_frames_of_the_open_roll(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    roll_id = rolls.recognize_folder(repo, old, name="roll_a")
    fake = _controller_double(repo, active_roll_id=roll_id, uploaded_files=[os.path.join(old, "a1.NEF")])

    assert AppController.relink_roll(fake, roll_id, new) == ""

    fake.session.rehome_folder_paths.assert_called_once_with(old, new)
    fake.open_roll.assert_not_called()


def test_relink_roll_refuses_a_virtual_roll_and_a_collision(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    virtual = rolls.create_virtual_roll(repo, "Portra", [])
    rolls.recognize_folder(repo, new)
    folder_roll = rolls.recognize_folder(repo, old)
    fake = _controller_double(repo)

    assert AppController.relink_roll(fake, virtual, new) == RELINK_NOT_A_FOLDER_ROLL
    assert AppController.relink_roll(fake, folder_roll, new) == RELINK_FOLDER_TAKEN
    assert AppController.relink_roll(fake, folder_roll, str(tmp_path / "gone")) == RELINK_NOT_A_FOLDER
    assert AppController.relink_roll(fake, "no-such-roll", new) == RELINK_NOT_A_FOLDER_ROLL
    fake.session.rehome_folder_paths.assert_not_called()
    fake.invalidate_library_walk.assert_not_called()

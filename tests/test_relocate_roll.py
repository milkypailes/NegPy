"""Relocate a folder roll after its folder moved on disk: only stored locations
change, while edits, marks and roll settings (all keyed by content hash) stay."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.controller import AppController
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls
from negpy.services.assets.composites import rehome_composites, remember_composites, saved_composites
from negpy.services.assets.triplets import rehome_triplets, remember_triplets, saved_triplets


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


# --- rolls.relocate_folder_roll -------------------------------------------------


def test_relocate_points_the_roll_at_the_new_folder(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    roll_id = rolls.recognize_folder(repo, old)

    assert rolls.relocate_folder_roll(repo, roll_id, new) == old
    assert rolls.roll_for_id(repo, roll_id)["folder_path"] == new
    assert rolls.folder_roll_id_for_path(repo, new) == roll_id
    assert rolls.folder_roll_id_for_path(repo, old) is None


def test_relocate_to_the_same_folder_is_a_noop_success(tmp_path):
    repo = _repo(tmp_path)
    old, _new = _roll_dirs(tmp_path)
    roll_id = rolls.recognize_folder(repo, old)

    assert rolls.relocate_folder_roll(repo, roll_id, old) == old
    assert rolls.roll_for_id(repo, roll_id)["folder_path"] == old


def test_relocate_refuses_a_folder_another_roll_points_at(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    roll_id = rolls.recognize_folder(repo, old)
    rolls.recognize_folder(repo, new)

    assert rolls.relocate_folder_roll(repo, roll_id, new) is None
    assert rolls.roll_for_id(repo, roll_id)["folder_path"] == old


def test_relocate_refuses_a_virtual_roll_and_a_missing_folder(tmp_path):
    repo = _repo(tmp_path)
    old, _new = _roll_dirs(tmp_path)
    virtual = rolls.create_virtual_roll(repo, "Portra", [])
    folder_roll = rolls.recognize_folder(repo, old)

    assert rolls.relocate_folder_roll(repo, virtual, old) is None
    assert rolls.relocate_folder_roll(repo, folder_roll, str(tmp_path / "gone")) is None
    assert rolls.relocate_folder_roll(repo, "no-such-roll", old) is None


def test_relocate_undismisses_the_new_folder(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    roll_id = rolls.recognize_folder(repo, old)
    doomed = rolls.recognize_folder(repo, new)
    rolls.delete_roll(repo, doomed)

    assert rolls.relocate_folder_roll(repo, roll_id, new) == old
    assert rolls.folder_roll_id_for_path(repo, new) == roll_id


# --- StorageRepository.rehome_file_paths ----------------------------------------


def test_rehome_file_paths_repoints_every_table_but_leaves_hashes_alone(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    config = WorkspaceConfig()
    repo.save_file_settings("ha", config, file_path=os.path.join(old, "a1.NEF"))
    repo.save_file_settings("hb", config, file_path=os.path.join(old, "sub", "b1.NEF"))
    repo.save_file_settings("hx", config, file_path="/elsewhere/x.NEF")
    repo.save_file_mark("ha", "keeper", file_path=os.path.join(old, "a1.NEF"))
    repo.save_file_mark("hx", "excluded", file_path="/elsewhere/x.NEF")
    repo.save_embedding("ha", np.zeros(4, dtype=np.float32), "v1", file_path=os.path.join(old, "a1.NEF"))

    repo.rehome_file_paths(old, new)

    assert repo.path_for_file_hash("ha") == os.path.join(new, "a1.NEF")
    assert repo.path_for_file_hash("hb") == os.path.join(new, "sub", "b1.NEF")
    assert repo.path_for_file_hash("hx") == "/elsewhere/x.NEF"
    assert repo.load_file_marks_by_path() == {os.path.join(new, "a1.NEF"): "keeper", "/elsewhere/x.NEF": "excluded"}
    assert repo.load_all_embeddings("v1")["ha"][0] == os.path.join(new, "a1.NEF")
    assert repo.load_file_settings("ha").to_dict() == config.to_dict()


def test_rehome_file_paths_matching_prefixes_only(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    repo.save_file_settings("ha", WorkspaceConfig(), file_path=old + "_other/a1.NEF")

    repo.rehome_file_paths(old, new)

    assert repo.path_for_file_hash("ha") == old + "_other/a1.NEF"


# --- composites / triplets -------------------------------------------------------


def test_rehome_composites_repoints_keys_and_parts(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    remember_composites(
        repo,
        [
            {
                "name": "1.tif",
                "path": os.path.join(old, "1.tif"),
                "hash": "hs",
                "stitch_paths": [os.path.join(old, "1.tif"), os.path.join(old, "2.tif")],
                "stitch_transforms": [[1, 0, 0], [0, 1, 0]],
                "stitch_canvas": [100, 100],
                "stitch_sizes": [[50, 100], [50, 100]],
            }
        ],
    )

    rehome_composites(repo, old, new)

    entry = saved_composites(repo)[os.path.join(new, "1.tif")]
    assert entry["paths"] == [os.path.join(new, "1.tif"), os.path.join(new, "2.tif")]


def test_rehome_triplets_repoints_keys_and_members(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    remember_triplets(
        repo,
        [
            {
                "name": "r (RGB)",
                "path": os.path.join(old, "r.tif"),
                "hash": "hr",
                "green_path": os.path.join(old, "g.tif"),
                "blue_path": os.path.join(old, "b.tif"),
                "green_hash": "hg",
                "blue_hash": "hb",
            }
        ],
    )

    rehome_triplets(repo, old, new)

    assert saved_triplets(repo) == {
        os.path.join(new, "r.tif"): [os.path.join(new, "g.tif"), os.path.join(new, "b.tif"), True, ["hr", "hg", "hb"]]
    }


# --- controller.request_relocate_roll --------------------------------------------


def _controller_double(repo):
    fake = SimpleNamespace(
        session=SimpleNamespace(repo=repo, rehome_folder_paths=MagicMock()),
        set_status=MagicMock(),
        invalidate_library_walk=MagicMock(),
    )
    fake._rehome_path_setting = lambda key, old, new: AppController._rehome_path_setting(fake, key, old, new)
    return fake


def test_request_relocate_roll_rehomes_every_store(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    roll_id = rolls.recognize_folder(repo, old, name="roll_a")
    frame = os.path.join(old, "a1.NEF")
    repo.save_file_settings("ha", WorkspaceConfig(), file_path=frame)
    repo.save_file_mark("ha", "keeper", file_path=frame)
    repo.save_global_setting("library_roots", [old])
    fake = _controller_double(repo)

    assert AppController.request_relocate_roll(fake, roll_id, new) is True

    assert rolls.roll_for_id(repo, roll_id)["folder_path"] == new
    assert repo.path_for_file_hash("ha") == os.path.join(new, "a1.NEF")
    assert repo.load_file_marks_by_path() == {os.path.join(new, "a1.NEF"): "keeper"}
    assert repo.get_global_setting("library_roots") == [new]
    fake.session.rehome_folder_paths.assert_called_once_with(old, new)
    fake.invalidate_library_walk.assert_called_once()


def test_request_relocate_roll_refuses_a_virtual_roll_and_a_collision(tmp_path):
    repo = _repo(tmp_path)
    old, new = _roll_dirs(tmp_path)
    virtual = rolls.create_virtual_roll(repo, "Portra", [])
    rolls.recognize_folder(repo, new)
    folder_roll = rolls.recognize_folder(repo, old)
    fake = _controller_double(repo)

    assert AppController.request_relocate_roll(fake, virtual, new) is False
    assert AppController.request_relocate_roll(fake, folder_roll, new) is False
    assert AppController.request_relocate_roll(fake, folder_roll, str(tmp_path / "gone")) is False
    assert AppController.request_relocate_roll(fake, "no-such-roll", new) is False
    fake.session.rehome_folder_paths.assert_not_called()

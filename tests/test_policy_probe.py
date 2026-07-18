import json

import numpy as np
import pytest

from rein_learning.common import PolicyProbeCorpus, make_policy_probe_corpus


def _records() -> list[dict[str, object]]:
    return [
        {
            "observation": np.asarray([index, index + 1], dtype=np.float32),
            "action_mask": np.asarray([True, False, True], dtype=np.bool_),
            "scenario": "medium",
            "source": "test",
            "phase": "initial" if index == 0 else "middle",
            "environment_seed": 40_000 + index,
            "episode_index": index,
            "step_index": index * 2,
        }
        for index in range(2)
    ]


def test_probe_corpus_round_trip_and_hash_are_deterministic(tmp_path) -> None:
    corpus = make_policy_probe_corpus(_records())
    manifest = corpus.save(tmp_path, metadata={"purpose": "test"})
    loaded = PolicyProbeCorpus.load(tmp_path)

    assert loaded.content_sha256() == corpus.content_sha256()
    assert manifest["content_sha256"] == corpus.content_sha256()
    assert manifest["phase_counts"] == {"initial": 1, "middle": 1}
    assert np.array_equal(loaded.observations, corpus.observations)


def test_probe_corpus_rejects_manifest_hash_mismatch(tmp_path) -> None:
    corpus = make_policy_probe_corpus(_records())
    corpus.save(tmp_path)
    manifest_path = tmp_path / "probe_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["content_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        PolicyProbeCorpus.load(tmp_path)

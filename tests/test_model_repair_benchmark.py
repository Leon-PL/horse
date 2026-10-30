import json
import pytest

from scripts.benchmark_model_repair import prepare


def test_registered_projection_is_not_rewritten_on_prepare(tmp_path):
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({}))
    (tmp_path / "manifest.json").write_text("{}")
    projection = tmp_path / "inputs.parquet"
    projection.write_bytes(b"preserve this registered input")
    with pytest.raises(ValueError, match="Already registered"):
        prepare(registration)
    assert projection.read_bytes() == b"preserve this registered input"

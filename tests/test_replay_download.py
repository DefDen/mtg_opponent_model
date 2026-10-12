import csv
import gzip
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "download_replay_sample.py"


def make_prefix(path):
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows([["draft_id", "value"], ["a", "quoted, value"], ["b", "next"]])


def test_extract_requested_complete_rows_from_local_prefix(tmp_path):
    prefix = tmp_path / "prefix.gz"
    output = tmp_path / "sample.gz"
    make_prefix(prefix)
    result = subprocess.run([sys.executable, str(SCRIPT), "--prefix", str(prefix),
                             "--games", "1", "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with gzip.open(output, "rt", newline="") as stream:
        assert list(csv.reader(stream)) == [["draft_id", "value"], ["a", "quoted, value"]]
    assert not output.with_suffix(".gz.partial").exists()


def test_short_prefix_preserves_previous_output_and_cleans_temporary_file(tmp_path):
    prefix = tmp_path / "prefix.gz"
    output = tmp_path / "sample.gz"
    make_prefix(prefix)
    output.write_bytes(b"previous artifact")
    result = subprocess.run([sys.executable, str(SCRIPT), "--prefix", str(prefix),
                             "--games", "3", "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Replay prefix too short" in result.stderr
    assert output.read_bytes() == b"previous artifact"
    assert not output.with_suffix(".gz.partial").exists()

import tarfile
import pytest

from payu_config.archive_scripts.archive_esmf_streams import (
    ARCHIVE_FILENAME,
    archive_traceout,
)


METADATA = "metadata"


def _make_traceout(tmp_path, stream_files):
    traceout_path = tmp_path / "traceout"
    traceout_path.mkdir()

    (traceout_path / METADATA).write_text("metadata")

    for name, content in stream_files.items():
        (traceout_path / name).write_bytes(content)

    return traceout_path


def test_archive_stream_files(tmp_path):
    stream_files = {
        "esmf_stream_0000": b"pet 0",
        "esmf_stream_0001": b"pet 1",
    }
    traceout_path = _make_traceout(tmp_path, stream_files)

    assert archive_traceout(traceout_path) == 2

    archive_path = traceout_path / ARCHIVE_FILENAME
    assert archive_path.is_file()

    assert (traceout_path / METADATA).is_file()
    assert not (traceout_path / "esmf_stream_0000").exists()
    assert not (traceout_path / "esmf_stream_0001").exists()

    with tarfile.open(archive_path) as archive:
        assert sorted(archive.getnames()) == sorted(stream_files)


def test_no_stream_files(tmp_path):
    traceout_path = tmp_path / "traceout"
    traceout_path.mkdir()

    assert archive_traceout(traceout_path) == 0


def test_recovers_from_interrupted_cleanup(tmp_path):
    traceout_path = _make_traceout(
        tmp_path,
        {"esmf_stream_0000": b"trace"},
    )
    archive_traceout(traceout_path)

    # Simulate interruption during cleanup
    loose_stream_path = traceout_path / "esmf_stream_0000"
    loose_stream_path.write_bytes(b"trace")

    assert archive_traceout(traceout_path) == 1
    assert not loose_stream_path.exists()


def test_does_not_delete_mismatched_stream(tmp_path):
    traceout_path = _make_traceout(
        tmp_path,
        {"esmf_stream_0000": b"trace"},
    )
    archive_traceout(traceout_path)

    # Simulate interruption during cleanup
    loose_stream_path = traceout_path / "esmf_stream_0000"
    loose_stream_path.write_bytes(b"mismatched")

    with pytest.raises(RuntimeError, match="refusing to remove it"):
        archive_traceout(traceout_path)

    assert loose_stream_path.is_file()

#!/usr/bin/env python3
# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# Archive ESMF per-PET trace streams to reduce inode usage.
# ESMF tracing produces one stream file per PET, for example,
# archive/output000/traceout/
#   metadata
#   esmf_stream_0000
#   esmf_stream_0001
#   ...
# This script consolidates the stream files into a single uncompressed tar:
# archive/output000/traceout/
#   metadata
#   esmf_stream.tar
# The metadata file remains outside the tar
# The loose stream files are removed only after the tar has been successfully created.

import argparse
import re
import tarfile
import tempfile
from pathlib import Path

STREAM_PREFIX = "esmf_stream"
ARCHIVE_FILENAME = f"{STREAM_PREFIX}.tar"
STREAM_RE = re.compile(rf"^{re.escape(STREAM_PREFIX)}_\d+$")


def _stream_files(traceout_path: Path) -> list[Path]:
    """
    Return a list of stream files in traceout_path
    """
    stream_files = [
        f
        for f in traceout_path.iterdir()
        if f.is_file() and STREAM_RE.fullmatch(f.name)
    ]
    return sorted(stream_files)


def _remove_streams_already_archived(
    archive_path: Path, stream_files: list[Path]
) -> int:
    """
    Remove loose stream files left behind after an interrupted cleanup. The final tar is only
    created after it has been completely written so if it exists along with loose stream files,
    verify those remaining streams are present in the tar with matching sizes before removing them.
    """
    with tarfile.open(archive_path, "r") as archive:
        archived = {
            member.name: member.size
            for member in archive.getmembers()
            if member.isfile()
        }

    for f in stream_files:
        if archived.get(f.name) != f.stat().st_size:
            raise RuntimeError(
                f"{f} does not match the archived copy in {archive_path}; refusing to remove it"
            )

    for f in stream_files:
        f.unlink()

    return len(stream_files)


def archive_traceout(traceout_path: Path) -> int:
    """
    Archive loose stream files in one traceout directory.
    Returns the number of loose stream files removed.
    """
    traceout_path = Path(traceout_path).expanduser().resolve()

    if not traceout_path.is_dir():
        raise ValueError(f"{traceout_path} is not a directory")

    stream_files = _stream_files(traceout_path)
    archive_path = traceout_path / ARCHIVE_FILENAME

    # If there are no stream files
    if not stream_files:
        return 0

    # A previous call may have completed the tar but been interrupted while removing the loose stream files
    if archive_path.exists():
        return _remove_streams_already_archived(archive_path, stream_files)

    # write beside the final archive so the rename stays on the same filesystem
    with tempfile.NamedTemporaryFile(
        dir=traceout_path,
        prefix=f".{ARCHIVE_FILENAME}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with tarfile.open(tmp_path, "w") as archive:
            for f in stream_files:
                archive.add(
                    f,
                    arcname=f.name,
                    recursive=False,
                )
        # NamedTemporaryFile creates files as 0600 default.
        # https://github.com/python/cpython/blob/a4f28a52b4b54c34100ee0891b9a15640ed3a7b2/Lib/tempfile.py#L244-L257
        # Change to 0640 so the archive is readable by group members
        tmp_path.chmod(0o640)

        # only expose the final archive after it has been completely written
        tmp_path.replace(archive_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    # raw stream files are removed only after the completed tar exists
    for f in stream_files:
        f.unlink()

    return len(stream_files)


def archive_output(output_path: Path) -> int:
    """
    Archive loose stream files under an output directory, e.g. archive/output000/traceout/
    """
    traceout_path = output_path / "traceout"
    if not traceout_path.is_dir():
        return 0

    return archive_traceout(traceout_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Archive ESMF per-PET trace streams to reduce inode usage"
    )
    parser.add_argument(
        "-d",
        "--directory",
        metavar="DIRECTORY",
        dest="output_dir",
        help="Process one or more output directories, if omitted, process all archive/output*/ directories",
    )
    args = parser.parse_args(argv)

    if args.output_dir:
        output_dirs = [Path(args.output_dir).expanduser().resolve()]
    else:
        output_dirs = sorted(Path("archive").glob("output*"))

    for output_dir in output_dirs:
        count = archive_output(output_dir)

        if count:
            print(
                f"Archived {count} stream files in {output_dir}/traceout/{ARCHIVE_FILENAME}"
            )


if __name__ == "__main__":
    main()

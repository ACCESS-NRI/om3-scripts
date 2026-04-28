# Copyright 2024 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# These are common functions which can get used in any of the om3 scripts
# =========================================================================================


import subprocess
import os
from warnings import warn
import io
import hashlib
from datetime import datetime


def get_git_url(file):
    """
    If the provided file is in a git repo, return the url to its most recent commit remote.origin.
    """
    dirname = os.path.dirname(file)

    try:
        url = (
            subprocess.check_output(
                ["git", "-C", dirname, "config", "--get", "remote.origin.url"]
            )
            .decode("ascii")
            .strip()
        )
        url = url.removesuffix(".git")
    except subprocess.CalledProcessError:
        return None

    if url.startswith("git@github.com:"):
        url = f"https://github.com/{url.removeprefix('git@github.com:')}"

    top_level_dir = (
        subprocess.check_output(["git", "-C", dirname, "rev-parse", "--show-toplevel"])
        .decode("ascii")
        .strip()
    )
    rel_path = file.removeprefix(top_level_dir)

    hash = (
        subprocess.check_output(["git", "-C", dirname, "rev-parse", "HEAD"])
        .decode("ascii")
        .strip()
    )

    return f"{url}/blob/{hash}/{rel_path}"


def git_status(file):
    """
    Return the git status of the file. Returns:
    - "unstaged" if the file has unstaged changes
    - "uncommitted" if the file has uncommited changes,
    - "unpushed" if the repo has unpushed commits
    - None otherwise
    """
    dirname = os.path.dirname(file)
    status = (
        subprocess.check_output(["git", "-C", dirname, "status", file])
        .decode("ascii")
        .strip()
    )

    if "Changes not staged for commit" in status:
        return "unstaged"
    elif "Changes to be committed" in status:
        return "uncommitted"
    elif "Your branch is ahead" in status:
        return "unpushed"
    else:
        return None


def username(file):
    """
    Return a string with the username of the current user. If possible, include the git username also.
    """
    dirname = os.path.dirname(file)

    name = os.environ["USER"]

    try:
        gitname = (
            subprocess.check_output(["git", "-C", dirname, "config", "user.name"])
            .decode("ascii")
            .strip()
        )
        name = f"{name} ({gitname})"
    except subprocess.CalledProcessError:
        pass

    return name


def get_provenance_metadata(file, runcmd):
    """
    Return a string with the provenance of the file being run. Warn if the file is not pushed to the git upstream repository.

    arguments:
        file: the path to the file being run
        runcmd: the command used to run the file (with any arguments)
    """

    prepend = (
        f"Created by {username(file)} on {datetime.now().strftime('%Y-%m-%d')}, using "
    )

    git_url = get_git_url(file)

    if git_url:
        status = git_status(file)
        if status in ["unstaged", "uncommitted"]:
            warn(
                f"{file} contains uncommitted changes! Commit and push your changes before generating any production output."
            )
        if status == "unpushed":
            warn(
                f"There are commits that are not pushed! Push your changes before generating any production output."
            )
        prepend += f"{git_url}: "
    else:
        warn(
            f"{file} not under git version control! Add your file to a repository before generating any production output."
        )
        prepend += f"{file}: "

    return prepend + runcmd


def md5sum(path):
    """
    Return the md5 hash of a provided file, reading in chunks to reduce memory usage for
    large files.
    From https://stackoverflow.com/a/40961519
    """
    length = io.DEFAULT_BUFFER_SIZE
    md5 = hashlib.md5()
    with io.open(path, mode="rb") as fd:
        for chunk in iter(lambda: fd.read(length), b""):
            md5.update(chunk)
    return md5.hexdigest()

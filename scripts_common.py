# Copyright 2024 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# These are common functions which can get used in any of the om3 scripts
# =========================================================================================


import subprocess
import os, sys
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


def get_email(file):
    """
    Return the git user.email configured for the repo containing file, or None if
    not under git version control or no email is configured.
    """
    dirname = os.path.dirname(file)

    try:
        return (
            subprocess.check_output(["git", "-C", dirname, "config", "user.email"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        return None


def write_readme(output_dir, file, output_filename, input_files, history):
    """
    Write a README.md to output_dir, following the ACCESS-NRI model-config-inputs
    template:
    https://github.com/ACCESS-NRI/model-config-inputs/blob/main/templates/README_template.md

    Everything is filled in automatically: title and file list from output_filename,
    contact details from the current user's git config, and links/relationships from
    input_files. Fields with no sensible automatic value (licensing, other locations,
    citation) are set to fixed defaults appropriate for ACCESS-NRI model input data.
    """
    contact_name = username(file)
    contact_email = get_email(file) or "[Email of data custodian]"

    if isinstance(output_filename, str):
        output_filenames = [output_filename]
    else:
        output_filenames = list(output_filename) if output_filename else []

    title = ", ".join(os.path.basename(f) for f in output_filenames)
    file_list = "\n".join(f"- {os.path.abspath(f)}" for f in output_filenames)
    links = (
        "\n".join(
            f"- {os.path.abspath(f)} (md5 hash: {md5sum(f)})" for f in input_files
        )
        if input_files
        else "N/A"
    )

    readme = f"""This readme file was generated on {datetime.now().strftime('%Y-%m-%d')} by {contact_name}

# {title}

## Contact(s)

Name: {contact_name}

Email: {contact_email}

Institution: ACCESS-NRI

## Data Access and Sharing

### Licensing/restrictions

CC-BY-4.0

### Other locations

N/A

### Links/relationships

Other files used to prepare this data:

{links}

### Recommended citation

N/A

## Data and Files

### File List

Files in this dataset:

{file_list}

## Methodological information

{history}
"""

    # give the README a unique name based on the output files
    if output_filenames:
        readme_name = (
            f"{os.path.splitext(os.path.basename(output_filenames[0]))[0]}.README.md"
        )
    else:
        readme_name = "README.md"

    readme_path = os.path.join(output_dir, readme_name)
    with open(readme_path, "w") as f:
        f.write(readme)

    return readme_path


def get_provenance_metadata(
    input_files=None, runcmd=None, output_dir=None, output_filename=None
):
    """
    Return a dictionary with the provenance of the file being run. Warn if the
    file is not pushed to the git upstream repository. Also writes a README.md
    alongside the output (see write_readme).

    arguments:
        input_files: list of input files being used in the script being run (optional)
        runcmd: the command used to run the file, with any arguments. Optional -
            defaults to the python executable + input arguments
        output_dir: directory to write the accompanying README.md into. Optional -
            defaults to the current working directory
        output_filename: name of the file being created (or list of names). Optional -
            used as the README's title and file list
    """

    file = os.path.abspath(sys.argv[0])  # script being run
    if runcmd is None:
        runcmd = f"{sys.executable} {' '.join(sys.argv)}"

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
        prepend += f"{file} (md5 hash: {md5sum(file)}): "

    attrs = {"history": prepend + runcmd}

    if input_files is not None:
        attrs["inputFile"] = get_provenance_input_files(input_files)

    write_readme(
        output_dir or os.getcwd(), file, output_filename, input_files, attrs["history"]
    )

    return attrs


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


def get_provenance_input_files(input_files):
    """
    Return a formatted string of provided input files and their md5 hashes
    """
    file_hashes = []
    for input in input_files:
        file_hashes.append(f"{os.path.abspath(input)} (md5 hash: {md5sum(input)})")
    return ", ".join(file_hashes)

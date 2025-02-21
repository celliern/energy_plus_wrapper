#!/usr/bin/env python
# coding=utf-8


import platform
import re
import shutil

from platformdirs import user_data_dir
import fasteners
import pexpect
import requests
from pathlib import Path
from tempfile import TemporaryDirectory as TempDir

eplus_filename_pattern = (
    r".*?(?P<filename>EnergyPlus-(?P<version>\d+.\d+.\d+)-"
    r"(?P<revision>\w+)-(?P<platform>.*?).sh)$"
)


def _is_downloadable(url: str):
    content_type = requests.head(url, allow_redirects=True).headers.get("content-type")
    if content_type is None:
        return False
    content_type = content_type.lower()
    if "text" in content_type:
        return False
    if "html" in content_type:
        return False
    return True


def _extract_filename_info(url: str):
    filename_match = re.match(pattern=eplus_filename_pattern, string=url)
    if filename_match is None:
        raise ValueError("URL does not match the EnergyPlus filename pattern.")
    return filename_match.groupdict()


def _download_eplus_version(url, path):
    if not _is_downloadable(url):
        raise ValueError("URL is not a downloadable file.")
    response = requests.get(url, allow_redirects=True)
    with open(path, "wb") as f:
        f.write(response.content)


def _extract_and_install(setup_script: str, eplus_folder: str):
    with pexpect.spawn(f"bash {setup_script}") as child:
        # child.logfile = sys.stderr
        child.expect("\r\n")
        # child.expect(r"Do you accept the license\? \[yN\]:")
        child.sendline("y")
        # child.expect(r"EnergyPlus install directory \[.*\]:")
        child.sendline(eplus_folder)
        child.expect("\r\n")
        # child.expect(r'Symbolic link location \(enter "n" for no links\) \[.*\]:')
        child.sendline("n")
        child.expect(pexpect.EOF)


def ensure_eplus_root(
    url: str,
    eplus_folder: Path | str = Path(user_data_dir(appname="energy_plus_wrapper")),
    installer_cache: Path | None = None,
) -> Path:
    """Check if the energy plus root is available in the provided eplus_folder,
    download it from the url, extract and install it if it's not the case. In any cases,
    return the EnergyPlus folder as needed by the EPlusRunner.

    This routine is only available for Linux (for now) !

    Arguments:
        url {str} -- the EnergyPlus installer URL. Look at
            `https://energyplus.net/downloads`


    Keyword Arguments:
        eplus_folder {Path} -- where EnergyPlus should be installed, as
            `{eplus_folder}/{eplus_version}/`.
            (default: user_data_dir(appname="energy_plus_wrapper"))
        installer_cache {Path} -- where to download the installation script. If None,
            a temporary folder will be created. (default: {None})

    Returns:
        [str] -- The EnergyPlus root.
    """

    if platform.system() != "Linux":
        raise ValueError(
            f"Your system ({platform.system()}) is not supported yet."
            " You have to install EnergyPlus by yourself."
        )
    eplus_folder = Path(eplus_folder)
    eplus_folder.mkdir(exist_ok=True, parents=True)
    with fasteners.InterProcessLock(eplus_folder / ".lock"):

        def url_to_installed(url, eplus_folder, script_path):
            if not script_path.exists():
                _download_eplus_version(url, script_path)
            _extract_and_install(str(script_path), str(eplus_folder))

        finfo = _extract_filename_info(url)
        filename = finfo["filename"]
        version = finfo["version"]
        expected_eplus_folder = eplus_folder / f"EnergyPlus-{version.replace('.', '-')}"
        if expected_eplus_folder.exists() and expected_eplus_folder.glob("*"):
            return expected_eplus_folder.absolute()
        shutil.rmtree(expected_eplus_folder, ignore_errors=True)
        if installer_cache is None:
            with TempDir() as d:
                url_to_installed(url, eplus_folder, Path(d) / filename)
        else:
            installer_cache = Path(installer_cache)
            installer_cache.mkdir(exist_ok=True, parents=True)
            url_to_installed(url, eplus_folder, installer_cache / filename)
        return expected_eplus_folder.absolute()

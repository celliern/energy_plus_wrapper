#!/usr/bin/env python
# coding=utf-8

import contextlib
import os
import shutil
import re
from typing import Callable, Hashable, Literal, Mapping, Sequence, TypeVar
from warnings import warn
from tempfile import gettempdir, TemporaryDirectory
from pathlib import Path
from charset_normalizer import detect

import plumbum
from coolname import generate_slug
from eppy.modeleditor import IDF as eppy_IDF
from joblib import Parallel, delayed
from plumbum import ProcessExecutionError
from loguru import logger


from .simulation import Simulation

eplus_version_pattern = re.compile(r"EnergyPlus, Version (\d\.\d)")
idf_version_pattern = re.compile(r"EnergyPlus Version (\d\.\d)")
idd_version_pattern = re.compile(r"IDD_Version (\d\.\d)")

IDF_TYPE = Path | eppy_IDF | str
AnyHashable = TypeVar("AnyHashable", bound=Hashable)


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


class EPlusRunner:
    """Object that contains all that is needed to run an EnergyPlus simulation.

    Attributes:
        energy_plus_root (Path): EnergyPlus root, where live the executable
            and the IDD file.
        temp_dir (Path, optional): where live the temporary files generated
            by EnergyPlus.
    """

    def __init__(
        self,
        energy_plus_root: str | Path,
        temp_dir: str | Path | None = None,
        mode: Literal["cli", "api"] = "cli",
    ):
        self.energy_plus_root = Path(energy_plus_root).absolute()
        self.temp_dir = Path(temp_dir) if temp_dir else Path(gettempdir())
        if mode not in ["cli", "api"]:
            raise ValueError("`mode` argument should be either 'cli' or 'api'.")
        self.mode = mode

    def get_idf_version(self, idf_file: Path) -> str:
        """extract the eplus version affiliated with the idf file.

        Arguments:
            idf_file {Path} -- idf file emplacement

        Returns:
            str -- the version as "{major}.{minor}" (e.g. "8.7")
        """
        with open(idf_file) as f:
            idf_str = f.read()
            try:
                version = idf_version_pattern.findall(idf_str)[0]
            except IndexError:
                version = False
        return str(version)

    @property
    def idd_version(self) -> str:
        """Get the eplus version affiliated with the idd file.

        Returns:
            str -- the version as "{major}.{minor}" (e.g. "8.7")
        """
        with open(self.idd_file) as f:
            idd_str = f.read()
            try:
                version = idd_version_pattern.findall(idd_str)[0]
            except IndexError:
                version = False
        return str(version)

    @property
    def eplus_version(self) -> str:
        """Get the eplus version for the executable itself.

        Returns:
            str -- the version as "{major}.{minor}" (e.g. "8.7")
        """
        version = eplus_version_pattern.findall(
            plumbum.local[str(self.eplus_bin)]("-v")
        )[0]
        return version

    @property
    def idd_file(self) -> Path:
        """Get the idd file given in the EnergyPlus folder.

        Returns:
            Path -- idd file emplacement
        """
        return self.energy_plus_root / "Energy+.idd"

    @property
    def eplus_bin(self) -> Path:
        """Get the EnergyPlus executable.

        Returns:
            Path -- Eplus binary emplacement
        """
        for bin_name in ["energyplus", "EnergyPlus.exe", "energyplus.exe"]:
            eplus_bin = self.energy_plus_root / bin_name
            if eplus_bin.exists():
                return eplus_bin
        raise FileNotFoundError(
            "Unable to find an e+ executable in the provided energy_plus_root."
        )

    def check_version_compat(self, idf_file, version_mismatch_action="raise") -> bool:
        """Check version compatibility between the IDF and the EnergyPlus
        binary. Raise an error or warn the user according to
        `version_mismatch_action`.

        Arguments:
            idf_file {Path} -- idf file emplacement
            version_mismatch_action {str} -- either ["raise", "warn", "ignore"]
        Returns:
            bool -- True if the versions are the same.
        """
        if version_mismatch_action not in ["raise", "warn", "ignore"]:
            raise ValueError(
                "`version_mismatch_action` argument should be either"
                " 'raise', 'warn', 'ignore'."
            )
        idf_version = self.get_idf_version(idf_file)
        eplus_version = self.eplus_version
        if idf_version != eplus_version:
            msg = (
                f"idf version ({idf_version}) and EnergyPlus version ({eplus_version}) "
                " does not match. According to the EnergyPlus versions, this can "
                " prevent the simulation to run or lead to silent error."
            )
            if version_mismatch_action == "raise":
                raise ValueError(msg)
            elif version_mismatch_action == "warn":
                warn(msg)
            return False
        return True

    def run_one(
        self,
        idf: IDF_TYPE,
        epw_file: Path | str,
        backup_strategy: Literal["on_error", "always"] | None = "on_error",
        backup_dir: Path = Path("./backup"),
        simulation_name: Hashable | None = None,
        custom_process: Callable[[Simulation], None] | None = None,
        version_mismatch_action: str = "raise",
        extra_files: Sequence[str] | None = None,
        encoding: str = "utf8",
    ) -> Simulation:
        """Run an EnergyPlus simulation with the provided idf and weather file.

        The IDF can be either a filename or an eppy IDF
        object.

        This function is process safe (as opposite as the one available in `eppy`).

        Arguments:
            idf {Union[Path, eppy_IDF, str]} -- idf file as filename or eppy IDF object.
            epw_file {Path} -- Weather file emplacement.

        Keyword Arguments:
            backup_strategy {str} -- when to save the files generated by e+
                (either"always", "on_error" or None) (default: {"on_error"})
            backup_dir {Path} -- where to save the files generated by e+
                (default: {"./backup"})
            simulation_name {str, optional} -- The simulation name. A random will be
                generated if not provided.
            custom_process {Callable[[Simulation], None], optional} -- overwrite the
                simulation post - process. Used to customize how the EnergyPlus files
                are treated after the simulation, but before cleaning the folder.
            version_mismatch_action {str} -- should be either ["raise", "warn",
                "ignore"] (default: {"raise"})

        Returns:
            Simulation -- the simulation object
        """
        if simulation_name is None:
            simulation_name = generate_slug()

        epw_file = Path(epw_file)

        if backup_strategy not in ["on_error", "always", None]:
            raise ValueError(
                "`backup_strategy` argument should be either 'on_error', 'always'"
                " or None."
            )
        backup_dir = Path(backup_dir).absolute()

        with TemporaryDirectory(prefix="energyplus_run_", dir=self.temp_dir) as td:
            td = Path(td)
            if extra_files is not None:
                for extra_file in extra_files:
                    shutil.copy(extra_file, td)
            if isinstance(idf, eppy_IDF):
                idf = idf.idfstr()
                idf_file = td / "eppy_idf.idf"
                idf_file.write_text(idf, encoding=encoding)
            else:
                idf_file = idf
                if version_mismatch_action in ["raise", "warn"]:
                    self.check_version_compat(
                        idf_file, version_mismatch_action=version_mismatch_action
                    )
            idf_file, epw_file = (Path(f).absolute() for f in (idf_file, epw_file))

            with working_directory(td):
                logger.debug((idf_file, epw_file, td))
                if idf_file not in td.glob("*"):
                    shutil.copy(idf_file, td)
                shutil.copy(epw_file, td)
                sim = Simulation(
                    simulation_name,
                    self.eplus_bin,
                    idf_file,
                    epw_file,
                    self.idd_file,
                    working_dir=td,
                    post_process=custom_process,
                    mode=self.mode,  # type: ignore
                )
                try:
                    sim.run()
                except (ProcessExecutionError, RuntimeError, KeyboardInterrupt):
                    if backup_strategy == "on_error":
                        logger.info("Backup the simulation (strategy: on_error).")
                        sim.backup(backup_dir)
                    raise
                finally:
                    if backup_strategy == "always":
                        logger.info("Backup the simulation (strategy: always).")
                        sim.backup(backup_dir)

        return sim

    def run_many(
        self,
        samples: Mapping[AnyHashable, tuple[IDF_TYPE, Path | str] | IDF_TYPE],
        epw_file: Path | str | None = None,
        backup_strategy: Literal["on_error", "always"] | None = "on_error",
        backup_dir: Path | str | None = Path("./backup"),
        custom_process: Callable[[Simulation], None] | None = None,
        version_mismatch_action: str = "raise",
    ) -> dict[AnyHashable, Simulation]:
        """Run multiple EnergyPlus simulation.

        Arguments:
            samples {mapping key: idf or (idf, weather_file)} -- A dict that contain a
                `run_one` arguments.
            epw_file {Path} -- Weather file emplacement. If None, it has to be in
                the samples. Otherwise, a unique weather file is used for each run.

        Keyword Arguments:
            backup_strategy {str} -- when to save the files generated by e+
                (either "always", "on_error" or None) (default: {"on_error"})
            backup_dir {Path} -- where to save the files generated by e+
                (default: {"./backup"})
            custom_process {Callable[[Simulation], None], optional} -- overwrite the
                simulation post - process. Used to customize how the EnergyPlus
                files are treated after the simulation, but before the folder clean.
            version_mismatch_action {str} -- should be either ["raise", "warn",
                "ignore"] (default: {"raise"})

        Returns:
            Dict[str, Simulation] -- the results put in a dictionnary with the same
                keys as the samples.
        """

        samples_: dict[AnyHashable, tuple[IDF_TYPE, Path | str]] = {}
        for key, value in samples.items():
            if isinstance(value, (tuple, list)) and epw_file is None:
                samples_[key] = value
            elif isinstance(value, IDF_TYPE) and epw_file is not None:
                samples_[key] = (value, Path(epw_file))
            else:
                raise ValueError(
                    "Either all the samples should be a tuple (idf, epw) and not epw file is given or "
                    "all the samples should be an idf file and an epw file should be given."
                )
        sims = Parallel()(
            delayed(self.run_one)(
                idf,
                epw_file,
                backup_strategy=backup_strategy,
                backup_dir=backup_dir,
                simulation_name=key,
                custom_process=custom_process,
                version_mismatch_action=version_mismatch_action,
            )
            for key, (idf, epw_file) in samples_.items()
        )
        return {
            key: raise_or_return_sim(sim) for key, sim in zip(samples_.keys(), sims)
        }


def raise_or_return_sim(value) -> Simulation:
    if not isinstance(value, Simulation):
        raise TypeError("The function should return a Simulation object.")
    return value

#!/usr/bin/env python
# coding=utf-8

from contextlib import contextmanager
from os import unlink
import shutil
from typing import Any, Callable, Hashable, Literal

import plumbum
from pathlib import Path
from plumbum import ProcessExecutionError

from .utils import process_eplus_html_report, process_eplus_time_series


def parse_generated_files_as_df(simulation):
    try:
        simulation.reports = process_eplus_html_report(
            simulation.working_dir / "eplus-table.htm"
        )
    except FileNotFoundError:
        pass
    simulation.time_series = process_eplus_time_series(simulation.working_dir)


@contextmanager
def eplus_api():
    try:
        import pyenergyplus.api
    except ImportError:
        raise ImportError(
            "Unable to find pyenergyplus. "
            "You may need to add the energyplus root folder to your PYTHONPATH with "
            "`export PYTHONPATH=$PYTHONPATH:/path/to/energyplus` in your shell "
            "or `sys.path.append('/path/to/energyplus')` in your script."
        )
    eplus_api = pyenergyplus.api.EnergyPlusAPI()
    state_manager = pyenergyplus.api.StateManager(eplus_api.api)
    eplus_runtime = pyenergyplus.api.Runtime(eplus_api.api)
    state = state_manager.new_state()
    yield state, eplus_runtime
    state_manager.delete_state(state)


class Simulation:
    """Object that contains all that is needed to run an EnergyPlus simulation.

    Attributes:
        name (str): simulation name
        eplus_bin (Path): EnergyPlus executable
        idf_file (Path): idf input file
        epw_file (Path): weather file
        idd_file (Path): idd file (should be in the EnergyPlus root)
        working_dir (Path): working folder, where the simulation will generate the files
        post_process (Callable): callable applied after a successful simulation.
            Take the simulation itself as argument.
        status (str): status of the simulation : either ["pending", "running",
            "interrupted", "failed"]
        reports (dict): if finished, contains the EPlus reports.
        time_series (dict): if finished, contains the EPlus time series results.
    """

    def __init__(
        self,
        name: Hashable,
        eplus_bin: str | Path,
        idf_file: str | Path,
        epw_file: str | Path,
        idd_file: str | Path,
        working_dir: str | Path,
        post_process: Callable | None = None,
        mode: Literal["cli", "api"] = "cli",
    ):
        self.name = name
        self.eplus_bin = Path(eplus_bin).absolute()
        self.idf_file = Path(idf_file).absolute()
        self.epw_file = Path(epw_file).absolute()
        self.idd_file = Path(idd_file).absolute()
        self.working_dir = Path(working_dir).absolute()
        if post_process is None:
            post_process = parse_generated_files_as_df
        self.post_process = post_process
        self.status = "pending"
        self.mode = mode
        self._log = ""
        self.reports: Any | None = None
        self.time_series: Any | None = None

    @property
    def log(self):
        """The log of finished simulation.

        Returns:
            str -- the log as a string
        """
        return self._log

    def _run_cli(self) -> Any | None:
        self.status = "running"
        try:
            self.status = "running"
            eplus_base_exec = plumbum.local[str(self.eplus_bin)]
            eplus_cmd = eplus_base_exec[
                "-s", "d", "-r", "-x", "-i", str(self.idd_file), "-w"
            ]
            self._log = eplus_cmd[str(self.epw_file), str(self.idf_file)]()
            self.status = "finished"
        except ProcessExecutionError:
            self.status = "failed"
            raise
        except KeyboardInterrupt:
            self.status = "interrupted"
            raise
        if self.post_process is not None:
            self.post_process(self)
        return self.reports

    def _run_api(self) -> Any | None:
        try:
            import pyenergyplus.api
        except ImportError:
            raise ImportError(
                "Unable to find pyenergyplus. "
                "You may need to add the energyplus root folder to your PYTHONPATH with "
                "`export PYTHONPATH=$PYTHONPATH:/path/to/energyplus` in your shell "
                "or `sys.path.append('/path/to/energyplus')` in your script."
            )
        self.status = "running"
        try:
            with eplus_api() as (eplus_state, eplus_runtime):
                return_status = eplus_runtime.run_energyplus(
                    eplus_state,
                    command_line_args=[
                        "-s",
                        "d",
                        "-r",
                        "-x",
                        "-i",
                        str(self.idd_file),
                        "-w",
                        str(self.epw_file),
                        str(self.idf_file),
                    ],
                )
            self._log = (self.working_dir / "eplus.err").read_text()
            if return_status != 0:
                raise RuntimeError("EnergyPlus exited with non-zero status code")
            self.status = "finished"
        except RuntimeError:
            self.status = "failed"
            raise
        except KeyboardInterrupt:
            self.status = "interrupted"
            raise
        if self.post_process is not None:
            self.post_process(self)
        return self.reports

    def run(self) -> Any | None:
        """Run the EPlus simulation

        Returns:
            Any | None -- the return depend on the post_process callable. By default, it will
                return the EPlus reports as a dict.
        """
        if self.mode == "cli":
            return self._run_cli()
        elif self.mode == "api":
            return self._run_api()

    def backup(self, backup_dir: Path):
        """Save all the files generated by energy-plus

        Files are saved in {backup_dir}/{sim.status}_{sim.name}

        Arguments:
            backup_dir {Path} -- where to save the files

        Returns:
            Path -- the exact folder where the data are saved.
        """
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        saved_data = backup_dir / f"{self.status}_{self.name}"
        if saved_data.exists():
            unlink(saved_data)
        shutil.copytree(self.working_dir, saved_data, ignore=lambda *_: ["backup"])
        return saved_data

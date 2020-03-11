import warnings
from typing import Generator, Tuple
import re

import bs4
import pandas as pd
from pandas import DataFrame
from path import Path
from box import Box, BoxList

re_section = re.compile(r"Report:(.*)", re.DOTALL)


def _eplus_html_report_gen(
    eplus_html_report: Path,
) -> Generator[Tuple[str, DataFrame], None, None]:
    """Extract the EnergyPlus html report into dataframes.

    Arguments:
        eplus_html_report {Path} -- the html report path

    Yields:
        Tuple[str, DataFrame] -- tuple of (report_title, report_data)
    """
    with open(eplus_html_report) as f:
        soup = bs4.BeautifulSoup(f.read(), features="lxml")
    for table in soup.find_all("table"):
        try:
            section = (
                table.find_previous(text=re.compile(r"Report:(.*)", re.DOTALL))
                .find_next_sibling("b")
                .text.strip()
            )
        except AttributeError:
            section = None
        title = table.find_previous_sibling("b").get_text()
        yield (section, title), pd.read_html(str(table), index_col=0, header=0)[
            0
        ].dropna(how="all")


def process_eplus_html_report(eplus_html_report: Path):
    """Extract the EnergyPlus html report into dataframes.

    Arguments:
        eplus_html_report {Path} -- the html report path

    Return:
        Box[str, DataFrame] -- Box of nested section - title : dataframe or custom-report: [dataframes]
            that contains the result of the reports.
    """
    reports = Box(box_intact_types=[pd.DataFrame], default_box=True)
    for (section, title), df in _eplus_html_report_gen(eplus_html_report):
        if section is None and title not in reports.keys():
            reports[title] = BoxList(box_intact_types=[pd.DataFrame])
        else:
            reports[section][title] = df
    return reports


def process_eplus_time_series(
    working_dir,
) -> Generator[Tuple[str, DataFrame], None, None]:
    """Extract the EnergyPlus csv outputs into dataframes.

    Arguments:
        working_dir {Path} -- path where live the generated csv outputs

    Yields:
        Tuple[str, DataFrame] -- tuple of (csv_name, csv_data)
    """
    time_series = {}
    for csv_file in working_dir.files("*.csv"):
        name = csv_file.basename().stripext()
        if name != "eplus":
            name = name.replace("eplus-", "")
        try:
            time_serie = pd.read_csv(csv_file)
        except Exception:
            warnings.warn(
                f"Unable to parse csv file {csv_file}. Return raw string as fallback."
            )
            with open(csv_file) as f:
                time_serie = f.read()
        time_series[str(name)] = time_serie
    return time_series

"""CDS downloader."""

import calendar
import itertools
import pathlib

import cdsapi
import xarray


def era_interim_model_levels(
    year: int,
    month: int,
    top_level: int,
    north: float,
    west: float,
    south: float,
    east: float,
):
    """Download a month of model-levels data from ERA-Interim."""
    fname = f"{year}-{month:02}.ml.nc"

    if pathlib.Path(fname).exists():
        return

    _, ndays = calendar.monthrange(year, month)

    cds = cdsapi.Client()
    cds.retrieve(
        "reanalysis-era-interim",
        {
            "date": f"{year}-{month:02}-01/to/{year}-{month:02}-{ndays}",
            "levelist": "/".join(str(l) for l in range(top_level, 60 + 1)),
            "levtype": "ml",
            "param": "t/q/w/u/v",
            "stream": "oper",
            "time": "00/06/12/18",
            "type": "an",
            "area": f"{north}/{west}/{south}/{east}",
            "grid": "0.75/0.75",
            "format": "netcdf",
        },
        fname,
    )


def era_interim_surface_level(
    year: int,
    month: int,
    north: float,
    west: float,
    south: float,
    east: float,
):
    """Download a month of surface data from ERA-Interim."""
    fname = f"{year}-{month:02}.sfc.nc"

    if pathlib.Path(fname).exists():
        return

    _, ndays = calendar.monthrange(year, month)

    cds = cdsapi.Client()
    cds.retrieve(
        "reanalysis-era-interim",
        {
            "date": f"{year}-{month:02}-01/to/{year}-{month:02}-{ndays}",
            "levelist": "1",
            "levtype": "ml",
            "param": "z/lnsp",
            "stream": "oper",
            "time": "00/06/12/18",
            "type": "an",
            "area": f"{north}/{west}/{south}/{east}",
            "grid": "0.75/0.75",
            "format": "netcdf",
        },
        fname,
    )


def main():
    """Test download."""

    for year, month in itertools.product(
        range(1981, 2020),
        range(1, 13),
    ):

        if year == 2019 and month > 8:
            break

        era_interim_model_levels(
            year,
            month,
            top_level=39,
            north=20,
            west=-90,
            south=-40,
            east=-30,
        )

        era_interim_surface_level(
            year,
            month,
            north=20,
            west=-90,
            south=-40,
            east=-30,
        )

    with xarray.open_mfdataset("*.ml.nc") as xds:
        print(xds)

    with xarray.open_mfdataset("*.sfc.nc") as xds:
        print(xds)


if __name__ == "__main__":
    main()

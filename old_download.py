import calendar
import datetime
import os.path

import ecmwfapi


def get_ecmwf_data(year, month, north, west, south, east, grid, initial_level=39):
    """Downloads data from ERA Interim model levels dataset.
    Data includes: wind (u,v,w), temperature (t) and specific humidity (q)
    at each level, as well as geopotential (z) and log pressure (lnsp) at the
    surface level.  TIP: ERA Interim's *w* is actually *omega*"""

    print(
        "[*] Requesting data for %s %d from ECMWF" % (calendar.month_abbr[month], year)
    )

    _, days_in_month = calendar.monthrange(year, month)
    start = datetime.date(year, month, 1)
    end = datetime.date(year, month, days_in_month)

    surface_file = "%s_z_lnsp_%s.nc" % (start, grid)
    levels_file = "%s_t_u_v_q_w_%s.nc" % (start, grid)

    server = ecmwfapi.ECMWFDataServer()

    if os.path.isfile(surface_file):
        print("    [+] %s is already present" % surface_file)
    else:
        server.retrieve(
            {
                "class": "ei",
                "dataset": "interim",
                "date": "%s/to/%s" % (start, end),
                "expver": "1",
                "grid": "%s/%s" % (grid, grid),
                "area": "%s/%s/%s/%s" % (north, west, south, east),
                "levelist": "1",
                "levtype": "ml",
                "param": "z/lnsp",  # http://apps.ecmwf.int/codes/grib/param-db
                "step": "0",
                "stream": "oper",
                "time": "00:00:00/06:00:00/12:00:00/18:00:00",
                "type": "an",
                "target": surface_file,
                "format": "netcdf",
            }
        )

    if os.path.isfile(levels_file):
        print("    [+] %s is already present" % levels_file)
    else:
        server.retrieve(
            {
                "class": "ei",
                "dataset": "interim",
                "date": "%s/to/%s" % (start, end),
                "expver": "1",
                "grid": "%s/%s" % (grid, grid),
                "area": "%s/%s/%s/%s" % (north, west, south, east),
                "levelist": "/".join(str(l) for l in range(initial_level, 60 + 1)),
                "levtype": "ml",
                "param": "t/u/v/q/w",  # http://apps.ecmwf.int/codes/grib/param-db
                "step": "0",
                "stream": "oper",
                "time": "00:00:00/06:00:00/12:00:00/18:00:00",
                "type": "an",
                "target": levels_file,
                "format": "netcdf",
            }
        )

    return surface_file, levels_file

"""One-year loop."""

import calendar
import sys
from datetime import datetime

import traj3d

# Lauch several runs in parallel:
#   for year in $(seq 1979 1979); do python traj3d_looper_ariporo_review.py $year > $year.log & done

if len(sys.argv) == 2:
    year = int(sys.argv[1])
    for month in range(1, 13):
        if year == 1979 and month > 8:
            break
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            for hour in range(0, 24, 6):
                traj3d.demo(datetime(year, month, day, hour), "ariporo")
else:
    print("Usage: python traj3d_loop.py [year]")

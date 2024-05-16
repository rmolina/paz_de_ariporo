import calendar
import collections
import datetime
import itertools
from timeit import default_timer as timer

import netCDF4
import numpy as np
from pyproj import Transformer

STANDARD_ACCELERATION_OF_GRAVITY = 9.80665  # m s**-2
MOLAR_MASS_OF_WATER_VAPOR = 0.018016  # kg mol**-1
MOLAR_MASS_OF_DRY_AIR = 0.028964  # kg mol**-1
UNIVERSAL_GAS_CONSTANT = 8.3144598  # J K**-1 mol**-1
SPECIFIC_GAS_CONSTANT_FOR_DRY_AIR = (
    UNIVERSAL_GAS_CONSTANT / MOLAR_MASS_OF_DRY_AIR
)  # J K**-1 mol**-1

ECMWF_DATA_TIMESTEP = datetime.timedelta(hours=6)
VERTICAL_DISCRETISATION_A = np.array(
    [
        0,
        20,
        38.425343,
        63.647804,
        95.636963,
        134.483307,
        180.584351,
        234.779053,
        298.495789,
        373.971924,
        464.618134,
        575.651001,
        713.218079,
        883.660522,
        1094.834717,
        1356.474609,
        1680.640259,
        2082.273926,
        2579.888672,
        3196.421631,
        3960.291504,
        4906.708496,
        6018.019531,
        7306.631348,
        8765.053711,
        10376.126953,
        12077.446289,
        13775.325195,
        15379.805664,
        16819.474609,
        18045.183594,
        19027.695313,
        19755.109375,
        20222.205078,
        20429.863281,
        20384.480469,
        20097.402344,
        19584.330078,
        18864.75,
        17961.357422,
        16899.46875,
        15706.447266,
        14411.124023,
        13043.21875,
        11632.758789,
        10209.500977,
        8802.356445,
        7438.803223,
        6144.314941,
        4941.77832,
        3850.91333,
        2887.696533,
        2063.779785,
        1385.912598,
        855.361755,
        467.333588,
        210.39389,
        65.889244,
        7.367743,
        0,
        0,
    ]
)

VERTICAL_DISCRETISATION_B = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.000076,
        0.000461,
        0.001815,
        0.005081,
        0.011143,
        0.020678,
        0.034121,
        0.05169,
        0.073534,
        0.099675,
        0.130023,
        0.164384,
        0.202476,
        0.243933,
        0.288323,
        0.335155,
        0.383892,
        0.433963,
        0.484772,
        0.53571,
        0.586168,
        0.635547,
        0.683269,
        0.728786,
        0.771597,
        0.811253,
        0.847375,
        0.879657,
        0.907884,
        0.93194,
        0.951822,
        0.967645,
        0.979663,
        0.98827,
        0.994019,
        0.99763,
        1,
    ]
)


INTEGRATION_TIMESTEP = datetime.timedelta(minutes=15)
SIMULATION_TIME = datetime.timedelta(days=11)
DATA_PATH = "/data/rmolina/ecmwf/interim-ml-075"
VERBOSE = True
VERBOSE = False
NON_CONVERGENT_TRESHOLD = 20
TOLERANCE = 0.01

Point = collections.namedtuple("Point", ["easting", "northing", "altitude", "time"])

Wind = collections.namedtuple("Wind", ["zonal", "meridional", "vertical"])


def get_virtual_temperature(temperature, specific_humidity):
    """estimates the virtual temperature (K) from the temperature (K) and the
    specific humidity [kg/kg].  NOTE: the virtual temperature of a moist air
    parcel is the temperature at which a theoretical dry air parcel would have
    a total pressure and density equal to the moist parcel of air"""
    epsilon = MOLAR_MASS_OF_WATER_VAPOR / MOLAR_MASS_OF_DRY_AIR
    return temperature * (1 + specific_humidity / epsilon) / (1 + specific_humidity)


def get_air_density(atmospheric_pressure, virtual_temperature):
    """estimates the air density [kg/m3] from the atmospheric pressure [Pa]
    and the virtual temperature [K].  ASSUMES: hydrostatic"""
    return atmospheric_pressure / (
        SPECIFIC_GAS_CONSTANT_FOR_DRY_AIR * virtual_temperature
    )


def get_vertical_velocity(lagrangian_tendency_of_pressure, air_density):
    """estimates the vertical velocity [m/s] from the lagrangian tendency of
    pressure [Pa/s] and the air density [kg/m3].  ASSUMES: ideal gas"""
    return -lagrangian_tendency_of_pressure / (
        air_density * STANDARD_ACCELERATION_OF_GRAVITY
    )


def get_pressure_at_half_levels_ml(surface_pressure, levels):
    """ Calculates atmospheric pressure [Pa] at model half-levels for a list
    of "levels".  NOTE:  In order to get the pressure at the full levels we
    need to calculate the average of two half levels, e.g.:

        pressure_at_half_levels = \
            get_atmospheric_pressure(surface_pressure, levels)

        atmospheric_pressure = 0.5 * (pressure_at_half_levels[:, +1:] +
                                      pressure_at_half_levels[:, :-1])

    In ERA Interim, pressure at model levels is defined as:

        pressure[level] = A[level] + B[level] * surface_pressure

    For the source of A an B values used here, see:
    http://www.ecmwf.int/en/forecasts/documentation-and-support/60-model-levels

    NOTE: Returns an array of the form: p[time, level, latitude, longitude] """

    return (
        VERTICAL_DISCRETISATION_A[np.newaxis, :, np.newaxis, np.newaxis]
        + VERTICAL_DISCRETISATION_B[np.newaxis, :, np.newaxis, np.newaxis]
        * surface_pressure[:, np.newaxis]
    )[:, 60 - levels.size :]


def integrate_from_the_surface(array):
    """Integrates from the surface upwards (ASSUMES: levels in axis=1).

    np.cumsum() performs the summation from the first to the last element.
    In ERA Interim's model levels, the first element (lev=1) is near the top
    of the atmosphere and the last element (lev=60) is near the surface.

    Since we need to integrate *from the surface*, we first use np.fliplr()
    to "flip" the array along axis=1 (levels), and after we have performed the
    summation, we use np.fliplr() again to restore the proper order."""
    return np.fliplr(np.cumsum(np.fliplr(array), axis=1))


def get_geopotential_altitude_ml(
    pressure_at_half_levels, virtual_temperature, surface_geopotential=None
):
    """Calculate geopotential altitude at model levels

    If surface_geopotential is ommited, we will return an altitude from the
    surface. If surface_geopotential is provided, we will return an altitude
    above sea level.

    For the equations, see:
    http://www.ecmwf.int/sites/default/files/elibrary/2015/9210-part-iii-dynamics-and-numerical-procedures.pdf
    """

    pressure_at_level_plus_half = pressure_at_half_levels[:, 1:]
    pressure_at_level_minus_half = pressure_at_half_levels[:, :-1]

    pratio = np.empty_like(virtual_temperature)
    pratio[:, 1:] = np.log(
        pressure_at_level_plus_half[:, 1:] / pressure_at_level_minus_half[:, 1:]
    )

    # alpha = log(2) for level=1, and as in EQ 2.23 for other levels
    alpha = (
        1.0
        - pressure_at_level_minus_half
        / np.diff(pressure_at_half_levels, axis=1)
        * pratio
    )
    alpha[:, 0] = np.log(2)

    # From EQ 2.21: integrate z_h to next half level
    geopotential_at_level_plus_half = integrate_from_the_surface(
        SPECIFIC_GAS_CONSTANT_FOR_DRY_AIR * virtual_temperature * pratio
    )

    # From EQ 2.22: integrate from previous half-level to the full level
    geopotential = alpha * SPECIFIC_GAS_CONSTANT_FOR_DRY_AIR * virtual_temperature
    geopotential[:, :-1] += geopotential_at_level_plus_half[:, 1:]

    # Add surface geopotential when available.  NOTE: if surface geopotential
    # is not provided we will return elevations above ground level, and if it
    # is provided we will return elevations above sea level
    if surface_geopotential is not None:
        geopotential += surface_geopotential[:, np.newaxis]

    return geopotential / STANDARD_ACCELERATION_OF_GRAVITY


def nc_read_data_ml(initial_datetime, final_datetime, top=38):
    """Reads data from the netCDF4 files, and calculates the geopotential
    altitude [m] and the vertical velocity [m/s]"""
    with netCDF4.MFDataset(
        DATA_PATH + "/????-??-ml-an-1.nc"
    ) as surface, netCDF4.MFDataset(DATA_PATH + "/????-??-ml-an-1to60.nc") as levels:

        times = netCDF4.num2date(
            levels.variables["time"][:], levels.variables["time"].units
        )

        initial = min(np.where(times == initial_datetime)[0])
        final = min(np.where(times == final_datetime)[0])

        if initial > final:
            initial, final = final, initial

        try:
            when = slice(initial.item(), final.item() + 1)
        except:
            raise IndexError(
                "Data is unavailable for the date range (%s, %s)"
                % (initial_datetime, final_datetime)
            )

        surface_pressure = np.exp(surface.variables["lnsp"][when, ...])

        pressure_at_half_levels = get_pressure_at_half_levels_ml(
            surface_pressure, levels.variables["level"][top:]
        )

        virtual_temperature = get_virtual_temperature(
            levels.variables["t"][when, top:], levels.variables["q"][when, top:]
        )

        altitude = get_geopotential_altitude_ml(
            pressure_at_half_levels, virtual_temperature
        )

        atmospheric_pressure = (
            pressure_at_half_levels[:, 1:] + pressure_at_half_levels[:, :-1]
        ) / 2.0

        air_density = get_air_density(atmospheric_pressure, virtual_temperature)

        vertical_velocity = get_vertical_velocity(
            levels.variables["w"][when, top:], air_density
        )

        return (
            levels.variables["longitude"][:],
            levels.variables["latitude"][:],
            altitude,
            Wind(
                levels.variables["u"][when, top:],
                levels.variables["v"][when, top:],
                vertical_velocity,
            ),
            levels.variables["q"][when, top:],
        )


def nc_read_data_pl(initial_datetime, final_datetime, grid="025"):
    """reads latitude, longitude, eastern wind, and northern wind data from
    netCDF files. Also reads geopotential and estimates altitude, and reads
    temperature, specific humidity, and lagrangian tendencies of air pressure
    (omega), and estimates vertical velocities"""
    with netCDF4.MFDataset(DATA_PATH + "/????-??.pl.%s.nc" % grid) as dataset:

        times = netCDF4.num2date(
            dataset.variables["time"][:], dataset.variables["time"].units
        )

        initial = np.where(times == initial_datetime)[0]
        final = np.where(times == final_datetime)[0]

        if initial > final:
            initial, final = final, initial
        try:
            when = slice(initial.item(), final.item() + 1)
        except:
            raise IndexError(
                "Data is unavailable for the date range (%s, %s)"
                % (initial_datetime, final_datetime)
            )

        virtual_temperature = get_virtual_temperature(
            dataset.variables["t"][when], dataset.variables["q"][when]
        )
        atmospheric_pressure = (
            100.0 * dataset.variables["level"][:][np.newaxis, :, np.newaxis, np.newaxis]
        )

        air_density = get_air_density(atmospheric_pressure, virtual_temperature)

        vertical_velocity = get_vertical_velocity(
            dataset.variables["w"][when], air_density
        )

        surface_geopotential = dataset.variables["z"][when, -1]

        altitude = (
            dataset.variables["z"][when] - surface_geopotential[:, np.newaxis]
        ) / STANDARD_ACCELERATION_OF_GRAVITY

        return (
            dataset.variables["longitude"][:],
            dataset.variables["latitude"][:],
            altitude,
            Wind(
                dataset.variables["u"][when],
                dataset.variables["v"][when],
                vertical_velocity,
            ),
            dataset.variables["q"][when],
        )


def geo_to_xy(longitude, latitude):
    """Transforms coordinates into a projected system"""
    transformer = Transformer.from_crs("epsg:4326", "epsg:3395")
    return transformer.transform(latitude, longitude)


def xy_to_geo(easting, northing):
    """Transforms coordinates into a geographic system"""
    transformer = Transformer.from_crs("epsg:3395", "epsg:4326")
    return transformer.transform(easting, northing)


def update_trajectories(trajectories, wind, grid, delta_time):
    """implements the Petterssen schema."""

    xpett = [0, 0, 0]
    ypett = [0, 0, 0]
    zpett = [0, 0, 0]

    upett = [0, 0]
    vpett = [0, 0]
    wpett = [0, 0]

    # Petterssen scheme: first iteration

    # Initial position (from previous step)
    xpett[0] = trajectories.easting
    ypett[0] = trajectories.northing
    zpett[0] = trajectories.altitude

    # initial velocities
    upett[0], vpett[0], wpett[0] = interpolate_wind(trajectories, grid, wind)

    # new position
    xpett[1] = xpett[0] + upett[0] * delta_time
    ypett[1] = ypett[0] + vpett[0] * delta_time
    zpett[1] = zpett[0] + wpett[0] * delta_time

    # Petterssen scheme: next iterations
    icountp = 0

    # "stopping the iteration at n = 40 is certainly justified
    # (one might even use 20 as a treshold)."
    # Source: 10.1175/1520-0450(1993)032<0558:CAAONM>2.0.CO;2
    while icountp < 20:

        # new velocity
        upett[1], vpett[1], wpett[1] = interpolate_wind(
            Point(xpett[1], ypett[1], zpett[1], trajectories.time + delta_time),
            grid,
            wind,
        )

        # corrected position
        ypett[2] = ypett[0] + 0.50 * delta_time * (vpett[0] + vpett[1])
        xpett[2] = xpett[0] + 0.50 * delta_time * (upett[0] + upett[1])
        zpett[2] = zpett[0] + 0.50 * delta_time * (wpett[0] + wpett[1])

        # check convergence
        if abs(
            np.concatenate(
                (xpett[2] - xpett[1], ypett[2] - ypett[1], zpett[2] - zpett[1])
            )
            < TOLERANCE
        ).all():
            break  # all differences less than tolerance? --> success!
        else:  # any differences greater than tolerance? --> iterate
            xpett[1], ypett[1], zpett[1] = xpett[2], ypett[2], zpett[2]

        # control to force break after 20 iterations
        icountp += 1
        if VERBOSE and icountp >= 20:
            print("NO CONVERGENCE - CHANGE THE TIMESTEP")

    return xpett[2], ypett[2], zpett[2], trajectories.time + delta_time


def get_polyhedra(x_values, y_values, easting, northing):
    """identify the indices for 8 vertices of the polyhedra containing the
    3D points (in time, northing, and easting) that we want to interpolate
    for each trajectory"""

    number_of_trajectories = x_values.size

    x_gt = np.minimum(np.searchsorted(easting, x_values), easting.size - 1)
    x_lt = np.maximum(x_gt - 1, 0)

    y_lt = np.minimum(
        northing.size - np.searchsorted(northing[::-1], y_values), northing.size - 1
    )
    y_gt = np.maximum(y_lt - 1, 0)

    vertices = np.empty((number_of_trajectories, 8, 3), dtype="intp")
    for traj in range(number_of_trajectories):
        vertices[traj] = list(
            itertools.product(
                [0, 1],  # time: already sliced
                [y_lt[traj], y_gt[traj]],
                [x_lt[traj], x_gt[traj]],
            )
        )
    return vertices


def get_polychora(z_values, altitude, polyhedra):
    """identify the indices for 16 vertices of the polychora containing the
    4D points (in time, altitude, northing, and easting) that we want to
    interpolate for each trajectory"""

    number_of_trajectories = z_values.size

    vertices = np.empty((number_of_trajectories, 16, 4), dtype="intp")
    for traj in range(number_of_trajectories):

        vertex = 0
        for t_ix, y_ix, x_ix in polyhedra[traj]:
            z_size = altitude[t_ix, :, y_ix, x_ix].size
            z_lt = np.minimum(
                z_size - np.searchsorted(altitude[t_ix, :, y_ix, x_ix][::-1], z_values),
                z_size - 1,
            )
            z_gt = np.maximum(z_lt - 1, 0)

            for z_ix in [z_lt[traj], z_gt[traj]]:
                vertices[traj, vertex] = t_ix, z_ix, y_ix, x_ix
                vertex += 1

    return vertices


def get_weights(vertices, trajectories, grid):
    """calculate inverse-distance weights for the weigthed average used in
    interpolations"""
    weights = np.empty((trajectories.easting.size, 16))

    grid_value_at_vertices = np.empty(shape=(16, 4))

    normalize = [1e6, 1e6, 1e3, 1e8]

    for traj in range(trajectories.easting.size):
        for vertex in range(16):
            t_ix, z_ix, y_ix, x_ix = vertices[traj, vertex]
            grid_value_at_vertices[vertex] = [
                grid.easting[x_ix],
                grid.northing[y_ix],
                grid.altitude[t_ix, z_ix, y_ix, x_ix],
                grid.time[t_ix],
            ]
        grid_value_at_vertices /= normalize

        points_to_interpolate = np.column_stack(
            [
                trajectories.easting[traj],
                trajectories.northing[traj],
                trajectories.altitude[traj],
                trajectories.time,
            ]
        )

        points_to_interpolate /= normalize

        distance_vertices = np.sum(
            (grid_value_at_vertices - points_to_interpolate) ** 2, axis=1
        )

        # does the point to interpolate match a vertex?
        if np.any(distance_vertices == 0):
            weights[traj] = (distance_vertices == 0).astype("int")
        else:
            weights[traj] = 1.0 / distance_vertices

    return weights


def interpolate_wind(points, grid, wind):
    """interpolate the value of a wind field at a point as the weighted
    average of the values of the field at each one of the 16 vertices of the
    polychoron surrounding the point.  The process is performed simultaneously
    for one point on each trajectory"""

    # first, we build a polyhedron (in time, easting and northing) for a point
    # on each trajectory
    polyhedra = get_polyhedra(
        points.easting, points.northing, grid.easting, grid.northing
    )

    # next, we evaluate the altitudes at the polyhedra, and use them to extend
    # the polyhedra into polychora (in time, altitude, easting and northing)
    polychora = get_polychora(points.altitude, grid.altitude, polyhedra)

    # finally, we calculate the inverse of the Euclidean distances between the
    # points and the vertices of the polychora to get the weights.
    weights = get_weights(polychora, points, grid)

    # http://stackoverflow.com/a/28491737/7691859
    zonal = np.average(wind.zonal[tuple(polychora.T)].T, weights=weights, axis=1)
    meridional = np.average(
        wind.meridional[tuple(polychora.T)].T, weights=weights, axis=1
    )
    vertical = np.average(wind.vertical[tuple(polychora.T)].T, weights=weights, axis=1)

    return Wind(zonal, meridional, vertical)


def interpolate_humidity(points, grid, humidity):
    """TODO: this should be merged with interpolate_wind() into a new function
    since both: the polychora and vertices, can be re-used here"""

    # first, we build a polyhedron (in time, easting and northing) for a point
    # on each trajectory
    polyhedra = get_polyhedra(
        points.easting, points.northing, grid.easting, grid.northing
    )

    # next, we evaluate the altitudes at the polyhedra, and use the to extend
    # the polyhedra into polychora (in time, altitude, easting and northing)
    polychora = get_polychora(points.altitude, grid.altitude, polyhedra)

    # finally, we calculate the inverse of the euclidean distances between the
    # points and the vertices of the polychora to get the weights.
    weights = get_weights(polychora, points, grid)

    # http://stackoverflow.com/a/28491737/7691859
    return np.average(humidity[tuple(polychora.T)].T, weights=weights, axis=1)


def main(
    initial_datetime,
    initial_latitudes,
    initial_longitudes,
    initial_altitudes,
    trajectory_type="backward",
):
    """This is the main program. It can trace multiple trajectories for the
    same initial_time.  You only need to pass the initial latitude, longitude
    and elevation for each trajectory."""

    assert trajectory_type == "backward" or trajectory_type == "forward"

    # In forward trajectories this variables are equal to the globals
    data_timestep = ECMWF_DATA_TIMESTEP
    integration_timestep = INTEGRATION_TIMESTEP
    simulation_time = SIMULATION_TIME

    # In backward trajectories we need to flip the sign
    if trajectory_type == "backward":
        integration_timestep *= -1
        data_timestep *= -1
        simulation_time *= -1

    final_datetime = initial_datetime + simulation_time
    number_of_trajectories = initial_altitudes.size
    number_of_integration_steps = int(
        simulation_time.total_seconds() / integration_timestep.total_seconds()
    )

    # define output arrays
    x_traj = np.zeros(shape=(number_of_integration_steps + 1, number_of_trajectories))
    y_traj = np.zeros_like(x_traj)
    z_traj = np.zeros_like(x_traj)
    t_traj = np.zeros(shape=(number_of_integration_steps + 1))
    q_traj = np.zeros_like(x_traj)

    # initialize trajectories with the origin positions
    initial_easting, initial_northing = geo_to_xy(initial_longitudes, initial_latitudes)
    x_traj[0] = initial_easting
    y_traj[0] = initial_northing
    z_traj[0] = initial_altitudes
    t_traj[0] = calendar.timegm(initial_datetime.timetuple())

    # initialize the counter for integration steps
    integration_counter = 0

    print(
        "[*] Running %d %s trajectories for %s"
        % (number_of_trajectories, trajectory_type, initial_datetime)
    )
    starttime = timer()

    current_datetime = initial_datetime
    current_data_step = initial_datetime

    # this loop will handle the simulation time: it reads data for a pair of
    # successive data steps, projects the coordinates, and runs an internal
    # integration loop using the Petterssen scheme
    while current_datetime != final_datetime:
        if VERBOSE:
            print("[*] Current data step:", current_data_step)
        next_data_step = current_data_step + data_timestep

        # read data for current and next data steps
        longitude, latitude, altitude, wind, humidity = nc_read_data_ml(
            current_data_step, next_data_step
        )

        # project coordinates
        easting, northing = geo_to_xy(*np.meshgrid(longitude, latitude))

        # this loop will handle the integration from current to next data steps
        # using the Petterssen scheme
        while current_datetime != next_data_step:

            if VERBOSE:
                print("    [+] Current integration step is", current_datetime)
            current_datetime += integration_timestep

            # times as seconds (for the interpolator)
            steps = np.array(
                [
                    calendar.timegm(tt.timetuple())
                    for tt in [current_data_step, next_data_step]
                ]
            )

            # interpolate humidity
            q_traj[integration_counter] = interpolate_humidity(
                Point(
                    x_traj[integration_counter],
                    y_traj[integration_counter],
                    z_traj[integration_counter],
                    t_traj[integration_counter],
                ),
                Point(easting[0, :], northing[:, 0], altitude, steps),
                humidity,
            )

            integration_counter += 1

            # run the Petterssen scheme to integrate the trajectories to the
            # next *integration step*
            (
                x_traj[integration_counter],
                y_traj[integration_counter],
                z_traj[integration_counter],
                t_traj[integration_counter],
            ) = update_trajectories(
                Point(
                    x_traj[integration_counter - 1],
                    y_traj[integration_counter - 1],
                    z_traj[integration_counter - 1],
                    t_traj[integration_counter - 1],
                ),
                wind,
                Point(easting[0, :], northing[:, 0], altitude, steps),
                integration_timestep.total_seconds(),
            )

        current_data_step = next_data_step

    # interpolate humidity at the last integration step
    q_traj[integration_counter] = interpolate_humidity(
        Point(
            x_traj[integration_counter],
            y_traj[integration_counter],
            z_traj[integration_counter],
            t_traj[integration_counter],
        ),
        Point(easting[0, :], northing[:, 0], altitude, steps),
        humidity,
    )

    print("[*] Runtime was %.2f seconds" % (timer() - starttime))

    lons, lats = xy_to_geo(x_traj, y_traj)

    dts = np.array(
        [
            datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=secs)
            for secs in t_traj
        ]
    )

    return lons, lats, z_traj, dts, q_traj


def demo(date_target, prefix="demo"):
    """an example on how to call main() and plot some outputs"""
    #    import matplotlib.pyplot as plt
    #    import cartopy.crs

    filename = prefix + "_%s" % date_target.strftime("%Y-%02m-%02dT%02H")

    elev_target = [
        10.0,
        34.97,
        71.89,
        124.48,
        195.85,
        288.55,
        404.72,
        546.06,
        713.97,
        909.57,
        1133.73,
        1387.12,
        1670.26,
        1983.49,
        2327.04,
        2701.0,
        3105.35,
        3539.96,
        4004.59,
        4498.91,
        5022.43,
        5574.58,
    ]
    num_traj = len(elev_target)
    # Paz de Ariporo
    long_target = (
        [-71.9016668] * num_traj  # ideam station
        + [-72.00] * num_traj
        + [-71.25] * num_traj
        + [-70.50] * num_traj
        + [-69.75] * num_traj
        + [-72.00] * num_traj
        + [-71.25] * num_traj
        + [-70.50] * num_traj
        + [-69.75] * num_traj
    )
    lati_target = (
        [5.8806384] * num_traj  # ideam station
        + [6.00] * (num_traj * 4)
        + [5.25] * (num_traj * 4)
    )
    elev_target = (
        elev_target
        + elev_target
        + elev_target
        + elev_target
        + elev_target
        + elev_target
        + elev_target
        + elev_target
        + elev_target
    )

    traj_x, traj_y, traj_z, traj_t, traj_q = main(
        initial_datetime=date_target,
        trajectory_type="backward",
        initial_latitudes=np.array(lati_target),
        initial_longitudes=np.array(long_target),
        initial_altitudes=np.array(elev_target),
    )
    np.savez_compressed(filename, x=traj_x, y=traj_y, z=traj_z, t=traj_t, q=traj_q)


# demo(datetime.datetime(1981, 9, 26, 12, 0), 'ariporo')

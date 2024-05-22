import calendar
from pathlib import Path

import numpy as np
from cartopy.feature import NaturalEarthFeature
from cartopy.io.shapereader import Reader as ShapeReader
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from tqdm import tqdm

TRAJECTORIES_PATH = Path("./trajectories")
OUTPUT_PATH = Path("./output")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

FOREST_SHP = r"./wwf/WWF_GLOBIL_Amazon_Ecoregion_WGS84.shp"
FOREST = next(ShapeReader(FOREST_SHP).geometries()).simplify(0.1)
LAND50 = NaturalEarthFeature("physical", "land", "50m")
AMERICA = list(LAND50.geometries())[1199]
DOMAIN = Polygon([(-90, -40), (-90, +40), (-20, +40), (-20, -40)])
CONTINENT_POLY = DOMAIN.intersection(AMERICA)


def wang_sde(
    lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Modified from: https://portailsig.org/content/ellipsewangpy.html
    meanx = np.mean(np.radians(lon))  # modified
    meany = np.mean(np.radians(lat))  # modified
    x_d = np.radians(lon) - meanx
    y_d = np.radians(lat) - meany
    xyw = sum(x_d * y_d)
    x2w = sum(np.power(x_d, 2))
    y2w = sum(np.power(y_d, 2))
    cov = np.array([[x2w, xyw], [xyw, y2w]]) / len(lon)
    eigenval, eigenvec = np.linalg.eig(cov)
    sigmay, sigmax = np.sqrt(eigenval)
    theta = np.arccos(eigenvec[0, 0])  # modified
    return (
        np.degrees(meanx),
        np.degrees(meany),
        np.degrees(sigmax),
        np.degrees(sigmay),
        np.degrees(theta),
    )


def read_monthly_trajectories(
    traj_path: Path, year: int | str, month: int
) -> tuple[np.ndarray, np.ndarray]:
    xlst, ylst = [], []
    files = list(traj_path.glob(f"ariporo_{year}-{month:02d}-??T??.npz"))
    for file in tqdm(files, desc=f"{year}-{month:02d}"):
        data = np.load(file)
        xlst.append(data["x"])
        ylst.append(data["y"])
    return np.array(xlst), np.array(ylst)


def calculate_monthly_ellipses():
    for year in range(1979, 2020):
        for month in range(1, 13):
            if year == 2019 and month == 9:
                break

            lon, lat = read_monthly_trajectories(TRAJECTORIES_PATH, year, month)

            elst = []
            nsteps = len(lon[0, :, 0])

            for step in range(0, nsteps, 6 * 4):
                x_val = lon[:, step, :].ravel()
                y_val = lat[:, step, :].ravel()
                meanx, meany, sigmax, sigmay, theta = wang_sde(x_val, y_val)
                elst.append([meanx, meany, sigmax, sigmay, theta])

            np.save(OUTPUT_PATH / f"ariporo_sde_{year}-{month:02d}", np.array(elst))


def calculate_ellipses_climatology():

    for month in range(1, 13):
        lon, lat = read_monthly_trajectories(TRAJECTORIES_PATH, "????", month)

        elst = []
        nsteps = len(lon[0, :, 0])
        for step in range(0, nsteps, 6 * 4):
            x_val = lon[:, step, :].ravel()
            y_val = lat[:, step, :].ravel()
            meanx, meany, sigmax, sigmay, theta = wang_sde(x_val, y_val)
            elst.append([meanx, meany, sigmax, sigmay, theta])
        np.save(OUTPUT_PATH / f"ariporo_sde_{month:02d}", np.array(elst))


def calculate_monthly_fractions(arg_numsd=1):
    elst = []
    for year in range(1979, 2020):
        for month in range(1, 13):
            if year == 2019 and month == 9:
                break

            sdes = np.load(OUTPUT_PATH / f"ariporo_sde_{year}-{month:02d}.npy")

            patches = []
            for meanx, meany, sigmax, sigmay, theta in sdes:
                patch = Ellipse(
                    xy=(meanx, meany),
                    width=arg_numsd * 2 * sigmay,
                    height=arg_numsd * 2 * sigmax,
                    angle=theta,
                )
                patches.append(patch)

            vertices = np.concatenate(
                [patch.get_verts() for patch in patches if len(patch.get_verts())]
            )

            hull = ConvexHull(vertices)
            qvert = vertices[hull.vertices]
            polyhull = Polygon(zip(*qvert.T))

            continental_fraction = (
                CONTINENT_POLY.intersection(polyhull).area / polyhull.area
            )
            oceanic_fraction = 1 - continental_fraction
            forest_fraction = FOREST.intersection(polyhull).area / polyhull.area
            land_fraction = continental_fraction - forest_fraction

            elst.append(
                [
                    year,
                    month,
                    oceanic_fraction,
                    continental_fraction,
                    land_fraction,
                    forest_fraction,
                ]
            )
    np.save(OUTPUT_PATH / f"ariporo_fractions", np.array(elst))
    np.savetxt(
        OUTPUT_PATH / f"ariporo_fractions.csv",
        np.array(elst),
        delimiter=",",
        header="year,month,oceanic_fraction,continental_fraction,land_fraction,forest_fraction",
    )


def calculate_fractions_climatology(arg_numsd=1):
    elst = []
    for month in range(1, 13):

        sdes = np.load(OUTPUT_PATH / f"ariporo_sde_{month:02d}.npy")

        patches = []
        for meanx, meany, sigmax, sigmay, theta in sdes:
            patch = Ellipse(
                xy=(meanx, meany),
                width=arg_numsd * 2 * sigmay,
                height=arg_numsd * 2 * sigmax,
                angle=theta,
            )
            patches.append(patch)

        vertices = np.concatenate(
            [patch.get_verts() for patch in patches if len(patch.get_verts())]
        )

        hull = ConvexHull(vertices)
        qvert = vertices[hull.vertices]
        polyhull = Polygon(zip(*qvert.T))

        continental_fraction = (
            CONTINENT_POLY.intersection(polyhull).area / polyhull.area
        )
        oceanic_fraction = 1 - continental_fraction
        forest_fraction = FOREST.intersection(polyhull).area / polyhull.area
        land_fraction = continental_fraction - forest_fraction

        elst.append(
            [
                month,
                oceanic_fraction,
                continental_fraction,
                land_fraction,
                forest_fraction,
            ]
        )
    np.save(OUTPUT_PATH / f"ariporo_fractions_means", np.array(elst))
    np.savetxt(
        OUTPUT_PATH / f"ariporo_fractions_means.csv",
        np.array(elst),
        delimiter=",",
        header="month,oceanic_fraction,continental_fraction,land_fraction,forest_fraction",
    )


def main():
    calculate_monthly_ellipses()
    calculate_ellipses_climatology()
    calculate_monthly_fractions()
    calculate_fractions_climatology()


if __name__ == "__main__":
    main()

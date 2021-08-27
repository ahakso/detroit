import pandas as pd
import geopandas as gpd
import kml2geojson
import os.path
import numpy as np

from scipy.spatial import KDTree


def kml_to_gpd(fn: str):
    """
    Should just be able to read kml, but it drops a bunch of columns.
    I tried & failed to solve that for a bit. Converting to json (annoying) solves the issue.
    """
    if ".kml" in fn:
        fn = fn.replace(".kml", "")
    if "json" in fn:
        fn = fn.replace(".json", "")
    if not os.path.isfile(fn + ".json"):
        kml2geojson.convert(fn + ".kml", fn + ".json")
    return gpd.read_file(f"{fn}.json/{fn}.geojson")


def csv_with_x_y_to_gpd(fn: str, crs="EPSG:4326", drop_null_cols: bool = True, read_csv_args: dict = {}):
    """assumes projection is consistent w/ detroit files with projection. if not true, supply crs

    optionally pass read_csv arguments convenient for reading in subset of rows w/ nrows
    """
    df = pd.read_csv(fn if "csv" in fn else fn + ".csv", **read_csv_args)
    if drop_null_cols:
        df = df.loc[:, df.notnull().sum() != 0]
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["X"], df["Y"])).set_crs("EPSG:4326")


def first_in_range_camera(a_df: pd.DataFrame, b_df: pd.DataFrame, distance_upper_bound: int = 50, max_in_range_cameras=10) -> pd.DataFrame:
    """return nearest point in B to each point in A

    both a_df and b_df must have a column called geometry with geopandas point values
    """

    #     get coordinates for each
    locations_a = np.array(list(a_df.geometry.apply(lambda x: (x.x, x.y))))
    locations_b = np.array(list(b_df.geometry.apply(lambda x: (x.x, x.y))))
    #     generate nearest neighbor lookup tree for point in B
    b_locations_map = KDTree(locations_b)
    #     calculate the first max_in_range closest cameras in B for every A closer than distance_upper_bound.
    dist_to_nearest_match, idx_in_b = b_locations_map.query(locations_a, k=max_in_range_cameras, distance_upper_bound=distance_upper_bound / 82410)
    # Get the first live date for within range cameras
    first_live_in_range_date = [
        [b_df.loc[idx_in_b[i_call][x], "live_day"] for x in range(max_in_range_cameras) if dist_to_nearest_match[i_call][x] != np.inf] for i_call in range(locations_a.shape[0])
    ]
    first_live_in_range_date = [min(x) if len(x) > 0 else np.nan for x in first_live_in_range_date]
    return a_df.assign(date_first_live_camera=first_live_in_range_date)

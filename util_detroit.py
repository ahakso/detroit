import os.path
from typing import Optional

import geopandas as gpd
import kml2geojson
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from detroit_geos import get_detroit_census_geos


def point_to_geo_id(
    df: gpd.GeoDataFrame,
    census_year: int = 2020,
    block_data_path: Optional[str] = "./",
    blocks: Optional[gpd.GeoDataFrame] = None,
) -> gpd.GeoSeries:
    """Return the geo ids for each row in the `geometry` column with a Point. <Returned serie>.index==df.index

    A unique identifier column "oid" for df is required to drop duplicates. If you don't have one, just assign one using .assign(oid=range(df.shape[0]))

    Args:
        df: DataFrame with a geometry column of type Point
        census_year: Year of the census data to use when looking up and returning block data
        block_data_path: Path to the census block data
        blocks: Optional GeoDataFrame of census blocks to avoid a load

    It's implemented in C, and very fast. About 250ms for 400k points and 16k polygons.
    """
    if blocks is None:
        blocks = get_detroit_census_geos(census_year, block_data_path)
    df = gpd.sjoin(df, blocks, how="left", predicate="within")
    # since lat/long is snapped, most of these are on block boundaries. Just pick one
    return df.drop_duplicates(subset=["oid"], keep="first").geo_id


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


def csv_with_x_y_to_gpd(fn: str, crs="epsg:3857", drop_null_cols: bool = True, read_csv_args: dict = {}):
    """uses a projection described here, https://epsg.io/3857, which projects to units of meters

    optionally pass read_csv arguments convenient for reading in subset of rows w/ nrows
    """
    df = pd.read_csv(fn if "csv" in fn else fn + ".csv", **read_csv_args)
    if drop_null_cols:
        df = df.loc[:, df.notnull().sum() != 0]
    if "Y" in df.columns:
        lat_col = "Y"
    elif "latitude" in df.columns:
        lat_col = "latitude"
    else:
        raise ValueError("No latitude column found")
    if "X" in df.columns:
        lon_col = "X"
    elif "longitude" in df.columns:
        lon_col = "longitude"
    else:
        raise ValueError("No longitude column found")

    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col])).set_crs(crs)


def first_in_range_camera(
    calls_df: pd.DataFrame, b_df: pd.DataFrame, distance_upper_bound: int = 50, max_in_range_cameras=10
) -> pd.DataFrame:
    """return nearest point in B to each point in A

    both a_df and b_df must have a column called geometry with geopandas point values
    """

    #     get coordinates for each
    locations_a = np.array(list(calls_df.geometry.apply(lambda x: (x.x, x.y))))
    locations_b = np.array(list(b_df.geometry.apply(lambda x: (x.x, x.y))))
    #     generate nearest neighbor lookup tree for point in B
    b_locations_map = KDTree(locations_b)
    #     calculate the first max_in_range closest cameras in B for every A closer than distance_upper_bound.
    dist_to_nearest_match, idx_in_b = b_locations_map.query(
        locations_a, k=max_in_range_cameras, distance_upper_bound=distance_upper_bound
    )
    closest = pd.DataFrame(dist_to_nearest_match).replace({np.inf: np.nan}).idxmax(axis=1)
    # return closest, idx_in_b
    # FIX THIS - THIS IS CLOSEST DISTANCE, I WANT EARLIEST TIME

    return calls_df.assign(
        first_live_camera=[
            idx_in_b[x][int(closest[x])] if not np.isnan(closest[x]) else np.nan for x in range(len(closest))
        ]
    )
    # return b_locations_map.query_ball_point(
    #     locations_a, distance_upper_bound / 82410
    # )  # This alternative is kind of interesting, but doesn't return distances. Nice for constraining problem of multiple matches per call


def get_normalized_time_series(df, background_rate):
    event_counts = df.groupby("call_day").calldescription.count().rename("n_calls")
    event_proportions = pd.merge(background_rate, event_counts, left_index=True, right_on="call_day").pipe(
        lambda df: df.n_calls / df.total_calls
    )
    event_proportions_normalized = (event_proportions / event_proportions.mean()).rename("response")
    return (
        event_proportions_normalized.to_frame()
        .merge(df.loc[:, ["call_day", "days_since_live"]].drop_duplicates(), on="call_day")
        .set_index("days_since_live")
        .response
    )

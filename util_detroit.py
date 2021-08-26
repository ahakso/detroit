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

def nearest_neighbor(a_df: pd.DataFrame, b_df: pd.DataFrame, target_nearness_rank: int=1, return_all_cols=False) -> pd.DataFrame:
    """return nearest point in B to each point in A
        
    both a_df and b_df must have a column called geometry with geopandas point values
    """
    
#     get coordinates for each
    locations_a = np.array(list(a_df.geometry.apply(lambda x: (x.x, x.y))))
    locations_b = np.array(list(b_df.geometry.apply(lambda x: (x.x, x.y))))
#     generate nearest neighbor lookup tree for point in B
    b_locations_map = KDTree(locations_b)
#     calculate the nearest point in B for every A
    dist_to_nearest_match, idx_in_b = b_locations_map.query(locations_a, k=[target_nearness_rank])
    if return_all_cols:
        b_nearest = b_df.iloc[idx_in_b.flatten()].drop(columns="geometry")
        b_nearest.columns = [str(col) + '_neighbor' for col in b_nearest.columns]
        gdf = pd.concat(
            [
                a_df.reset_index(drop=True),
                b_nearest.reset_index(drop=True),
                pd.Series(dist_to_nearest_match.flatten()*111139, name='meters_to_nearest_match')
            ], 
            axis=1)
        return gdf
    else:
        return a_df.assign(meters_to_nearest_match=dist_to_nearest_match*111139, idx_in_b=idx_in_b)

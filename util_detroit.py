import pandas as pd
import geopandas as gpd
import kml2geojson
import os.path


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

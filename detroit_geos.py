from typing import Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import GEO_GRAIN_LEN_MAP


def get_detroit_census_geos(
    decennial_census_year: int,
    data_path: str = "./",
    target_geo_grain: Optional[str] = "block",
    return_polygons: Optional[bool] = True,
    inclusion_grain: Optional[str] = "block",
    inclusion_criteria: Optional[str] = "intersects",
) -> gpd.GeoDataFrame:
    """
    Returns a geometries of the census polygons in Detroit for the given decennial census year.

    2010 census blocks are available from the following URL: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Blocks
    2020 census blocks are available from the following URL: https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/

    The shape files read in are available on box, and include only polygons with blocks at least partially intersecting the city of
    Detroit polygon, as defined in kx-city-of-detroit-michigan-city-boundary-SHP/city-of-detroit-michigan-city-boundary.shp,
    which was sourced from the following URL: https://koordinates.com/search/?q=detroit+city+boundary
    For generation of this file, see the section 'Census blocks in city boundaries' in the notebook census_blocks.ipynb in this repo.

    Geos are filtered by the inclusion_grain and inclusion_criteria parameters.

    Args:
        decennial_census_year -- The decennial census year to get the census blocks for.
        target_geo_grain -- The target geo grain to return. If None, return blocks
        return_polygons -- If True, return the geometry column, otherwise avoid computational overhead of dissolving polygons
        inclusIOn_grain -- Determines which spatial grain the inclusion critera operates on. Must be one of (block, block_group, tract)
        inclusion_criteria -- whether the inclusion_grain must be completely within detroit, or just touching.
            Must be one of ('intersects', 'within')
    """
    colmap = {
        "block_id": "block_id",
        "tract_id": "tract_id",
        "bg_id": "block_group_id",
        "block_x": "block_intersects",
        "block_in": "block_within",
        "tract_x": "tract_intersects",
        "tract_in": "tract_within",
        "bg_x": "block_group_intersects",
        "bg_in": "block_group_within",
        "geometry": "geometry",
    }

    df = (
        gpd.read_file(
            data_path + f"detroit_census_blocks_{decennial_census_year}/geos_in_detroit_{decennial_census_year}.shp"
        )
        .to_crs("epsg:4326")
        .astype({"block_id": float})
        .rename(columns=colmap)
    )
    # Filter to inclusion params:
    df = df.loc[df[inclusion_grain + "_" + inclusion_criteria] == 1, ["block_id", "geometry"]]
    if target_geo_grain != "block":
        column_aggs = "mean"
        df = df.assign(
            geo_id=lambda x: x.block_id // 10 ** (GEO_GRAIN_LEN_MAP["block"] - GEO_GRAIN_LEN_MAP[target_geo_grain])
        )
        if return_polygons:
            return df.dissolve(by="geo_id", aggfunc=column_aggs).reset_index()
        else:
            return df.drop(columns=["geometry"]).groupby("geo_id").agg(column_aggs).reset_index()
    else:
        df = df.rename(columns={"block_id": "geo_id"})
        if return_polygons:
            return df
        else:
            return df.drop(columns=["geometry"])


def get_detroit_boundaries():
    """
    Returns a geometries of the city of Detroit boundaries.

    The shape files read in are available on box in the directory kx-city-of-detroit-michigan-city-boundary-SHP

    sourced from the following URL: https://koordinates.com/search/?q=detroit+city+boundary
    """
    return gpd.read_file("kx-city-of-detroit-michigan-city-boundary-SHP/city-of-detroit-michigan-city-boundary.shp")


def attach_polygons(
    df: Union[gpd.GeoDataFrame, gpd.GeoSeries, pd.Series, pd.DataFrame],
    polygons_df: Optional[gpd.GeoDataFrame] = None,
    decennial_census_year: Optional[int] = 2010,
) -> gpd.GeoDataFrame:
    """
    Attaches the census polygons to the target geos.

    Args:
        df -- The target geos to attach the polygons to. Must have index of geo_id

    the polygons are retrieved at the grain of the geo_id index inclusion_grain, at the tract intersection level.
    If you need a specific inclusion_grain or inclusion_criteria, you'll need to pass the optional `polygons`
    GeoDataFrame
    """
    if isinstance(df, pd.Series) or isinstance(df, gpd.GeoSeries):
        df = df.to_frame()
    n_index_digits = np.floor(np.log10(df.index.values[0])) + 1
    if (not isinstance(df.index.values[0], float)) or (n_index_digits not in GEO_GRAIN_LEN_MAP.values()):
        raise ValueError("df must have float index of geo_ids")
    polygon_grain = {v: k for k, v in GEO_GRAIN_LEN_MAP.items()}.get(n_index_digits)
    if polygons_df is None:
        polygons_df = get_detroit_census_geos(
            decennial_census_year=decennial_census_year,
            data_path="./",
            target_geo_grain=polygon_grain,
            return_polygons=True,
            inclusion_grain="tract",
            inclusion_criteria="intersects",
        ).set_index("geo_id")
    if np.floor(np.log10(df.index.values[0])) != np.floor(np.log10(polygons_df.index.values[0])):
        raise ValueError("Polygons must have same index grain as df")
    return gpd.GeoDataFrame(df.merge(polygons_df.geometry, left_index=True, right_index=True, how="outer"))


def nice_detroit_plot(
    df: Union[gpd.GeoDataFrame, pd.Series, pd.DataFrame],
    color_column: str,
    ax=None,
    polygons_df: Optional[gpd.GeoDataFrame] = None,
    decennial_census_year: Optional[int] = 2010,
    save_to_filename: Optional[str] = None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    if "geometry" not in df.columns:
        df = attach_polygons(df, decennial_census_year=decennial_census_year, polygons_df=polygons_df)
    df.plot(color_column, ax=ax, legend=True)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.set(title=color_column)
    if save_to_filename is not None:
        plt.savefig(save_to_filename, dpi=120, transparent=True)

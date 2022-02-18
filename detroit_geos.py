from typing import Optional

import geopandas as gpd

from constants import GEO_GRAIN_LEN_MAP


def get_detroit_census_geos(
    decennial_census_year: int,
    data_path: str = "./",
    target_geo_grain: Optional[str] = "block",
    return_polygons: Optional[bool] = True,
) -> gpd.GeoDataFrame:
    """
    Returns a geometries of the census polygons in Detroit for the given decennial census year.

    2010 census blocks are available from the following URL: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Blocks
    2020 census blocks are available from the following URL: https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/

    The shape files read in are available on box, and include only polygons at least partially intersecting the city of
    Detroit polygon, as defined in kx-city-of-detroit-michigan-city-boundary-SHP/city-of-detroit-michigan-city-boundary.shp,
    which was sourced from the following URL: https://koordinates.com/search/?q=detroit+city+boundary
    For generation of this file, see the section 'Census blocks in city boundaries' in the notebook census_blocks.ipynb in this repo.

    Args:
        decennial_census_year -- The decennial census year to get the census blocks for.
        target_geo_grain -- The target geo grain to return. If None, return blocks

    """
    if decennial_census_year == 2010:
        df = gpd.read_file(data_path + "detroit_census_blocks_2010/blocks_in_detroit.shp")
    elif decennial_census_year == 2020:
        df = gpd.read_file(data_path + "detroit_census_blocks_2020/blocks_in_detroit_2020.shp")
    else:
        raise ValueError(f"decennial_census_year must be 2010 or 2020")
    df = df.to_crs("epsg:4326").astype({"block_id": float})
    if target_geo_grain != "block":
        column_aggs = {"longitude": "mean", "latitude": "mean"} if decennial_census_year == 2010 else "mean"
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

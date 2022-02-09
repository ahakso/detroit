import geopandas as gpd


def get_detroits_census_blocks(decennial_census_year: int) -> gpd.GeoDataFrame:
    """
    Returns a geometries of the census blocks in Detroit for the given decennial census year.

    2010 census blocks are available from the following URL: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Blocks
    2020 census blocks are available from the following URL: https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/

    The shape files read in are available on box, and include only polygons at least partially intersecting the city of
    Detroit polygon, as defined in kx-city-of-detroit-michigan-city-boundary-SHP/city-of-detroit-michigan-city-boundary.shp,
    which was sourced from the following URL: https://koordinates.com/search/?q=detroit+city+boundary
    For generation of this file, see the section 'Census blocks in city boundaries' in the notebook census_blocks.ipynb in this repo.

    :param decennial_census_year: The decennial census year to get the census blocks for.
    """
    if decennial_census_year == 2020:
        return gpd.read_file("detroit_census_blocks_2020/blocks_in_detroit.shp")
    elif decennial_census_year == 2010:
        return gpd.read_file("detroit_census_blocks_2010/blocks_in_detroit.shp")
    else:
        raise ValueError(f"decennial_census_year must be 2010 or 2020")

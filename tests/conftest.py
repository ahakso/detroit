import pandas as pd
import pytest

BLOCKS_PER_YEAR_GEO = {
    2010: {"block": 16341, "block group": 970, "tract": 346},
    2020: {"block": 14691, "block group": 698, "tract": 324},
}


@pytest.fixture()
def partial_geo_data():
    """Reads 5 row dataframes in each grain and year, using a sample of population data"""
    d = {grain: {} for grain in ["block", "block group", "tract"]}
    for grain in ("block", "block group", "tract"):
        for year in (2010, 2020):
            d[grain][year] = pd.read_csv(f"./tests/test_data/{grain}_{year}.csv")
    return d

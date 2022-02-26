import re
from logging import warn
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
from util_detroit import point_to_geo_id

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class smartbusstops(Feature):
    # Only read in the columns we want
    COLS_bus_stops = [
        "stop_lat",
        "stop_lon",
        "stop_id",
    ]
    TYPES_bus_stops = [float, float, int]

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": "SMART_bus_stops",
                "box_url": "https://bloombergdotorg.box.com/s/bpi2l5g4h8gascym621g7k7y029p7tah",
                "source_url": "https://data.detroitmi.gov/datasets/smart-bus-stops/explore",
                "min_geo_grain": "lat/long",
                "filename": "open_data/SMART_Bus_Stops.csv",
            },
            **kwargs,
        )

    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "SMART bus stops\n\n" + super_str

    def load_data(
        self,
        sample_rows: Optional[int] = None,
        use_lat_long: bool = False,
    ) -> None:
        """Bring in the granular data as an attribute of the class of type gpd.GeoDataframe: self.data

        Arguments:
            sample_rows -- Small file with the only useful row info is bus stop location
            use_lat_long -- use coordinates and census tracts rather than assigned ID. If using 2010 census, it's more accurate to use their id
            call_whitelist_strings: determines the whitelist filter on call descriptions. Pass 'close_proxy', 'near_proxy', or a list of custom whitelist strings

        Has no overlapping stops with the DDOT bus stops based on lat/lon matching.

        Dataset contains stops outside of Detroit.
        """

        # use a generator function to select rows we want in chunks rather than loading everything into memory at once

        df = pd.read_csv(
            self.data_path + self.meta.get("filename"),
            nrows=sample_rows,
            usecols=self.COLS_bus_stops,
            dtype=dict(zip(self.COLS_bus_stops, self.TYPES_bus_stops)),
        )

        stops = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.stop_lon, df.stop_lat), crs="epsg:4326"
        ).rename(columns={"stop_id": "oid"})
        stops = stops.assign(
            block_id=point_to_geo_id(
                stops.loc[:, ["oid", "geometry"]],
                self.decennial_census_year,
            )
        ).dropna(subset=["block_id"]).astype({'block_id':float}).rename(columns={"block_id": "geo_id"})
        self.data = stops
        print(f"Loaded {self.data.shape[0] if sample_rows is None else sample_rows:,} rows of data")

    @cleanse_decorator
    def cleanse_data(self) -> None:
        self.clean_data = self.data.copy().dropna(subset=["geo_id"])
        return self.clean_data

    @classmethod
    def null_handler(s: pd.Series) -> pd.Series:
        return s.fillna(0)

    @data_loader
    def construct_feature(self, target_geo_grain: str) -> pd.Series:
        """Return a Series of counts of stops by geo entity

        target_geo_grain should be one of "block", "block group", "tract"

        By default, will load and cleanse data if not already done
        """
        bus_stops = (
            self.assign_geo_column(target_geo_grain).groupby("geo").oid.count()
        )
        return bus_stops.reindex(self.index)

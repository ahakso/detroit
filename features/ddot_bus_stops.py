import re
from logging import warn
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
from util_detroit import point_to_geo_id

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class DDotBusStops(Feature):
    # Only read in the columns we want
    COLS_bus_stops = [
        "Latitude",
        "Longitude",
        "Shelter",
        "Bench",
        "Trash",
        "StopID",
    ]
    TYPES_bus_stops = [float, float, int, int, int, int]

    def __init__(
        self,
        decennial_census_year: Optional[int] = 2010,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": "DDOT_bus_stops",
                "box_url": "https://bloombergdotorg.box.com/s/o9cs59l0uzhfgia6k9xzi46gasyypgfx",
                "source_url": "https://data.detroitmi.gov/datasets/ddot-bus-stops/explore",
                "min_geo_grain": "lat/long",
                "filename": "open_data/DDOT_Bus_Stops.csv",
            },
            decennial_census_year=decennial_census_year,
            **kwargs,
        )

    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "DDOT bus stops\n\n" + super_str

    def load_data(
        self,
        sample_rows: Optional[int] = None,
    ) -> None:
        """
        Data Notes:

            5546 total stops
            1148 with trash bins
            306 with benches
            208 with shelters
        """

        # use a generator function to select rows we want in chunks rather than loading everything into memory at once

        df = pd.read_csv(
            self.data_path + self.meta.get("filename"),
            nrows=sample_rows,
            usecols=self.COLS_bus_stops,
            dtype=dict(zip(self.COLS_bus_stops, self.TYPES_bus_stops)),
        )

        stops = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="epsg:4326").rename(
            columns={"StopID": "oid"}
        )
        stops = (
            stops.assign(
                block_id=point_to_geo_id(
                    stops.loc[:, ["oid", "geometry"]],
                    self.decennial_census_year,
                )
            )
            .dropna(subset=["block_id"])
            .astype({"block_id": float})
            .rename(columns={"block_id": "geo_id"})
        )
        self.data = stops
        print(f"Loaded {self.data.shape[0] if sample_rows is None else sample_rows:,} rows of data")

    @cleanse_decorator
    def cleanse_data(self) -> None:
        self.clean_data = self.data.copy().dropna(subset=["geo_id"])
        return self.clean_data

    @data_loader
    def construct_feature(self, target_geo_grain: str) -> pd.Series:
        """Return a Series of counts of stops by geo entity

        target_geo_grain should be one of "block", "block group", "tract"

        By default, will load and cleanse data if not already done
        """
        bus_stops = self.assign_geo_column(target_geo_grain).groupby("geo").oid.count().rename("bus_stops")
        return bus_stops.reindex(self.index).fillna(0)

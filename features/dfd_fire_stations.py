import re
from logging import warn
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
from util_detroit import point_to_geo_id

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class dfdfirestations(Feature):
    # Only read in the columns we want
    COLS_fire_stations = [
        "Lat",
        "Long",
        "FID",
    ]
    TYPES_fire_stations = [float, float, int]

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": "DFD Fire Stations Locations",
                "box_url": "https://bloombergdotorg.box.com/s/jolyncbdjpqcbg475q3vg7dm71gq9fm0",
                "source_url": "https://data.detroitmi.gov/datasets/dfd-fire-station-locations/explore",
                "min_geo_grain": "lat/long",
                "filename": "open_data/DFD_Fire_Station_Locations.csv",
            },
            **kwargs,
        )

    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "DFD Fire Stations\n\n" + super_str

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

        
        """

        # use a generator function to select rows we want in chunks rather than loading everything into memory at once

        df = pd.read_csv(
            self.data_path + self.meta.get("filename"),
            nrows=sample_rows,
            usecols=self.COLS_fire_stations,
            dtype=dict(zip(self.COLS_fire_stations, self.TYPES_fire_stations)),
        )

        stations = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.Lat, df.Long), crs="epsg:4326"
        ).rename(columns={"FID": "oid"})
        stations = stations.assign(
            block_id=point_to_geo_id(
                stations.loc[:, ["oid", "geometry"]],
                self.decennial_census_year,
            )
        ).dropna(subset=["block_id"]).astype({'block_id':float}).rename(columns={"block_id": "geo_id"})
        self.data = stations
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
        stations = (
            self.assign_geo_column(target_geo_grain).groupby("geo").oid.count()
        )
        return stations.reindex(self.index)

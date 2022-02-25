import re
from logging import warn
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
from util_detroit import point_to_geo_id

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class projectgreenlightlocations(Feature):
    # Only read in the columns we want
    COLS_green_light_loc = [
        "X",
        "Y",
        "business_type",
        "precinct",
        "live_date",
        "ObjectId",
    ]
    TYPES_green_light_loc = [float, float, str, int, str, int]

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": "Project_Green_Light_Locations",
                "box_url": "https://bloombergdotorg.box.com/s/a9sn1hbdb6bahuwkyk2515y508iuls6f",
                "source_url": "https://data.detroitmi.gov/datasets/project-green-light-locations/explore?location=42.362386%2C-83.100770%2C11.18",
                "min_geo_grain": "lat/long",
                "filename": "open_data/Project_Green_Light_Locations.csv",
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
        
        Data Notes:
        * Business types: 'Retail', 'Services', 'Residential', 'Restaurant / Bar', 'Community', 'Party / Liquor Store'
        * Precint: 2-12

        """

        # use a generator function to select rows we want in chunks rather than loading everything into memory at once

        df = pd.read_csv(
            self.data_path + self.meta.get("filename"),
            nrows=sample_rows,
            usecols=self.COLS_green_light_loc,
            dtype=dict(zip(self.COLS_green_light_loc, self.TYPES_green_light_loc)),
        )

        locations = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.X, df.Y), crs="epsg:4326"
        )
        if use_lat_long:
            if self.decennial_census_year == 2010:
                warn("More accurate to use their block_id for 2010 census context")
            locations.assign(
                block_id=point_to_geo_id(
                    locations.loc[:, ["oid", "geometry"]],
                    self.decennial_census_year,
                )
            )
        self.data = locations
        print(f"Loaded {self.data.shape[0] if sample_rows is None else sample_rows:,} rows of data")

    @cleanse_decorator
    def cleanse_data(self) -> None:
        self.clean_data = self.data.copy().dropna(subset=["block_id"])
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
        green_light_locations = (
            self.assign_geo_column(target_geo_grain).groupby("geo").oid.count()
        )
        return green_light_locations.reindex(self.index)

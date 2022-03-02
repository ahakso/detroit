import re
from logging import warn
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
from util_detroit import point_to_geo_id

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class RentalStatuses(Feature):
    # Only read in the columns we want
    COLS_RENTALS = [
        "X",
        "Y",
        "date_status",
        "record_type",
        "oid",
    ]
    TYPES_RENTALS = [float, float, str, str, int]

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": "Rental_Statuses",
                "box_url": "https://bloombergdotorg.box.com/s/a5vqlnjp0w7g6nkmndmcs54s4pd7zadj",
                "source_url": "https://data.detroitmi.gov/datasets/rental-statuses-1/explore",
                "min_geo_grain": "lat/long",
                "filename": "open_data/Rental_Statuses.csv",
            },
            **kwargs,
        )

    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "Rental Statuses\n\n" + super_str

    def load_data(
        self,
        sample_rows: Optional[int] = None,
    ) -> None:
        """
        kept record_type but unsure how to use it yet, has 3 values: Registion Only, Initial Registration, and Renewal Registration
        """

        df = pd.read_csv(
            self.data_path + self.meta.get("filename"),
            nrows=sample_rows,
            usecols=self.COLS_RENTALS,
            dtype=dict(zip(self.COLS_RENTALS, self.TYPES_RENTALS)),
        )

        rentals = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs="epsg:4326")
        self.data = rentals.assign(
            geo_id=point_to_geo_id(
                rentals.loc[:, ["oid", "geometry"]],
                self.decennial_census_year,
            )
        ).astype({"geo_id": float})
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
        """Return a Series of counts of rental registrations by geo entity

        target_geo_grain should be one of "block", "block group", "tract"

        By default, will load and cleanse data if not already done
        """
        rentals = self.assign_geo_column(target_geo_grain).groupby("geo").oid.count().rename("rental_counts")
        return rentals.reindex(self.index).fillna(0)

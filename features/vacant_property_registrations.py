import re
from logging import warn
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
from util_detroit import point_to_geo_id

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class VacantPropertyRegistrations(Feature):
    # Only read in the columns we want
    COLS_VACANT_PROPERTIES = [
        "lat",
        "lon",
        "record_id",
        "date_status",
        "ObjectId",
    ]

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": "Vacant property registrations issued by BSEED",
                "box_url": "https://bloombergdotorg.box.com/s/kyibz9e8k2kdo6g4sfo5869yt27bpeqx",
                "source_url": "https://data.detroitmi.gov/datasets/vacant-property-registrations-1/explore",
                "min_geo_grain": "lat/long",
                "filename": "open_data/Vacant_Property_Registrations.csv",
            },
            **kwargs,
        )

    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "Vacant Property Registrations\n\n" + super_str

    def load_data(
        self,
        sample_rows: Optional[int] = None,
    ) -> None:

        df = pd.read_csv(
            self.data_path + self.meta.get("filename"),
            nrows=sample_rows,
            usecols=self.COLS_VACANT_PROPERTIES,
        )

        registrations = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="epsg:4326").rename(
            columns={"ObjectId": "oid"}
        )
        registrations = (
            registrations.assign(
                block_id=point_to_geo_id(
                    registrations.loc[:, ["oid", "geometry"]],
                    self.decennial_census_year,
                )
            )
            .dropna(subset=["block_id"])
            .astype({"block_id": float})
            .rename(columns={"block_id": "geo_id"})
        )
        self.data = registrations
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
        stations = self.assign_geo_column(target_geo_grain).groupby("geo").oid.count()
        return stations.reindex(self.index).fillna(0)

import re
from logging import warn
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
from util_detroit import point_to_geo_id

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class LiquorLicenses(Feature):
    # Only read in the columns we want
    COLS_LIQUOR_LICENSE = [
        "X",
        "Y",
        "business_id",
        "status",
        "number",
        "ObjectId",
    ]
    TYPES_LIQUOR_LICENSE = [float, float, int, str, str]

    def __init__(
        self,
        decennial_census_year=2010,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": "Active liquor licenses issued by the State of Michigan",
                "box_url": "https://bloombergdotorg.box.com/s/xr35jkvuk2j15mipkj3f27fk7pakxh4l",
                "source_url": "https://data.detroitmi.gov/datasets/liquor-licenses/explore",
                "min_geo_grain": "lat/long",
                "filename": "open_data/Liquor_Licenses.csv",
            },
            decennial_census_year=decennial_census_year,
            **kwargs,
        )

    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "Active Liquor Licenses\n\n" + super_str

    def load_data(
        self,
        sample_rows: Optional[int] = None,
    ) -> None:
        """
        Number is the license id and should be used to filter out duplicates.
        """

        df = pd.read_csv(
            self.data_path + self.meta.get("filename"),
            nrows=sample_rows,
            usecols=self.COLS_LIQUOR_LICENSE,
            dtype=dict(zip(self.COLS_LIQUOR_LICENSE, self.TYPES_LIQUOR_LICENSE)),
        )

        # Use only Active licenses
        df = df[df.status == "Active"]

        licenses = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs="epsg:4326").rename(
            columns={"ObjectId": "oid"}
        )
        licenses = (
            licenses.assign(
                block_id=point_to_geo_id(
                    licenses.loc[:, ["oid", "geometry"]],
                    self.decennial_census_year,
                )
            )
            .dropna(subset=["block_id"])
            .astype({"block_id": float})
            .rename(columns={"block_id": "geo_id"})
        )
        self.data = licenses
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
        stations = self.assign_geo_column(target_geo_grain).groupby("geo").number.nunique()
        return stations.reindex(self.index).fillna(0)

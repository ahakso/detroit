from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class Households(Feature):
    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "2019 American Community survey on household counts\n\n" + super_str

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": ("households", "families", "married_families", "non_family_households"),
                "box_url": "https://bloombergdotorg.box.com/s/3y35ojoubnv4lgda3b7wo674om5kb1uq",
                "source_url": "https://data.census.gov/cedsci/table?q=income&g=0500000US26163%241400000",
                "min_geo_grain": "tract",
                "filename": "productDownload_2022-02-15T172253/ACSST5Y2019.S1901_data_with_overlays_2022-02-15T172238.csv",
            },
            decennial_census_year=2010,
            **kwargs,
        )

    def load_data(
        self,
    ) -> None:
        HOUSEHOLD_COLS = {
            "id": "geo_id",
            "Estimate!!Households!!Total": "households",
            "Estimate!!Married-couple families!!Total": "married_families",
            "Estimate!!Nonfamily households!!Total": "non_family_households",
        }
        raw = pd.read_csv(self.meta.get("filename"), usecols=HOUSEHOLD_COLS.keys(), skiprows=[0])

        df = (
            raw.rename(columns=HOUSEHOLD_COLS)
            .assign(
                geo_id=lambda x: x.geo_id.apply(lambda y: y.split("US")[1]),
            )
            .astype({"geo_id": float})
        )
        self.data = df

    @cleanse_decorator
    def cleanse_data(self) -> None:
        self.clean_data = self.data.dropna(subset=["geo_id"]).copy()
        return self.clean_data

    @data_loader
    def construct_feature(self, target_geo_grain: str = "block") -> pd.DataFrame:
        return self.assign_geo_column(target_geo_grain).set_index("geo").reindex(self.index).drop(columns=["geo_id"])

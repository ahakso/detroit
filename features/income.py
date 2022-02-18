from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class Income(Feature):
    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "2010 census income statistics\n\n" + super_str

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": ("mean_household_income", "per_capita_income"),
                "box_url": "",
                "source_url": "https://data.census.gov/cedsci/table?q=income&g=0500000US26163%241400000",
                "min_geo_grain": "tract",
                "filename": "productDownload_2022-02-15T172253/ACSST5Y2019.S1902_data_with_overlays_2022-02-15T172238.csv",
            },
            decennial_census_year=2010,
            **kwargs,
        )

    def load_data(
        self,
    ) -> None:
        INCOME_COLS = {
            "GEO_ID": "geo_id",
            "S1902_C01_019E": "per_capita_income",
            "S1902_C03_001E": "per_household_income",
        }
        raw = pd.read_csv(self.meta.get("filename"), usecols=INCOME_COLS.keys(), skiprows=[1])

        df = (
            raw.rename(columns=INCOME_COLS)
            .astype({"per_capita_income": float})
            .assign(
                geo_id=lambda x: x.geo_id.apply(lambda y: y.split("US")[1]),
                per_household_income=lambda x: x.per_household_income.str.replace("-|N", "nan", regex=True),
                per_capita_income=lambda x: x.per_capita_income.replace(0, np.nan),
            )
            .astype({"per_household_income": float, "geo_id": float})
        )
        self.data = df

    @cleanse_decorator
    def cleanse_data(self) -> None:
        self.clean_data = self.data.dropna(subset=["geo_id"]).copy()
        return self.clean_data

    @data_loader
    def construct_feature(self, target_geo_grain: str = "block") -> pd.DataFrame:
        return self.assign_geo_column(target_geo_grain).set_index("geo")

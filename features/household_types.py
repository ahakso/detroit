import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class HouseholdTypes(Feature):
    """Block-level householdtypes for the census

    Arguments:
        decennial_census_year: year from which to draw the housing_types and block_ids
        data_path: path to housing_types data *must be within data_path*
    """

    def __init__(
        self,
        decennial_census_year: Optional[int] = 2020,
        **kwargs,
    ) -> None:
        if decennial_census_year == 2020:
            raise ValueError("2020 data not found. Year must be 2010")
        elif decennial_census_year == 2010:
            source_url = "https://data.census.gov/cedsci/table?q=DECENNIALSF12010.P18&tid=DECENNIALSF12010.P18"
            box_url = "https://bloombergdotorg.box.com/s/2r7cdthnlx8ncru4u1y9w3wp5p5yxsnq"
            fn = "DECENNIALSF12010.P18_data_with_overlays_2022-02-10T193949.csv"
        else:
            raise ValueError("Year must be 2010")
        super().__init__(
            meta={
                "feature_name": "housing_type",
                "box_url": box_url,
                "source_url": source_url,
                "min_geo_grain": "block",
                "filename": fn,
            },
            decennial_census_year=decennial_census_year,
            **kwargs,
        )

    def load_data(self):
        # this is a smaller categorical file; we can read it in
        df = pd.read_csv(os.path.join(self.data_path + self.meta.get("filename")), nrows=2)
        if self.decennial_census_year == 2010:
            # the first non header row in the table is
            cols = {}
            for col in df.columns:
                cols[col] = df[col].iloc[0]
            # cols
            cols["GEO_ID"] = "geo_id"
            del cols["NAME"]

        elif self.decennial_census_year == 2020:
            cols = {}
            raise ValueError("Year must be 2010")
        data = (
            pd.read_csv(
                os.path.join(self.data_path + self.meta.get("filename")),
                usecols=cols.keys(),
                skiprows=[1],
            )
            .rename(columns=cols)
            .assign(geo_id=lambda x: x.geo_id.str.split("US").apply(lambda s: s[1]))
        )
        del cols["GEO_ID"]
        self.features = list(cols.values())
        # cast geo_id to int
        data["geo_id"] = data["geo_id"].astype(float)

        # clean up Total column
        data["Total"] = data["Total"].apply(lambda x: re.sub(r"\([^()]*\)", "", x)).astype(np.int64)
        self.data = data
        self.data = self.remove_geos_outside_detroit(self.data)

        if self.verbose:
            print(f"Loaded {self.data.shape[0]} rows")

    @cleanse_decorator
    def cleanse_data(self):
        return self.data.copy()

    @data_loader
    def construct_feature(self, target_geo_grain: str) -> pd.Series:
        housing_types = self.assign_geo_column(target_geo_grain).groupby("geo")[self.features].sum().reindex(self.index)

        if self.verbose:
            print(f"{housing_types.isna().sum()} of {housing_types.shape[0]} {target_geo_grain}s are unaccounted for")
        return housing_types.fillna(0)

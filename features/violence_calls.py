import re
from logging import warn
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
from util_detroit import point_to_geo_id

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class ViolenceCalls(Feature):
    """A count of 911 calls received in the city of detroit"""

    CLOSE_PROXY_CALL_STRINGS = (
        "ASSAULT",
        "SHOTS",
        "SHOOTING",
        "CUTTING",
        "HOLD UP",
        "WEAP",
        "ROBBERY ARMED",
        "VIOLENT - ARMED",
        "RAPE",
        "STABBED",
        "SHOT",
    )
    NEAR_PROXY_CALL_STRINGS = ("WITH WEAPON", "DV", "ABUSE", "BREAKING AND ENTERING", "BREAKING & ENTERING")

    # Only read in the columns we want
    COLS_911 = [
        "calldescription",
        "call_timestamp",
        "block_id",
        "category",
        "officerinitiated",
        "priority",
        "oid",
        "longitude",
        "latitude",
    ]
    TYPES_911 = [str, str, float, str, str, str, int, float, float]

    def __init__(
        self,
        decennial_census_year: int = 2010,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": ("violence_calls",),
                "box_url": "https://bloombergdotorg.box.com/s/pci8u0mqij9kusq1ce9wq2lzjtznap58",
                "source_url": "https://data.detroitmi.gov/datasets/911-calls-for-service/explore",
                "min_geo_grain": "lat/long",
                "filename": "calls_for_service_from_jimmy.csv",
            },
            decennial_census_year=decennial_census_year,
            **kwargs,
        )

    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "Violence calls feature\n\n" + super_str

    def load_data(
        self,
        sample_rows: Optional[int] = None,
        use_lat_long: bool = False,
        call_whitelist_strings: Optional[Union[List[str], str]] = "close_proxy",
    ) -> None:
        """Bring in the granular data as an attribute of the class of type gpd.GeoDataframe: self.data

        Arguments:
            sample_rows -- This is a big file (~4M rows). Getting 100k rows is enough to play with, but defaults to full load
            use_lat_long -- use coordinates and census tracts rather than assigned ID. If using 2010 census, it's more accurate to use their block_id
            call_whitelist_strings: determines the whitelist filter on call descriptions. Pass 'close_proxy', 'near_proxy', or a list of custom whitelist strings
        """

        # use a generator function to select rows we want in chunks rather than loading everything into memory at once

        if call_whitelist_strings == "close_proxy":
            call_whitelist_strings = self.CLOSE_PROXY_CALL_STRINGS
        elif call_whitelist_strings == "near_proxy":
            call_whitelist_strings = self.NEAR_PROXY_CALL_STRINGS + self.CLOSE_PROXY_CALL_STRINGS
        expr = re.compile("|".join(call_whitelist_strings))
        generator = pd.read_csv(
            self.data_path + self.meta.get("filename"),
            nrows=sample_rows,
            usecols=self.COLS_911,
            parse_dates=["call_timestamp"],
            chunksize=1e4,
            dtype=dict(zip(self.COLS_911, self.TYPES_911)),
        )

        calls = pd.concat(
            [x.loc[lambda x: x.calldescription.str.contains(expr)] for x in generator],
            ignore_index=True,
        )

        calls = gpd.GeoDataFrame(
            calls, geometry=gpd.points_from_xy(calls.longitude, calls.latitude), crs="epsg:4326"
        ).rename(columns={"block_id": "geo_id"})
        if use_lat_long:
            if self.decennial_census_year == 2010:
                warn("More accurate to use their block_id for 2010 census context")
            calls = calls.assign(
                geo_id=point_to_geo_id(
                    calls.loc[:, ["oid", "geometry"]],
                    self.decennial_census_year,
                )
            )
        self.data = calls
        print(f"Loaded {self.data.shape[0] if sample_rows is None else sample_rows:,} rows of data")

    @cleanse_decorator
    def cleanse_data(self) -> None:
        self.clean_data = self.data.copy().dropna(subset=["geo_id"])
        return self.clean_data

    @classmethod
    def null_handler(s: pd.Series) -> pd.Series:
        return s.fillna(0)

    @data_loader
    def construct_feature(self, target_geo_grain: str, features: Tuple[str] = None) -> pd.DataFrame:
        """Return a Dataframe of counts of calls by geo entity

        target_geo_grain should be one of "block", "block group", "tract"

        By default, will load and cleanse data if not already done
        """
        if features is None:
            features = self.meta.get("supported_features")
        if "violence_calls" in features:
            n_violent_calls = self.assign_geo_column(target_geo_grain).groupby("geo").oid.count()
            return n_violent_calls.reindex(self.index).to_frame(name="violence_calls").fillna(0)

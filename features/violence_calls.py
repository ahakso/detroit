import re
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd

from feature_constructor import Feature


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
    TYPES_911 = [str, str, str, str, str, str, int, float, float]

    def __init__(self) -> None:
        super().__init__(
            meta={
                "feature_name": "violence_calls",
                "box_url": "https://bloombergdotorg.box.com/s/pci8u0mqij9kusq1ce9wq2lzjtznap58",
                "source_url": "https://data.detroitmi.gov/datasets/911-calls-for-service/explore",
                "min_geo_grain": "lat/long",
                "filename": "calls_for_service_from_jimmy.csv",
            }
        )

    def __repr__(self) -> str:
        super_str = super().__repr__()
        if self.data is None:
            data = "No data loaded"
        else:
            data = f"{self.data.shape[0]} rows"

        if self.clean_data is None:
            clean_data = "No data cleaned"
        else:
            clean_data = f"{self.clean_data.shape[0]} rows after cleaning"

        if self.index is None:
            index = "No index generated"
        else:
            index = f"Indexed to {self.index.name}"
        return super_str + "\n\n" + "\n\n".join([data, clean_data, index])

    def load_data(
        self,
        sample_rows: Optional[int] = None,
        call_whitelist_strings: Optional[Union[List[str], str]] = "close_proxy",
    ) -> None:
        """Bring in the granular data as an attribute of the class of type gpd.GeoDataframe: self.data

        This is a big file (~4M rows). Getting 100k rows is enough to play with, but defaults to full load

        call_whitelist_strings: determines the whitelist filter on call descriptions. Pass 'close_proxy', 'near_proxy', or a list of custom whitelist strings
        """

        # use a generator function to select rows we want in chunks rather than loading everything into memory at once

        if call_whitelist_strings == "close_proxy":
            call_whitelist_strings = self.CLOSE_PROXY_CALL_STRINGS
        elif call_whitelist_strings == "near_proxy":
            call_whitelist_strings = self.NEAR_PROXY_CALL_STRINGS + self.CLOSE_PROXY_CALL_STRINGS
        expr = re.compile("|".join(call_whitelist_strings))
        generator = pd.read_csv(
            self._data_path + self.meta.get("filename"),
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

        calls = gpd.GeoDataFrame(calls, geometry=gpd.points_from_xy(calls.longitude, calls.latitude), crs="epsg:4327")
        self.data = calls
        print(f"Loaded {sample_rows:,} rows of data")

    def cleanse_data(self) -> None:
        self.clean_data = self.data.copy().dropna(subset=["block_id"])
        self.standardize_block_id()
        self.validate_cleansed_data()

    @classmethod
    def null_handler(s: pd.Series) -> pd.Series:
        return s.fillna(0)

    def construct_feature(self, target_geo_grain: str, load_missing_data: bool = True) -> pd.Series:
        """Return a Series of counts of calls by geo entity

        target_geo_grain should be one of "block", "block group", "tract"

        By default, will load and cleanse data if not already done
        """
        if load_missing_data:
            if self.data is None:
                self.load_data()
            if self.clean_data is None:
                self.cleanse_data()

        if (self.index is None) or (self.index.name != target_geo_grain):
            self.index = self.generate_index(target_geo_grain)

        n_violent_calls = (
            self.assign_geo_column(target_geo_grain).groupby("geo").oid.count().rename(self.meta.get("feature_name"))
        )
        return n_violent_calls.reindex(self.index)

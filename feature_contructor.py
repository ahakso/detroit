from logging import warn
import geopandas as gpd
import numpy as np
from typing import Dict, List, Optional, Union
import pandas as pd
import re
import webbrowser

CLOSE_PROXY_CALL_STRINGS = [
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
]
NEAR_PROXY_CALL_STRINGS = ["WITH WEAPON", "DV", "ABUSE", "BREAKING AND ENTERING", "BREAKING & ENTERING"]


class Feature:
    """Parent class from which additional features constructors inherit

    This class is not intended to be used directly, but rather to be inherited from.
    Feature standardization, imputation etc is better handled by 3rd party libraries, and should not be done here.
    Features that require multiple data assets are not handled by this class. I suspect they can always be
    combined downstream.

    The following methods are required to be implemented:
        - load_data(), which should be an opinionated import of the raw data, selecting appropriate columns, performing
          obvious cleaning steps etc
        - cleanse_data(), which should be where experimentation on the roughly cleaned data happens. block_id must be
          assigned_here, and self.standardize_block_id() must be run
        - construct_feature(), which should reshape the data to output a Series indexed by the geo entity.
    """

    def __init__(self, meta=Dict) -> None:
        if meta.get("min_geo_grain") not in ("lat/long", "block", "block group", "tract"):
            raise ValueError("min_geo_grain must be one of 'lat/long', 'block', 'block group', 'tract'")
        self.meta = meta
        self.data = None
        self.clean_data = None

    def open_data_url(self, source: Optional[str] = "box") -> None:
        if source == "box":
            if self.meta.get("box_url") is None:
                raise ValueError("box_url must be defined in meta")
            webbrowser.open(self.meta.get("box_url"))
        elif source == "source":
            if self.meta.get("source_url") is None:
                raise ValueError("source_url must be defined in meta")
            webbrowser.open(self.meta.get("source_url"))
        if source not in ("box", "source"):
            raise ValueError("source must be one of 'box', 'source'")

    def load_data(self):
        """Method should be an opinionated import of the raw data, selecting appropriate columns, performing obvious cleaning steps etc

        Requires a local file. If you don't have it, get it from self.open_data_url()
        """
        raise NotImplementedError("load_data() must be implemented")

    def cleanse_data(self):
        """This method should be where experimentation on the rough cleaned data happens

        Requires self.data to be populated, and assigns the result to self.clean_data

        self.clean_data must be a dataframe, and must contain the column block_id, of type str and len(block_id) == 15
        Use self.standardize_block_id() to standardize block_id to length 15
        """
        raise NotImplementedError("clean_data() must be implemented")

    def construct_feature(self, target_geo_grain) -> pd.Series:
        """Reshape the data to output a Series indexed by the geo entity.

        target_geo_grain should be one of "block", "block group", "tract"

        Change of grain, including opinionated aggregations, should be implemented here.

        No additional munging of data should be carried out.

        Requires self.clean_data to be populated
        """

        raise NotImplementedError("construct_feature() must be implemented")

    def assign_geo_column(self, target_geo_grain: str) -> pd.DataFrame:
        """
        Takes a dataframe with a block_id of the appropriate length and truncates or extends it to the desired granularity
        """

        if target_geo_grain not in ("block", "block group", "tract"):
            raise ValueError("target_geo_grain must be one of 'block', 'block group', 'tract'")
        GEO_ENTITY_INDEX_MAP = {"block": 15, "block group": 12, "tract": 11}

        return self.clean_data.assign(geo=lambda x: x.block_id.str[: GEO_ENTITY_INDEX_MAP[target_geo_grain]])

    def validate_cleansed_data(self):
        """
        Ensures loaded data has columns and datatypes required downstream
        """
        if "block_id" not in self.clean_data.columns:
            raise ValueError("block_id must be in dataframe")
        if not isinstance(self.clean_data.block_id.iloc[0], str):
            raise ValueError("block_id must be a string")
        if np.any(self.clean_data.block_id.str.len() != (self.clean_data.block_id.iloc[0])):
            raise ValueError("block_id is of inconsistent length")
        print("block_id looks good")

    def standardize_block_id(self):
        """
        Standardizes block_id to all be of length 15 by right padding with zeros. Should be run for any feature of min_geo_grain != 'block'
        """
        if np.any(self.clean_data.block_id.str.len() != (self.clean_data.block_id.iloc[0])):
            warn("block_id is of inconsistent length")
        self.clean_data.block_id = self.clean_data.block_id.apply(lambda x: x.ljust(15, "0"))


class ViolenceCalls(Feature):
    def __init__(self) -> None:
        super().__init__(
            meta={
                "feature_name": "violence_calls",
                "box_url": "https://bloombergdotorg.box.com/s/pci8u0mqij9kusq1ce9wq2lzjtznap58",
                "source_url": "https://data.detroitmi.gov/datasets/911-calls-for-service/explore",
                "min_geo_grain": "lat/long",
                "standard_filename": "calls_for_service_from_jimmy.csv",
            }
        )

    def load_data(
        self,
        sample_rows: Optional[int] = None,
        data_path: Optional[str] = None,
        call_strings: Optional[Union[List[str], str]] = "close_proxy",
    ) -> None:
        """Bring in the granular data as an attribute of the class of type gpd.GeoDataframe: self.data

        This is a big file (~4M rows). Getting 100k rows is enough to play with, but defaults to full load

        call_strings: determines the whitelist filter on call descriptions. Pass 'close_proxy', 'near_proxy', or a list of custom whitelist strings
        """
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
        # use a generator function to select rows we want in chunks rather than loading everything into memory at once

        if call_strings == "close_proxy":
            call_strings = CLOSE_PROXY_CALL_STRINGS
        elif call_strings == "near_proxy":
            call_strings = NEAR_PROXY_CALL_STRINGS + CLOSE_PROXY_CALL_STRINGS
        expr = re.compile("|".join(call_strings))
        generator = pd.read_csv(
            self.meta.get("standard_filename") if data_path is None else data_path,
            nrows=sample_rows,
            usecols=COLS_911,
            parse_dates=["call_timestamp"],
            chunksize=1e4,
            dtype=dict(zip(COLS_911, TYPES_911)),
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
        self.validate_cleansed_data()

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

        return self.assign_geo_column(target_geo_grain).groupby("geo").oid.count().rename("n_violent_calls")

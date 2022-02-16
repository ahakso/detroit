import pprint
import webbrowser
from logging import warn
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from detroit_geos import get_detroit_census_blocks

GEO_GRAIN_LEN_MAP = {"block": 15, "block group": 12, "tract": 11}


def cleanse_decorator(func):
    def standardize_and_validate(self, *args, **kwargs):
        self.clean_data = func(self)
        if self.verbose:
            print(f"clean data has {self.clean_data.shape[0]} rows")
        self.standardize_block_id()
        self.validate_cleansed_data()

    return standardize_and_validate


def data_loader(func):
    """Loads and cleans data + assigns index. Useful for methods that require all three"""

    def load_data(self, target_geo_grain):
        if self.data is None:
            if self.verbose:
                print("Data not yet loaded, loading all data")
            self.load_data()
        if self.clean_data is None:
            if self.verbose:
                print("Data not yet cleansed, cleaning")
            self.cleanse_data()

        if (self.index is None) or (self.index.name != target_geo_grain):
            if self.verbose:
                print(
                    f"Generate index not run, or was run on the wrong grain. Creating index on {target_geo_grain} grain"
                )
            self.index = self.generate_index(target_geo_grain)
        return func(self, target_geo_grain)

    return load_data


class Feature:
    """Parent class from which additional features constructors inherit

    This class is not intended to be used directly, but rather to be inherited from.
    Feature standardization, imputation etc is better handled by 3rd party libraries, and should not be done here.
    Features that require multiple data assets are not handled by this class. I suspect they can always be
    combined downstream.

    Missing geographic entities are included with generate_index

    Arguments:
        meta -- metadata for the feature, hardcoded into child class
        data_path -- path to local data files
        decennial_census_year -- year of reference geo data

    Attributes:
        meta {dict}: A dictionary of metadata about the feature, including where to get the data, the minimum granularity, and the feature name
        data {pd.Dataframe}: An opinionated initial load of the data
        clean_data {pd.Dataframe}: data ready for feature construction
        index {pd.Index}: The geo index of the feature

    The following methods must be implemented in the child classes:
        - load_data(), which should be an opinionated import of the raw data, selecting appropriate columns, performing
          obvious cleaning steps etc
        - cleanse_data(), which should be where experimentation on the roughly cleaned data happens. block_id must be
          assigned_here
        - construct_feature(), which should reshape the data to output a Series indexed by the geo entity.
    """

    def __init__(
        self,
        meta: Dict,
        data_path: Optional[str] = ".",
        decennial_census_year: Optional[int] = 2020,
        verbose: Optional[bool] = True,
        **kwargs,
    ) -> None:
        if meta.get("min_geo_grain") not in ("lat/long", "block", "block group", "tract"):
            raise ValueError("min_geo_grain must be one of 'lat/long', 'block', 'block group', 'tract'")
        if decennial_census_year not in (2010, 2020):
            raise ValueError("decennial_census_year must be one of 2010, 2020")
        self.meta = meta
        self.data = None
        self.clean_data = None
        self.index = None
        self.data_path = data_path.rstrip("/") + "/"
        self.decennial_census_year = decennial_census_year
        self.verbose = verbose

    def __repr__(self) -> str:
        meta = f"Function metadata:\n{pprint.pformat(self.meta)}"
        ref_year = f"Using {self.decennial_census_year} as reference geo"
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
        return "\n\n".join([meta, ref_year, data, clean_data, index])

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

        Run self.standardize_block_id() to standardize block_id to length 15 and self.validate_cleansed_data() before exiting the method
        """
        raise NotImplementedError("clean_data() must be implemented")

    def generate_index(self, target_geo_grain: str) -> pd.Index:
        """Reads in the census blocks in detroit and generates a pandas index for the target_geo_grain"""
        blocks = get_detroit_census_blocks(self.decennial_census_year, data_path=self.data_path)
        self.index = pd.Index(blocks.block_id[: GEO_GRAIN_LEN_MAP[target_geo_grain]].unique(), name=target_geo_grain)

    def construct_feature(self) -> pd.DataFrame:
        """Reshape the data to output a Series indexed by the geo entity.

        target_geo_grain must be an argument, and should be one of "block", "block group", "tract"

        Change of grain, including opinionated aggregations, should be implemented here.

        No additional munging of data should be carried out.

        Requires self.clean_data to be populated

        we could implement a Tuple[pd.Series] return that indicates any filled nulls, but not clearly necessary
        """

        raise NotImplementedError("construct_feature() must be implemented")

    def null_handler(self) -> None:
        """Method to handle null values (permitting nulls to go through if appropriate)

        This method is not required to be implemented, but is makes null handling explicit.
        Should be called after reindexing in construct_feature().
        """
        raise NotImplementedError("null_handler() must be implemented")

    def assign_geo_column(self, target_geo_grain: str) -> pd.DataFrame:
        """
        take block_id from self.clean_data and truncates or extends it to the desired granularity
        """

        if target_geo_grain not in ("block", "block group", "tract"):
            raise ValueError("target_geo_grain must be one of 'block', 'block group', 'tract'")

        n_chars_to_truncate = GEO_GRAIN_LEN_MAP.get(self.meta.get("min_geo_grain")) - GEO_GRAIN_LEN_MAP.get(
            target_geo_grain
        )
        is_adequately_granular = n_chars_to_truncate >= 0

        if self.clean_data.per_capita_income.dtype == "float64":
            if is_adequately_granular:
                return self.clean_data.assign(geo=lambda x: x.block_id // (10 ** n_chars_to_truncate))
            else:
                n_chars_to_truncate = GEO_GRAIN_LEN_MAP.get("block") - GEO_GRAIN_LEN_MAP.get(
                    self.meta.get("min_geo_grain")
                )
                blocks = get_detroit_census_blocks(self.decennial_census_year, self.data_path)
                return (
                    blocks.loc[:, ["block_id"]]
                    .assign(join_column=lambda x: x.block_id // 10 ** (n_chars_to_truncate))
                    .rename(columns={"block_id": "geo"})
                    .merge(self.clean_data, left_on="join_column", right_on="block_id")
                    .drop(columns=["join_column"])
                    .astype({"geo": float})
                )
        elif self.clean_data.per_capita_income.dtype == "object":
            if is_adequately_granular:
                return self.clean_data.assign(geo=lambda x: x.block_id.str[: GEO_GRAIN_LEN_MAP[target_geo_grain]])
            else:
                raise NotImplementedError("Ability to go to more granular than available not implemented")

    def validate_cleansed_data(self):
        """
        Ensures loaded data has columns and datatypes required downstream
        """
        if "block_id" not in self.clean_data.columns:
            raise ValueError("block_id must be in dataframe")
        if self.clean_data.block_id.dtype not in ("object", "float64"):
            raise ValueError("block_id must be a string or float")
        if self.clean_data.block_id.dtype == "object":
            block_id_len = self.clean_data.block_id.str.len()
        else:
            block_id_len = self.clean_data.block_id.transform(lambda x: np.floor(np.log10(x)) + 1)
        if np.any(block_id_len != block_id_len.iloc[0]):
            raise ValueError("block_id is of inconsistent length")
        if self.verbose:
            print("cleansed data validator: block_id looks good")

    def standardize_block_id(self):
        """
        Standardizes block_id to all be of length corresponding with self.meta.min_geo_grain by right padding with zeros.
         Should be run in load_data (automatic with @data_loader)
        This may be a no-op

        For features of self.meta.min_geo_grain > target_geo_grain passed to self.construct_feature(), full geo_id
        will need to be introduced
        """
        target_len = GEO_GRAIN_LEN_MAP.get(self.meta.get("min_geo_grain"))
        if self.clean_data.block_id.dtype == "object":
            if (self.clean_data.block_id == "nan").sum():
                warn("Null block ids exist prior to standardization")
            no_null_blocks = self.clean_data.loc[lambda x: x.block_id != "nan"].block_id
            if np.any(no_null_blocks.str.len() != len(no_null_blocks.iloc[0])):
                warn(
                    "block_id is of inconsistent length in self.clean_data prior to standardization. Investigate self.data"
                )
            self.clean_data.block_id = self.clean_data.block_id.apply(lambda x: x.ljust(target_len, "0"))
        else:
            if self.clean_data.block_id.isna().sum():
                warn("Null block ids exist prior to standardization")
            self.clean_data.block_id = self.clean_data.block_id.apply(
                lambda x: x * 10 ** (target_len - (np.floor(np.log10(x)) + 1))
            )

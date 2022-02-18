import pprint
import webbrowser
from logging import warn
from typing import Dict, Optional

import numpy as np
import pandas as pd
from constants import GEO_GRAIN_LEN_MAP
from detroit_geos import get_detroit_census_geos


def cleanse_decorator(func):
    def standardize_and_validate(self, *args, **kwargs):
        self.clean_data = func(self)
        if self.verbose:
            print(f"clean data has {self.clean_data.shape[0]} rows")
        self.standardize_geo_id()
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
        - cleanse_data(), which should be where experimentation on the roughly cleaned data happens. geo_id must be
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

        self.clean_data must be a dataframe, and must contain the column geo_id, of type float,
        and the floor(log10(geo_id))+1 == GEO_GRAIN_LEN_MAP[self.meta.get("min_geo_grain")]

        Run self.standardize_geo_id() to standardize geo_id to consistent length and self.validate_cleansed_data() before exiting the method
        """
        raise NotImplementedError("clean_data() must be implemented")

    def generate_index(self, target_geo_grain: str) -> pd.Index:
        """Reads in the census blocks in detroit and generates a pandas index for the target_geo_grain"""
        geos = get_detroit_census_geos(
            self.decennial_census_year,
            data_path=self.data_path,
            target_geo_grain=target_geo_grain,
            return_polygons=False,
        )
        self.index = pd.Index(
            geos.geo_id.unique(),
            name=target_geo_grain,
        )

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
        if self.clean_data.geo_id.dtype != "float64":
            raise ValueError("geo_id must be of type float64")
        n_chars_from_target_to_min = GEO_GRAIN_LEN_MAP.get(self.meta.get("min_geo_grain")) - GEO_GRAIN_LEN_MAP.get(
            target_geo_grain
        )
        n_chars_from_block_to_target = GEO_GRAIN_LEN_MAP.get("block") - GEO_GRAIN_LEN_MAP.get(target_geo_grain)
        n_chars_from_block_to_min = GEO_GRAIN_LEN_MAP.get("block") - GEO_GRAIN_LEN_MAP.get(
            self.meta.get("min_geo_grain")
        )
        is_coarser_than_target = n_chars_from_target_to_min < 0
        if is_coarser_than_target:
            # Use ground truth census block assign a column for both target_geo_grain (geo) and min_geo_grain (join_column)
            blocks = get_detroit_census_geos(self.decennial_census_year, self.data_path)
            return (
                blocks.loc[:, ["geo_id"]]
                .assign(join_column=lambda x: x.geo_id // 10 ** (n_chars_from_block_to_min))
                .assign(geo=lambda x: x.geo_id // 10 ** (n_chars_from_block_to_target))
                .drop(columns=["geo_id"])
                .merge(self.clean_data, left_on="join_column", right_on="geo_id")
                .drop(columns=["join_column"])
                .drop_duplicates(subset=["geo"])
                .astype({"geo": float})
            )
        else:
            return self.clean_data.assign(geo=lambda x: x.geo_id // (10 ** n_chars_from_target_to_min))

    def validate_cleansed_data(self):
        """
        Ensures loaded data has columns and datatypes required downstream
        """
        if "geo_id" not in self.clean_data.columns:
            raise ValueError("geo_id must be in dataframe")
        if self.clean_data.geo_id.dtype != "float64":
            raise ValueError("geo_id must be a float")
        geo_id_len = self.clean_data.geo_id.transform(lambda x: np.floor(np.log10(x)) + 1)
        if np.any(geo_id_len != geo_id_len.iloc[0]):
            raise ValueError("geo_id is of inconsistent length")
        if self.verbose:
            print("cleansed data validator: geo_id looks good")

    def standardize_geo_id(self):
        """
        Standardizes geo_id to all be of length corresponding with self.meta.min_geo_grain by right padding with zeros.
        Should be run in load_data (automatic with @data_loader)
        This may be a no-op

        For features of self.meta.min_geo_grain > target_geo_grain passed to self.construct_feature(), full geo_id
        will need to be introduced
        """
        target_len = GEO_GRAIN_LEN_MAP.get(self.meta.get("min_geo_grain"))
        if self.clean_data.geo_id.isna().sum():
            warn("Null geo ids exist prior to standardization")
        self.clean_data.geo_id = self.clean_data.geo_id.apply(
            lambda x: x * 10 ** (target_len - (np.floor(np.log10(x)) + 1))
        )

    def remove_geos_outside_detroit(self, df, target_geo_grain: Optional[str] = None):
        """
        Removes geos outside of detroit. Useful in the load_data() method when data contains geos we don't want

        Could easily be extended to use polygons to do this with geopandas.
        """
        if target_geo_grain is None:
            target_geo_grain = self.meta.get("min_geo_grain")
        geos_in_detroit = get_detroit_census_geos(
            self.decennial_census_year, self.data_path, target_geo_grain, return_polygons=False
        ).loc[:, ["geo_id"]]
        return pd.merge(df, geos_in_detroit, on="geo_id", how="inner")

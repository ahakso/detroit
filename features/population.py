from typing import Optional

import pandas as pd

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class Population(Feature):
    """Block-level population for the census

    Arguments:
        year -- decennial census year (affects both block_id and population counts)
        data_path: path of population csv

    """

    def __init__(self, year: int, data_path: Optional[str] = ".", verbose: Optional[bool] = True) -> None:
        self.year = year
        if year == 2020:
            source_url = "https://data.census.gov/cedsci/table?q=Population%20Total&t=Counts,%20Estimates,%20and%20Projections&g=0500000US26163%241000000&tid=DECENNIALPL2020.P1"
            box_url = "https://bloombergdotorg.box.com/s/og2qmb948k5aj7koch94kf60sfori73t"
            fn = "DECENNIALPL2020.P1_data_with_overlays_2022-02-06T092022.csv"
        elif year == 2010:
            source_url = "https://data.census.gov/cedsci/table?q=Population%20Total&t=Counts,%20Estimates,%20and%20Projections&g=0500000US26163%241000000&tid=DECENNIALPL2010.P1"
            box_url = "https://bloombergdotorg.box.com/s/zvsd9depnwj6nctmahhjo7baiekt86vn"
            fn = "DECENNIALPL2020.P1_data_with_overlays_2022-02-06T092022.csv"
        else:
            raise ValueError("Year must be 2010 or 2020")
        super().__init__(
            meta={
                "feature_name": "population",
                "box_url": box_url,
                "source_url": source_url,
                "min_geo_grain": "block",
                "filename": fn,
            },
            decennial_census_year=year,
            data_path=data_path,
            verbose=verbose,
        )
        self.year = year

    def load_data(self):
        if self.year == 2010:
            cols = {"GEO_ID": "block_id", "P001001": "population"}
        elif self.year == 2020:
            cols = {"GEO_ID": "block_id", "P1_001N": "population"}
        self.data = (
            pd.read_csv(
                self._data_path + self.meta.get("filename"),
                usecols=cols.keys(),
                skiprows=[1],
            )
            .rename(columns=cols)
            .assign(block_id=lambda x: x.block_id.str.split("US").apply(lambda s: s[1]))
        )
        if self.verbose:
            print(f"Loaded {self.data.shape[0]} rows")

    @cleanse_decorator
    def cleanse_data(self):
        return self.data.copy()

    @data_loader
    def construct_feature(self, target_geo_grain: str) -> pd.Series:
        population = (
            self.assign_geo_column(target_geo_grain)
            .groupby("geo")
            .population.sum()
            .rename(self.meta.get("feature_name"))
            .reindex(self.index)
        )
        if self.verbose:
            print(f"{population.isna().sum()} of {population.shape[0]} {target_geo_grain}s are unaccounted for")
        return population

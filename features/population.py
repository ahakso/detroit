from typing import Optional

import pandas as pd

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class Population(Feature):
    """Block-level population for the census

    Arguments:
        decennial_census_year: year from which to draw the population and block_ids
        population_data_path: path to population data *must be within data_path*
    """

    def __init__(
        self,
        decennial_census_year: Optional[int] = 2020,
        population_data_path: Optional[str] = "",
        **kwargs,
    ) -> None:
        if decennial_census_year == 2010:
            source_url = "https://data2.nhgis.org/main"
            box_url = "https://bloombergdotorg.box.com/s/qq9da9oknv6bb7pd8w802oalp27tg22b"
            fn = "nhgis0001_ds172_2010_block.csv"
        elif decennial_census_year == 2020:
            source_url = "https://data2.nhgis.org/main"
            box_url = "https://bloombergdotorg.box.com/s/6stydvct9exh1fycfxbjjvmw4kj96acb"
            fn = "nhgis0002_ds248_2020_block.csv"
        else:
            raise ValueError("Year must be 2010 or 2020")
        if "meta" in kwargs:
            meta = kwargs.pop("meta")
        else:
            meta = {
                "supported_features": ("population",),
                "box_url": box_url,
                "source_url": source_url,
                "min_geo_grain": "block",
                "filename": fn,
            }
        super().__init__(
            meta=meta,
            decennial_census_year=decennial_census_year,
            **kwargs,
        )
        self.population_data_path = self.data_path.rstrip("/") + "/" + population_data_path.rstrip("/") + "/"

    def load_data(self):
        if self.decennial_census_year == 2010:
            cols = {
                "H7V001": "population",
                "STATEA": "state_code",
                "COUNTYA": "county_code",
                "BLOCKA": "block_code",
                "TRACTA": "tract_code",
            }
            self.data = (
                pd.read_csv(
                    self.population_data_path + self.meta.get("filename"),
                    usecols=cols.keys(),
                )
                .loc[lambda x: x.COUNTYA == 163]
                .assign(
                    geo_id=lambda t: t.STATEA.astype(str)
                    + t.COUNTYA.astype(str)
                    + t.TRACTA.astype(str)
                    + t.BLOCKA.astype(str)
                )
                .loc[lambda x: x.geo_id.astype(str).str.len() == 15, ["geo_id", "H7V001"]]
                .astype({"geo_id": float})
                .rename(columns={"H7V001": "population"})
            )
        elif self.decennial_census_year == 2020:
            cols = {"GEOCODE": "geo_id", "U7B001": "population"}
            self.data = (
                pd.read_csv(
                    self.population_data_path + self.meta.get("filename"),
                    usecols=["GEOCODE", "U7B001"],
                )
                .rename(columns=cols)
                .astype({"geo_id": float})
            )

        self.data = self.remove_geos_outside_detroit(
            self.data, target_geo_grain="block", inclusion_grain="tract", inclusion_criteria="intersects"
        )
        if self.data.population.dtype == "object":
            self.data = self.data.assign(population=lambda x: x.population.apply(lambda x: x.split("(")[0]))
        self.data = self.data.astype({"population": int})
        if self.verbose:
            print(f"Loaded {self.data.shape[0]} rows")

    @cleanse_decorator
    def cleanse_data(self):
        return self.data.copy()

    @data_loader
    def construct_feature(self, target_geo_grain: str) -> pd.DataFrame:
        population = self.assign_geo_column(target_geo_grain).groupby("geo").population.sum().reindex(self.index)
        if self.verbose:
            print(f"{population.isna().sum()} of {population.shape[0]} {target_geo_grain}s are unaccounted for")
        return population.to_frame(name=self.meta.get("supported_features")[0])

from typing import Optional

import pandas as pd

from features.feature_constructor import Feature


class Population(Feature):
    def __init__(self, year: int, population_data_path: Optional[str] = "."):
        """
        args:
            year: select which decenial census to use, 2010 or 2020
            population_data_path: data_path in super is still the base, but this can be nested
        """
        self.year = year
        self.pop = population_data_path
        if year == 2020:
            source_url = "https://data.census.gov/cedsci/table?q=Population%20Total&t=Counts,%20Estimates,%20and%20Projections&g=0500000US26163%241000000&tid=DECENNIALPL2020.P1"
            box_url = "https://bloombergdotorg.box.com/s/og2qmb948k5aj7koch94kf60sfori73t"
            fn = "DECENNIALSF12010.P10_data_with_overlays_2022-01-28T162836.csv"
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
                "filename": "csv",
            },
            decennial_census_year=year,
        )
        self.year = year
        self.population_data_path = self._data_path + population_data_path.rstrip("/") + "/"

    def load_data(self):
        decennial_p1 = (
            pd.read_csv(
                self.population_data_path + self.filename,
                usecols=["GEO_ID", "P1_001N", "NAME"],
                skiprows=[1],
            )
            .rename(columns={"GEO_ID": "block_id", "P1_001N": "population"})
            .assign(block_id=lambda x: x.block_id.str.split("US").apply(lambda s: s[1]))
            .astype({"block_id": float})
        )

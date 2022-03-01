from logging import warn
from typing import Optional

import geopandas as gpd
import pandas as pd
from detroit_geos import get_detroit_census_geos
from util_detroit import point_to_geo_id

from features.population import Population, cleanse_decorator, data_loader


class PopulationDensity(Population):
    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "Population density (people per sq km).\n\n" + super_str

    def __init__(
        self,
        decennial_census_year: Optional[int] = 2010,
        population_data_path: Optional[str] = "",
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": ("population_density",),
                "box_url": "Requires two files, specified in the superclass Population and get_detroit_census_geos",
                "source_url": "Requires two files, specified in the superclass Population and get_detroit_census_geos",
                "min_geo_grain": "block",
                "filename": "Requires two files, specified in the superclass Population and get_detroit_census_geos",
            },
            decennial_census_year=decennial_census_year,
            population_data_path=population_data_path,
            **kwargs,
        )

    def load_data(
        self,
    ) -> None:
        warn("No independent data source to load. See Population class for details.")

    def cleanse_data(self) -> None:
        warn("No independent data source here. See Population class for details.")
        # self.clean_data = self.data.dropna(subset=["geo_id"]).copy()
        # return self.clean_data

    def open_data_url(self, source: Optional[str]) -> None:
        warn("No independent data source here. See Population class for details.")

    def construct_feature(self, target_geo_grain: str = "block") -> pd.DataFrame:
        geo = get_detroit_census_geos(
            self.decennial_census_year,
            self.data_path,
            target_geo_grain,
            inclusion_grain="tract",
            inclusion_criteria="intersects",
        ).assign(sq_km=lambda x: x.geometry.to_crs("EPSG:3857").area / 1e6)
        population = Population(self.decennial_census_year, self.population_data_path, verbose=False).construct_feature(
            target_geo_grain
        )
        return (population.population / geo.set_index("geo_id").sq_km).rename("population_density")

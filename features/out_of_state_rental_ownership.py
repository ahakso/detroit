from typing import Optional

import geopandas as gpd
import pandas as pd
from util_detroit import point_to_geo_id

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class OutOfStateRentalOwnership(Feature):
    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "Out of state rental ownership proportion \n\n" + super_str

    def __init__(
        self,
        decennial_census_year: Optional[int] = 2010,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": ("out_of_state_rental_ownership",),
                "box_url": "https://bloombergdotorg.box.com/s/a5vqlnjp0w7g6nkmndmcs54s4pd7zadj",
                "source_url": "https://data.detroitmi.gov/datasets/detroitmi::rental-statuses-1/about",
                "min_geo_grain": "lat/long",
                "filename": "Rental_Statuses.csv",
            },
            decennial_census_year=decennial_census_year,
            **kwargs,
        )

    def load_data(
        self,
    ) -> None:
        raw = pd.read_csv(self.meta.get("filename"))
        df = gpd.GeoDataFrame(raw, geometry=gpd.points_from_xy(raw.X, raw.Y), crs="epsg:4326")
        df = df.assign(
            geo_id=lambda df: point_to_geo_id(
                df.loc[:, ["oid", "geometry"]],
                self.decennial_census_year,
            ),
            owner_state=lambda df: df.owner_state.str.upper(),
        ).assign(owner_state=lambda df: df.owner_state.str.replace(" *MI *|MICHIGAN|MI +(MI)|MICH", "MI", regex=True))
        self.data = df

    @cleanse_decorator
    def cleanse_data(self) -> None:
        self.clean_data = self.data.dropna(subset=["geo_id"]).copy()
        return self.clean_data

    @data_loader
    def construct_feature(self, target_geo_grain: str = "block") -> pd.DataFrame:
        return (
            self.assign_geo_column(target_geo_grain)
            .groupby("geo")
            .apply(lambda df: (df.owner_state != "MI").sum() / df.shape[0])
        ).reindex(self.index)

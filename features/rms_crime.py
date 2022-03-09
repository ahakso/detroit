import re
from logging import warn
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
from util_detroit import point_to_geo_id

from features.feature_constructor import Feature, cleanse_decorator, data_loader


class RmsCrime(Feature):
    """A count of violent rms crime incidents in detroit"""

    WHITELIST_STRINGS = [
        "MURDER",
        "HOMICIDE",
        "CSC",
        "ROBBERY",
        "CARJACKING",
        "ASSAULT",
        "SHOOTING",
        "WEAPONS OFFENSE",
    ]
    COLNAMES = [
        "crime_id",
        "report_number",
        "address",
        "offense_description",
        "offense_category",
        "state_offense_code",
        "arrest_charge",
        "charge_description",
        "incident_timestamp",
        "day_of_week",
        "hour_of_day",
        "year",
        "scout_car_area",
        "precinct",
        "geo_id",
        "neighborhood",
        "council_district",
        "zip_code",
        "longitude",
        "latitude",
        "oid",
        "geometry",
    ]
    COLS_TO_KEEP = ["offense_description", "arrest_charge", "geo_id", "longitude", "latitude", "oid", "geometry"]

    def __init__(
        self,
        decennial_census_year: int = 2010,
        **kwargs,
    ) -> None:
        super().__init__(
            meta={
                "supported_features": ("rms_crime",),
                "box_url": "https://bloombergdotorg.box.com/s/ng6xv921tcpe4xldb5fk0bjcu8bo388u",
                "source_url": "https://data.detroitmi.gov/datasets/detroitmi::rms-crime-incidents/about",
                "min_geo_grain": "lat/long",
                "filename": "RMS_Crime_Incidents/RMS_Crime_Incidents.shp",
            },
            decennial_census_year=decennial_census_year,
            **kwargs,
        )

    def __repr__(self) -> str:
        super_str = super().__repr__()
        return "Violent RMS crimes feature\n\n" + super_str

    def load_data(
        self,
        sample_rows: Optional[int] = None,
        use_lat_long: bool = False,
    ) -> None:
        """Bring in the granular data as an attribute of the class of type gpd.GeoDataframe: self.data

        Arguments:
            sample_rows -- This is a big file (~4M rows). Getting 100k rows is enough to play with, but defaults to full load
            use_lat_long -- use coordinates and census tracts rather than assigned ID. If using 2010 census, it's more accurate to use their block_id

        arrest codes for michigan can be found at https://www.michigan.gov/documents/MICRArrestCodes_June06_163082_7.pdf
        """

        expr = re.compile("|".join(self.WHITELIST_STRINGS))
        raw = gpd.read_file(self.data_path + self.meta.get("filename"), rows=sample_rows)
        raw.columns = self.COLNAMES
        df = raw.loc[lambda x: x.offense_description.fillna("").str.contains(expr), self.COLS_TO_KEEP]

        if use_lat_long:
            if self.decennial_census_year == 2010:
                warn("More accurate to use their block_id for 2010 census context")
            df = df.assign(
                geo_id=point_to_geo_id(
                    df.loc[:, ["oid", "geometry"]],
                    self.decennial_census_year,
                )
            )
        else:
            if self.decennial_census_year == 2020:
                raise ValueError("Must use lat/long to map to 2020 census, detroit assigns 2010 census blocks")
        self.data = df.astype({"geo_id": float})
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
        """Return a Dataframe of counts of violent crimes by geo entity

        target_geo_grain should be one of "block", "block group", "tract"

        By default, will load and cleanse data if not already done
        """
        if features is None:
            features = self.meta.get("supported_features")
        if "rms_crime" in features:
            n_rms_crimes = self.assign_geo_column(target_geo_grain).groupby("geo").oid.count()
            return n_rms_crimes.reindex(self.index).to_frame(name="rms_crime").fillna(0)

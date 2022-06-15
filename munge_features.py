import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def transform_1(
    feat_df,
    cols_to_log=[
        "greenlight_density",
        "rental_density",
        "bus_density",
        "per_household_income",
        "liquor_license_density",
        "vacant_property_density",
    ],
):
    """The _1 is an id, just add transformers if you want to keep this one around
    Args:
        feat_df (pandas.DataFrame): The dataframe comprised of simply concatenated feature dataframes output by a feature class
    Returns:
        pd.DataFrame: A dataframe ready to modeling
    """

    df = feat_df.copy().loc[lambda x: (x.population >= 10) & (x.households > 0)]
    imputer = SimpleImputer(strategy="median")

    cols_with_nulls = df.isna().sum().loc[lambda x: x > 0].index
    df.loc[:, cols_with_nulls] = imputer.fit_transform(
        df.loc[
            :,
            cols_with_nulls,
        ]
    )

    df0 = df.assign(
        # Convert to calls per 1k people per year
        call_rate=lambda x: 1000 * (x.violence_calls / x.population) / 4.5,
        married_household_prop=lambda x: (x.married_families.fillna(0)) / x.households,
        non_family_household_prop=lambda x: x.non_family_households / x.households,
        area=lambda df: df.population_density / df.population,
        people_per_household=lambda x: (x.population / x.households).clip(upper=5),
        greenlight_density=lambda x: x.greenlights / x.area,
        rental_density=lambda x: x.rental_counts / x.area,
        bus_density=lambda x: x.bus_stops / x.area,
        per_household_income=lambda x: x.per_household_income,
        vacant_property_density=lambda x: x.vacant_properties / x.area,
        liquor_license_density=lambda x: x.liquor_licenses / x.area,
        # liquor_license_density=lambda x: pd.qcut(x.liquor_licenses / x.area, 3),
    ).drop(
        columns=[
            "population",
            "violence_calls",
            "households",
            "married_families",
            "non_family_households",
            "smart_bus_stops",
            "bus_stops",
            "rental_counts",
            "greenlights",
            "area",
            "per_capita_income",
            "non_family_household_prop",
            "liquor_licenses",
            "vacant_properties",
        ]
    )
    df = df0.copy()
    df[cols_to_log] = df[cols_to_log].apply(lambda x: np.log(x + 1))

    cols_with_nulls = df.isna().sum().loc[lambda x: x > 0].index

    if len(cols_with_nulls) > 0:
        df.loc[:, cols_with_nulls] = imputer.fit_transform(
            df.loc[
                :,
                cols_with_nulls,
            ]
        )

    return df, df0

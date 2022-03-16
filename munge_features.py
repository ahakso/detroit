import numpy as np
from sklearn.impute import SimpleImputer


def transform_1(feat_df, cols_to_log=["greenlight_density", "rental_density", "bus_density", "per_household_income"]):
    """The _1 is an id, just add transformers if you want to keep this one around
    Args:
        feat_df (pandas.DataFrame): The dataframe comprised of simply concatenated feature dataframes output by a feature class
    Returns:
        pd.DataFrame: A dataframe ready to modeling
    """

    df = feat_df.copy()

    # Quite a few tracts would be lost if we let these nulls live
    df.isna().sum()

    imputer = SimpleImputer(strategy="median")

    cols_with_nulls = df.isna().sum().loc[lambda x: x > 0].index

    df.loc[:, cols_with_nulls] = imputer.fit_transform(
        df.loc[
            :,
            cols_with_nulls,
        ]
    )

    df0 = (
        df.loc[lambda x: x.population >= 10]
        .assign(
            # Convert to calls per 1k people per year
            call_rate=lambda x: 1000 * (x.violence_calls / x.population) / 4.5,
            married_household_prop=lambda x: x.married_families / x.households,
            non_family_household_prop=lambda x: x.non_family_households / x.households,
            area=lambda df: df.population_density / df.population,
            people_per_household=lambda x: (x.population / x.households).clip(upper=5),
            greenlight_density=lambda x: x.greenlights / x.area,
            rental_density=lambda x: x.rental_counts / x.area,
            bus_density=lambda x: x.bus_stops / x.area,
            per_household_income=lambda x: x.per_household_income,
        )
        .drop(
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
            ]
        )
    )
    df = df0.copy()
    df[cols_to_log] = df[cols_to_log].apply(lambda x: np.log(x + 1))

    cols_with_nulls = df.isna().sum().loc[lambda x: x > 0].index

    df.loc[:, cols_with_nulls] = imputer.fit_transform(
        df.loc[
            :,
            cols_with_nulls,
        ]
    )

    return df, df0

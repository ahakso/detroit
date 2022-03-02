from sklearn.impute import SimpleImputer


def transform_1(feat_df):
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

    df = (
        df.loc[lambda x: x.population >= 5]
        .assign(
            call_rate=lambda x: x.violence_calls / x.population,
            married_household_prop=lambda x: x.married_families / x.households,
            non_family_household_prop=lambda x: x.non_family_households / x.households,
            area=lambda df: df.population_density / df.population,
            rental_density=lambda x: x.rental_counts / x.area,
            bus_density=lambda x: x.bus_stops / x.area,
            greenlight_density=lambda x: x.greenlights / x.area,
            people_per_household=lambda x: (x.population / x.households).clip(upper=5),
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

    cols_with_nulls = df.isna().sum().loc[lambda x: x > 0].index

    df.loc[:, cols_with_nulls] = imputer.fit_transform(
        df.loc[
            :,
            cols_with_nulls,
        ]
    )

    return df

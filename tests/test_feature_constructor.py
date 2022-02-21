import pytest
from features.feature_constructor import Feature

from tests.conftest import BLOCKS_PER_YEAR_GEO


class TestFeastureConstructor:
    def test_remove_geos_outside_detroit(self):
        pass

    def test_standardize_geo_id(self):
        pass

    def test_validate_cleansed_data(self):
        pass

    def test_assign_geo_column(self):
        pass

    @pytest.mark.parametrize("decennial_census_year", [2010, 2020])
    @pytest.mark.parametrize("target_geo_grain", ["block", "block group", "tract"])
    def test_generate_index(self, decennial_census_year, target_geo_grain, partial_geo_data):
        ftr = Feature(meta={"min_geo_grain": target_geo_grain}, decennial_census_year=decennial_census_year)
        ftr.generate_index(target_geo_grain)
        assert (
            len(ftr.index) == BLOCKS_PER_YEAR_GEO[decennial_census_year][target_geo_grain]
        ), f"Index does not correspond withall geos for {target_geo_grain} in {decennial_census_year}"
        faux_geo_data = partial_geo_data[target_geo_grain][decennial_census_year].set_index("geo_id")
        n_geos_with_values = faux_geo_data.shape[0]
        geos_found_in_index = faux_geo_data.reindex(ftr.index).population.notna().sum()
        assert (
            geos_found_in_index == n_geos_with_values
        ), f"not all geos in sample frame contained in the index for {target_geo_grain} in {decennial_census_year}"

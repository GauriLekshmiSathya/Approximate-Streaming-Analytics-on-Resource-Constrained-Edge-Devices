import math

import pytest

from streamapprox.control import AdaptiveController, LinearCostFunction
from streamapprox.estimators import compare_with_exact
from streamapprox.models import EstimationResult


def test_weighted_count_is_population_estimate_without_predicate():
    from streamapprox.oasrs import OASRS
    from streamapprox.aggregators import approximate_count

    sampler = OASRS(sample_size=2, stratum_key="source", seed=7)
    sampler.update_batch([{"source": "a"} for _ in range(9)] + [{"source": "b"}])
    assert approximate_count(sampler.snapshot()) == pytest.approx(10.0)


def test_cost_function_maps_budget_and_divides_across_strata():
    cost = LinearCostFunction(items_per_budget_unit=100, minimum=2, maximum=50)
    assert cost.sample_size(1, strata=2) == 50
    assert cost.sample_size(100, strata=2) == 50


def test_controller_increases_for_excess_error_and_caps():
    controller = AdaptiveController(target_relative_error=0.05, maximum_sample_size=40)
    assert controller.next_sample_size(20, 10, 2) == 40


def test_empirical_error_zero_exact_is_not_nan():
    result = compare_with_exact(0, 0)
    assert result.accuracy_loss == 0
    nonzero = compare_with_exact(2, 0)
    assert math.isinf(nonzero.accuracy_loss)


def test_estimation_result_exposes_three_sigma_ranges():
    result = EstimationResult(estimate=10, variance=4, standard_error=2)
    assert result.sigma_margins == {1: 2, 2: 4, 3: 6}
    assert result.sigma_ranges[3] == (4, 16)

"""Unit tests for core/domain/metrics.py — pure math, no mocks needed."""

import pytest
from core.domain.metrics import compute_throughput, compute_cost_efficiency


class TestComputeThroughput:
    def test_basic(self):
        assert compute_throughput(100, 2000) == 50.0

    def test_zero_latency(self):
        assert compute_throughput(100, 0) is None

    def test_none_latency(self):
        assert compute_throughput(100, None) is None

    def test_zero_tokens(self):
        assert compute_throughput(0, 2000) is None

    def test_none_tokens(self):
        assert compute_throughput(None, 2000) is None

    def test_negative_latency(self):
        assert compute_throughput(100, -100) is None


class TestComputeCostEfficiency:
    def test_basic(self):
        assert compute_cost_efficiency(10, 0.5) == 20.0

    def test_none_quality(self):
        assert compute_cost_efficiency(None, 0.5) is None

    def test_none_cost(self):
        assert compute_cost_efficiency(10, None) is None

    def test_zero_cost(self):
        assert compute_cost_efficiency(10, 0) is None

    def test_negative_cost(self):
        assert compute_cost_efficiency(10, -0.5) is None

"""
Tests for performance.py.

These tests check CORRECTNESS (partition counts, equivalent outputs, valid
statistics). They never assert that anything is faster than something else
or finishes within a time limit, because timings depend on the computer.
"""

import pytest

from src import performance as perf
from src.data_generator import generate_synthetic_records
from src.rdd_engine import PASS_MARK


def test_canonicalize_rounds_floats_and_normalises_containers():
    value = {"a": 1.23456789, "b": [(1.00000049, True), 2], "c": "text", "d": False}
    assert perf.canonicalize(value) == {"a": 1.234568, "b": [[1.0, True], 2], "c": "text", "d": False}


def test_prepare_input_gives_exact_partition_count(spark_context):
    records = generate_synthetic_records(50, seed=1)
    rdd, seconds = perf.prepare_input(spark_context, records, 4)
    try:
        assert rdd.getNumPartitions() == 4
        assert rdd.count() == 50
        assert seconds > 0
    finally:
        rdd.unpersist()


def test_workload_is_correct_against_plain_python(spark_context):
    records = generate_synthetic_records(200, seed=3)
    rdd = spark_context.parallelize(records, 3)
    output = perf.run_workload(rdd)

    for row in output["departments"]:
        members = [r for r in records if r.department == row["department"]]
        expected_average = sum((r.maths + r.science + r.computer) / 3 for r in members) / len(members)
        assert row["students"] == len(members)
        assert row["average_marks"] == pytest.approx(expected_average, abs=1e-5)
        assert row["pass_count"] == sum(
            1 for r in members if min(r.maths, r.science, r.computer) >= PASS_MARK
        )
    assert sum(row["students"] for row in output["departments"]) == 200

    subjects = {row["subject"]: row for row in output["subjects"]}
    assert subjects["maths"]["average"] == pytest.approx(sum(r.maths for r in records) / 200, abs=1e-5)

    best = sorted(records, key=lambda r: (-(r.maths + r.science + r.computer) / 3, r.student_id))[:5]
    assert [row["student_id"] for row in output["top_students"]] == [r.student_id for r in best]


def test_benchmark_uses_requested_partitions_and_gives_equivalent_outputs(spark_context):
    report = perf.run_benchmark(
        dataset_sizes=[300], partition_counts=[1, 2, 4], trials=2, seed=42, warmup_runs=1
    )
    assert [c.requested_partitions for c in report.configurations] == [1, 2, 4]
    for config in report.configurations:
        assert config.actual_partitions == config.requested_partitions
        assert len(config.trial_seconds) == 2
        assert all(t > 0 for t in config.trial_seconds)
        assert config.min_seconds <= config.median_seconds <= config.max_seconds
        assert config.input_preparation_seconds > 0
        assert config.matches_reference is True
    assert report.equivalent_by_size == {300: True}
    assert report.all_equivalent is True
    assert report.data_generation_seconds[300] >= 0
    assert report.spark_init_seconds >= 0 and report.trials == 2 and report.warmup_runs == 1


def test_benchmark_rows_and_progress_callback(spark_context):
    calls = []
    report = perf.run_benchmark(
        dataset_sizes=[100, 200], partition_counts=[1, 3], trials=1, warmup_runs=0,
        progress=lambda fraction, message: calls.append((fraction, message)),
    )
    rows = report.to_rows()
    assert len(rows) == 4
    assert {(r["dataset_size"], r["partitions"]) for r in rows} == {(100, 1), (100, 3), (200, 1), (200, 3)}
    baseline = [r for r in rows if r["partitions"] == 1]
    assert all(r["speedup_vs_fewest_partitions"] == pytest.approx(1.0) for r in baseline)
    assert [round(f, 2) for f, _ in calls] == [0.25, 0.5, 0.75, 1.0]
    assert report.equivalent_by_size == {100: True, 200: True}


def test_benchmark_is_reproducible_in_content(spark_context):
    first = perf.run_workload(spark_context.parallelize(generate_synthetic_records(150, 9), 2))
    second = perf.run_workload(spark_context.parallelize(generate_synthetic_records(150, 9), 5))
    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dataset_sizes=[], partition_counts=[1]),
        dict(dataset_sizes=[100], partition_counts=[]),
        dict(dataset_sizes=[0], partition_counts=[1]),
        dict(dataset_sizes=[100], partition_counts=[0]),
        dict(dataset_sizes=[100], partition_counts=[1, 1]),
        dict(dataset_sizes=[100], partition_counts=[1000]),
        dict(dataset_sizes=[100], partition_counts=[1], trials=0),
        dict(dataset_sizes=[100], partition_counts=[1], warmup_runs=-1),
        dict(dataset_sizes=[10**9], partition_counts=[1]),
    ],
)
def test_invalid_benchmark_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        perf.run_benchmark(**kwargs)


def test_documentation_constants_are_present():
    assert len(perf.METHODOLOGY) >= 5 and len(perf.LIMITATIONS) >= 4

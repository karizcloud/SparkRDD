"""
performance.py
==============
A reproducible experiment: does changing the number of RDD partitions change
how long the SAME Spark job takes on the SAME data?

Honesty rules built into this file
----------------------------------
* No times are made up or stored. Every number comes from time.perf_counter().
* Every partition count is checked with rdd.getNumPartitions().
* Every configuration runs the identical workload on identical data
  (deterministic synthetic records, fixed seed).
* The outputs of all configurations are compared; the report says whether they
  were equivalent.
* Several trials are run and the MEDIAN is reported (single timings are noisy).
* Data generation, Spark start-up, input preparation and the measured
  operation are timed separately.
"""

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from pyspark import RDD, SparkContext

from src.data_generator import generate_synthetic_records
from src.rdd_engine import (
    MAX_PARTITIONS,
    department_statistics,
    get_spark_session,
    subject_statistics,
    to_result,
    top_students,
)

DEFAULT_DATASET_SIZES = (1_000, 10_000, 50_000)
AVAILABLE_DATASET_SIZES = (1_000, 5_000, 10_000, 50_000, 100_000)
DEFAULT_PARTITION_COUNTS = (1, 2, 4, 8)
MAX_DATASET_SIZE = 500_000

METHODOLOGY = (
    "Synthetic students are generated with a fixed random seed, so every run and every "
    "partition configuration sees exactly the same records.",
    "For each configuration the records are sent to Spark with sc.parallelize(records, n), "
    "cached, and materialised with count(). This 'input preparation' time is reported "
    "separately and is NOT part of the measured operation.",
    "The real partition count is read back with getNumPartitions() and the experiment "
    "stops with an error if it differs from the requested one.",
    "The measured operation is one workload: map to student results, department "
    "statistics (reduceByKey), subject statistics (flatMap + reduceByKey) and the top 5 "
    "students (sortBy + take). Because Spark is lazy, the timer stops only after the "
    "final actions have returned.",
    "Each configuration gets warm-up runs (ignored) and then several timed trials. "
    "The median of the trials is the headline number.",
    "Every output is compared with the first configuration's output (floats rounded to "
    "6 decimals). The report states whether all configurations were equivalent.",
)

LIMITATIONS = (
    "Spark runs in local mode on ONE computer. 'Executors' are threads of one JVM sharing "
    "the same CPU cores and memory, so there is no network cost and no real cluster scaling.",
    "More partitions than CPU cores cannot run at the same time. Extra partitions mostly add "
    "scheduling and serialization overhead.",
    "Small datasets finish in fractions of a second, where job start-up, task scheduling and "
    "Python worker communication dominate the real computation, so results can be noisy.",
    "Timings depend on other programs running on your computer, CPU boosting, and "
    "background activity. Repeat the experiment before drawing conclusions.",
    "RDD operations in PySpark run Python code in separate worker processes, which adds "
    "serialization cost that Scala/Java RDDs or DataFrames would not have.",
)


class PerformanceExperimentError(Exception):
    """The experiment could not be run correctly."""


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------
@dataclass
class ConfigurationResult:
    dataset_size: int
    requested_partitions: int
    actual_partitions: int
    input_preparation_seconds: float
    trial_seconds: List[float]
    median_seconds: float
    min_seconds: float
    max_seconds: float
    matches_reference: bool


@dataclass
class BenchmarkReport:
    seed: int
    trials: int
    warmup_runs: int
    cpu_count: Optional[int]
    spark_master: str
    default_parallelism: int
    spark_init_seconds: float
    spark_session_reused: bool
    data_generation_seconds: Dict[int, float] = field(default_factory=dict)
    configurations: List[ConfigurationResult] = field(default_factory=list)
    equivalent_by_size: Dict[int, bool] = field(default_factory=dict)

    @property
    def all_equivalent(self) -> bool:
        return all(self.equivalent_by_size.values()) if self.equivalent_by_size else False

    def to_rows(self) -> List[Dict[str, Any]]:
        """Flat table rows for display. speedup is relative to the smallest
        partition count of the same dataset size (values below 1 mean SLOWER)."""
        rows = []
        for config in self.configurations:
            same_size = [c for c in self.configurations if c.dataset_size == config.dataset_size]
            baseline = min(same_size, key=lambda c: c.requested_partitions)
            speedup = (
                baseline.median_seconds / config.median_seconds
                if config.median_seconds > 0
                else float("nan")
            )
            rows.append(
                {
                    "dataset_size": config.dataset_size,
                    "partitions": config.requested_partitions,
                    "actual_partitions": config.actual_partitions,
                    "input_prep_s": config.input_preparation_seconds,
                    "median_s": config.median_seconds,
                    "min_s": config.min_seconds,
                    "max_s": config.max_seconds,
                    "trials": len(config.trial_seconds),
                    "speedup_vs_fewest_partitions": speedup,
                    "outputs_equivalent": config.matches_reference,
                }
            )
        return rows


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def canonicalize(value: Any, digits: int = 6) -> Any:
    """Round floats and convert containers so two results can be compared with ==."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: canonicalize(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item, digits) for item in value]
    return value


def run_workload(records: RDD) -> Dict[str, Any]:
    """The measured computation (identical for every configuration)."""
    results = records.map(to_result)
    return canonicalize(
        {
            "departments": department_statistics(results),
            "subjects": subject_statistics(results),
            "top_students": top_students(results, 5),
        }
    )


def prepare_input(sc: SparkContext, records: Sequence, num_partitions: int):
    """Create a cached RDD with exactly `num_partitions` partitions.

    Returns (rdd, seconds_spent). Raises PerformanceExperimentError if Spark
    did not give the requested partition count.
    """
    started = time.perf_counter()
    rdd = sc.parallelize(list(records), num_partitions)
    rdd.cache()
    rdd.count()  # ACTION: materialises the cache so later timings exclude loading
    elapsed = time.perf_counter() - started

    actual = rdd.getNumPartitions()
    if actual != num_partitions:
        rdd.unpersist()
        raise PerformanceExperimentError(
            f"Requested {num_partitions} partitions but Spark created {actual}."
        )
    return rdd, elapsed


def _validate_arguments(dataset_sizes, partition_counts, trials, warmup_runs) -> None:
    if not dataset_sizes:
        raise ValueError("Choose at least one dataset size.")
    if not partition_counts:
        raise ValueError("Choose at least one partition count.")
    for size in dataset_sizes:
        if not isinstance(size, int) or size < 1 or size > MAX_DATASET_SIZE:
            raise ValueError(f"Dataset size must be a whole number from 1 to {MAX_DATASET_SIZE}.")
    for count in partition_counts:
        if not isinstance(count, int) or count < 1 or count > MAX_PARTITIONS:
            raise ValueError(f"Partition count must be a whole number from 1 to {MAX_PARTITIONS}.")
    if len(set(partition_counts)) != len(partition_counts):
        raise ValueError("Partition counts must not repeat.")
    if not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be 1 or more.")
    if not isinstance(warmup_runs, int) or warmup_runs < 0:
        raise ValueError("warmup_runs must be 0 or more.")


def run_benchmark(
    dataset_sizes: Sequence[int] = DEFAULT_DATASET_SIZES,
    partition_counts: Sequence[int] = DEFAULT_PARTITION_COUNTS,
    trials: int = 3,
    seed: int = 42,
    warmup_runs: int = 1,
    progress: Optional[Callable[[float, str], None]] = None,
) -> BenchmarkReport:
    """Run the experiment and return a BenchmarkReport.

    `progress(fraction, message)` is called after each configuration if given.
    """
    import os

    _validate_arguments(dataset_sizes, partition_counts, trials, warmup_runs)

    session_existed = SparkContext._active_spark_context is not None
    started = time.perf_counter()
    spark = get_spark_session()
    spark_init_seconds = time.perf_counter() - started
    sc = spark.sparkContext

    report = BenchmarkReport(
        seed=seed,
        trials=trials,
        warmup_runs=warmup_runs,
        cpu_count=os.cpu_count(),
        spark_master=sc.master,
        default_parallelism=sc.defaultParallelism,
        spark_init_seconds=spark_init_seconds,
        spark_session_reused=session_existed,
    )

    total_steps = len(dataset_sizes) * len(partition_counts)
    finished_steps = 0

    for size in dataset_sizes:
        generation_started = time.perf_counter()
        records = generate_synthetic_records(size, seed)
        report.data_generation_seconds[size] = time.perf_counter() - generation_started

        reference_output = None
        size_is_equivalent = True

        for partitions in partition_counts:
            rdd, prep_seconds = prepare_input(sc, records, partitions)
            try:
                for _ in range(warmup_runs):
                    warm_output = run_workload(rdd)
                    if reference_output is None:
                        reference_output = warm_output

                trial_seconds: List[float] = []
                matches = True
                for _ in range(trials):
                    trial_started = time.perf_counter()
                    output = run_workload(rdd)  # ends with actions -> includes all work
                    trial_seconds.append(time.perf_counter() - trial_started)
                    if reference_output is None:
                        reference_output = output
                    if output != reference_output:
                        matches = False

                report.configurations.append(
                    ConfigurationResult(
                        dataset_size=size,
                        requested_partitions=partitions,
                        actual_partitions=rdd.getNumPartitions(),
                        input_preparation_seconds=prep_seconds,
                        trial_seconds=trial_seconds,
                        median_seconds=statistics.median(trial_seconds),
                        min_seconds=min(trial_seconds),
                        max_seconds=max(trial_seconds),
                        matches_reference=matches,
                    )
                )
                size_is_equivalent = size_is_equivalent and matches
            finally:
                rdd.unpersist()  # release cached data before the next configuration

            finished_steps += 1
            if progress is not None:
                progress(
                    finished_steps / total_steps,
                    f"Finished {size:,} records with {partitions} partition(s)",
                )

        report.equivalent_by_size[size] = size_is_equivalent

    return report

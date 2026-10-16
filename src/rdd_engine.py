"""
rdd_engine.py
=============
The Apache Spark RDD processing engine.

* No Streamlit code here: the dashboard (app.py) only *calls* these functions.
* Every analysis below is computed with real PySpark RDD operations
  (map, filter, flatMap, reduce, reduceByKey, distinct, sortBy, count, take,
  collect, groupByKey, mapPartitions).
* Results sent back to the driver are always small and bounded
  (take(n), or aggregated rows such as one row per department).

Reading guide for students
--------------------------
    transformation  -> builds a NEW RDD, runs LAZILY (nothing happens yet)
    action          -> returns a value to Python and TRIGGERS the real work
"""

import inspect
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from operator import add
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence

from pyspark import RDD, SparkContext
from pyspark.sql import SparkSession

from src.data_generator import SAMPLE_STUDENTS
from src.validators import (
    ATTENDANCE_THRESHOLD,
    PASS_MARK,
    SUBJECTS,
    EmptyDatasetError,
    StudentRecord,
    parse_header,
    try_parse_student_row,
)

# --------------------------------------------------------------------------
# Limits that keep every result sent to the driver small
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_TABLE_ROWS = 1000      # most student rows ever returned to the dashboard
MAX_GROUP_ROWS = 500       # most groups (e.g. departments) ever returned
MAX_ERRORS_SHOWN = 50      # most validation messages kept for display
MAX_PARTITIONS = 64        # sanity limit for user-chosen partition counts

PERFORMANCE_BANDS = ("Below 40", "40-49", "50-59", "60-69", "70-79", "80-89", "90-100")
ATTENDANCE_BANDS = ("Below 60", "60 to below 75", "75 to below 85", "85 and above")


# --------------------------------------------------------------------------
# Result row: a validated record plus calculated fields
# --------------------------------------------------------------------------
class StudentResult(NamedTuple):
    student_id: str
    name: str
    department: str
    maths: float
    science: float
    computer: float
    attendance: float
    total: float
    average: float
    passed: bool
    low_attendance: bool


# --------------------------------------------------------------------------
# Small pure business-rule functions (single source of truth)
# --------------------------------------------------------------------------
def compute_total(record) -> float:
    """Sum of the three subject marks."""
    return record.maths + record.science + record.computer


def compute_average(record) -> float:
    """Average of the three subject marks."""
    return compute_total(record) / len(SUBJECTS)


def is_passing(record) -> bool:
    """A student passes only if ALL three marks are at least PASS_MARK."""
    return all(getattr(record, subject) >= PASS_MARK for subject in SUBJECTS)


def has_low_attendance(record) -> bool:
    """True when attendance is strictly below ATTENDANCE_THRESHOLD."""
    return record.attendance < ATTENDANCE_THRESHOLD


def to_result(record: StudentRecord) -> StudentResult:
    """Turn a validated StudentRecord into a StudentResult."""
    return StudentResult(
        student_id=record.student_id,
        name=record.name,
        department=record.department,
        maths=record.maths,
        science=record.science,
        computer=record.computer,
        attendance=record.attendance,
        total=compute_total(record),
        average=compute_average(record),
        passed=is_passing(record),
        low_attendance=has_low_attendance(record),
    )


def performance_band(average: float) -> str:
    """Label for the average-marks histogram (e.g. 73.0 -> '70-79')."""
    if average < 40:
        return PERFORMANCE_BANDS[0]
    if average >= 90:
        return PERFORMANCE_BANDS[-1]
    lower = int(average // 10) * 10
    return f"{lower}-{lower + 9}"


def attendance_band(attendance: float) -> str:
    """Label for the attendance histogram."""
    if attendance < 60:
        return ATTENDANCE_BANDS[0]
    if attendance < ATTENDANCE_THRESHOLD:
        return ATTENDANCE_BANDS[1]
    if attendance < 85:
        return ATTENDANCE_BANDS[2]
    return ATTENDANCE_BANDS[3]


def rows_as_dicts(rows: Sequence[NamedTuple]) -> List[Dict[str, Any]]:
    """Convert a small list of named tuples to dictionaries."""
    return [row._asdict() for row in rows]


# --------------------------------------------------------------------------
# Spark session management
# --------------------------------------------------------------------------
def get_spark_session(
    app_name: str = "SparkRDD-Lab", master: str = "local[*]"
) -> SparkSession:
    """Create (or return the existing) local Spark session.

    Windows notes
    -------------
    * Spark's Python workers must use the SAME Python as this program,
      otherwise you get "Python worker failed to connect back".
    * Workers must also be able to import the `src` package, so the project
      root is added to PYTHONPATH *before* Spark starts its JVM.
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    root = str(PROJECT_ROOT)
    existing = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if root not in existing:
        os.environ["PYTHONPATH"] = os.pathsep.join([root] + existing)

    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def is_spark_context_active(sc: Optional[SparkContext]) -> bool:
    """Return whether a SparkContext still has a live JVM context."""
    return sc is not None and sc._jsc is not None


def stop_spark_session() -> None:
    """Stop the active Spark session (safe to call when none is running)."""
    active = SparkSession.getActiveSession()
    if active is not None:
        active.stop()


def release_rdds(*rdds: Optional[RDD]) -> None:
    """Remove cached RDDs from memory. None values are ignored."""
    for rdd in rdds:
        if rdd is not None:
            try:
                rdd.unpersist()
            except Exception:  # the SparkContext may already be stopped
                pass


# --------------------------------------------------------------------------
# Loading and validating data
# --------------------------------------------------------------------------
@dataclass
class LoadResult:
    """Everything the dashboard needs to know about a loaded file."""

    records: RDD                      # cached RDD of valid StudentRecord objects
    source: str
    load_method: str                  # how the lines reached Spark
    total_data_lines: int             # non-blank data lines examined
    valid_count: int                  # records kept
    rejected_count: int               # lines rejected (invalid or duplicate)
    errors: List[str] = field(default_factory=list)
    errors_truncated: bool = False
    duplicate_ids: List[str] = field(default_factory=list)
    num_partitions: int = 0

    @property
    def is_usable(self) -> bool:
        return self.valid_count > 0


def _check_partition_request(num_partitions: Optional[int]) -> None:
    if num_partitions is None:
        return
    if not isinstance(num_partitions, int) or isinstance(num_partitions, bool):
        raise ValueError("Partition count must be a whole number.")
    if num_partitions < 1 or num_partitions > MAX_PARTITIONS:
        raise ValueError(f"Partition count must be between 1 and {MAX_PARTITIONS}.")


def _summarize_parsed_partition(items):
    """Return counts and a bounded error sample for parsed CSV rows."""
    ok_count = 0
    error_count = 0
    errors = []
    for item in items:
        if item[0] == "ok":
            ok_count += 1
        else:
            error_count += 1
            if len(errors) < MAX_ERRORS_SHOWN:
                errors.append(item[2])
    yield ok_count, error_count, errors


def _load_from_lines_rdd(
    lines: RDD,
    source: str,
    load_method: str,
    num_partitions: Optional[int],
    first_line: Optional[List[str]] = None,
) -> LoadResult:
    """Validate an RDD of CSV text lines (first line = header)."""
    _check_partition_request(num_partitions)

    if first_line is None:
        first_line = lines.take(1)  # ACTION: tiny job to read the header
    if not first_line:
        raise EmptyDatasetError("The file is empty: no header line was found.")
    layout = parse_header(first_line[0])  # raises MissingColumnsError etc.

    def parse_indexed_line(pair):
        line, index = pair
        line_number = index + 1  # the header was line 1
        record, problem = try_parse_student_row(line, layout)
        if problem is None:
            return ("ok", line_number, record)
        return ("error", line_number, f"Line {line_number}: {problem}")

    # zipWithIndex gives each line its position so errors can quote line numbers.
    parsed = (
        lines.zipWithIndex()
        .filter(lambda pair: pair[1] > 0 and pair[0].strip() != "")
        .map(parse_indexed_line)
    )
    parsed.cache()

    try:
        partition_summaries = parsed.mapPartitions(_summarize_parsed_partition).collect()
        ok_count = sum(summary[0] for summary in partition_summaries)
        error_count = sum(summary[1] for summary in partition_summaries)
        errors = [
            message
            for summary in partition_summaries
            for message in summary[2]
        ][:MAX_ERRORS_SHOWN]
        errors_truncated = error_count > len(errors)

        ok_pairs = parsed.filter(lambda item: item[0] == "ok").map(
            lambda item: (item[1], item[2])  # (line_number, StudentRecord)
        )

        # Duplicate student_id values would distort every statistic.
        duplicate_counts = (
            ok_pairs.map(lambda pair: (pair[1].student_id, 1))
            .reduceByKey(add)
            .filter(lambda kv: kv[1] > 1)
            .take(MAX_ERRORS_SHOWN)
        )
        duplicate_ids = sorted(student_id for student_id, _ in duplicate_counts)

        if duplicate_ids:
            # Keep only the first occurrence (lowest line number) of each id.
            records = (
                ok_pairs.map(lambda pair: (pair[1].student_id, pair))
                .reduceByKey(lambda a, b: a if a[0] <= b[0] else b)
                .map(lambda kv: kv[1][1])
            )
            for student_id, copies in sorted(duplicate_counts):
                errors.append(
                    f"Duplicate student_id '{student_id}' appears {copies} times; "
                    "only the first occurrence was kept."
                )
        else:
            records = ok_pairs.map(lambda pair: pair[1])

        if num_partitions is not None:
            records = records.repartition(num_partitions)

        records.cache()
        valid_count = records.count() if duplicate_ids else ok_count
    finally:
        parsed.unpersist()

    return LoadResult(
        records=records,
        source=source,
        load_method=load_method,
        total_data_lines=ok_count + error_count,
        valid_count=valid_count,
        rejected_count=ok_count + error_count - valid_count,
        errors=errors,
        errors_truncated=errors_truncated,
        duplicate_ids=duplicate_ids,
        num_partitions=records.getNumPartitions(),
    )


def load_students_from_lines(
    sc: SparkContext,
    lines: Sequence[str],
    num_partitions: Optional[int] = None,
    source: str = "in-memory lines",
) -> LoadResult:
    """Load CSV text lines (header first) given as a Python list.

    Mainly used by the tests and demos.
    """
    slices = max(1, min(len(lines), 2))
    lines_rdd = sc.parallelize(list(lines), slices)
    return _load_from_lines_rdd(lines_rdd, source, "sc.parallelize", num_partitions)


def load_students_from_csv(
    sc: SparkContext, path, num_partitions: Optional[int] = None
) -> LoadResult:
    """Load a CSV file into a validated, cached RDD of StudentRecord objects.

    Spark reads the file with sc.textFile(). If Spark cannot read the path
    (this can happen with unusual Windows paths) the lines are read with
    Python and given to Spark with sc.parallelize(); `load_method` in the
    result says which route was used.

    Raises
    ------
    FileNotFoundError, EmptyDatasetError, MissingColumnsError, DatasetError
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        lines = sc.textFile(csv_path.resolve().as_posix())
        first_line = lines.take(1)  # forces Spark to open the file now, not later
        method = "sc.textFile"
    except Exception:
        text_lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
        lines = sc.parallelize(text_lines, max(1, min(len(text_lines), 2)))
        first_line = None
        method = "sc.parallelize (textFile fallback)"

    return _load_from_lines_rdd(
        lines, csv_path.name, method, num_partitions, first_line=first_line
    )


def build_results_rdd(records: RDD) -> RDD:
    """TRANSFORMATION (map): StudentRecord -> StudentResult with totals etc."""
    return records.map(to_result)


# --------------------------------------------------------------------------
# Merge helpers used by reduceByKey (module-level so they are easy to read)
# --------------------------------------------------------------------------
def _merge_subject_stats(a, b):
    """(sum, count, min, max, passes) + (sum, count, min, max, passes)."""
    return (a[0] + b[0], a[1] + b[1], min(a[2], b[2]), max(a[3], b[3]), a[4] + b[4])


def _merge_department_stats(a, b):
    """(count, sum_avg, passed, sum_att, min_avg, max_avg, low_att) merged."""
    return (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        min(a[4], b[4]),
        max(a[5], b[5]),
        a[6] + b[6],
    )


def _department_row(key_value) -> Dict[str, Any]:
    department, (count, sum_avg, passed, sum_att, low_avg, high_avg, low_att) = key_value
    return {
        "department": department,
        "students": count,
        "average_marks": sum_avg / count,
        "pass_count": passed,
        "pass_percentage": 100.0 * passed / count,
        "average_attendance": sum_att / count,
        "lowest_average": low_avg,
        "highest_average": high_avg,
        "low_attendance_count": low_att,
    }


def _rank_key_descending(result: StudentResult):
    return (-result.average, result.student_id)


def _rank_key_ascending(result: StudentResult):
    return (result.average, result.student_id)


# --------------------------------------------------------------------------
# Analytics (all return small Python objects)
# --------------------------------------------------------------------------
def count_students(results: RDD) -> int:
    """ACTION: count()."""
    return results.count()


def overall_summary(results: RDD) -> Dict[str, Any]:
    """Headline numbers for the Overview page, using map + reduce."""
    empty = {
        "total_students": 0,
        "overall_average": 0.0,
        "pass_count": 0,
        "fail_count": 0,
        "pass_percentage": 0.0,
        "average_attendance": 0.0,
        "low_attendance_count": 0,
    }
    try:
        total, avg_sum, passed, att_sum, low_att = results.map(
            lambda r: (
                1,
                r.average,
                1 if r.passed else 0,
                r.attendance,
                1 if r.low_attendance else 0,
            )
        ).reduce(  # ACTION: reduce() combines pairs of tuples until one is left
            lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4])
        )
    except ValueError:  # PySpark raises ValueError when reducing an empty RDD
        return empty

    return {
        "total_students": total,
        "overall_average": avg_sum / total,
        "pass_count": passed,
        "fail_count": total - passed,
        "pass_percentage": 100.0 * passed / total,
        "average_attendance": att_sum / total,
        "low_attendance_count": low_att,
    }


def subject_statistics(results: RDD) -> List[Dict[str, Any]]:
    """Average, min, max and pass rate per subject.

    flatMap turns each student into three (subject, mark) pairs, then
    reduceByKey combines them per subject.
    """
    per_subject = (
        results.flatMap(lambda r: [(subject, getattr(r, subject)) for subject in SUBJECTS])
        .map(lambda pair: (pair[0], (pair[1], 1, pair[1], pair[1], 1 if pair[1] >= PASS_MARK else 0)))
        .reduceByKey(_merge_subject_stats)
    )
    rows = []
    for subject, (total, count, low, high, passes) in per_subject.collect():  # 3 rows
        rows.append(
            {
                "subject": subject,
                "average": total / count,
                "lowest": low,
                "highest": high,
                "pass_percentage": 100.0 * passes / count,
                "students": count,
            }
        )
    order = {subject: position for position, subject in enumerate(SUBJECTS)}
    return sorted(rows, key=lambda row: order[row["subject"]])


def department_statistics(results: RDD) -> List[Dict[str, Any]]:
    """Per-department statistics using reduceByKey, best average first."""
    return (
        results.map(
            lambda r: (
                r.department,
                (
                    1,
                    r.average,
                    1 if r.passed else 0,
                    r.attendance,
                    r.average,
                    r.average,
                    1 if r.low_attendance else 0,
                ),
            )
        )
        .reduceByKey(_merge_department_stats)
        .map(_department_row)
        .sortBy(lambda row: (-row["average_marks"], row["department"]))
        .take(MAX_GROUP_ROWS)
    )


def top_students(results: RDD, n: int = 5) -> List[Dict[str, Any]]:
    """The n students with the highest average (ties broken by student_id)."""
    if n < 1:
        return []
    return rows_as_dicts(results.sortBy(_rank_key_descending).take(min(n, MAX_TABLE_ROWS)))


def sort_students_by_average(
    results: RDD, descending: bool = True, limit: int = 10
) -> List[Dict[str, Any]]:
    """Students ordered by average marks (bounded by `limit`)."""
    key = _rank_key_descending if descending else _rank_key_ascending
    return rows_as_dicts(results.sortBy(key).take(max(0, min(limit, MAX_TABLE_ROWS))))


def students_needing_support(results: RDD, limit: int = 20) -> Dict[str, Any]:
    """Students who fail at least one subject OR have low attendance."""
    needs_support = results.filter(lambda r: (not r.passed) or r.low_attendance)
    count = needs_support.count()
    rows = needs_support.sortBy(_rank_key_ascending).take(max(0, min(limit, MAX_TABLE_ROWS)))
    return {"count": count, "students": rows_as_dicts(rows)}


def low_attendance_students(results: RDD, limit: int = 20) -> Dict[str, Any]:
    """Students whose attendance is below ATTENDANCE_THRESHOLD, lowest first."""
    low = results.filter(lambda r: r.low_attendance)
    count = low.count()
    rows = low.sortBy(lambda r: (r.attendance, r.student_id)).take(
        max(0, min(limit, MAX_TABLE_ROWS))
    )
    return {"count": count, "students": rows_as_dicts(rows)}


def pass_fail_distribution(results: RDD) -> Dict[str, int]:
    """{'Pass': n, 'Fail': m} computed with reduceByKey."""
    counts = dict(
        results.map(lambda r: ("Pass" if r.passed else "Fail", 1)).reduceByKey(add).collect()
    )
    return {"Pass": counts.get("Pass", 0), "Fail": counts.get("Fail", 0)}


def performance_distribution(results: RDD) -> List[Dict[str, Any]]:
    """How many students fall in each average-marks band (all bands listed)."""
    counts = dict(
        results.map(lambda r: (performance_band(r.average), 1)).reduceByKey(add).collect()
    )
    return [{"band": band, "students": counts.get(band, 0)} for band in PERFORMANCE_BANDS]


def attendance_distribution(results: RDD) -> List[Dict[str, Any]]:
    """How many students fall in each attendance band (all bands listed)."""
    counts = dict(
        results.map(lambda r: (attendance_band(r.attendance), 1)).reduceByKey(add).collect()
    )
    return [{"band": band, "students": counts.get(band, 0)} for band in ATTENDANCE_BANDS]


def distinct_departments(results: RDD) -> List[str]:
    """Sorted list of unique department names (distinct + sortBy)."""
    return (
        results.map(lambda r: r.department)
        .distinct()
        .sortBy(lambda name: name)
        .take(MAX_GROUP_ROWS)
    )


def group_by_department(results: RDD, max_names: int = 10) -> List[Dict[str, Any]]:
    """Group students by department with groupByKey.

    groupByKey ships EVERY value across the network, so for numeric totals
    reduceByKey (used above) is preferred. Here we really need the names,
    and only a small preview of each group is returned.
    """
    return (
        results.map(lambda r: (r.department, r.name))
        .groupByKey()
        .mapValues(sorted)
        .map(lambda kv: {"department": kv[0], "students": len(kv[1]), "names": kv[1][:max_names]})
        .sortBy(lambda row: row["department"])
        .take(MAX_GROUP_ROWS)
    )


def query_students(
    results: RDD,
    search: str = "",
    department: Optional[str] = None,
    status: str = "All",
    descending: bool = True,
    limit: int = MAX_TABLE_ROWS,
) -> Dict[str, Any]:
    """Search / filter / sort for the Student Explorer.

    status is 'All', 'Pass' or 'Fail'. Only up to `limit` rows are returned.
    """
    text = (search or "").strip().lower()
    filtered = results
    if text:
        filtered = filtered.filter(
            lambda r: text in r.student_id.lower() or text in r.name.lower()
        )
    if department and department != "All":
        filtered = filtered.filter(lambda r: r.department == department)
    if status == "Pass":
        filtered = filtered.filter(lambda r: r.passed)
    elif status == "Fail":
        filtered = filtered.filter(lambda r: not r.passed)

    limit = max(0, min(limit, MAX_TABLE_ROWS))
    total_matches = filtered.count()
    key = _rank_key_descending if descending else _rank_key_ascending
    rows = filtered.sortBy(key).take(limit)
    return {
        "total_matches": total_matches,
        "rows": rows_as_dicts(rows),
        "truncated": total_matches > len(rows),
    }


def _dashboard_partition_summary(partition_index: int, rows, top_limit: int = 5):
    """Compute compact dashboard summaries for one cached RDD partition."""
    department_stats = {}
    subject_stats = {}
    pass_fail = {"Pass": 0, "Fail": 0}
    performance_bands = {}
    attendance_bands = {}
    top_candidates = []
    support_candidates = []
    low_attendance_candidates = []
    scatter_sample = []
    department_names = set()
    total = 0
    average_sum = 0.0
    passed = 0
    attendance_sum = 0.0
    low_attendance_count = 0
    support_count = 0
    for result in rows:
        total += 1
        average_sum += result.average
        passed += int(result.passed)
        attendance_sum += result.attendance
        low_attendance_count += int(result.low_attendance)
        pass_fail["Pass" if result.passed else "Fail"] += 1

        band = performance_band(result.average)
        performance_bands[band] = performance_bands.get(band, 0) + 1
        band = attendance_band(result.attendance)
        attendance_bands[band] = attendance_bands.get(band, 0) + 1

        department_names.add(result.department)
        department = department_stats.setdefault(
            result.department,
            [0, 0.0, 0, 0.0, result.average, result.average, 0],
        )
        department[0] += 1
        department[1] += result.average
        department[2] += int(result.passed)
        department[3] += result.attendance
        department[4] = min(department[4], result.average)
        department[5] = max(department[5], result.average)
        department[6] += int(result.low_attendance)

        for subject in SUBJECTS:
            mark = getattr(result, subject)
            stats = subject_stats.setdefault(subject, [0.0, 0, mark, mark, 0])
            stats[0] += mark
            stats[1] += 1
            stats[2] = min(stats[2], mark)
            stats[3] = max(stats[3], mark)
            stats[4] += int(mark >= PASS_MARK)

        if top_limit:
            top_candidates.append(result)
            top_candidates.sort(key=_rank_key_descending)
            del top_candidates[top_limit:]

        if (not result.passed) or result.low_attendance:
            support_count += 1
            support_candidates.append(result)
            support_candidates.sort(key=_rank_key_ascending)
            del support_candidates[20:]
        if result.low_attendance:
            low_attendance_candidates.append(result)
            low_attendance_candidates.sort(key=lambda r: (r.attendance, r.student_id))
            del low_attendance_candidates[20:]
        if len(scatter_sample) < MAX_TABLE_ROWS:
            scatter_sample.append(result)

    yield {
        "partition": partition_index,
        "summary": (total, average_sum, passed, attendance_sum, low_attendance_count),
        "departments": department_stats,
        "subjects": subject_stats,
        "pass_fail": pass_fail,
        "performance_bands": performance_bands,
        "attendance_bands": attendance_bands,
        "top_students": top_candidates,
        "support_count": support_count,
        "support_students": support_candidates,
        "low_attendance_students": low_attendance_candidates,
        "department_names": sorted(department_names)[:MAX_GROUP_ROWS],
        "scatter_sample": scatter_sample,
    }


def compute_dashboard_analytics(results: RDD, top_n: int = 5) -> Dict[str, Any]:
    """Compute every dashboard metric in one Spark action over cached results."""
    top_limit = max(0, min(top_n, MAX_TABLE_ROWS))
    partitions = results.mapPartitionsWithIndex(
        lambda index, rows: _dashboard_partition_summary(index, rows, top_limit)
    ).collect()

    total = sum(part["summary"][0] for part in partitions)
    average_sum = sum(part["summary"][1] for part in partitions)
    pass_count = sum(part["summary"][2] for part in partitions)
    attendance_sum = sum(part["summary"][3] for part in partitions)
    low_attendance_count = sum(part["summary"][4] for part in partitions)
    summary = {
        "total_students": total,
        "overall_average": average_sum / total if total else 0.0,
        "pass_count": pass_count,
        "fail_count": total - pass_count,
        "pass_percentage": 100.0 * pass_count / total if total else 0.0,
        "average_attendance": attendance_sum / total if total else 0.0,
        "low_attendance_count": low_attendance_count,
    }

    merged_departments = {}
    merged_subjects = {}
    pass_fail = {"Pass": 0, "Fail": 0}
    performance_counts = {}
    attendance_counts = {}
    for part in partitions:
        for key, value in part["departments"].items():
            merged_departments[key] = (
                _merge_department_stats(merged_departments[key], value)
                if key in merged_departments
                else tuple(value)
            )
        for key, value in part["subjects"].items():
            merged_subjects[key] = (
                _merge_subject_stats(merged_subjects[key], value)
                if key in merged_subjects
                else tuple(value)
            )
        for key in pass_fail:
            pass_fail[key] += part["pass_fail"][key]
        for key, value in part["performance_bands"].items():
            performance_counts[key] = performance_counts.get(key, 0) + value
        for key, value in part["attendance_bands"].items():
            attendance_counts[key] = attendance_counts.get(key, 0) + value

    departments = sorted(
        (_department_row((key, value)) for key, value in merged_departments.items()),
        key=lambda row: (-row["average_marks"], row["department"]),
    )[:MAX_GROUP_ROWS]
    subject_order = {subject: index for index, subject in enumerate(SUBJECTS)}
    subjects = [
        {
            "subject": subject,
            "average": values[0] / values[1],
            "lowest": values[2],
            "highest": values[3],
            "pass_percentage": 100.0 * values[4] / values[1],
            "students": values[1],
        }
        for subject, values in sorted(
            merged_subjects.items(), key=lambda item: subject_order[item[0]]
        )
    ]
    top = sorted(
        (row for part in partitions for row in part["top_students"]),
        key=_rank_key_descending,
    )[:top_limit]
    support_count = sum(part["support_count"] for part in partitions)
    support_students = sorted(
        (row for part in partitions for row in part["support_students"]),
        key=_rank_key_ascending,
    )[:20]
    low_attendance_students_rows = sorted(
        (row for part in partitions for row in part["low_attendance_students"]),
        key=lambda row: (row.attendance, row.student_id),
    )[:20]
    department_names = sorted(
        {name for part in partitions for name in part["department_names"]}
    )[:MAX_GROUP_ROWS]
    scatter_sample = [
        row for part in partitions for row in part["scatter_sample"]
    ][:MAX_TABLE_ROWS]

    return {
        "summary": summary,
        "departments": departments,
        "subjects": subjects,
        "pass_fail": pass_fail,
        "performance_bands": [
            {"band": band, "students": performance_counts.get(band, 0)}
            for band in PERFORMANCE_BANDS
        ],
        "attendance_bands": [
            {"band": band, "students": attendance_counts.get(band, 0)}
            for band in ATTENDANCE_BANDS
        ],
        "top_students": rows_as_dicts(top),
        "needs_support": {
            "count": support_count,
            "students": rows_as_dicts(support_students),
        },
        "low_attendance": {
            "count": low_attendance_count,
            "students": rows_as_dicts(low_attendance_students_rows),
        },
        "department_names": department_names,
        "scatter_sample": rows_as_dicts(scatter_sample),
    }


# --------------------------------------------------------------------------
# Partitions
# --------------------------------------------------------------------------
def partition_count(rdd: RDD) -> int:
    """The real number of partitions Spark uses for this RDD."""
    return rdd.getNumPartitions()


def _count_partition(rows):
    """mapPartitions function: receives ALL rows of ONE partition."""
    count = 0
    for _ in rows:
        count += 1
    yield count


def partition_sizes(rdd: RDD) -> List[int]:
    """Number of records in each partition, found with mapPartitions()."""
    return rdd.mapPartitions(_count_partition).take(MAX_GROUP_ROWS)


def partition_details(rdd: RDD, preview: int = 3) -> List[Dict[str, Any]]:
    """For each partition: its index, size and the first few student ids."""

    def describe_partition(index, rows):
        count = 0
        sample_ids = []
        for row in rows:
            count += 1
            if len(sample_ids) < preview:
                sample_ids.append(row.student_id)
        yield {"partition": index, "records": count, "sample_ids": sample_ids}

    return rdd.mapPartitionsWithIndex(describe_partition).take(MAX_GROUP_ROWS)


def set_partition_count(rdd: RDD, num_partitions: int) -> RDD:
    """TRANSFORMATION: return an RDD with exactly `num_partitions` partitions.

    repartition() does a full shuffle, so records are spread across partitions.
    """
    _check_partition_request(num_partitions)
    if num_partitions is None:
        raise ValueError("num_partitions is required.")
    return rdd.repartition(num_partitions)


# --------------------------------------------------------------------------
# RDD Operations Lab: small, safe, predefined demonstrations
# --------------------------------------------------------------------------
DEMO_STUDENT_IDS = ("S001", "S005", "S009", "S010", "S013", "S014", "S018", "S021")


def demo_sample_records() -> List[StudentRecord]:
    """Eight fixed students from the sample dataset used by every demo."""
    wanted = set(DEMO_STUDENT_IDS)
    return [StudentRecord(*row) for row in SAMPLE_STUDENTS if row[0] in wanted]


def make_demo_students_rdd(sc: SparkContext) -> RDD:
    """RDD of StudentResult rows (2 partitions) used as demo input."""
    records = demo_sample_records()
    return sc.parallelize(records, 2).map(to_result)


# Each demo function below is BOTH executed and displayed (via inspect), so the
# code shown in the dashboard is exactly the code that produced the result.
def demo_map(students):
    name_and_average = students.map(lambda s: (s.name, round(s.average, 1)))
    return name_and_average.take(8)


def demo_filter(students):
    passing = students.filter(lambda s: s.passed)
    return passing.map(lambda s: s.name).collect()


def demo_flat_map(students):
    marks = students.flatMap(
        lambda s: [
            (s.student_id, "maths", s.maths),
            (s.student_id, "science", s.science),
            (s.student_id, "computer", s.computer),
        ]
    )
    return {
        "students_in": students.count(),
        "rows_out": marks.count(),
        "first_six_rows": marks.take(6),
    }


def demo_reduce(students):
    totals = students.map(lambda s: s.total)
    return {
        "sum_of_all_totals": totals.reduce(lambda a, b: a + b),
        "highest_total": totals.reduce(max),
    }


def demo_reduce_by_key(students):
    pairs = students.map(lambda s: (s.department, (s.average, 1)))
    sums = pairs.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    averages = sums.mapValues(lambda v: round(v[0] / v[1], 2))
    return sorted(averages.collect())


def demo_distinct(students):
    departments = students.map(lambda s: s.department).distinct()
    return sorted(departments.collect())


def demo_sort_by(students):
    ranked = students.sortBy(lambda s: s.average, ascending=False)
    return ranked.map(lambda s: (s.name, round(s.average, 1))).take(5)


def demo_count(students):
    return {
        "all_students": students.count(),
        "passing_students": students.filter(lambda s: s.passed).count(),
    }


def demo_take(students):
    first_three = students.take(3)
    return [(s.student_id, s.name) for s in first_three]


def demo_collect(students):
    failing = students.filter(lambda s: not s.passed)
    return [(s.student_id, s.name) for s in failing.collect()]


def demo_group_by_key(students):
    grouped = students.map(lambda s: (s.department, s.name)).groupByKey()
    return sorted((dept, sorted(names)) for dept, names in grouped.collect())


def demo_map_partitions(students):
    def summarise_partition(rows):
        count = 0
        average_sum = 0.0
        for row in rows:
            count += 1
            average_sum += row.average
        yield (count, round(average_sum, 1))

    return students.mapPartitions(summarise_partition).collect()


@dataclass(frozen=True)
class RDDOperation:
    name: str
    kind: str                # 'Transformation' or 'Action'
    dependency: str          # 'Narrow', 'Wide (shuffle)' or 'Action'
    summary: str
    input_description: str
    output_description: str
    scenario: str
    function: Callable
    notes: str


OPERATIONS: Dict[str, RDDOperation] = {
    op.name: op
    for op in (
        RDDOperation(
            "map()", "Transformation", "Narrow",
            "Applies a function to every element and returns exactly one output per input.",
            "RDD of StudentResult rows.",
            "RDD of (name, average) pairs.",
            "Turn every student row into a short (name, average marks) pair.",
            demo_map,
            "Lazy: nothing runs until an action. This demo ends with take(8), which is the action.",
        ),
        RDDOperation(
            "filter()", "Transformation", "Narrow",
            "Keeps only the elements for which the function returns True.",
            "RDD of StudentResult rows.",
            "A smaller RDD containing only passing students.",
            "Find students who scored at least 40 in all three subjects.",
            demo_filter,
            "Lazy. The final collect() is safe because only 8 students exist.",
        ),
        RDDOperation(
            "flatMap()", "Transformation", "Narrow",
            "Like map(), but each input can produce zero, one or MANY outputs, which are flattened.",
            "RDD of StudentResult rows (1 row per student).",
            "RDD of (student_id, subject, mark) rows (3 rows per student).",
            "Split each student into one row per subject for subject-wise analysis.",
            demo_flat_map,
            "Lazy. count() and take() are the actions in this demo.",
        ),
        RDDOperation(
            "reduce()", "Action", "Action",
            "Combines all elements into ONE value using a function that is associative and commutative.",
            "RDD of numbers (student totals).",
            "A single number returned to Python.",
            "Add up all student totals and find the highest total.",
            demo_reduce,
            "Action: starts a Spark job immediately. reduce() fails on an empty RDD.",
        ),
        RDDOperation(
            "reduceByKey()", "Transformation", "Wide (shuffle)",
            "Combines values that share a key. Values are pre-combined inside each partition before the shuffle.",
            "RDD of (department, (average, 1)) pairs.",
            "RDD of (department, (sum, count)) pairs, then averaged.",
            "Average marks per department without moving every record across the network.",
            demo_reduce_by_key,
            "Lazy. Preferred over groupByKey() for aggregations because it moves less data.",
        ),
        RDDOperation(
            "distinct()", "Transformation", "Wide (shuffle)",
            "Removes duplicate elements.",
            "RDD of department names (one per student).",
            "RDD of unique department names.",
            "List every department that appears in the data.",
            demo_distinct,
            "Lazy. Needs a shuffle because equal values may sit in different partitions.",
        ),
        RDDOperation(
            "sortBy()", "Transformation", "Wide (shuffle)",
            "Sorts the RDD by a key function.",
            "RDD of StudentResult rows.",
            "RDD ordered by average marks (highest first).",
            "Rank students by average marks.",
            demo_sort_by,
            "Counted as a transformation, but Spark may run a small sampling job to choose range boundaries.",
        ),
        RDDOperation(
            "count()", "Action", "Action",
            "Returns the number of elements in the RDD.",
            "Any RDD.",
            "One integer.",
            "Count all students and count passing students.",
            demo_count,
            "Action: triggers execution of every transformation it depends on.",
        ),
        RDDOperation(
            "take()", "Action", "Action",
            "Returns the first n elements to Python without scanning the whole RDD if possible.",
            "Any RDD and a number n.",
            "A Python list with at most n items.",
            "Preview the first three students safely.",
            demo_take,
            "Action. Safe for large data because the result size is limited to n.",
        ),
        RDDOperation(
            "collect()", "Action", "Action",
            "Returns ALL elements to the driver as a Python list.",
            "A small RDD.",
            "A Python list with every element.",
            "List the (small) set of failing students.",
            demo_collect,
            "Action. Dangerous on big RDDs because everything must fit in driver memory; use only on small, bounded results.",
        ),
        RDDOperation(
            "groupByKey()", "Transformation", "Wide (shuffle)",
            "Groups all values that share a key into one iterable per key.",
            "RDD of (department, student name) pairs.",
            "RDD of (department, iterable of names).",
            "List the names of the students in each department.",
            demo_group_by_key,
            "Lazy. Every value crosses the network, so prefer reduceByKey() when you only need totals or counts.",
        ),
        RDDOperation(
            "mapPartitions()", "Transformation", "Narrow",
            "Runs a function ONCE per partition, receiving an iterator over all rows of that partition.",
            "RDD of StudentResult rows split into 2 partitions.",
            "RDD with one summary tuple per partition.",
            "Count the students in each partition and sum their averages.",
            demo_map_partitions,
            "Lazy. Useful for per-partition setup work and for inspecting how data is divided.",
        ),
    )
}


def operation_names() -> List[str]:
    """Names of all demonstrations, in teaching order."""
    return list(OPERATIONS.keys())


def run_operation(sc: SparkContext, name: str) -> Dict[str, Any]:
    """Run one predefined demo on the built-in 8-student sample.

    Only functions from the OPERATIONS table can run: the dashboard never
    executes text typed by the user.
    """
    if name not in OPERATIONS:
        raise KeyError(f"Unknown RDD operation: {name!r}")
    operation = OPERATIONS[name]
    students = make_demo_students_rdd(sc)

    started = time.perf_counter()
    result = operation.function(students)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return {
        "name": operation.name,
        "kind": operation.kind,
        "dependency": operation.dependency,
        "summary": operation.summary,
        "input_description": operation.input_description,
        "output_description": operation.output_description,
        "scenario": operation.scenario,
        "notes": operation.notes,
        "code": textwrap.dedent(inspect.getsource(operation.function)),
        "sample_input": [
            {
                "student_id": r.student_id,
                "name": r.name,
                "department": r.department,
                "maths": r.maths,
                "science": r.science,
                "computer": r.computer,
                "attendance": r.attendance,
            }
            for r in demo_sample_records()
        ],
        "partitions": students.getNumPartitions(),
        "result": result,
        "elapsed_ms": elapsed_ms,
    }


def lazy_evaluation_demo(sc: SparkContext) -> Dict[str, Any]:
    """Show that transformations are lazy and that actions run them.

    An accumulator counts how many times the tracked function really runs.
    """
    processed = sc.accumulator(0)

    def track(student):
        processed.add(1)
        return student

    students = sc.parallelize(demo_sample_records(), 2).map(to_result)
    pipeline = students.map(track).filter(lambda s: s.passed).map(lambda s: s.name)
    count_after_transformations = processed.value  # no action yet -> still 0

    lineage = pipeline.toDebugString().decode("utf-8")

    passing_names = pipeline.collect()  # first ACTION: the work happens now
    count_after_first_action = processed.value

    pipeline.count()  # second ACTION on an uncached RDD: work happens again
    count_after_second_action = processed.value

    return {
        "rows_in_input": len(demo_sample_records()),
        "rows_processed_after_transformations": count_after_transformations,
        "rows_processed_after_first_action": count_after_first_action,
        "rows_processed_after_second_action": count_after_second_action,
        "result_of_first_action": passing_names,
        "lineage": lineage,
        "code": textwrap.dedent(inspect.getsource(lazy_evaluation_demo)),
    }

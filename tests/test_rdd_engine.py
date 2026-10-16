"""
Tests for rdd_engine.py. These start a real local Spark session (see conftest.py).

Expected values come from a tiny hand-checkable dataset:

    id  name   dept  maths science computer attendance | total  average pass
    T1  Alice  CS     90     80      70       95        | 240    80.0    yes
    T2  Bob    CS     60     60      60       70        | 180    60.0    yes (low attendance)
    T3  Cara   EE     40     40      40       75        | 120    40.0    yes (all exactly 40)
    T4  Dan    EE     39     90      90       80        | 219    73.0    NO  (maths 39)
    T5  Eve    ME    100    100     100      100        | 300   100.0    yes
    T6  Finn   ME     20     30      40       50        |  90    30.0    NO  (low attendance)
"""

from pathlib import Path

import pytest

from src import rdd_engine as engine
from src.data_generator import SAMPLE_STUDENTS
from src.performance import canonicalize
from src.validators import EmptyDatasetError, MissingColumnsError

HEADER = "student_id,name,department,maths,science,computer,attendance"
SMALL_LINES = [
    HEADER,
    "T1,Alice,CS,90,80,70,95",
    "T2,Bob,CS,60,60,60,70",
    "T3,Cara,EE,40,40,40,75",
    "T4,Dan,EE,39,90,90,80",
    "T5,Eve,ME,100,100,100,100",
    "T6,Finn,ME,20,30,40,50",
]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def loaded(spark_context):
    """Load the small dataset; release the cached RDDs after each test."""
    load = engine.load_students_from_lines(spark_context, SMALL_LINES, num_partitions=2)
    results = engine.build_results_rdd(load.records).cache()
    yield load, results
    engine.release_rdds(results, load.records)


@pytest.fixture()
def results(loaded):
    return loaded[1]


# ------------------------- pure business rules --------------------------
def test_pass_requires_every_subject_to_reach_40():
    make = lambda m, s, c: engine.StudentRecord("x", "x", "d", m, s, c, 80)
    assert engine.is_passing(make(40, 40, 40)) is True
    assert engine.is_passing(make(39.9, 90, 90)) is False
    assert engine.is_passing(make(90, 90, 0)) is False


def test_attendance_exactly_75_is_not_low():
    make = lambda a: engine.StudentRecord("x", "x", "d", 50, 50, 50, a)
    assert engine.has_low_attendance(make(75)) is False
    assert engine.has_low_attendance(make(74.9)) is True


def test_total_and_average():
    record = engine.StudentRecord("x", "x", "d", 90, 80, 70, 95)
    assert engine.compute_total(record) == 240
    assert engine.compute_average(record) == 80


@pytest.mark.parametrize(
    "average, band",
    [(0, "Below 40"), (39.99, "Below 40"), (40, "40-49"), (59.9, "50-59"),
     (73, "70-79"), (89.99, "80-89"), (90, "90-100"), (100, "90-100")],
)
def test_performance_band_edges(average, band):
    assert engine.performance_band(average) == band


# ------------------------------- loading --------------------------------
def test_load_small_dataset(loaded):
    load, _ = loaded
    assert load.valid_count == 6
    assert load.rejected_count == 0
    assert load.errors == []
    assert load.is_usable
    assert load.num_partitions == 2
    assert load.records.is_cached


def test_load_reports_each_bad_row_with_its_line_number(spark_context):
    lines = [
        HEADER,                              # line 1
        "A1,Ann,CS,80,80,80,90",             # line 2  valid
        "A2,Bad,CS,abc,80,80,90",            # line 3  text in marks
        "A3,Neg,CS,-1,80,80,90",             # line 4  negative
        "A4,Over,CS,80,80,101,90",           # line 5  mark above 100
        "A5,Att,CS,80,80,80,150",            # line 6  attendance above 100
        "A6,Short,CS,80,80",                 # line 7  too few fields
        "",                                  # line 8  blank (ignored)
        "A7,,CS,80,80,80,90",                # line 9  empty name
        "A8,Nan,CS,nan,80,80,90",            # line 10 nan
        "A9,Fine,EE,70,70,70,70",            # line 11 valid
    ]
    load = engine.load_students_from_lines(spark_context, lines)
    try:
        assert load.valid_count == 2
        assert load.rejected_count == 7
        assert load.total_data_lines == 9
        bad_lines = [int(message.split(":")[0].split()[1]) for message in load.errors]
        assert bad_lines == [3, 4, 5, 6, 7, 9, 10]
        assert "maths must be a number" in load.errors[0]
        assert "expected 7 fields" in load.errors[4]
        names = sorted(r.name for r in load.records.collect())
        assert names == ["Ann", "Fine"]
    finally:
        engine.release_rdds(load.records)


def test_duplicate_ids_keep_first_occurrence_and_are_reported(spark_context):
    lines = [HEADER, "D1,First,CS,80,80,80,90", "D2,Other,CS,70,70,70,70", "D1,Second,CS,10,10,10,10"]
    load = engine.load_students_from_lines(spark_context, lines)
    try:
        assert load.valid_count == 2
        assert load.rejected_count == 1
        assert load.duplicate_ids == ["D1"]
        assert any("Duplicate student_id 'D1'" in message for message in load.errors)
        kept = {r.student_id: r.name for r in load.records.collect()}
        assert kept == {"D1": "First", "D2": "Other"}
    finally:
        engine.release_rdds(load.records)


def test_missing_columns_raise_clear_error(spark_context):
    with pytest.raises(MissingColumnsError) as error:
        engine.load_students_from_lines(spark_context, ["student_id,name,maths", "S1,A,50"])
    assert "department" in error.value.missing


def test_completely_empty_input_raises(spark_context):
    with pytest.raises(EmptyDatasetError):
        engine.load_students_from_lines(spark_context, [])


def test_header_only_file_loads_zero_students(spark_context):
    load = engine.load_students_from_lines(spark_context, [HEADER])
    assert load.valid_count == 0 and not load.is_usable
    assert load.records.count() == 0
    results = engine.build_results_rdd(load.records)
    assert engine.count_students(results) == 0
    assert engine.overall_summary(results)["total_students"] == 0
    assert engine.department_statistics(results) == []
    assert engine.top_students(results, 3) == []
    assert engine.pass_fail_distribution(results) == {"Pass": 0, "Fail": 0}
    engine.release_rdds(load.records)


def test_file_where_every_row_is_invalid_is_not_usable(spark_context):
    load = engine.load_students_from_lines(spark_context, [HEADER, "S1,A,CS,abc,1,1,1", "S2,B,CS,1,1,1,500"])
    assert not load.is_usable
    assert load.rejected_count == 2 and len(load.errors) == 2
    engine.release_rdds(load.records)


def test_load_csv_file_from_disk(spark_context, tmp_path):
    csv_file = tmp_path / "small.csv"
    csv_file.write_text("\n".join(SMALL_LINES) + "\n", encoding="utf-8")
    load = engine.load_students_from_csv(spark_context, csv_file, num_partitions=3)
    try:
        assert load.valid_count == 6
        assert load.num_partitions == 3
        assert load.load_method in ("sc.textFile", "sc.parallelize (textFile fallback)")
    finally:
        engine.release_rdds(load.records)


def test_csv_with_bom_and_reordered_columns(spark_context, tmp_path):
    csv_file = tmp_path / "reordered.csv"
    csv_file.write_text(
        "\ufeffattendance,computer,science,maths,department,name,student_id\n"
        "88,70,80,90,CS,Alice,T1\n",
        encoding="utf-8",
    )
    load = engine.load_students_from_csv(spark_context, csv_file)
    try:
        record = load.records.collect()[0]
        assert (record.student_id, record.maths, record.science, record.computer, record.attendance) == ("T1", 90, 80, 70, 88)
    finally:
        engine.release_rdds(load.records)


def test_missing_file_raises_file_not_found(spark_context, tmp_path):
    with pytest.raises(FileNotFoundError):
        engine.load_students_from_csv(spark_context, tmp_path / "nope.csv")


@pytest.mark.parametrize("bad", [0, -1, 65, 2.5, "4", True])
def test_invalid_partition_requests_are_rejected(spark_context, bad):
    with pytest.raises(ValueError):
        engine.load_students_from_lines(spark_context, SMALL_LINES, num_partitions=bad)


def test_shipped_sample_csv_loads_cleanly(spark_context):
    load = engine.load_students_from_csv(spark_context, PROJECT_ROOT / "data" / "students.csv", num_partitions=4)
    try:
        assert load.valid_count == len(SAMPLE_STUDENTS) == 25
        assert load.rejected_count == 0 and load.errors == []
        assert load.num_partitions == 4
    finally:
        engine.release_rdds(load.records)


# ----------------------------- calculations -----------------------------
def test_results_have_correct_totals_averages_and_status(results):
    rows = {r.student_id: r for r in results.collect()}
    expected = {  # id: (total, average, passed, low_attendance)
        "T1": (240, 80.0, True, False),
        "T2": (180, 60.0, True, True),
        "T3": (120, 40.0, True, False),
        "T4": (219, 73.0, False, False),
        "T5": (300, 100.0, True, False),
        "T6": (90, 30.0, False, True),
    }
    for student_id, (total, average, passed, low) in expected.items():
        row = rows[student_id]
        assert row.total == total and row.average == pytest.approx(average)
        assert row.passed is passed and row.low_attendance is low


def test_count_students(results):
    assert engine.count_students(results) == 6


def test_overall_summary(results):
    summary = engine.overall_summary(results)
    assert summary["total_students"] == 6
    assert summary["overall_average"] == pytest.approx(383 / 6)
    assert summary["pass_count"] == 4 and summary["fail_count"] == 2
    assert summary["pass_percentage"] == pytest.approx(400 / 6)
    assert summary["average_attendance"] == pytest.approx(470 / 6)
    assert summary["low_attendance_count"] == 2


def test_subject_statistics(results):
    by_subject = {row["subject"]: row for row in engine.subject_statistics(results)}
    assert list(by_subject) == ["maths", "science", "computer"]
    assert by_subject["maths"]["average"] == pytest.approx(349 / 6)
    assert by_subject["science"]["average"] == pytest.approx(400 / 6)
    assert by_subject["computer"]["average"] == pytest.approx(400 / 6)
    assert (by_subject["maths"]["lowest"], by_subject["maths"]["highest"]) == (20, 100)
    assert by_subject["maths"]["pass_percentage"] == pytest.approx(400 / 6)    # 4 of 6
    assert by_subject["science"]["pass_percentage"] == pytest.approx(500 / 6)  # 5 of 6
    assert by_subject["computer"]["pass_percentage"] == pytest.approx(100.0)   # 6 of 6


def test_department_statistics_are_sorted_and_correct(results):
    rows = engine.department_statistics(results)
    assert [r["department"] for r in rows] == ["CS", "ME", "EE"]  # 70, 65, 56.5
    cs, me, ee = rows
    assert cs["average_marks"] == pytest.approx(70.0) and cs["students"] == 2
    assert cs["pass_count"] == 2 and cs["average_attendance"] == pytest.approx(82.5)
    assert me["average_marks"] == pytest.approx(65.0) and me["pass_count"] == 1
    assert me["lowest_average"] == pytest.approx(30.0) and me["highest_average"] == pytest.approx(100.0)
    assert ee["average_marks"] == pytest.approx(56.5) and ee["pass_percentage"] == pytest.approx(50.0)
    assert ee["low_attendance_count"] == 0 and cs["low_attendance_count"] == 1


def test_top_students(results):
    assert [r["student_id"] for r in engine.top_students(results, 3)] == ["T5", "T1", "T4"]
    assert engine.top_students(results, 0) == []
    assert len(engine.top_students(results, 100)) == 6


def test_sorting_by_average(results):
    descending = [r["student_id"] for r in engine.sort_students_by_average(results, True, 10)]
    ascending = [r["student_id"] for r in engine.sort_students_by_average(results, False, 10)]
    assert descending == ["T5", "T1", "T4", "T2", "T3", "T6"]
    assert ascending == list(reversed(descending))
    assert len(engine.sort_students_by_average(results, True, 2)) == 2


def test_students_needing_support(results):
    support = engine.students_needing_support(results)
    assert support["count"] == 3
    assert [r["student_id"] for r in support["students"]] == ["T6", "T2", "T4"]


def test_low_attendance_students(results):
    low = engine.low_attendance_students(results)
    assert low["count"] == 2
    assert [r["student_id"] for r in low["students"]] == ["T6", "T2"]  # 50 then 70


def test_pass_fail_distribution(results):
    assert engine.pass_fail_distribution(results) == {"Pass": 4, "Fail": 2}


def test_performance_distribution_lists_every_band(results):
    bands = {r["band"]: r["students"] for r in engine.performance_distribution(results)}
    assert bands == {"Below 40": 1, "40-49": 1, "50-59": 0, "60-69": 1,
                     "70-79": 1, "80-89": 1, "90-100": 1}


def test_attendance_distribution(results):
    bands = {r["band"]: r["students"] for r in engine.attendance_distribution(results)}
    assert bands == {"Below 60": 1, "60 to below 75": 1, "75 to below 85": 2, "85 and above": 2}


def test_distinct_departments(results):
    assert engine.distinct_departments(results) == ["CS", "EE", "ME"]


def test_group_by_department(results):
    groups = {g["department"]: g for g in engine.group_by_department(results)}
    assert groups["CS"]["names"] == ["Alice", "Bob"]
    assert groups["EE"]["names"] == ["Cara", "Dan"]
    assert all(g["students"] == 2 for g in groups.values())
    limited = engine.group_by_department(results, max_names=1)
    assert all(len(g["names"]) == 1 and g["students"] == 2 for g in limited)


def test_query_students_search_filter_and_sort(results):
    by_name = engine.query_students(results, search="ALI")
    assert [r["student_id"] for r in by_name["rows"]] == ["T1"]
    by_id = engine.query_students(results, search="t3")
    assert [r["name"] for r in by_id["rows"]] == ["Cara"]
    ee = engine.query_students(results, department="EE")
    assert [r["student_id"] for r in ee["rows"]] == ["T4", "T3"]
    failing = engine.query_students(results, status="Fail", descending=False)
    assert [r["student_id"] for r in failing["rows"]] == ["T6", "T4"]
    passing_cs = engine.query_students(results, department="CS", status="Pass")
    assert passing_cs["total_matches"] == 2
    nothing = engine.query_students(results, search="zzz")
    assert nothing["total_matches"] == 0 and nothing["rows"] == []


def test_query_students_limit_marks_truncation(results):
    page = engine.query_students(results, limit=2)
    assert len(page["rows"]) == 2 and page["total_matches"] == 6 and page["truncated"]


def test_spark_results_match_plain_python_on_shipped_dataset(spark_context):
    """Independent cross-check: recompute with ordinary Python and compare."""
    load = engine.load_students_from_csv(spark_context, PROJECT_ROOT / "data" / "students.csv")
    results = engine.build_results_rdd(load.records).cache()
    try:
        python_pass = sum(1 for r in SAMPLE_STUDENTS if min(r[3:6]) >= 40)
        python_avg = sum(sum(r[3:6]) / 3 for r in SAMPLE_STUDENTS) / len(SAMPLE_STUDENTS)
        python_low = sum(1 for r in SAMPLE_STUDENTS if r[6] < 75)
        summary = engine.overall_summary(results)
        assert summary["pass_count"] == python_pass
        assert summary["overall_average"] == pytest.approx(python_avg)
        assert summary["low_attendance_count"] == python_low
        # department averages
        for row in engine.department_statistics(results):
            members = [r for r in SAMPLE_STUDENTS if r[2] == row["department"]]
            assert row["students"] == len(members)
            assert row["average_marks"] == pytest.approx(sum(sum(m[3:6]) / 3 for m in members) / len(members))
        # top student is the highest-average student in plain Python
        best = max(SAMPLE_STUDENTS, key=lambda r: (sum(r[3:6]) / 3, -ord(r[0][-1])))
        assert engine.top_students(results, 1)[0]["student_id"] == best[0]
    finally:
        engine.release_rdds(results, load.records)


# ------------------------------ partitions ------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 6, 8])
def test_requested_partition_count_is_real(spark_context, n):
    load = engine.load_students_from_lines(spark_context, SMALL_LINES, num_partitions=n)
    try:
        assert engine.partition_count(load.records) == n
        sizes = engine.partition_sizes(load.records)
        assert len(sizes) == n and sum(sizes) == 6
    finally:
        engine.release_rdds(load.records)


def test_partition_details_cover_every_student_once(results):
    details = engine.partition_details(results, preview=2)
    assert [d["partition"] for d in details] == list(range(len(details)))
    assert sum(d["records"] for d in details) == 6
    assert all(len(d["sample_ids"]) <= 2 for d in details)


def test_set_partition_count_changes_partitions_but_not_data(results):
    resized = engine.set_partition_count(results, 5)
    assert engine.partition_count(resized) == 5
    assert sorted(r.student_id for r in resized.collect()) == ["T1", "T2", "T3", "T4", "T5", "T6"]
    with pytest.raises(ValueError):
        engine.set_partition_count(results, 0)


@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_results_are_identical_for_every_partition_count(spark_context, n):
    baseline_load = engine.load_students_from_lines(spark_context, SMALL_LINES, num_partitions=1)
    load = engine.load_students_from_lines(spark_context, SMALL_LINES, num_partitions=n)
    base = engine.build_results_rdd(baseline_load.records)
    other = engine.build_results_rdd(load.records)
    try:
        for function in (engine.overall_summary, engine.department_statistics,
                         engine.subject_statistics, engine.pass_fail_distribution,
                         engine.performance_distribution):
            assert canonicalize(function(base)) == canonicalize(function(other))
        assert engine.top_students(base, 6) == engine.top_students(other, 6)
    finally:
        engine.release_rdds(baseline_load.records, load.records)


def test_release_rdds_unpersists_and_ignores_none(spark_context):
    rdd = spark_context.parallelize(range(10), 2).cache()
    rdd.count()
    assert rdd.is_cached
    engine.release_rdds(rdd, None)
    assert not rdd.is_cached


def test_dashboard_analytics_bundle_is_bounded_and_complete(results):
    analytics = engine.compute_dashboard_analytics(results, top_n=2)
    assert set(analytics) == {
        "summary", "departments", "subjects", "pass_fail", "performance_bands",
        "attendance_bands", "top_students", "needs_support", "low_attendance",
        "department_names", "scatter_sample",
    }
    assert len(analytics["top_students"]) == 2
    assert len(analytics["scatter_sample"]) == 6


# --------------------------- RDD operations lab --------------------------
EXPECTED_DEMO_RESULTS = {
    "map()": [("Aarav Sharma", 95.0), ("Kabir Singh", 45.0), ("Arjun Reddy", 48.0),
              ("Saanvi Kapoor", 80.0), ("Krishna Das", 41.7), ("Tara Bose", 34.7),
              ("Sneha Pillai", 41.7), ("Devansh Kulkarni", 86.0)],
    "filter()": ["Aarav Sharma", "Saanvi Kapoor", "Krishna Das", "Devansh Kulkarni"],
    "reduce()": {"sum_of_all_totals": 1416, "highest_total": 285},
    "reduceByKey()": [("Civil", 86.0), ("Computer Science", 70.0), ("Electronics", 38.17),
                      ("Information Technology", 64.0), ("Mechanical", 41.67)],
    "distinct()": ["Civil", "Computer Science", "Electronics", "Information Technology", "Mechanical"],
    "sortBy()": [("Aarav Sharma", 95.0), ("Devansh Kulkarni", 86.0), ("Saanvi Kapoor", 80.0),
                 ("Arjun Reddy", 48.0), ("Kabir Singh", 45.0)],
    "count()": {"all_students": 8, "passing_students": 4},
    "take()": [("S001", "Aarav Sharma"), ("S005", "Kabir Singh"), ("S009", "Arjun Reddy")],
    "collect()": [("S005", "Kabir Singh"), ("S009", "Arjun Reddy"),
                  ("S014", "Tara Bose"), ("S018", "Sneha Pillai")],
    "mapPartitions()": [(4, 268.0), (4, 204.0)],
}


@pytest.mark.parametrize("name, expected", list(EXPECTED_DEMO_RESULTS.items()))
def test_operation_demo_results(spark_context, name, expected):
    outcome = engine.run_operation(spark_context, name)
    assert outcome["result"] == expected
    assert outcome["kind"] in ("Transformation", "Action")


def test_flat_map_demo_expands_each_student_to_three_rows(spark_context):
    result = engine.run_operation(spark_context, "flatMap()")["result"]
    assert result["students_in"] == 8 and result["rows_out"] == 24
    assert result["first_six_rows"][0] == ("S001", "maths", 92)


def test_group_by_key_demo(spark_context):
    result = engine.run_operation(spark_context, "groupByKey()")["result"]
    assert dict(result)["Computer Science"] == ["Aarav Sharma", "Kabir Singh"]
    assert len(result) == 5


def test_every_required_operation_is_listed_and_classified():
    required = {"map()", "filter()", "flatMap()", "reduce()", "reduceByKey()", "distinct()",
                "sortBy()", "count()", "take()", "collect()", "groupByKey()", "mapPartitions()"}
    assert set(engine.operation_names()) == required
    actions = {n for n, op in engine.OPERATIONS.items() if op.kind == "Action"}
    assert actions == {"reduce()", "count()", "take()", "collect()"}


def test_demo_shows_the_code_that_actually_ran(spark_context):
    outcome = engine.run_operation(spark_context, "reduceByKey()")
    assert "def demo_reduce_by_key" in outcome["code"]
    assert ".reduceByKey(" in outcome["code"]
    assert outcome["partitions"] == 2 and len(outcome["sample_input"]) == 8


def test_unknown_operation_is_rejected(spark_context):
    with pytest.raises(KeyError):
        engine.run_operation(spark_context, "os.system('x')")


def test_lazy_evaluation_demo_shows_work_only_happens_on_actions(spark_context):
    demo = engine.lazy_evaluation_demo(spark_context)
    assert demo["rows_processed_after_transformations"] == 0
    assert demo["rows_processed_after_first_action"] == 8
    assert demo["rows_processed_after_second_action"] == 16   # recomputed, not cached
    assert len(demo["result_of_first_action"]) == 4
    assert "PythonRDD" in demo["lineage"] or "MapPartitionsRDD" in demo["lineage"]

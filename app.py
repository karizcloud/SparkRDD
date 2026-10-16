"""
app.py - SparkRDD Lab dashboard (Streamlit)
===========================================
Run with:   streamlit run app.py

This file only builds the user interface. Every number shown comes from
src/rdd_engine.py (real PySpark RDD operations) or src/performance.py.
"""

import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import performance as perf
from src import rdd_engine as engine
from src.data_generator import sample_csv_text, write_sample_csv
from src.validators import (
    ATTENDANCE_THRESHOLD,
    PASS_MARK,
    REQUIRED_COLUMNS,
    DatasetError,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_CSV_PATH = PROJECT_ROOT / "data" / "students.csv"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
PARTITION_CHOICES = [1, 2, 3, 4, 6, 8]
DEFAULT_PARTITIONS = 4

PAGES = [
    "Overview",
    "Student Explorer",
    "Academic Analytics",
    "RDD Operations Lab",
    "Partition Explorer",
    "Performance Explorer",
    "Dataset Manager",
]

INK = "#1f2937"
MUTED = "#6b7280"
PRIMARY = "#1d4ed8"
TEAL = "#0f766e"
AMBER = "#b45309"
RED = "#b91c1c"
GRID = "#e5e7eb"
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

st.set_page_config(page_title="SparkRDD Lab", page_icon="⚡", layout="wide")

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{ font-family: {FONT}; }}
    .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }}
    h1 {{ font-size: 1.9rem !important; font-weight: 650 !important; letter-spacing: -0.01em; margin-bottom: 0.1rem; }}
    h2, h3 {{ font-weight: 600 !important; letter-spacing: -0.005em; }}
    .page-subtitle {{ color: {MUTED}; font-size: 1.0rem; margin: 0 0 1.4rem 0; }}
    [data-testid="stMetric"] {{ background: #f8fafc; border: 1px solid {GRID}; border-radius: 10px; padding: 0.85rem 1rem; }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    [data-testid="stMetricValue"] {{ font-size: 1.7rem; font-weight: 650; }}
    section[data-testid="stSidebar"] {{ background: #f4f6fa; border-right: 1px solid {GRID}; }}
    .brand {{ font-weight: 700; font-size: 1.15rem; color: {INK}; margin-bottom: 0.1rem; }}
    .brand-sub {{ color: {MUTED}; font-size: 0.82rem; margin-bottom: 1rem; }}
    .badge {{ display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; margin-right: 0.4rem; }}
    .badge-transformation {{ background: #dbeafe; color: #1e40af; }}
    .badge-action {{ background: #ffedd5; color: #9a3412; }}
    .badge-neutral {{ background: #e5e7eb; color: #374151; }}
    .small-note {{ color: {MUTED}; font-size: 0.86rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Spark and dataset state
# --------------------------------------------------------------------------
@st.cache_resource(
    show_spinner="Starting Apache Spark (the first start can take a few seconds)...",
    validate=lambda spark: engine.is_spark_context_active(spark.sparkContext),
)
def start_spark():
    return engine.get_spark_session()


def activate_dataset(load, label: str) -> None:
    """Make `load` the active dataset and recompute every dashboard number."""
    results = engine.build_results_rdd(load.records).cache()
    try:
        analytics = engine.compute_dashboard_analytics(results)
    except Exception:
        engine.release_rdds(results)
        raise

    engine.release_rdds(st.session_state.get("results_rdd"))
    old_load = st.session_state.get("load")
    if old_load is not None and old_load.records is not load.records:
        engine.release_rdds(old_load.records)

    st.session_state["load"] = load
    st.session_state["results_rdd"] = results
    st.session_state["analytics"] = analytics
    st.session_state["dataset_label"] = label
    st.session_state.pop("pending", None)


def _rdd_is_reusable(rdd, sc) -> bool:
    return (
        rdd is not None
        and rdd.context is sc
        and rdd.is_cached
        and engine.is_spark_context_active(sc)
    )


def _dataset_is_reusable(sc) -> bool:
    load = st.session_state.get("load")
    results = st.session_state.get("results_rdd")
    return (
        engine.is_spark_context_active(sc)
        and isinstance(load, engine.LoadResult)
        and _rdd_is_reusable(load.records, sc)
        and _rdd_is_reusable(results, sc)
        and isinstance(st.session_state.get("analytics"), dict)
        and bool(st.session_state.get("dataset_label"))
    )


def _discard_dataset_state() -> None:
    load = st.session_state.get("load")
    pending = st.session_state.get("pending")
    engine.release_rdds(
        st.session_state.get("results_rdd"),
        getattr(load, "records", None),
        getattr(pending.get("load"), "records", None) if isinstance(pending, dict) else None,
    )
    for key in ("load", "results_rdd", "analytics", "dataset_label", "pending"):
        st.session_state.pop(key, None)


def ensure_dataset(sc) -> None:
    """Load the bundled sample dataset the first time the app starts."""
    if _dataset_is_reusable(sc):
        pending = st.session_state.get("pending")
        if isinstance(pending, dict) and not _rdd_is_reusable(
            getattr(pending.get("load"), "records", None), sc
        ):
            engine.release_rdds(getattr(pending.get("load"), "records", None))
            st.session_state.pop("pending", None)
        return
    _discard_dataset_state()
    if not SAMPLE_CSV_PATH.is_file():
        write_sample_csv(SAMPLE_CSV_PATH)
    load = engine.load_students_from_csv(sc, SAMPLE_CSV_PATH, num_partitions=DEFAULT_PARTITIONS)
    activate_dataset(load, "Sample dataset (data/students.csv)")


# --------------------------------------------------------------------------
# Small presentation helpers
# --------------------------------------------------------------------------
def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=44, b=8),
        font=dict(family=FONT, size=13, color=INK),
        title=dict(font=dict(size=15)),
        showlegend=legend,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def students_dataframe(rows) -> pd.DataFrame:
    """Student rows (dicts from the engine) -> display table."""
    columns = ["student_id", "name", "department", "maths", "science", "computer",
               "average", "attendance", "status"]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    frame["status"] = frame["passed"].map({True: "Pass", False: "Fail"})
    frame["average"] = frame["average"].round(1)
    return frame[columns]


STUDENT_COLUMN_CONFIG = {
    "student_id": st.column_config.TextColumn("ID", width="small"),
    "name": st.column_config.TextColumn("Name"),
    "department": st.column_config.TextColumn("Department"),
    "maths": st.column_config.NumberColumn("Maths", format="%.0f"),
    "science": st.column_config.NumberColumn("Science", format="%.0f"),
    "computer": st.column_config.NumberColumn("Computer", format="%.0f"),
    "average": st.column_config.NumberColumn("Average", format="%.1f"),
    "attendance": st.column_config.ProgressColumn("Attendance %", min_value=0, max_value=100, format="%.0f"),
    "status": st.column_config.TextColumn("Status", width="small"),
}


def show_students_table(rows, key: str) -> None:
    frame = students_dataframe(rows)
    st.dataframe(frame, hide_index=True, column_config=STUDENT_COLUMN_CONFIG, key=key)


def badge(text: str, css: str) -> str:
    return f'<span class="badge badge-{css}">{text}</span>'


def empty_state(message: str) -> None:
    st.info(message)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar() -> st.delta_generator.DeltaGenerator:
    st.sidebar.markdown('<div class="brand">SparkRDD Lab</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="brand-sub">Student analytics with Apache Spark RDDs</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    dataset_status = st.sidebar.empty()
    dataset_status.caption("Starting Spark and loading the dataset...")
    return dataset_status


def update_sidebar_dataset(dataset_status, sc) -> None:
    load = st.session_state["load"]
    with dataset_status.container():
        st.markdown("**Active dataset**")
        st.caption(st.session_state["dataset_label"])
        st.caption(
            f"{load.valid_count:,} students · {st.session_state['results_rdd'].getNumPartitions()} partitions"
        )
        st.caption(f"Spark {sc.version} · {sc.master}")


# --------------------------------------------------------------------------
# Page: Overview
# --------------------------------------------------------------------------
def page_overview() -> None:
    page_header("Overview", "A quick read on how the cohort is doing, computed by Spark RDD jobs.")
    analytics = st.session_state["analytics"]
    summary = analytics["summary"]

    columns = st.columns(4)
    columns[0].metric("Total students", f"{summary['total_students']:,}")
    columns[1].metric("Overall average", f"{summary['overall_average']:.1f}")
    columns[2].metric("Passing", f"{summary['pass_percentage']:.1f}%")
    columns[3].metric("Average attendance", f"{summary['average_attendance']:.1f}%")
    st.caption(
        f"Pass rule (demonstration): every subject mark of at least {PASS_MARK:g}. "
        f"Attendance flag: below {ATTENDANCE_THRESHOLD:g}%."
    )

    left, right = st.columns(2)
    departments = pd.DataFrame(analytics["departments"])
    with left:
        if departments.empty:
            empty_state("No department data.")
        else:
            ordered = departments.sort_values("average_marks")
            fig = px.bar(ordered, x="average_marks", y="department", orientation="h",
                         text=ordered["average_marks"].round(1), title="Average marks by department",
                         color_discrete_sequence=[PRIMARY])
            fig.update_xaxes(range=[0, 100], title="")
            fig.update_yaxes(title="")
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(style_figure(fig), key="overview_departments")
    with right:
        subjects = pd.DataFrame(analytics["subjects"])
        if subjects.empty:
            empty_state("No subject data.")
        else:
            fig = px.bar(subjects, x="subject", y="average", text=subjects["average"].round(1),
                         title="Average marks by subject", color_discrete_sequence=[TEAL])
            fig.update_yaxes(range=[0, 100], title="")
            fig.update_xaxes(title="")
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(style_figure(fig), key="overview_subjects")

    bands = pd.DataFrame(analytics["performance_bands"])
    fig = px.bar(bands, x="band", y="students", title="Academic performance distribution (average marks)",
                 color_discrete_sequence=[PRIMARY])
    fig.update_xaxes(title="Average marks band")
    fig.update_yaxes(title="Students")
    st.plotly_chart(style_figure(fig, 320), key="overview_distribution")


# --------------------------------------------------------------------------
# Page: Student Explorer
# --------------------------------------------------------------------------
def page_student_explorer() -> None:
    page_header("Student Explorer", "Search, filter and sort students. Every query runs as a Spark job.")
    results = st.session_state["results_rdd"]
    departments = ["All"] + st.session_state["analytics"]["department_names"]

    c1, c2, c3, c4 = st.columns([2, 2, 1.3, 1.7])
    search = c1.text_input("Search by ID or name", placeholder="e.g. S012 or Aarav")
    department = c2.selectbox("Department", departments)
    status = c3.selectbox("Status", ["All", "Pass", "Fail"])
    sort_label = c4.selectbox("Sort by average", ["High to low", "Low to high"])

    with st.spinner("Running Spark job..."):
        outcome = engine.query_students(
            results, search=search, department=department, status=status,
            descending=(sort_label == "High to low"),
        )

    if outcome["total_matches"] == 0:
        empty_state("No students match these filters. Try clearing the search box or choosing 'All'.")
        return

    caption = f"{outcome['total_matches']:,} matching students."
    if outcome["truncated"]:
        caption += f" Showing the first {len(outcome['rows']):,}."
    st.caption(caption)
    show_students_table(outcome["rows"], key="explorer_table")

    frame = students_dataframe(outcome["rows"])
    st.download_button("Download this table (CSV)", frame.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_students.csv", mime="text/csv")

    st.subheader("Student detail")
    chosen_id = st.selectbox("Choose a student", frame["student_id"].tolist(),
                             format_func=lambda sid: f"{sid} · {frame.loc[frame.student_id == sid, 'name'].iloc[0]}")
    student = frame[frame["student_id"] == chosen_id].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Average marks", f"{student['average']:.1f}")
    m2.metric("Attendance", f"{student['attendance']:.0f}%")
    m3.metric("Status", student["status"])
    marks = pd.DataFrame({"subject": ["maths", "science", "computer"],
                          "marks": [student["maths"], student["science"], student["computer"]]})
    fig = px.bar(marks, x="subject", y="marks", text="marks", color_discrete_sequence=[PRIMARY],
                 title=f"{student['name']}: marks by subject")
    fig.add_hline(y=PASS_MARK, line_dash="dash", line_color=AMBER,
                  annotation_text=f"pass mark {PASS_MARK:g}", annotation_position="top left")
    fig.update_yaxes(range=[0, 105], title="")
    fig.update_xaxes(title="")
    st.plotly_chart(style_figure(fig, 300), key="explorer_detail")


# --------------------------------------------------------------------------
# Page: Academic Analytics
# --------------------------------------------------------------------------
def page_academic_analytics() -> None:
    page_header("Academic Analytics", "Department, subject, pass/fail and attendance analysis.")
    analytics = st.session_state["analytics"]
    tabs = st.tabs(["Departments", "Subjects", "Pass / Fail", "Attendance", "Students"])

    with tabs[0]:
        departments = pd.DataFrame(analytics["departments"])
        if departments.empty:
            empty_state("No department data.")
        else:
            fig = px.bar(departments, x="department", y=["average_marks", "average_attendance"],
                         barmode="group", title="Average marks and attendance by department",
                         color_discrete_sequence=[PRIMARY, TEAL],
                         labels={"value": "", "variable": "", "department": ""})
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(style_figure(fig, legend=True), key="analytics_dept")
            table = departments.rename(columns={
                "department": "Department", "students": "Students", "average_marks": "Avg marks",
                "pass_count": "Passed", "pass_percentage": "Pass %", "average_attendance": "Avg attendance %",
                "lowest_average": "Lowest avg", "highest_average": "Highest avg",
                "low_attendance_count": "Low attendance",
            })
            st.dataframe(table.round(1), hide_index=True, key="analytics_dept_table")
            st.caption("Computed with reduceByKey: (count, sum, ...) tuples are merged per department, then divided.")

    with tabs[1]:
        subjects = pd.DataFrame(analytics["subjects"])
        if subjects.empty:
            empty_state("No subject data.")
        else:
            fig = px.bar(subjects, x="subject", y="average", text=subjects["average"].round(1),
                         title="Average marks by subject", color_discrete_sequence=[TEAL])
            fig.update_yaxes(range=[0, 100], title="")
            fig.update_xaxes(title="")
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(style_figure(fig), key="analytics_subject")
            st.dataframe(subjects.rename(columns={
                "subject": "Subject", "average": "Average", "lowest": "Lowest", "highest": "Highest",
                "pass_percentage": f"Marks >= {PASS_MARK:g} (%)", "students": "Students"}).round(1),
                hide_index=True, key="analytics_subject_table")
            st.caption("Computed with flatMap (one row per student per subject) followed by reduceByKey.")

    with tabs[2]:
        pass_fail = analytics["pass_fail"]
        left, right = st.columns(2)
        if pass_fail["Pass"] + pass_fail["Fail"] == 0:
            empty_state("No students to analyse.")
        else:
            fig = go.Figure(go.Pie(labels=list(pass_fail), values=list(pass_fail.values()), hole=0.6,
                                   marker=dict(colors=[TEAL, RED]), sort=False))
            fig.update_layout(title="Pass / fail distribution")
            left.plotly_chart(style_figure(fig, legend=True), key="analytics_passfail")
            departments = pd.DataFrame(analytics["departments"])
            departments["failed"] = departments["students"] - departments["pass_count"]
            fig = px.bar(departments, x="department", y=["pass_count", "failed"], barmode="stack",
                         title="Pass and fail counts by department",
                         color_discrete_sequence=[TEAL, RED], labels={"value": "", "variable": "", "department": ""})
            right.plotly_chart(style_figure(fig, legend=True), key="analytics_passfail_dept")

    with tabs[3]:
        bands = pd.DataFrame(analytics["attendance_bands"])
        left, right = st.columns(2)
        fig = px.bar(bands, x="band", y="students", title="Attendance distribution",
                     color_discrete_sequence=[PRIMARY])
        fig.update_xaxes(title="")
        left.plotly_chart(style_figure(fig), key="analytics_att_bands")
        sample = pd.DataFrame(analytics["scatter_sample"])
        if sample.empty:
            right.info("No data.")
        else:
            sample["status"] = sample["passed"].map({True: "Pass", False: "Fail"})
            fig = px.scatter(sample, x="attendance", y="average", color="status", hover_name="name",
                             title="Attendance vs average marks",
                             color_discrete_map={"Pass": TEAL, "Fail": RED})
            fig.add_vline(x=ATTENDANCE_THRESHOLD, line_dash="dash", line_color=AMBER)
            fig.update_xaxes(title="Attendance %", range=[0, 102])
            fig.update_yaxes(title="Average marks", range=[0, 102])
            right.plotly_chart(style_figure(fig, legend=True), key="analytics_att_scatter")
            if st.session_state["load"].valid_count > len(sample):
                right.caption(f"Scatter plot shows the first {len(sample):,} students only (bounded for speed).")
        low = analytics["low_attendance"]
        st.markdown(f"**{low['count']} student(s) below {ATTENDANCE_THRESHOLD:g}% attendance**")
        if low["students"]:
            show_students_table(low["students"], key="analytics_low_att")
            if low["count"] > len(low["students"]):
                st.caption(f"Showing the {len(low['students'])} lowest attendance records.")
        else:
            empty_state("Nobody is below the attendance threshold.")

    with tabs[4]:
        st.markdown("**Top-performing students**")
        show_students_table(analytics["top_students"], key="analytics_top")
        st.caption("Found with sortBy() followed by take(5): only five rows leave Spark.")
        support = analytics["needs_support"]
        st.markdown(f"**Students needing academic support: {support['count']}**")
        st.caption("Failing at least one subject, or attendance below the threshold. Lowest averages first.")
        if support["students"]:
            show_students_table(support["students"], key="analytics_support")
            if support["count"] > len(support["students"]):
                st.caption(f"Showing the first {len(support['students'])} of {support['count']}.")
        else:
            empty_state("Nobody currently needs support under these rules.")


# --------------------------------------------------------------------------
# Page: RDD Operations Lab
# --------------------------------------------------------------------------
def render_result(result) -> None:
    if isinstance(result, (dict, list, tuple)):
        st.json(result)
    else:
        st.write(result)


def page_rdd_lab(sc) -> None:
    page_header("RDD Operations Lab",
                "Predefined, safe examples on 8 sample students. The code shown is the code that ran.")
    name = st.selectbox("Choose an RDD operation", engine.operation_names())
    with st.spinner("Running on Spark..."):
        outcome = engine.run_operation(sc, name)

    css = "transformation" if outcome["kind"] == "Transformation" else "action"
    st.markdown(
        badge(outcome["kind"], css) + badge(outcome["dependency"], "neutral")
        + badge(f"{outcome['partitions']} partitions", "neutral"),
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**What it does**")
        st.write(outcome["summary"])
        st.markdown("**Student-performance example**")
        st.write(outcome["scenario"])
    with right:
        st.markdown("**Input**")
        st.write(outcome["input_description"])
        st.markdown("**Output**")
        st.write(outcome["output_description"])
    st.info(outcome["notes"])

    with st.expander("Sample input (8 students, split into 2 partitions)"):
        st.dataframe(pd.DataFrame(outcome["sample_input"]), hide_index=True, key="lab_input")

    st.markdown("**PySpark code**")
    st.code(outcome["code"], language="python")
    st.markdown("**Result produced by Spark**")
    render_result(outcome["result"])
    st.caption(f"Wall-clock time for this call: {outcome['elapsed_ms']:.0f} ms (includes Spark scheduling; not a benchmark).")

    st.divider()
    st.subheader("Transformations vs actions")
    overview = pd.DataFrame([{"Operation": n, "Type": op.kind, "Data movement": op.dependency}
                             for n, op in engine.OPERATIONS.items()])
    st.dataframe(overview, hide_index=True, key="lab_overview")

    st.subheader("Lazy evaluation")
    st.write(
        "Transformations only describe a plan; Spark does the work when an action is called. "
        "The demo below counts, with an accumulator, how many students the `map` step really processed."
    )
    if st.button("Run lazy evaluation demo"):
        with st.spinner("Running on Spark..."):
            st.session_state["lazy_demo"] = engine.lazy_evaluation_demo(sc)
    demo = st.session_state.get("lazy_demo")
    if demo:
        c1, c2, c3 = st.columns(3)
        c1.metric("After transformations only", demo["rows_processed_after_transformations"])
        c2.metric("After 1st action (collect)", demo["rows_processed_after_first_action"])
        c3.metric("After 2nd action (count)", demo["rows_processed_after_second_action"])
        st.write(
            f"The input has {demo['rows_in_input']} students. Nothing was processed until `collect()` ran. "
            "The second action processed them again because the RDD was not cached."
        )
        st.markdown("**Lineage (the plan Spark built)**")
        st.code(demo["lineage"])
        with st.expander("Show the demo code"):
            st.code(demo["code"], language="python")


# --------------------------------------------------------------------------
# Page: Partition Explorer
# --------------------------------------------------------------------------
def page_partition_explorer(sc) -> None:
    page_header("Partition Explorer", "How Spark splits your data. Numbers come from the real RDD.")
    results = st.session_state["results_rdd"]
    count = engine.partition_count(results)
    sizes = engine.partition_sizes(results)

    c1, c2, c3 = st.columns(3)
    c1.metric("Partitions in this RDD", count)
    c2.metric("Records", f"{sum(sizes):,}")
    c3.metric("Spark default parallelism", sc.defaultParallelism)

    fig = px.bar(x=[f"P{i}" for i in range(len(sizes))], y=sizes, text=sizes,
                 title="Records per partition (measured with mapPartitions)",
                 color_discrete_sequence=[PRIMARY])
    fig.update_xaxes(title="Partition")
    fig.update_yaxes(title="Records")
    st.plotly_chart(style_figure(fig, 300), key="partition_sizes")

    details = pd.DataFrame(engine.partition_details(results, preview=4))
    if not details.empty:
        details["sample_ids"] = details["sample_ids"].map(", ".join)
        st.dataframe(details.rename(columns={"partition": "Partition", "records": "Records",
                                             "sample_ids": "First student IDs"}),
                     hide_index=True, key="partition_details")
    if any(size == 0 for size in sizes):
        st.warning("Some partitions are empty: there are more partitions than records (or the data is uneven).")

    st.subheader("Try a different partition count")
    st.write("This repartitions a temporary copy with `repartition(n)`; your dataset is not changed.")
    choice = st.selectbox("Partitions", [1, 2, 3, 4, 6, 8, 12, 16], index=3)
    temporary = engine.set_partition_count(results, choice)
    temp_sizes = engine.partition_sizes(temporary)
    fig = px.bar(x=[f"P{i}" for i in range(len(temp_sizes))], y=temp_sizes, text=temp_sizes,
                 title=f"After repartition({choice}): actual partitions = {temporary.getNumPartitions()}",
                 color_discrete_sequence=[TEAL])
    fig.update_xaxes(title="Partition")
    fig.update_yaxes(title="Records")
    st.plotly_chart(style_figure(fig, 300), key="partition_preview")

    with st.expander("How data is divided and why it matters", expanded=True):
        st.markdown(
            """
- An RDD is a collection split into **partitions**. Each partition is processed by one **task**, and tasks can run in parallel.
- With `local[*]`, Spark uses the CPU cores of this computer. More cores can process more partitions at once.
- `repartition(n)` shuffles records into exactly `n` partitions. `reduceByKey` also shuffles, grouping equal keys together.
- Too few partitions leave cores idle. Too many add scheduling overhead and can leave partitions empty or tiny.
- If a partition is lost, Spark recomputes it from the RDD's **lineage** (fault tolerance).
            """
        )


# --------------------------------------------------------------------------
# Page: Performance Explorer
# --------------------------------------------------------------------------
def page_performance_explorer() -> None:
    page_header("Performance Explorer",
                "Measure the same Spark workload with different partition counts on this computer.")
    st.warning(
        "Results are measured on your computer during this run and are not stored or pre-written. "
        "Small datasets are dominated by Spark overhead, so more partitions are often NOT faster."
    )

    c1, c2 = st.columns(2)
    sizes = c1.multiselect("Dataset sizes (synthetic students)", list(perf.AVAILABLE_DATASET_SIZES),
                           default=[1_000, 10_000], format_func=lambda n: f"{n:,}")
    partitions = c2.multiselect("Partition counts", [1, 2, 4, 8, 16], default=[1, 2, 4, 8])
    c3, c4 = st.columns(2)
    trials = c3.slider("Timed trials per configuration", 1, 7, 3)
    warmups = c4.slider("Warm-up runs (not timed)", 0, 3, 1)

    if st.button("Run experiment", type="primary"):
        if not sizes or not partitions:
            st.error("Choose at least one dataset size and one partition count.")
        else:
            bar = st.progress(0.0, text="Starting...")
            try:
                report = perf.run_benchmark(
                    sorted(sizes), sorted(partitions), trials=trials, warmup_runs=warmups,
                    progress=lambda fraction, message: bar.progress(fraction, text=message),
                )
                st.session_state["benchmark"] = report
            except (ValueError, perf.PerformanceExperimentError) as error:
                st.error(f"Experiment failed: {error}")
            finally:
                bar.empty()

    report = st.session_state.get("benchmark")
    if report is None:
        empty_state("Choose settings and press 'Run experiment'. Nothing is shown until real measurements exist.")
    else:
        render_benchmark_report(report)

    with st.expander("Methodology"):
        for item in perf.METHODOLOGY:
            st.markdown(f"- {item}")
    with st.expander("Limitations of running Spark on one computer"):
        for item in perf.LIMITATIONS:
            st.markdown(f"- {item}")
    with st.expander("Why don't more partitions always help?"):
        st.markdown(
            """
Partitions let Spark run tasks in parallel, but every task has a fixed cost: it must be scheduled, serialised,
sent to a Python worker, and its result collected. If the real work per task is tiny, that cost is bigger than
the saving. On a laptop, parallelism is also capped by the number of CPU cores, so partitions beyond the core
count only queue up. Larger datasets and heavier computations give parallelism more room to pay off.
            """
        )


def render_benchmark_report(report: perf.BenchmarkReport) -> None:
    if report.all_equivalent:
        st.success("All partition configurations produced equivalent results for every dataset size.")
    else:
        mismatched = [str(size) for size, ok in report.equivalent_by_size.items() if not ok]
        st.error("Outputs differed between configurations for dataset size(s): " + ", ".join(mismatched))

    st.caption(
        f"CPU cores: {report.cpu_count} · Spark master: {report.spark_master} · "
        f"default parallelism: {report.default_parallelism} · seed: {report.seed} · "
        f"{report.trials} trial(s), {report.warmup_runs} warm-up run(s) per configuration"
    )
    rows = pd.DataFrame(report.to_rows())
    chart = rows.assign(
        partitions_label=rows["partitions"].astype(str),
        size_label=rows["dataset_size"].map(lambda n: f"{n:,} records"),
        err_up=rows["max_s"] - rows["median_s"],
        err_down=rows["median_s"] - rows["min_s"],
    )
    fig = px.bar(chart, x="partitions_label", y="median_s", color="size_label", barmode="group",
                 error_y="err_up", error_y_minus="err_down",
                 title="Median time of the measured operation (bars show min to max)",
                 labels={"partitions_label": "Partitions", "median_s": "Seconds", "size_label": "Dataset"},
                 color_discrete_sequence=[PRIMARY, TEAL, AMBER, RED, MUTED])
    st.plotly_chart(style_figure(fig, 380, legend=True), key="perf_chart")

    table = rows.rename(columns={
        "dataset_size": "Records", "partitions": "Requested partitions", "actual_partitions": "Verified partitions",
        "input_prep_s": "Input prep (s)", "median_s": "Median (s)", "min_s": "Min (s)", "max_s": "Max (s)",
        "trials": "Trials", "speedup_vs_fewest_partitions": "Speed vs fewest partitions (x)",
        "outputs_equivalent": "Outputs equivalent",
    })
    st.dataframe(table.round(4), hide_index=True, key="perf_table")

    st.markdown("**What was in this run (honest summary)**")
    for size in sorted(report.equivalent_by_size):
        subset = rows[rows["dataset_size"] == size]
        fastest = subset.loc[subset["median_s"].idxmin()]
        st.markdown(
            f"- {size:,} records: fastest median was **{int(fastest['partitions'])} partition(s)** "
            f"({fastest['median_s']:.4f} s). Differences between close values may just be noise."
        )

    with st.expander("Time breakdown: what was NOT part of the measured operation"):
        st.write(
            f"Spark initialisation: {report.spark_init_seconds:.3f} s "
            + ("(session already running, so this is near zero)." if report.spark_session_reused else "(cold start).")
        )
        generation = pd.DataFrame(
            [{"Records": size, "Data generation (s)": seconds}
             for size, seconds in report.data_generation_seconds.items()]
        )
        st.dataframe(generation.round(4), hide_index=True, key="perf_generation")
        st.caption("Input preparation (sending records to Spark and caching them) is listed per configuration above.")


# --------------------------------------------------------------------------
# Page: Dataset Manager
# --------------------------------------------------------------------------
def save_upload(data: bytes) -> Path:
    UPLOAD_DIR.mkdir(exist_ok=True)
    path = UPLOAD_DIR / f"upload_{int(time.time() * 1000)}.csv"  # safe file name chosen by us
    path.write_bytes(data)
    return path


def page_dataset_manager(sc) -> None:
    page_header("Dataset Manager", "Use the sample data or upload your own CSV. Files are validated by Spark.")
    load = st.session_state["load"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valid students", f"{load.valid_count:,}")
    c2.metric("Rejected rows", f"{load.rejected_count:,}")
    c3.metric("Partitions", load.num_partitions)
    c4.metric("Loaded with", "textFile" if load.load_method == "sc.textFile" else "parallelize")
    st.caption(f"Active: {st.session_state['dataset_label']}")

    st.subheader("Required format")
    st.code(",".join(REQUIRED_COLUMNS) + "\nS001,Aarav Sharma,Computer Science,92,95,98,96", language="text")
    st.markdown(
        f"Marks and attendance must be numbers from 0 to 100. Column order does not matter. "
        f"A student passes only with every mark at least {PASS_MARK:g}. Invalid rows are rejected with a reason, never converted to zero."
    )

    d1, d2 = st.columns(2)
    d1.download_button("Download sample dataset (CSV)", sample_csv_text().encode("utf-8"),
                       file_name="students.csv", mime="text/csv")
    partitions = d2.selectbox("Partitions for the loaded RDD", PARTITION_CHOICES,
                              index=PARTITION_CHOICES.index(DEFAULT_PARTITIONS))

    if st.button("Reload the sample dataset"):
        if not SAMPLE_CSV_PATH.is_file():
            write_sample_csv(SAMPLE_CSV_PATH)
        activate_dataset(engine.load_students_from_csv(sc, SAMPLE_CSV_PATH, partitions),
                         "Sample dataset (data/students.csv)")
        st.success("Sample dataset loaded and the analysis was refreshed.")
        st.rerun()

    st.subheader("Upload your own CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None and st.button("Validate uploaded file"):
        validate_upload(sc, uploaded, partitions)

    pending = st.session_state.get("pending")
    if pending:
        render_pending(pending)


def validate_upload(sc, uploaded, partitions: int) -> None:
    data = uploaded.getvalue()
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError:
        st.error("The file is not UTF-8 text. In Excel use 'Save as > CSV UTF-8 (Comma delimited)'.")
        return
    old = st.session_state.pop("pending", None)
    if old:
        engine.release_rdds(old["load"].records)
    try:
        with st.spinner("Validating with Spark..."):
            load = engine.load_students_from_csv(sc, save_upload(data), num_partitions=partitions)
    except DatasetError as error:
        st.error(f"Cannot use this file: {error}")
        return
    st.session_state["pending"] = {"name": uploaded.name, "load": load}


def render_pending(pending: dict) -> None:
    load = pending["load"]
    st.markdown(f"**Validation result for `{pending['name']}`**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Data rows checked", f"{load.total_data_lines:,}")
    m2.metric("Valid", f"{load.valid_count:,}")
    m3.metric("Rejected", f"{load.rejected_count:,}")

    if load.errors:
        st.warning("Some rows were rejected. Each message names the line number and the reason.")
        st.dataframe(pd.DataFrame({"Problem": load.errors}), hide_index=True, key="pending_errors")
        if load.errors_truncated:
            st.caption(f"Only the first {engine.MAX_ERRORS_SHOWN} problems are listed.")

    if not load.is_usable:
        st.error("No valid student rows were found, so this file cannot be analysed.")
        return

    st.markdown("**Preview (first 10 valid rows)**")
    preview = [engine.to_result(record) for record in load.records.take(10)]
    show_students_table(engine.rows_as_dicts(preview), key="pending_preview")

    label = "Use this dataset" if not load.errors else f"Use the {load.valid_count:,} valid rows"
    if st.button(label, type="primary"):
        activate_dataset(load, f"Uploaded: {pending['name']}")
        st.success("Dataset loaded. Every page now reflects the new data.")
        st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    shell = st.empty()
    shell.title("SparkRDD Lab")
    shell.caption("Student analytics with Apache Spark RDDs")
    tabs = st.tabs(
        PAGES,
        width="stretch",
        default="Overview",
        on_change="rerun",
        key="dashboard_tabs",
    )
    dataset_status = render_sidebar()

    try:
        spark = start_spark()
    except Exception as error:  # Java/PySpark problems are common on first setup
        st.error(
            "Apache Spark could not start. Run `python verify_setup.py` in the terminal to diagnose "
            f"Java/PySpark problems.\n\nDetails: {error}"
        )
        st.stop()
    sc = spark.sparkContext

    if _dataset_is_reusable(sc):
        ensure_dataset(sc)
    else:
        with st.status("Loading the dataset and computing dashboard analytics...", expanded=False) as status:
            try:
                ensure_dataset(sc)
            except (DatasetError, FileNotFoundError) as error:
                status.update(label="The sample dataset could not be loaded.", state="error", expanded=True)
                st.error(f"The sample dataset could not be loaded: {error}")
                st.stop()
            except Exception as error:
                status.update(label="Dashboard data could not be prepared.", state="error", expanded=True)
                st.error(f"Spark could not prepare the dashboard data: {error}")
                st.stop()
            status.update(label="Dashboard data is ready.", state="complete", expanded=False)

    update_sidebar_dataset(dataset_status, sc)
    shell.empty()
    renderers = [
        page_overview,
        page_student_explorer,
        page_academic_analytics,
        lambda: page_rdd_lab(sc),
        lambda: page_partition_explorer(sc),
        page_performance_explorer,
        lambda: page_dataset_manager(sc),
    ]
    for tab, render_page in zip(tabs, renderers):
        if tab.open:
            with tab:
                render_page()


main()

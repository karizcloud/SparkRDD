# SparkRDD Lab: Intelligent Student Performance Analytics Using Apache Spark RDDs

An interactive educational dashboard that teaches **basic Apache Spark RDD operations** through student-performance analysis. All analytics are computed with real PySpark RDDs; Pandas and Plotly are used only to display results.

## Abstract
SparkRDD Lab loads a CSV of student marks and attendance into a Spark RDD, validates every row inside Spark, and answers questions such as average marks, pass/fail status, top performers and department comparisons. A Streamlit interface lets learners see each RDD operation (with the exact PySpark code that ran), inspect partitions, and run a reproducible partition-count timing experiment.

## Problem statement
Spark RDD concepts (lazy transformations, actions, partitions, shuffles) are abstract. Students need a small, runnable, honest example where they can see the code, the result and the effect of partitioning on a familiar dataset.

## Objectives
- Demonstrate 12 RDD operations: `map, filter, flatMap, reduce, reduceByKey, distinct, sortBy, count, take, collect, groupByKey, mapPartitions`.
- Analyse marks, attendance, departments and subjects with real RDD jobs.
- Show partitions and measure the effect of partition counts without fabricated numbers.
- Let users upload their own CSV with clear validation messages.

## Features
- 7 dashboard pages: Overview, Student Explorer, Academic Analytics, RDD Operations Lab, Partition Explorer, Performance Explorer, Dataset Manager.
- Row-level validation with line numbers (marks and attendance must be numbers from 0 to 100; invalid values are rejected, never turned into zero).
- Duplicate `student_id` detection.
- Lazy-evaluation demo using an accumulator.
- Benchmark: deterministic synthetic data, verified partition counts, warm-up, repeated trials, median timing, output-equivalence check.
- No arbitrary code execution from the dashboard: the Lab runs only predefined functions.

## Technology stack
Python, PySpark 3.5.3 (RDD API), Streamlit, Plotly, Pandas (display only), pytest. Data is stored in local CSV files. No database, API, or cloud service.

## Architecture
```
app.py (Streamlit UI)  ->  src/rdd_engine.py  (all RDD processing)
                       ->  src/performance.py (benchmark; reuses rdd_engine)
rdd_engine.py          ->  src/validators.py  (pure-Python row/header validation)
performance.py         ->  src/data_generator.py (sample data + deterministic synthetic data)
```
The engine never imports Streamlit, so it can be tested with pytest.

## Folder structure
```
SparkRDD-Lab/
|-- app.py                  Streamlit dashboard
|-- requirements.txt
|-- pytest.ini              makes `src` importable in tests
|-- verify_setup.py         checks Java + PySpark
|-- .streamlit/config.toml  light theme, 10 MB upload limit
|-- src/
|   |-- __init__.py
|   |-- validators.py       header/row validation, constants (pass mark 40, attendance 75)
|   |-- data_generator.py   25 fictional students + synthetic generator
|   |-- rdd_engine.py       Spark session, loading, analytics, partitions, RDD Lab demos
|   |-- performance.py      partition-count experiment
|-- data/students.csv       sample dataset (25 fictional students)
|-- tests/
|   |-- conftest.py         one shared Spark session for tests
|   |-- test_validators.py
|   |-- test_rdd_engine.py
|   |-- test_performance.py
|-- docs/project_report.md, docs/viva_questions.md
```
Additions compared with the original plan: `tests/conftest.py` (shared Spark fixture), `.streamlit/config.toml` (theme) and `verify_setup.py`.

## Java and Python compatibility
Spark runs on the JVM, so Java must be installed separately; `pip install pyspark` does not include it.

| Component | Recommended | Notes |
|---|---|---|
| PySpark | 3.5.3 (pinned) | The RDD API is the same across 3.x. |
| Java | 17 (Temurin JDK) | The development run of this project used Java 21 with PySpark 3.5.3 on Linux. Other versions: check the official Spark documentation for your PySpark version. |
| Python | 3.10 or 3.11 | The project was also run with Python 3.12 on Linux. Newer versions have not been tried. |

Always confirm the current support matrix on spark.apache.org before choosing versions.

## Installation (Windows, PowerShell)
```powershell
cd SparkRDD-Lab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python verify_setup.py
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
Install the JDK from adoptium.net and tick "Set JAVA_HOME" in the installer. Keep the project in a path **without spaces** (for example `C:\Projects\SparkRDD-Lab`).

## Run the application
```powershell
streamlit run app.py
```
The browser opens at http://localhost:8501. The first start takes several seconds because Spark launches.

## Run the tests
```powershell
pytest
```
Useful variants: `pytest tests\test_validators.py` (fast, no Spark), `pytest -k partition`, `pytest -v`. The Spark tests start a local Spark session, so a full run takes a few minutes. Tests check outputs against hand-calculated values and never assert timing thresholds.

## Dataset format
```
student_id,name,department,maths,science,computer,attendance
S001,Aarav Sharma,Computer Science,92,95,98,96
```
- Header names are case-insensitive; column order does not matter; extra columns are ignored.
- `maths`, `science`, `computer`, `attendance`: plain decimal numbers from 0 to 100 (`nan`, `inf`, `1e2` are rejected).
- Text fields must not be empty. `student_id` must be unique (later duplicates are dropped and reported).
- Demonstration rules (not official policy): pass = all three marks at least 40; low attendance = below 75.
- Quoted fields are supported, but a quoted value that spans several lines is not.
- Files must be UTF-8 (a BOM is accepted).

## RDD operations demonstrated
| Operation | Type | Where used |
|---|---|---|
| `map` | transformation | parse/validate rows, build student results |
| `filter` | transformation | pass/fail, low attendance, search filters |
| `flatMap` | transformation | one (subject, mark) row per subject |
| `reduce` | action | overall summary totals |
| `reduceByKey` | transformation | department/subject statistics, distributions |
| `distinct` | transformation | list of departments |
| `sortBy` | transformation | rankings, sorted tables |
| `count` | action | student counts |
| `take` | action | bounded previews and top-N |
| `collect` | action | only small aggregated results |
| `groupByKey` | transformation | names grouped by department |
| `mapPartitions` | transformation | per-partition record counts |

## Performance experiment methodology
1. Deterministic synthetic students from a fixed seed, identical for every configuration.
2. `sc.parallelize(records, n)`, cached and materialised; the real partition count is read back with `getNumPartitions()` and must equal `n`.
3. Identical workload each time (department stats, subject stats, top 5).
4. Warm-up runs are discarded; several timed trials use `time.perf_counter()`; the **median** is reported.
5. Data generation, Spark initialisation, input preparation and the measured operation are timed separately.
6. All outputs are compared (floats rounded to 6 decimals); the dashboard states whether they were equivalent.

No results are stored in the repository. Run the experiment yourself and report what you measure.

## Known limitations
- Spark runs in local mode on one machine: no real cluster, no network cost.
- Small datasets are dominated by Spark overhead, so more partitions are often not faster and timings are noisy.
- PySpark RDDs run Python in worker processes, which adds serialisation cost.
- Multi-line quoted CSV fields are not supported.
- The dashboard scatter plot and tables are capped (1,000 rows) to keep results bounded.
- Uploaded files are saved to `uploads/` and are not deleted automatically.
- The marks/attendance rules are demonstration rules only.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `JAVA_HOME is not set` / `java not found` | Install JDK 17, enable "Set JAVA_HOME", reopen VS Code, run `java -version`. |
| `Python worker failed to connect back` | Activate the venv; the code already sets `PYSPARK_PYTHON`. Allow Java through Windows Firewall; avoid spaces in the project path. |
| `ModuleNotFoundError: No module named 'src'` | Run `pytest` / `streamlit run app.py` from the project root. |
| Warnings about `winutils` / `native-hadoop` | Normally harmless for local CSV work. |
| `Exception while deleting Spark temp dir` on exit | Known harmless Windows message. |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| Dashboard says Spark could not start | Run `python verify_setup.py` and read its message. |
| Upload rejected as "not UTF-8" | Excel: Save As > CSV UTF-8. |

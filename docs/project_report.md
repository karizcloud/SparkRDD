# SparkRDD Lab: Intelligent Student Performance Analytics Using Apache Spark RDDs

## 1. Title page information

| Field | Value |
|---|---|
| Project title | SparkRDD Lab: Intelligent Student Performance Analytics Using Apache Spark RDDs |
| Topic | Implement Basic RDD Operations in Apache Spark |
| Student name | [YOUR NAME] |
| Roll / enrollment number | [ROLL NUMBER] |
| Department / programme | [DEPARTMENT, PROGRAMME] |
| Guide / supervisor | [GUIDE NAME] |
| Institution | [COLLEGE NAME] |
| Academic year | [YEAR] |
| Date of submission | [DATE] |

## 2. Abstract
SparkRDD Lab is an interactive Streamlit application that teaches Apache Spark RDD programming through student-performance analytics. A CSV of marks and attendance is loaded into a Spark RDD, validated row by row inside Spark, and analysed with twelve core RDD operations. The application also exposes partition information and a reproducible experiment that times the same workload under different partition counts. All figures shown in the application are computed at run time; none are pre-written.

## 3. Introduction
Data-processing frameworks such as Spark divide data into partitions and process them in parallel. Beginners often find concepts like lazy evaluation and shuffles abstract. This project uses a familiar domain, student results, to make them visible.

## 4. Problem statement
Learners need a small, runnable, trustworthy example that shows (a) how RDD transformations and actions behave, (b) how data is partitioned, and (c) that more parallelism does not automatically mean faster execution, while using genuine Spark processing rather than ordinary Python lists.

## 5. Objectives
1. Demonstrate `map, filter, flatMap, reduce, reduceByKey, distinct, sortBy, count, take, collect, groupByKey, mapPartitions` with real PySpark RDDs.
2. Analyse marks, attendance, departments and subjects.
3. Validate CSV input with clear, line-numbered error messages.
4. Explore partitions using real RDD information.
5. Run an honest, reproducible partition-count timing experiment.
6. Provide automated tests and documentation.

## 6. Hardware and software requirements
**Hardware:** a Windows computer with at least 8 GB RAM recommended (record your actual specification: [CPU MODEL, CORES, RAM]).
**Software:** Windows [VERSION]; Python 3.10 or 3.11; Java JDK 17; PySpark 3.5.3; Streamlit; Plotly; Pandas; pytest; Visual Studio Code. Record the exact versions you used: [PYTHON VERSION], [JAVA VERSION], [PYSPARK VERSION].

## 7. Theoretical background of RDDs
An RDD (Resilient Distributed Dataset) is Spark's fundamental data abstraction: an immutable, partitioned collection that can be processed in parallel. It is **resilient** because lost partitions can be recomputed from lineage; **distributed** because partitions can live on different executors; and **immutable** because each transformation creates a new RDD. RDDs may be cached in memory to avoid recomputation. In local mode all executors run on one machine.

## 8. Transformations and actions
- **Transformations** build a new RDD lazily: `map, filter, flatMap, distinct, sortBy, reduceByKey, groupByKey, mapPartitions`. *Narrow* transformations (`map`, `filter`, `flatMap`, `mapPartitions`) work inside one partition. *Wide* transformations (`reduceByKey`, `groupByKey`, `distinct`, `sortBy`) need a shuffle.
- **Actions** trigger execution and return values: `reduce, count, take, collect`.
- **Lazy evaluation:** Spark records a plan and executes it only when an action runs. The application demonstrates this with an accumulator that counts processed rows before and after actions.
- `reduceByKey` is preferred to `groupByKey` for aggregation because values are combined inside each partition before shuffling.

## 9. System architecture
```
Streamlit UI (app.py)
   |-- rdd_engine.py  : Spark session, loading, validation pipeline, analytics, partitions, RDD Lab demos
   |-- performance.py : benchmark (uses rdd_engine analytics)
rdd_engine.py -> validators.py (pure-Python row checks), data_generator.py (sample + synthetic data)
Data: local CSV files (data/students.csv, uploaded files in uploads/)
```
[INSERT ARCHITECTURE DIAGRAM IF REQUIRED BY YOUR COLLEGE]

## 10. Methodology
1. Load text lines with `sc.textFile` (with a `sc.parallelize` fallback if the path cannot be read).
2. Read the header; verify required columns; build a column layout.
3. `zipWithIndex` attaches line numbers; `map` validates each row and tags it `ok` or `error`.
4. Detect duplicate `student_id` values with `reduceByKey`; keep the first occurrence.
5. Convert valid records to `StudentResult` rows (total, average, pass flag, low-attendance flag) with `map`.
6. Compute analytics with RDD transformations and bounded actions.
7. Present results with Streamlit, Plotly and Pandas (display only).

Business rules (demonstration only, not official policy): marks and attendance are numbers from 0 to 100; a student passes only if all three marks are at least 40; attendance below 75 is flagged.

## 11. Implementation details
| File | Responsibility |
|---|---|
| `src/validators.py` | Header and row validation, shared constants, exception types |
| `src/data_generator.py` | 25 fictional students; deterministic synthetic generator (seeded) |
| `src/rdd_engine.py` | Spark session; `load_students_from_csv`; analytics (summary, department/subject statistics, top students, distributions); partition tools; RDD Lab demos; lazy-evaluation demo |
| `src/performance.py` | Benchmark: input preparation, partition verification, warm-up, trials, median, equivalence check |
| `app.py` | Seven dashboard pages |

Design decisions: results sent to the driver are always bounded (`take`, or one row per group); cached RDDs are released with `unpersist()` when the dataset changes; Spark workers are given the same Python and the project path through environment variables.

## 12. Testing strategy
pytest tests use small hand-calculated datasets with known answers (for example six students whose totals, averages and pass status can be checked by hand) and verify:
- header validation, missing columns, invalid marks, invalid attendance, malformed rows, empty datasets;
- pass/fail classification (including the boundary values 40 and 75);
- average marks, subject averages, department aggregation, top students, distinct departments, sorting;
- partition counts and equivalence of results across partition counts;
- each RDD Lab demo result;
- the benchmark: requested = actual partitions, equivalent outputs, argument validation.

Tests never assert timing thresholds. Run with `pytest` from the project root.
Record your own test run here: [PASTE YOUR `pytest` SUMMARY LINE AND DATE].

## 13. Results and discussion
Do not copy numbers from anywhere else. Fill this section from your own runs.

**13.1 Analytics results.** Open the Overview page with the sample dataset and record the metrics: total students [ ], overall average [ ], pass percentage [ ], average attendance [ ]. Discuss the department comparison and the students needing support.

**13.2 Partition behaviour.** Record the partition count and sizes shown in the Partition Explorer: [ ]. Discuss how `repartition` changed the distribution.

**13.3 Performance experiment.** Record your settings (dataset sizes, partition counts, trials), your CPU core count, the median times, and whether outputs were equivalent: [TABLE FROM YOUR RUN]. Discuss honestly whether additional partitions helped, referring to Spark overhead, the number of CPU cores and noise. If results are inconsistent between runs, say so.

## 14. Screenshot placeholders
- Figure 1: Overview page [SCREENSHOT]
- Figure 2: Student Explorer [SCREENSHOT]
- Figure 3: Academic Analytics [SCREENSHOT]
- Figure 4: RDD Operations Lab (one operation and the lazy-evaluation demo) [SCREENSHOT]
- Figure 5: Partition Explorer [SCREENSHOT]
- Figure 6: Performance Explorer after a run [SCREENSHOT]
- Figure 7: Dataset Manager with validation errors [SCREENSHOT]
- Figure 8: pytest output [SCREENSHOT]

## 15. Limitations
- Spark runs in local mode on one computer; no network or cluster effects.
- Small datasets are dominated by overhead; timings are noisy and depend on other running programs.
- PySpark RDDs execute Python in worker processes with serialisation cost; DataFrames would normally be faster.
- CSV fields spanning multiple lines are not supported.
- Marks and attendance rules are demonstration rules.
- Only synthetic data was used for benchmarking.

## 16. Future scope
- A DataFrame/Spark SQL version for comparison with the RDD version.
- Running on a real cluster (e.g. Databricks or a university cluster).
- More statistics (percentiles, correlation of attendance and marks).
- Per-subject pass thresholds and grade bands.
- Exporting reports as PDF.

## 17. Conclusion
The project shows how Spark RDD transformations and actions can be applied to a realistic analytics problem, with real validation, real partition information and honest timing experiments. [ADD YOUR OWN CONCLUSIONS BASED ON YOUR MEASURED RESULTS.]

## 18. References
1. Apache Spark Documentation, "RDD Programming Guide", spark.apache.org/docs/latest/rdd-programming-guide.html
2. Apache Spark Documentation, "PySpark API Reference", spark.apache.org/docs/latest/api/python/
3. M. Zaharia et al., "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing", NSDI 2012.
4. Streamlit Documentation, docs.streamlit.io
5. Plotly Python Documentation, plotly.com/python/
6. pytest Documentation, docs.pytest.org
7. [ADD ANY ADDITIONAL SOURCES YOU USED]

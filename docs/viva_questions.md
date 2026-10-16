# Viva Questions and Answers: SparkRDD Lab

## Spark and RDD basics

**1. What is Apache Spark?**
An open-source engine for processing large amounts of data in parallel. It keeps data in memory where possible, which makes it much faster than disk-based MapReduce for many jobs.

**2. What is an RDD?**
A Resilient Distributed Dataset: an immutable collection of records split into partitions that can be processed in parallel. "Resilient" means lost partitions can be rebuilt from their lineage.

**3. Why did this project use RDDs instead of DataFrames?**
The topic is basic RDD operations. RDDs expose `map`, `filter`, `reduceByKey` and partitions directly, which makes the concepts easy to see. DataFrames are usually better for real projects because Spark can optimise them.

**4. What is the difference between a transformation and an action?**
A transformation (e.g. `map`) creates a new RDD and runs lazily. An action (e.g. `count`) returns a value to Python and triggers the computation.

**5. What is lazy evaluation and where does the project show it?**
Spark records the transformations as a plan and only runs it when an action is called. The Lab's lazy-evaluation demo uses an accumulator: it shows 0 rows processed after the transformations, then 8 after `collect()`.

**6. Why did the second action in the lazy demo process the data again?**
The RDD was not cached, so each action recomputes the lineage from the start. Caching avoids this.

**7. What is lineage and how does it give fault tolerance?**
Lineage is the recorded chain of transformations that produced an RDD. If a partition is lost, Spark replays only the needed steps to rebuild it instead of keeping copies.

**8. What is the difference between narrow and wide transformations?**
Narrow transformations (`map`, `filter`) work within one partition. Wide ones (`reduceByKey`, `groupByKey`, `distinct`, `sortBy`) need a shuffle, moving data between partitions.

**9. What is a shuffle and why is it expensive?**
Redistributing records across partitions by key. It involves serialisation, disk and network, so it is usually the slowest part of a job.

## Operations used

**10. Explain `map` versus `flatMap` with an example from the project.**
`map` gives one output per input. `flatMap` can give many: each student row becomes three (subject, mark) rows, so 8 students become 24 rows.

**11. Why is `reduceByKey` preferred over `groupByKey`?**
`reduceByKey` combines values inside each partition before the shuffle, so far less data moves. `groupByKey` sends every value across the network. The project uses `groupByKey` only where the actual list of names is needed.

**12. How is the department average computed correctly?**
Each student becomes `(department, (average, 1))`. `reduceByKey` adds the sums and counts, then the average is `sum / count`. Averaging averages of groups of different sizes would give a wrong result.

**13. What does `reduce` do and what happens on an empty RDD?**
It combines all elements into one value with an associative, commutative function. On an empty RDD PySpark raises `ValueError`; the project catches this and returns zeros.

**14. Why is `collect()` dangerous and how does the project handle it?**
It brings all data into the driver's memory. The project calls `collect()` only on small aggregated results (e.g. three subject rows) and uses `take(n)` for everything else.

**15. Is `sortBy` a transformation or an action?**
A transformation. However, Spark may run a small sampling job to choose range boundaries, so you can see a job even before an action.

**16. What does `distinct` do and why does it need a shuffle?**
It removes duplicates. Equal values can be in different partitions, so they must be brought together.

**17. What does `mapPartitions` do and how is it different from `map`?**
It calls your function once per partition with an iterator over that partition's rows. The project uses it to count records per partition. It is useful for per-partition setup and to inspect partitions.

## Partitions and parallelism

**18. What is a partition?**
A slice of the RDD. Each partition is processed by one task.

**19. How do you find the number of partitions?**
`rdd.getNumPartitions()`. The Partition Explorer shows this value and the sizes found with `mapPartitions`.

**20. How does the project make sure the requested partition count is real?**
`prepare_input` creates the RDD with `parallelize(records, n)` and then compares `getNumPartitions()` with `n`; it raises an error if they differ.

**21. What is the difference between `repartition` and `coalesce`?**
`repartition(n)` always shuffles and can increase or decrease partitions. `coalesce(n)` avoids a full shuffle and is mainly used to reduce partitions. The project uses `repartition` because it must reach an exact count in both directions.

**22. What is parallelism and what limits it in this project?**
Running several tasks at once. In local mode it is limited by the CPU cores of one computer, so partitions beyond the core count simply queue.

**23. Why can a partition be empty?**
If there are more partitions than records, or data is unevenly distributed. The Partition Explorer warns about this.

## Caching and performance

**24. What does `cache()` do and why does the project release RDDs?**
It keeps computed partitions in memory for reuse. `unpersist()` frees that memory when the data is no longer needed, e.g. when the dataset changes.

**25. How does the project measure performance honestly?**
It uses `time.perf_counter()`, a fixed random seed, the same workload for every configuration, warm-up runs, several trials and the median. It verifies partition counts and compares outputs. Nothing is hard-coded.

**26. Why are more partitions not always faster?**
Every task has a fixed overhead (scheduling, serialisation, Python worker communication). With small data or limited cores the overhead exceeds the benefit.

**27. Why is the median used instead of the mean?**
A single slow trial (caused by garbage collection or other programs) can distort the mean. The median is less affected.

**28. Why are warm-up runs used?**
The first run includes one-off costs such as starting Python workers and loading code. Excluding it makes later timings more comparable.

**29. Why is input preparation timed separately?**
Sending data to Spark is a different cost from computing on it. Mixing them would hide what the experiment is about.

## Implementation choices

**30. How is validation done at scale?**
`validators.py` checks a single row in pure Python and is applied inside an RDD `map`, so Spark validates all rows in parallel. Invalid rows keep their line number and reason.

**31. Why are invalid values rejected instead of set to zero?**
Zero would silently change averages and pass/fail results. Rejecting with a message keeps the statistics trustworthy.

**32. Why is the engine separate from the Streamlit app?**
So the logic can be tested without a UI and the interface can change without touching the calculations.

**33. How does the RDD Lab show "the actual code"?**
Each demo is a normal Python function; the dashboard runs it and displays `inspect.getsource()` of the same function, so the code shown is the code executed. Users cannot type their own code.

**34. What is a SparkSession and SparkContext?**
`SparkSession` is the main entry point; `SparkContext` (from `spark.sparkContext`) is used to create RDDs. Only one context can be active per program.

**35. What are the limitations of this project?**
It runs Spark in local mode on one computer, uses small data, and the pass/attendance rules are demonstration rules. Results from the performance experiment cannot be generalised to a real cluster.

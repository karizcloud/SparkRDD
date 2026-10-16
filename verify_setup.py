"""
verify_setup.py
Checks that Python, Java and PySpark work together on this computer.

Run from the project root with the virtual environment active:
    python verify_setup.py
"""

import os
import re
import shutil
import subprocess
import sys

# These must be set BEFORE PySpark starts, so that Spark's Python workers
# use the same Python as this virtual environment (important on Windows).
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def check_python():
    print("[1/4] Python")
    print(f"      Version : {sys.version.split()[0]}")
    print(f"      Path    : {sys.executable}")
    if sys.prefix == sys.base_prefix:
        print("      WARNING : You do not appear to be inside a virtual environment.")
    if sys.version_info < (3, 9) or sys.version_info >= (3, 13):
        print("      WARNING : Python 3.10 or 3.11 is recommended for this project.")
    print()


def check_java():
    print("[2/4] Java")
    java_home = os.environ.get("JAVA_HOME")
    print(f"      JAVA_HOME : {java_home if java_home else '(not set)'}")

    java_path = shutil.which("java")
    if java_path is None:
        print("      ERROR : 'java' was not found on PATH.")
        print("      Fix   : Install Temurin JDK 17 (adoptium.net), enable")
        print("              'Set JAVA_HOME' in the installer, then reopen VS Code.")
        return False

    print(f"      java.exe  : {java_path}")
    result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    # Java prints its version information on stderr.
    version_text = (result.stderr or result.stdout).strip()
    first_line = version_text.splitlines()[0] if version_text else "(no output)"
    print(f"      Output    : {first_line}")

    match = re.search(r'version "(\d+)(?:\.(\d+))?', version_text)
    if match:
        major = int(match.group(1))
        if major == 1:  # old style "1.8.0_xxx" means Java 8
            major = int(match.group(2))
        print(f"      Major ver : {major}")
        if major != 17:
            print("      NOTE      : Java 17 is the recommended version for this project.")
            print("                  Other versions may work but are not what we planned for.")
    print()
    return True


def check_pyspark_and_run_job():
    print("[3/4] PySpark import")
    try:
        import pyspark
        from pyspark.sql import SparkSession
    except ImportError as error:
        print(f"      ERROR : Could not import pyspark ({error}).")
        print("      Fix   : Activate the virtual environment, then run:")
        print("              pip install -r requirements.txt")
        return False
    print(f"      PySpark version : {pyspark.__version__}")
    print()

    print("[4/4] Running a small Spark RDD job (first start can take 10-30 seconds)")
    spark = None
    try:
        spark = (
            SparkSession.builder.appName("SparkRDD-Lab-Verify")
            .master("local[2]")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )
        sc = spark.sparkContext
        sc.setLogLevel("ERROR")

        numbers = sc.parallelize(range(1, 11), 4)

        partitions = numbers.getNumPartitions()
        sum_of_squares = numbers.map(lambda x: x * x).reduce(lambda a, b: a + b)
        even_count = numbers.filter(lambda x: x % 2 == 0).count()
        parity_sums = sorted(
            numbers.map(lambda x: ("even" if x % 2 == 0 else "odd", x))
            .reduceByKey(lambda a, b: a + b)
            .collect()
        )

        print(f"      Spark master          : {sc.master}")
        print(f"      Partitions            : {partitions}")
        print(f"      Sum of squares 1..10  : {sum_of_squares}")
        print(f"      Count of even numbers : {even_count}")
        print(f"      Sums by parity        : {parity_sums}")
        print()

        expected_ok = (
            partitions == 4
            and sum_of_squares == 385
            and even_count == 5
            and parity_sums == [("even", 30), ("odd", 25)]
        )
        if expected_ok:
            print("SUCCESS: Java, Python and PySpark are working together.")
        else:
            print("WARNING: Spark ran, but the results differ from the expected values.")
            print("         Expected: partitions=4, sum=385, evens=5,")
            print("         parity=[('even', 30), ('odd', 25)]")
        return expected_ok
    except Exception as error:  # noqa: BLE001 - we want a friendly message here
        print(f"      ERROR : Spark failed to run: {error}")
        print()
        print("Common fixes:")
        print("  - 'JAVA_HOME is not set' / 'java not found': install JDK 17 and reopen VS Code.")
        print("  - 'Python worker failed to connect back': make sure the venv is active,")
        print("    and allow Java through Windows Firewall if prompted.")
        print("  - Path problems: move the project to a folder without spaces, e.g. C:\\Projects.")
        return False
    finally:
        if spark is not None:
            spark.stop()


def main():
    print("=" * 60)
    print("SparkRDD Lab - environment verification")
    print("=" * 60)
    print()
    check_python()
    java_ok = check_java()
    if not java_ok:
        sys.exit(1)
    spark_ok = check_pyspark_and_run_job()
    sys.exit(0 if spark_ok else 1)


if __name__ == "__main__":
    main()

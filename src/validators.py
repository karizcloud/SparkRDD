"""
validators.py
=============
Pure-Python validation for the student CSV format.

Nothing in this file needs Spark, so it is easy to unit-test.
The Spark engine (rdd_engine.py) calls these functions *inside* RDD
transformations, which means validation of a big file is still done by Spark.

Expected CSV header (column order does not matter, extra columns are ignored):

    student_id,name,department,maths,science,computer,attendance

Demonstration rules (NOT official college policy):
    * a mark/attendance value must be a number from 0 to 100
    * a student passes only if ALL three marks are >= PASS_MARK (40)
    * attendance below ATTENDANCE_THRESHOLD (75) is flagged
"""

import csv
import re
from typing import Dict, NamedTuple, Optional, Tuple

# --------------------------------------------------------------------------
# Constants shared by the whole project
# --------------------------------------------------------------------------
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "student_id",
    "name",
    "department",
    "maths",
    "science",
    "computer",
    "attendance",
)
SUBJECTS: Tuple[str, ...] = ("maths", "science", "computer")
PASS_MARK: float = 40.0
ATTENDANCE_THRESHOLD: float = 75.0
MIN_VALUE: float = 0.0
MAX_VALUE: float = 100.0
MAX_TEXT_LENGTH: int = 100

# Plain decimal numbers only: "75", "75.5", ".5", "-3".
# This deliberately rejects "nan", "inf", "1e2" and "1_000".
_NUMBER_PATTERN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
class StudentRecord(NamedTuple):
    """One validated row of the CSV file."""

    student_id: str
    name: str
    department: str
    maths: float
    science: float
    computer: float
    attendance: float


class ColumnLayout(NamedTuple):
    """Where each required column sits in the CSV, and how wide the file is."""

    index: Dict[str, int]
    width: int


# --------------------------------------------------------------------------
# Exceptions (each gives a clear message to show the user)
# --------------------------------------------------------------------------
class DatasetError(Exception):
    """Base class for all dataset problems."""


class EmptyDatasetError(DatasetError):
    """The file has no header or no student rows."""


class MissingColumnsError(DatasetError):
    """One or more required columns are missing from the header."""

    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__(
            "Missing required column(s): "
            + ", ".join(self.missing)
            + ". Required header: "
            + ",".join(REQUIRED_COLUMNS)
        )


class RowValidationError(DatasetError):
    """A single data row is invalid (the message explains why)."""


# --------------------------------------------------------------------------
# Header validation
# --------------------------------------------------------------------------
def parse_header(header_line: str) -> ColumnLayout:
    """Validate the header line and return the column layout.

    Raises
    ------
    EmptyDatasetError     if the header line is blank.
    MissingColumnsError   if a required column is absent.
    DatasetError          if the header contains a duplicated column name.
    """
    if header_line is None or not header_line.strip():
        raise EmptyDatasetError("The file is empty: no header line was found.")

    # Excel sometimes saves a hidden BOM character at the start of the file.
    cleaned = header_line.lstrip("\ufeff")
    try:
        fields = next(csv.reader([cleaned]))
    except (csv.Error, StopIteration) as error:
        raise DatasetError(f"The header line could not be read as CSV: {error}")

    names = [field.strip().lower() for field in fields]

    seen = set()
    for column_name in names:
        if column_name in seen:
            raise DatasetError(f"Duplicate column name in header: '{column_name}'.")
        seen.add(column_name)

    missing = [column for column in REQUIRED_COLUMNS if column not in names]
    if missing:
        raise MissingColumnsError(missing)

    index = {column: names.index(column) for column in REQUIRED_COLUMNS}
    return ColumnLayout(index=index, width=len(names))


# --------------------------------------------------------------------------
# Field validation
# --------------------------------------------------------------------------
def validate_number(raw_value: str, field_name: str) -> float:
    """Convert text to a float between 0 and 100, or raise RowValidationError."""
    text = (raw_value or "").strip()
    if text == "":
        raise RowValidationError(f"{field_name} is empty (a number from 0 to 100 is required)")
    if not _NUMBER_PATTERN.match(text):
        raise RowValidationError(
            f"{field_name} must be a number from 0 to 100, but got '{text}'"
        )
    value = float(text)
    if value < MIN_VALUE or value > MAX_VALUE:
        raise RowValidationError(
            f"{field_name} must be between {MIN_VALUE:g} and {MAX_VALUE:g}, but got {text}"
        )
    return value


def validate_text(raw_value: str, field_name: str) -> str:
    """Return trimmed text, or raise RowValidationError if empty or too long."""
    text = " ".join((raw_value or "").split())  # trims and collapses spaces
    if text == "":
        raise RowValidationError(f"{field_name} is empty")
    if len(text) > MAX_TEXT_LENGTH:
        raise RowValidationError(
            f"{field_name} is longer than {MAX_TEXT_LENGTH} characters"
        )
    return text


# --------------------------------------------------------------------------
# Row validation
# --------------------------------------------------------------------------
def parse_student_row(line: str, layout: ColumnLayout) -> StudentRecord:
    """Validate one CSV data line and return a StudentRecord.

    Raises RowValidationError with a human-readable reason on any problem.
    The line number is NOT included here; the caller adds it.
    """
    try:
        rows = list(csv.reader([line]))
    except csv.Error as error:
        raise RowValidationError(f"malformed CSV row ({error})")

    if len(rows) != 1:
        raise RowValidationError("malformed CSV row (unexpected line structure)")
    fields = rows[0]

    if len(fields) != layout.width:
        raise RowValidationError(
            f"expected {layout.width} fields but found {len(fields)}"
        )

    def field(column: str) -> str:
        return fields[layout.index[column]]

    return StudentRecord(
        student_id=validate_text(field("student_id"), "student_id"),
        name=validate_text(field("name"), "name"),
        department=validate_text(field("department"), "department"),
        maths=validate_number(field("maths"), "maths"),
        science=validate_number(field("science"), "science"),
        computer=validate_number(field("computer"), "computer"),
        attendance=validate_number(field("attendance"), "attendance"),
    )


def try_parse_student_row(
    line: str, layout: ColumnLayout
) -> Tuple[Optional[StudentRecord], Optional[str]]:
    """Like parse_student_row, but returns (record, None) or (None, message)."""
    try:
        return parse_student_row(line, layout), None
    except RowValidationError as error:
        return None, str(error)

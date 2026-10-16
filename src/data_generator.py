"""
data_generator.py
=================
Creates data for the project:

1. SAMPLE_STUDENTS - 25 fictional students used for data/students.csv.
2. generate_synthetic_records() - deterministic fake students for the
   performance experiment (same seed -> exactly the same records).

All names are fictional.
"""

import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

from src.validators import REQUIRED_COLUMNS, StudentRecord

# --------------------------------------------------------------------------
# Fixed sample dataset
# (student_id, name, department, maths, science, computer, attendance)
# It contains high, average and low performers, marks below 40,
# attendance below 75, and borderline cases (a mark of exactly 40,
# attendance of exactly 75).
# --------------------------------------------------------------------------
SAMPLE_STUDENTS: Tuple[Tuple, ...] = (
    ("S001", "Aarav Sharma", "Computer Science", 92, 95, 98, 96),
    ("S002", "Diya Patel", "Computer Science", 85, 88, 91, 92),
    ("S003", "Rohan Mehta", "Computer Science", 72, 68, 75, 85),
    ("S004", "Ananya Iyer", "Computer Science", 58, 62, 66, 78),
    ("S005", "Kabir Singh", "Computer Science", 35, 48, 52, 70),
    ("S006", "Ishita Nair", "Information Technology", 88, 84, 93, 94),
    ("S007", "Vihaan Gupta", "Information Technology", 67, 71, 64, 80),
    ("S008", "Meera Joshi", "Information Technology", 54, 49, 58, 73),
    ("S009", "Arjun Reddy", "Information Technology", 45, 38, 61, 68),
    ("S010", "Saanvi Kapoor", "Information Technology", 78, 82, 80, 89),
    ("S011", "Aditya Verma", "Electronics", 81, 76, 69, 90),
    ("S012", "Navya Menon", "Electronics", 63, 59, 70, 82),
    ("S013", "Krishna Das", "Electronics", 41, 44, 40, 76),
    ("S014", "Tara Bose", "Electronics", 29, 33, 42, 62),
    ("S015", "Yash Malhotra", "Electronics", 74, 79, 83, 87),
    ("S016", "Pooja Rao", "Mechanical", 69, 65, 57, 84),
    ("S017", "Rahul Choudhury", "Mechanical", 52, 47, 44, 74),
    ("S018", "Sneha Pillai", "Mechanical", 38, 42, 45, 71),
    ("S019", "Manav Khanna", "Mechanical", 76, 70, 72, 91),
    ("S020", "Nisha Agarwal", "Mechanical", 61, 55, 60, 79),
    ("S021", "Devansh Kulkarni", "Civil", 90, 86, 82, 97),
    ("S022", "Riya Banerjee", "Civil", 66, 73, 68, 88),
    ("S023", "Harsh Vardhan", "Civil", 47, 51, 43, 75),
    ("S024", "Kavya Naidu", "Civil", 33, 36, 39, 58),
    ("S025", "Imran Qureshi", "Civil", 80, 77, 85, 93),
)

SYNTHETIC_DEPARTMENTS: Tuple[str, ...] = (
    "Computer Science",
    "Information Technology",
    "Electronics",
    "Mechanical",
    "Civil",
)

SAMPLE_CSV_RELATIVE_PATH = Path("data") / "students.csv"


# --------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------
def _format_number(value: Union[int, float]) -> str:
    """90.0 -> '90', 87.5 -> '87.5' (keeps CSV files tidy)."""
    return f"{value:g}"


def rows_to_csv_text(rows: Iterable[Sequence]) -> str:
    """Convert rows (tuples in REQUIRED_COLUMNS order) to CSV text.

    Names in the project never contain commas or quotes, so a plain join is
    safe here.
    """
    lines = [",".join(REQUIRED_COLUMNS)]
    for row in rows:
        lines.append(
            ",".join(
                _format_number(item) if isinstance(item, (int, float)) else str(item)
                for item in row
            )
        )
    return "\n".join(lines) + "\n"


def sample_csv_text() -> str:
    """The sample dataset as CSV text (used by the download button)."""
    return rows_to_csv_text(SAMPLE_STUDENTS)


def write_sample_csv(path: Union[str, Path]) -> Path:
    """Write the sample dataset to `path` (creating folders if needed)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(sample_csv_text(), encoding="utf-8", newline="\n")
    return target


# --------------------------------------------------------------------------
# Deterministic synthetic data for the performance experiment
# --------------------------------------------------------------------------
def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def generate_synthetic_records(
    num_records: int, seed: int = 42
) -> List[StudentRecord]:
    """Create `num_records` fake but valid students.

    The same (num_records, seed) always gives exactly the same list, which is
    what makes the performance experiment reproducible.

    Raises ValueError if num_records is negative.
    """
    if num_records < 0:
        raise ValueError("num_records must be 0 or more")

    generator = random.Random(seed)  # private generator: no global state used
    records: List[StudentRecord] = []
    for number in range(1, num_records + 1):
        ability = generator.gauss(65, 15)  # overall ability of this student
        marks = [
            round(_clamp(ability + generator.gauss(0, 10)), 1) for _ in range(3)
        ]
        attendance = round(_clamp(generator.gauss(82, 12), 30.0, 100.0), 1)
        records.append(
            StudentRecord(
                student_id=f"G{number:07d}",
                name=f"Student {number:07d}",
                department=generator.choice(SYNTHETIC_DEPARTMENTS),
                maths=marks[0],
                science=marks[1],
                computer=marks[2],
                attendance=attendance,
            )
        )
    return records

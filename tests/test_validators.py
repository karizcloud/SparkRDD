"""Tests for validators.py and data_generator.py (no Spark needed)."""

from pathlib import Path

import pytest

from src.data_generator import (
    SAMPLE_STUDENTS,
    generate_synthetic_records,
    sample_csv_text,
    write_sample_csv,
)
from src.validators import (
    REQUIRED_COLUMNS,
    DatasetError,
    EmptyDatasetError,
    MissingColumnsError,
    RowValidationError,
    StudentRecord,
    parse_header,
    parse_student_row,
    try_parse_student_row,
    validate_number,
)

HEADER = "student_id,name,department,maths,science,computer,attendance"
LAYOUT = parse_header(HEADER)


# ----------------------------- header tests -----------------------------
def test_valid_header_gives_expected_layout():
    assert LAYOUT.width == 7
    assert LAYOUT.index["student_id"] == 0
    assert LAYOUT.index["attendance"] == 6


def test_header_columns_can_be_in_any_order_and_case():
    layout = parse_header("Attendance, NAME ,department,student_id,maths,science,computer")
    assert layout.index["attendance"] == 0
    assert layout.index["name"] == 1
    assert layout.index["student_id"] == 3


def test_header_with_bom_is_accepted():
    assert parse_header("\ufeff" + HEADER).width == 7


def test_header_with_extra_column_is_accepted():
    layout = parse_header(HEADER + ",notes")
    assert layout.width == 8


def test_missing_columns_are_reported_by_name():
    with pytest.raises(MissingColumnsError) as error:
        parse_header("student_id,name,department,maths")
    assert error.value.missing == ["science", "computer", "attendance"]
    assert "science" in str(error.value)


def test_duplicate_column_is_rejected():
    with pytest.raises(DatasetError, match="Duplicate column"):
        parse_header(HEADER + ",maths")


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_header_is_empty_dataset(blank):
    with pytest.raises(EmptyDatasetError):
        parse_header(blank)


# ------------------------------ row tests -------------------------------
def test_valid_row_is_parsed_into_record():
    record = parse_student_row("S1,  Asha   Rao ,CS,90,80.5,70,95", LAYOUT)
    assert record == StudentRecord("S1", "Asha Rao", "CS", 90.0, 80.5, 70.0, 95.0)


def test_boundary_values_0_and_100_are_valid():
    record = parse_student_row("S1,A,CS,0,100,0,100", LAYOUT)
    assert (record.maths, record.science, record.attendance) == (0.0, 100.0, 100.0)


def test_quoted_name_with_comma_is_parsed():
    record = parse_student_row('S1,"Rao, Asha",CS,50,50,50,50', LAYOUT)
    assert record.name == "Rao, Asha"


@pytest.mark.parametrize(
    "line, expected_text",
    [
        ("S1,A,CS,abc,50,50,80", "maths must be a number"),
        ("S1,A,CS,50,xyz,50,80", "science must be a number"),
        ("S1,A,CS,50,50,,80", "computer is empty"),
        ("S1,A,CS,-1,50,50,80", "maths must be between"),
        ("S1,A,CS,50,50,100.5,80", "computer must be between"),
        ("S1,A,CS,50,50,50,101", "attendance must be between"),
        ("S1,A,CS,50,50,50,-0.5", "attendance must be between"),
        ("S1,A,CS,nan,50,50,80", "maths must be a number"),
        ("S1,A,CS,50,inf,50,80", "science must be a number"),
        ("S1,A,CS,1e2,50,50,80", "maths must be a number"),
        ("S1,A,CS,50,50,50,", "attendance is empty"),
        ("S1,,CS,50,50,50,80", "name is empty"),
        (",A,CS,50,50,50,80", "student_id is empty"),
        ("S1,A,   ,50,50,50,80", "department is empty"),
    ],
)
def test_invalid_rows_are_rejected_with_clear_message(line, expected_text):
    with pytest.raises(RowValidationError, match=expected_text):
        parse_student_row(line, LAYOUT)


@pytest.mark.parametrize("line", ["S1,A,CS,50,50,50", "S1,A,CS,50,50,50,80,extra", ""])
def test_wrong_number_of_fields_is_rejected(line):
    with pytest.raises(RowValidationError, match="expected 7 fields"):
        parse_student_row(line, LAYOUT)


def test_try_parse_returns_record_or_message_never_both():
    good, no_problem = try_parse_student_row("S1,A,CS,50,50,50,80", LAYOUT)
    assert isinstance(good, StudentRecord) and no_problem is None
    bad, problem = try_parse_student_row("S1,A,CS,xx,50,50,80", LAYOUT)
    assert bad is None and "maths" in problem


def test_validate_number_does_not_silently_turn_bad_values_into_zero():
    with pytest.raises(RowValidationError):
        validate_number("abc", "maths")
    with pytest.raises(RowValidationError):
        validate_number("", "maths")


# --------------------- sample dataset and generator ----------------------
def test_sample_dataset_has_at_least_20_valid_students():
    assert len(SAMPLE_STUDENTS) >= 20
    ids = [row[0] for row in SAMPLE_STUDENTS]
    assert len(set(ids)) == len(ids)
    for line in sample_csv_text().splitlines()[1:]:
        assert isinstance(parse_student_row(line, LAYOUT), StudentRecord)


def test_sample_dataset_covers_required_situations():
    departments = {row[2] for row in SAMPLE_STUDENTS}
    assert len(departments) >= 3
    assert any(row[6] < 75 for row in SAMPLE_STUDENTS)                 # low attendance
    assert any(min(row[3:6]) < 40 for row in SAMPLE_STUDENTS)          # failing mark
    assert any(sum(row[3:6]) / 3 >= 85 for row in SAMPLE_STUDENTS)     # high performer
    assert any(40 <= sum(row[3:6]) / 3 < 60 for row in SAMPLE_STUDENTS)  # lower-middle


def test_csv_file_in_data_folder_matches_the_generator(tmp_path):
    project_root = Path(__file__).resolve().parent.parent
    on_disk = (project_root / "data" / "students.csv").read_text(encoding="utf-8")
    assert on_disk == sample_csv_text()
    assert sample_csv_text().splitlines()[0] == ",".join(REQUIRED_COLUMNS)

    written = write_sample_csv(tmp_path / "nested" / "copy.csv")
    assert written.read_text(encoding="utf-8") == sample_csv_text()


def test_synthetic_records_are_deterministic():
    assert generate_synthetic_records(50, seed=7) == generate_synthetic_records(50, seed=7)


def test_synthetic_records_change_with_seed():
    assert generate_synthetic_records(50, seed=1) != generate_synthetic_records(50, seed=2)


def test_synthetic_records_are_valid_and_unique():
    records = generate_synthetic_records(300, seed=42)
    assert len(records) == 300
    assert len({r.student_id for r in records}) == 300
    for r in records:
        for value in (r.maths, r.science, r.computer, r.attendance):
            assert 0.0 <= value <= 100.0


def test_synthetic_zero_records_and_negative_count():
    assert generate_synthetic_records(0) == []
    with pytest.raises(ValueError):
        generate_synthetic_records(-1)

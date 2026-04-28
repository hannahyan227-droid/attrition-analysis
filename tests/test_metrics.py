import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


# ---------------------------------------------------------------------------
# attrition_rate
# ---------------------------------------------------------------------------

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_all_stay():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leave():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# ---------------------------------------------------------------------------
# attrition_by_department
# ---------------------------------------------------------------------------

def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_values():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "department": ["Sales", "Sales", "HR", "HR", "HR"],
            "attrition": ["Yes", "No", "Yes", "Yes", "No"],
        }
    )
    result = attrition_by_department(df)
    sales = result[result["department"] == "Sales"].iloc[0]
    hr = result[result["department"] == "HR"].iloc[0]

    assert sales["employees"] == 2
    assert sales["leavers"] == 1
    assert sales["attrition_rate"] == 50.0

    assert hr["employees"] == 3
    assert hr["leavers"] == 2
    assert round(hr["attrition_rate"], 2) == 66.67


def test_attrition_by_department_sorted_descending_by_rate():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["HR", "HR", "Sales", "Sales"],
            "attrition": ["No", "No", "Yes", "Yes"],
        }
    )
    result = attrition_by_department(df)
    assert result.iloc[0]["department"] == "Sales"
    assert result.iloc[1]["department"] == "HR"


# ---------------------------------------------------------------------------
# attrition_by_overtime
# ---------------------------------------------------------------------------

def test_attrition_by_overtime_values():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "overtime": ["Yes", "Yes", "No", "No", "No"],
            "attrition": ["Yes", "No", "No", "No", "No"],
        }
    )
    result = attrition_by_overtime(df)
    ot_yes = result[result["overtime"] == "Yes"].iloc[0]
    ot_no = result[result["overtime"] == "No"].iloc[0]

    assert ot_yes["employees"] == 2
    assert ot_yes["leavers"] == 1
    assert ot_yes["attrition_rate"] == 50.0

    assert ot_no["employees"] == 3
    assert ot_no["leavers"] == 0
    assert ot_no["attrition_rate"] == 0.0


def test_attrition_by_overtime_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "overtime": ["Yes", "No"],
            "attrition": ["Yes", "No"],
        }
    )
    result = attrition_by_overtime(df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


# ---------------------------------------------------------------------------
# average_income_by_attrition
# ---------------------------------------------------------------------------

def test_average_income_by_attrition_values():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "monthly_income": [4000, 6000, 5000, 8000],
            "attrition": ["Yes", "No", "Yes", "No"],
        }
    )
    result = average_income_by_attrition(df)
    yes_row = result[result["attrition"] == "Yes"].iloc[0]
    no_row = result[result["attrition"] == "No"].iloc[0]

    assert yes_row["avg_monthly_income"] == 4500.0   # (4000 + 5000) / 2
    assert no_row["avg_monthly_income"] == 7000.0    # (6000 + 8000) / 2


def test_average_income_by_attrition_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "monthly_income": [5000, 7000],
            "attrition": ["Yes", "No"],
        }
    )
    result = average_income_by_attrition(df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


# ---------------------------------------------------------------------------
# satisfaction_summary
# ---------------------------------------------------------------------------

def test_satisfaction_summary_rate_uses_group_headcount_not_total_leavers():
    # Each group has 2 employees; group 1 has 2 leavers, group 4 has 0.
    # Correct: 100% and 0%. Wrong (old bug): 2/2*100=100% and 0/2*100=0%
    # The bug divided by total leavers (2) instead of group size (2), which
    # would give the same result here — so use unequal group sizes to expose it.
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "job_satisfaction": [1, 1, 4, 4, 4],
            "attrition": ["Yes", "Yes", "No", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    sat1 = result[result["job_satisfaction"] == 1].iloc[0]
    sat4 = result[result["job_satisfaction"] == 4].iloc[0]

    # group 1: 2 leavers / 2 employees = 100%
    assert sat1["attrition_rate"] == 100.0
    # group 4: 0 leavers / 3 employees = 0%
    assert sat4["attrition_rate"] == 0.0


def test_satisfaction_summary_partial_attrition_within_group():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [2, 2, 2, 2],
            "attrition": ["Yes", "Yes", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    assert result.iloc[0]["attrition_rate"] == 50.0


def test_satisfaction_summary_sorted_ascending_by_satisfaction():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [4, 1, 3, 2],
            "attrition": ["No", "Yes", "No", "Yes"],
        }
    )
    result = satisfaction_summary(df)
    assert list(result["job_satisfaction"]) == [1, 2, 3, 4]


def test_satisfaction_summary_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "job_satisfaction": [1, 4],
            "attrition": ["Yes", "No"],
        }
    )
    result = satisfaction_summary(df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]

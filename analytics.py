"""Data loading and statistics helpers for the Titanic dashboard.

Every function here returns plain Python data (dicts / lists / numbers) so it can
be used three ways:
  1. by the Streamlit dashboard to draw charts,
  2. by the Claude chatbot as a "tool" it can call,
  3. by the offline (no API key) chatbot that answers with simple rules.

Keeping the math in one place means the chatbot and the charts can never disagree.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "titanic.csv"

# Human-friendly descriptions of each column (the Kaggle data dictionary).
COLUMN_INFO = {
    "PassengerId": "Row id (1-891), no meaning on its own",
    "Survived": "0 = died, 1 = survived (the target variable)",
    "Pclass": "Ticket class: 1 = Upper, 2 = Middle, 3 = Lower (proxy for wealth)",
    "Name": "Passenger name including title (Mr, Mrs, Miss, Master, ...)",
    "Sex": "male / female",
    "Age": "Age in years (fractional for infants); 177 values missing",
    "SibSp": "Number of siblings + spouses aboard",
    "Parch": "Number of parents + children aboard",
    "Ticket": "Ticket number",
    "Fare": "Ticket price in 1912 British pounds",
    "Cabin": "Cabin number; mostly missing (687 of 891)",
    "Embarked": "Port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton",
    # Derived columns added in load_data()
    "AgeGroup": "Derived: age bucket (Child 0-12, Teen 13-19, Adult 20-39, Middle 40-59, Senior 60+)",
    "FamilySize": "Derived: SibSp + Parch + 1",
    "Title": "Derived: title extracted from Name (Mr, Mrs, Miss, Master, Rare)",
    "Class": "Derived: Pclass as text (1st / 2nd / 3rd)",
    "Port": "Derived: Embarked as a full port name",
}

NUMERIC_COLUMNS = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
CATEGORICAL_COLUMNS = ["Sex", "Class", "Pclass", "Port", "Embarked", "AgeGroup", "Title", "FamilySize", "SibSp", "Parch"]

AGE_BINS = [0, 12, 19, 39, 59, 120]
AGE_LABELS = ["Child (0-12)", "Teen (13-19)", "Adult (20-39)", "Middle-aged (40-59)", "Senior (60+)"]
PORTS = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Read the Kaggle CSV and add a few easy-to-read derived columns."""
    df = pd.read_csv(DATA_PATH)
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)  # ordered categorical
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    title = df["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    df["Class"] = df["Pclass"].map({1: "1st", 2: "2nd", 3: "3rd"})
    df["Port"] = df["Embarked"].map(PORTS)
    return df


def _clean(value):
    """Turn numpy / pandas scalars into JSON-friendly Python values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 3)
    return value


def _check_column(name: str, df: pd.DataFrame) -> str:
    """Case-insensitive column lookup with a helpful error."""
    for col in df.columns:
        if col.lower() == str(name).lower():
            return col
    raise ValueError(f"Unknown column '{name}'. Available columns: {', '.join(df.columns)}")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

_OPS = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
}


def apply_filters(df: pd.DataFrame, filters: list[dict] | None) -> pd.DataFrame:
    """Apply a list of {"column", "op", "value"} filters (no eval, so it is safe).

    op is one of ==, !=, >, >=, <, <=, in. For "in", value is a list.
    """
    if not filters:
        return df
    out = df
    for f in filters:
        col = _check_column(f["column"], out)
        op = f.get("op", "==")
        value = f.get("value")
        series = out[col]
        if op == "in":
            values = value if isinstance(value, list) else [value]
            if pd.api.types.is_numeric_dtype(series):
                values = [float(v) for v in values]
            else:
                values = [str(v).lower() for v in values]
                series = series.astype("string").str.lower()
            out = out[series.isin(values)]
            continue
        if op not in _OPS:
            raise ValueError(f"Unsupported operator '{op}'. Use one of {list(_OPS) + ['in']}.")
        if pd.api.types.is_numeric_dtype(series):
            value = float(value)
        else:
            series = series.astype("string").str.lower()
            value = str(value).lower()
        out = out[_OPS[op](series, value).fillna(False)]
    return out


# ---------------------------------------------------------------------------
# Statistics used by the dashboard AND exposed to the chatbot as tools
# ---------------------------------------------------------------------------

def dataset_overview() -> dict:
    """Shape, columns, missing values and the headline survival rate."""
    df = load_data()
    return {
        "dataset": "Titanic - Machine Learning from Disaster (Kaggle)",
        "rows": int(len(df)),
        "columns": {c: COLUMN_INFO.get(c, "") for c in df.columns},
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "overall_survival_rate_pct": _clean(df["Survived"].mean() * 100),
        "survivors": int(df["Survived"].sum()),
        "deaths": int((1 - df["Survived"]).sum()),
    }


def describe_column(column: str, filters: list[dict] | None = None) -> dict:
    """Descriptive statistics for one column (numeric or categorical)."""
    df = apply_filters(load_data(), filters)
    col = _check_column(column, df)
    s = df[col]
    result = {"column": col, "meaning": COLUMN_INFO.get(col, ""), "rows_considered": int(len(s)),
              "missing": int(s.isna().sum())}
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
        d = s.describe()
        result.update({
            "type": "numeric",
            "mean": _clean(d["mean"]), "median": _clean(s.median()), "std": _clean(d["std"]),
            "min": _clean(d["min"]), "q1": _clean(d["25%"]), "q3": _clean(d["75%"]), "max": _clean(d["max"]),
            "skew": _clean(s.skew()),
        })
    else:
        counts = s.value_counts(dropna=False)
        result.update({
            "type": "categorical",
            "unique_values": int(s.nunique()),
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percent": {str(k): _clean(v / len(s) * 100) for k, v in counts.items()},
        })
    return result


def survival_rate_by(group_by: list[str], filters: list[dict] | None = None) -> list[dict]:
    """Survival rate (%) and passenger count for each combination of the group_by columns."""
    df = apply_filters(load_data(), filters)
    cols = [_check_column(c, df) for c in group_by]
    g = df.groupby(cols, dropna=True, observed=True)["Survived"].agg(["mean", "sum", "count"]).reset_index()
    rows = []
    for _, r in g.iterrows():
        row = {c: _clean(r[c]) for c in cols}
        row.update({"survival_rate_pct": _clean(r["mean"] * 100), "survivors": int(r["sum"]), "passengers": int(r["count"])})
        rows.append(row)
    return rows


def group_stats(metric: str, group_by: list[str] | None = None, agg: str = "mean",
                filters: list[dict] | None = None) -> list[dict] | dict:
    """Aggregate a numeric column (mean/median/sum/min/max/std/count), optionally per group."""
    df = apply_filters(load_data(), filters)
    m = _check_column(metric, df)
    if agg not in {"mean", "median", "sum", "min", "max", "std", "count"}:
        raise ValueError("agg must be one of mean, median, sum, min, max, std, count")
    if not group_by:
        return {"metric": m, "agg": agg, "value": _clean(getattr(df[m], agg)()), "rows_considered": int(len(df))}
    cols = [_check_column(c, df) for c in group_by]
    g = df.groupby(cols, dropna=True, observed=True)[m].agg([agg, "count"]).reset_index()
    return [{**{c: _clean(r[c]) for c in cols}, f"{agg}_{m}": _clean(r[agg]), "rows": int(r["count"])}
            for _, r in g.iterrows()]


def correlation(columns: list[str] | None = None) -> dict:
    """Pearson correlation matrix between numeric columns (defaults to all numeric ones)."""
    df = load_data()
    if columns is None:
        cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
    else:
        cols = [_check_column(c, df) for c in columns]
    corr = df[cols].corr(numeric_only=True)
    return {a: {b: _clean(corr.loc[a, b]) for b in corr.columns} for a in corr.index}


def count_rows(filters: list[dict] | None = None) -> dict:
    """How many passengers match the filters, and how many of them survived."""
    df = apply_filters(load_data(), filters)
    n = len(df)
    return {"passengers": int(n), "survivors": int(df["Survived"].sum()),
            "survival_rate_pct": _clean(df["Survived"].mean() * 100) if n else None}


def sample_rows(filters: list[dict] | None = None, n: int = 5, sort_by: str | None = None,
                ascending: bool = False) -> list[dict]:
    """Return up to n example rows (optionally sorted), e.g. 'the 5 most expensive tickets'."""
    df = apply_filters(load_data(), filters)
    if sort_by:
        df = df.sort_values(_check_column(sort_by, df), ascending=ascending)
    cols = ["Name", "Sex", "Age", "Class", "Fare", "Port", "Survived"]
    return [{c: _clean(v) for c, v in row.items()} for row in df[cols].head(min(int(n), 25)).to_dict("records")]

"""The "Ask the Data" chatbot.

Two engines, same interface (`answer(question, history) -> str`):

* ClaudeEngine  - uses the Anthropic API with *tool use*. Claude never guesses
  numbers: it calls the functions in analytics.py, reads the real results and
  then explains them in plain English.
* OfflineEngine - a small keyword-based engine that works with no API key and no
  internet. It answers the most common questions (survival rate by X, average
  of Y by X, correlations, counts, column summaries) so the project can be
  demoed anywhere.
"""

from __future__ import annotations

import json
import os
import re

import analytics as A

# ---------------------------------------------------------------------------
# Claude (LLM) engine
# ---------------------------------------------------------------------------

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are "Data Buddy", a friendly data analyst embedded in a dashboard about the
Kaggle Titanic dataset (891 passengers, 12 original columns + 5 derived ones).

Columns: {columns}

Rules:
- ALWAYS call a tool to get numbers. Never invent or estimate a statistic from memory.
- After getting results, explain what they mean in simple language a first-year student
  understands. Mention the actual numbers (rounded sensibly) and one insight or caveat
  (e.g. small sample size, missing Age values, correlation is not causation).
- Use short paragraphs or a small markdown table when comparing groups. Keep answers under
  ~200 words unless the user asks for detail.
- Text values in filters are case-insensitive (e.g. Sex == "female", Class == "1st",
  Port == "Southampton", AgeGroup == "Child (0-12)").
- If a question is outside this dataset, say so politely and suggest a related question
  the data CAN answer."""


def _filters_arg(filters):
    """Tools receive filters as a JSON list; accept None, list, or JSON string."""
    if filters in (None, "", []):
        return None
    if isinstance(filters, str):
        filters = json.loads(filters)
    return filters


def _build_tools():
    from anthropic import beta_tool

    @beta_tool
    def dataset_overview() -> str:
        """Get the dataset's size, every column with its meaning, missing-value counts and the overall survival rate. Call this first for general 'what is this data' questions."""
        return json.dumps(A.dataset_overview())

    @beta_tool
    def describe_column(column: str, filters: list[dict] | None = None) -> str:
        """Descriptive statistics for one column: mean/median/std/quartiles for numeric columns, counts and percentages for categorical ones.

        Args:
            column: Column name, e.g. Age, Fare, Sex, Class, Port, AgeGroup, Title, FamilySize.
            filters: Optional list of {"column": str, "op": one of ==,!=,>,>=,<,<=,in, "value": any} to restrict rows first.
        """
        return json.dumps(A.describe_column(column, _filters_arg(filters)))

    @beta_tool
    def survival_rate_by(group_by: list[str], filters: list[dict] | None = None) -> str:
        """Survival rate (%), survivors and passenger count for each group. Use for 'who survived more', 'survival by class/sex/age group', etc.

        Args:
            group_by: One or two column names to group by, e.g. ["Sex"] or ["Class", "Sex"].
            filters: Optional list of {"column", "op", "value"} filters applied before grouping.
        """
        return json.dumps(A.survival_rate_by(group_by, _filters_arg(filters)))

    @beta_tool
    def group_stats(metric: str, group_by: list[str] | None = None, agg: str = "mean",
                    filters: list[dict] | None = None) -> str:
        """Aggregate a numeric column, overall or per group. Example: average Fare by Class, median Age by Sex.

        Args:
            metric: Numeric column to aggregate (Age, Fare, SibSp, Parch, FamilySize, Survived, Pclass).
            group_by: Optional list of columns to group by.
            agg: One of mean, median, sum, min, max, std, count.
            filters: Optional list of {"column", "op", "value"} filters.
        """
        return json.dumps(A.group_stats(metric, group_by, agg, _filters_arg(filters)))

    @beta_tool
    def correlation(columns: list[str] | None = None) -> str:
        """Pearson correlation matrix between numeric columns (Survived, Pclass, Age, SibSp, Parch, Fare, FamilySize).

        Args:
            columns: Optional subset of numeric columns; defaults to all of them.
        """
        return json.dumps(A.correlation(columns))

    @beta_tool
    def count_rows(filters: list[dict] | None = None) -> str:
        """Count passengers matching filters and how many of them survived. Example: how many children in 3rd class survived?

        Args:
            filters: List of {"column", "op", "value"} filters.
        """
        return json.dumps(A.count_rows(_filters_arg(filters)))

    @beta_tool
    def sample_rows(filters: list[dict] | None = None, n: int = 5, sort_by: str | None = None,
                    ascending: bool = False) -> str:
        """Return up to 25 example passenger rows, optionally sorted. Example: the 5 oldest passengers, highest fares.

        Args:
            filters: Optional list of {"column", "op", "value"} filters.
            n: Number of rows (max 25).
            sort_by: Optional column to sort by.
            ascending: Sort ascending (True) or descending (False).
        """
        return json.dumps(A.sample_rows(_filters_arg(filters), n, sort_by, ascending))

    return [dataset_overview, describe_column, survival_rate_by, group_stats,
            correlation, count_rows, sample_rows]


class ClaudeEngine:
    name = f"Claude ({MODEL})"

    def __init__(self, api_key: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.tools = _build_tools()
        cols = ", ".join(f"{k} ({v})" for k, v in A.COLUMN_INFO.items())
        self.system = SYSTEM_PROMPT.format(columns=cols)

    def answer(self, question: str, history: list[dict]) -> str:
        # Only plain text turns are kept in history; tool calls happen fresh each turn.
        messages = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
        messages.append({"role": "user", "content": question})

        runner = self.client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=16000,
            system=self.system,
            tools=self.tools,
            messages=messages,
            output_config={"effort": "low"},          # chat-style Q&A: fast and cheap
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                       # re-run on a fallback model if a request is declined
            max_iterations=8,
        )
        final = runner.until_done()
        if final.stop_reason == "refusal":
            return "Sorry, I can't help with that request. Try asking about the Titanic passengers instead."
        text = "".join(b.text for b in final.content if b.type == "text").strip()
        return text or "I couldn't produce an answer - try rephrasing the question."


# ---------------------------------------------------------------------------
# Offline (rule-based) engine
# ---------------------------------------------------------------------------

# Words a user might type -> dataset column
_SYNONYMS = {
    "Sex": ["sex", "gender", "male", "female", "men", "women"],
    "Class": ["class", "pclass", "ticket class", "first class", "second class", "third class"],
    "Port": ["port", "embark", "embarked", "boarded", "cherbourg", "queenstown", "southampton"],
    "AgeGroup": ["age group", "age bucket", "children", "child", "kids", "teen", "senior", "elderly"],
    "Title": ["title", "mr", "mrs", "miss", "master"],
    "FamilySize": ["family", "family size", "relatives", "alone", "siblings", "sibsp", "parch"],
}
_METRICS = {"Age": ["age", "old", "young"], "Fare": ["fare", "price", "ticket price", "cost", "paid"],
            "FamilySize": ["family size"], "SibSp": ["siblings", "spouse"], "Parch": ["parents", "parch"]}
_AGGS = {"median": ["median"], "max": ["max", "maximum", "highest", "oldest", "most expensive"],
         "min": ["min", "minimum", "lowest", "youngest", "cheapest"], "sum": ["total", "sum"],
         "std": ["std", "standard deviation", "spread", "variation"], "mean": ["average", "mean", "avg", "typical"]}


def _has(text: str, words) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def _find_groups(text: str) -> list[str]:
    """Columns to group by. If the question says 'by ...', only look after that word."""
    m = re.search(r"\b(by|per|across|between|each|for each)\b", text)
    scope = text[m.end():] if m else text
    hits = []
    for col, words in _SYNONYMS.items():
        pos = [mm.start() for w in words for mm in re.finditer(rf"\b{re.escape(w)}\b", scope)]
        if pos:
            hits.append((min(pos), col))
    return [c for _, c in sorted(hits)][:2]


def _find_filters(text: str) -> list[dict]:
    f = []
    if _has(text, ["women", "female", "females", "woman"]):
        f.append({"column": "Sex", "op": "==", "value": "female"})
    elif _has(text, ["men", "male", "males", "man"]):
        f.append({"column": "Sex", "op": "==", "value": "male"})
    for word, cls in [("first", "1st"), ("1st", "1st"), ("second", "2nd"), ("2nd", "2nd"), ("third", "3rd"), ("3rd", "3rd")]:
        if re.search(rf"\b{word}\s*class\b", text):
            f.append({"column": "Class", "op": "==", "value": cls})
            break
    if _has(text, ["children", "child", "kids"]):
        f.append({"column": "AgeGroup", "op": "==", "value": "Child (0-12)"})
    for port in ["cherbourg", "queenstown", "southampton"]:
        if port in text:
            f.append({"column": "Port", "op": "==", "value": port})
    m = re.search(r"(older than|over|above|more than|greater than)\s+(\d+)", text)
    if m:
        f.append({"column": "Age", "op": ">", "value": int(m.group(2))})
    m = re.search(r"(younger than|under|below|less than)\s+(\d+)", text)
    if m:
        f.append({"column": "Age", "op": "<", "value": int(m.group(2))})
    return f


def _describe_filters(filters) -> str:
    if not filters:
        return "all passengers"
    return ", ".join(f"{x['column']} {x['op']} {x['value']}" for x in filters)


def _fmt(v) -> str:
    return f"{v:,.1f}" if isinstance(v, float) else str(v)


def _table(rows: list[dict]) -> str:
    if not rows:
        return "_No rows matched._"
    keys = list(rows[0].keys())
    head = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
    return head + "\n".join("| " + " | ".join(_fmt(r[k]) for k in keys) + " |" for r in rows)


class OfflineEngine:
    name = "Offline rule-based engine"

    def answer(self, question: str, history: list[dict]) -> str:
        q = question.lower().strip()
        try:
            return self._route(q)
        except Exception as exc:  # never crash the UI
            return f"I hit an error working that out ({exc}). Try rephrasing, e.g. *'survival rate by class'*."

    def _route(self, q: str) -> str:
        if _has(q, ["hi", "hello", "hey", "help"]) and len(q) < 25:
            return self._help()

        if _has(q, ["correlation", "correlated", "correlate", "relationship", "related"]):
            corr = A.correlation()["Survived"]
            pairs = sorted(((k, v) for k, v in corr.items() if k != "Survived"), key=lambda kv: -abs(kv[1]))
            lines = "\n".join(f"| {k} | {v:+.2f} |" for k, v in pairs)
            return ("Correlation of each numeric column with **Survived** (Pearson r, -1 to +1):\n\n"
                    f"| Column | r |\n|---|---|\n{lines}\n\n"
                    f"**{pairs[0][0]}** has the strongest link. A negative r for Pclass means a *higher class number* "
                    "(3rd class) goes with *lower* survival. Fare is positive: people who paid more survived more. "
                    "Remember: correlation is not causation - fare and class are tangled together.")

        if _has(q, ["overview", "summary", "summarize", "describe the data", "about the data", "what is this", "columns", "dataset"]):
            o = A.dataset_overview()
            missing = ", ".join(f"{k}: {v}" for k, v in o["missing_values"].items())
            cols = "\n".join(f"- **{k}** - {v}" for k, v in o["columns"].items())
            return (f"This is the **{o['dataset']}** dataset with **{o['rows']} passengers**. "
                    f"**{o['survivors']}** survived and **{o['deaths']}** died, an overall survival rate of "
                    f"**{o['overall_survival_rate_pct']:.1f}%**.\n\nMissing values - {missing}.\n\n**Columns**\n{cols}")

        if _has(q, ["missing", "null", "nan", "empty"]):
            o = A.dataset_overview()
            lines = "\n".join(f"- **{k}**: {v} missing ({v / o['rows'] * 100:.1f}%)" for k, v in o["missing_values"].items())
            return (f"Missing values:\n{lines}\n\n**Cabin** is mostly empty so it's rarely useful. "
                    "**Age** is missing for about 1 in 5 passengers - age statistics only use the rows that have it.")

        metric = next((m for m, words in _METRICS.items() if _has(q, words)), None)
        agg = next((a for a, words in _AGGS.items() if _has(q, words)), None)
        groups = _find_groups(q)
        filters = _find_filters(q)
        # Don't group by a column we are also filtering on
        groups = [g for g in groups if g not in {f["column"] for f in filters}]

        if _has(q, ["survival", "survive", "survived", "survivors", "died", "death", "lived"]):
            if _has(q, ["how many", "number of", "count"]) or not groups:
                r = A.count_rows(filters)
                if r["passengers"] == 0:
                    return f"No passengers match ({_describe_filters(filters)})."
                return (f"For **{_describe_filters(filters)}**: {r['survivors']} of {r['passengers']} passengers survived - "
                        f"a survival rate of **{r['survival_rate_pct']:.1f}%** (overall average is 38.4%).")
            rows = A.survival_rate_by(groups, filters)
            best = max(rows, key=lambda r: r["survival_rate_pct"])
            worst = min(rows, key=lambda r: r["survival_rate_pct"])
            label = lambda r: " / ".join(str(r[g]) for g in groups)
            return (f"Survival rate by **{' and '.join(groups)}** ({_describe_filters(filters)}):\n\n{_table(rows)}\n\n"
                    f"Highest: **{label(best)}** at {best['survival_rate_pct']:.1f}%. "
                    f"Lowest: **{label(worst)}** at {worst['survival_rate_pct']:.1f}%. "
                    "During the evacuation the rule was largely *'women and children first'*, and upper-class cabins "
                    "were closer to the lifeboats - both show up clearly in the data.")

        if metric and (agg or groups):
            agg = agg or "mean"
            if agg in ("max", "min") and not groups:
                rows = A.sample_rows(filters, n=5, sort_by=metric, ascending=(agg == "min"))
                return f"Top 5 passengers by **{agg} {metric}** ({_describe_filters(filters)}):\n\n{_table(rows)}"
            res = A.group_stats(metric, groups or None, agg, filters)
            if isinstance(res, dict):
                return (f"The **{agg} {metric}** for {_describe_filters(filters)} is **{res['value']}** "
                        f"(based on {res['rows_considered']} rows).")
            return f"**{agg.title()} {metric}** by {' and '.join(groups)} ({_describe_filters(filters)}):\n\n{_table(res)}"

        if metric:
            d = A.describe_column(metric, filters)
            return (f"**{metric}** for {_describe_filters(filters)} ({d['rows_considered']} rows, {d['missing']} missing):\n\n"
                    f"| mean | median | std | min | Q1 | Q3 | max |\n|---|---|---|---|---|---|---|\n"
                    f"| {d['mean']} | {d['median']} | {d['std']} | {d['min']} | {d['q1']} | {d['q3']} | {d['max']} |\n\n"
                    + ("The mean is well above the median, so the distribution is **right-skewed** - a few very "
                       "expensive tickets pull the average up." if d["skew"] and d["skew"] > 1 else
                       "Mean and median are close, so the distribution is fairly symmetric."))

        if _has(q, ["how many", "number of", "count"]):
            r = A.count_rows(filters)
            return f"There are **{r['passengers']}** passengers matching {_describe_filters(filters)}."

        if groups:
            d = A.describe_column(groups[0], filters)
            rows = [{"value": k, "count": v, "percent": d["percent"][k]} for k, v in d["counts"].items()]
            return f"Breakdown of **{groups[0]}** ({_describe_filters(filters)}):\n\n{_table(rows)}"

        return self._help()

    @staticmethod
    def _help() -> str:
        return ("I can explain the Titanic data. Try asking:\n"
                "- *Give me an overview of the dataset*\n"
                "- *What was the survival rate by class and sex?*\n"
                "- *Average fare by port*\n"
                "- *How many children in third class survived?*\n"
                "- *Which columns are correlated with survival?*\n"
                "- *Describe the age column* / *Which values are missing?*\n\n"
                "_Running in offline mode. Add an Anthropic API key in the sidebar for free-form questions answered by Claude._")


def get_engine(api_key: str | None = None):
    """Use Claude when a key is available, otherwise fall back to the offline engine."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            return ClaudeEngine(key)
        except Exception:
            pass
    return OfflineEngine()

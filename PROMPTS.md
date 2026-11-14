# Prompts Used to Build the Project

This file lists, in order, the prompts a person would give an AI coding assistant to build
this project from an empty folder. They are written the way a student would actually type
them. Each prompt has a short note on **why** it is phrased that way, which is the part
of prompt engineering worth learning.

The project has two kinds of prompts:

* **Part A - build prompts:** what the developer typed to get the code written.
* **Part B - runtime prompts:** the system prompt and tool descriptions that live inside
  `chatbot.py` and steer the chatbot every time a user asks a question.
* **Part C - test prompts:** questions used to check the chatbot's answers against the dashboard.

---

## Part A - Build prompts (developer → AI assistant)

### 1. Kick-off and scope

> I'm building a college data-science project. I want an interactive dashboard on a Kaggle
> dataset with a chatbot section where I can ask questions about the data and get
> explanations and statistics. Use Python and Streamlit. Suggest a Kaggle dataset that is
> small, clean enough for beginners, and has both numeric and categorical columns.

*Why:* it gives the goal, the audience (college), the tech stack and the selection criteria
for the dataset, so the assistant doesn't have to guess.

### 2. Get the data

> Use the Kaggle "Titanic - Machine Learning from Disaster" dataset (train.csv, 891 rows).
> Download it into `data/titanic.csv`. If the Kaggle CLI isn't set up, use a public mirror of
> the same file and tell me how to get it from Kaggle with `kaggle competitions download -c titanic`.

*Why:* it names the exact file and plans for missing Kaggle credentials.

### 3. Explore before building

> Load the CSV with pandas and give me a quick EDA: shape, dtypes, missing values per column,
> overall survival rate, and survival rate by Sex, Pclass and Embarked. Suggest 3-5 derived
> columns that would make the data easier to explain (e.g. age groups, family size, title from name).

*Why:* asking for exploration first means the dashboard is built around what is actually in the data.

### 4. A single analytics module

> Create `analytics.py` with a cached `load_data()` that adds the derived columns
> (AgeGroup, FamilySize, Title, Class as "1st/2nd/3rd", Port as full names) and a dictionary
> describing every column. Then add pure functions that return JSON-friendly dicts:
> `dataset_overview()`, `describe_column(column, filters)`, `survival_rate_by(group_by, filters)`,
> `group_stats(metric, group_by, agg, filters)`, `correlation(columns)`, `count_rows(filters)`
> and `sample_rows(filters, n, sort_by)`. Filters should be a list of
> `{"column", "op", "value"}` objects. Do NOT use `eval` or `df.query` on user text.

*Why:* one source of truth for the numbers; the dashboard and the chatbot can never disagree.
The "no eval" line is a safety requirement: the chatbot passes text that came from users.

### 5. The dashboard

> Build `app.py` in Streamlit with a wide layout. Sidebar filters: ticket class, sex, port
> (multiselects) and an age-range slider with a checkbox for unknown ages. Top row: 5 KPI cards
> (passengers, survival rate with delta vs. all passengers, survivors, median age, median fare).
> Then three tabs: "Dashboard", "Ask the Data (Chatbot)" and "Data & Dictionary".
> In the Dashboard tab use Plotly for: survival rate by sex, stacked passengers by class and
> outcome, age histogram by outcome, fare box plot by class (log scale), a sex × class
> survival heatmap, survival by age group, a correlation matrix and survival by port.

*Why:* a concrete list of charts gives a predictable result; tabs keep the chatbot one click away.

### 6. Make the charts readable and accessible

> Make all charts use one consistent style: a colour-blind-safe palette (blue = survived,
> orange = died), a single-hue blue scale for the heatmap, a diverging red–grey–blue scale
> centred on 0 for correlations, light gridlines, no chart junk, value labels on bars,
> and hover tooltips with the exact numbers and sample size.

*Why:* without this the assistant uses default rainbow colours; the colour roles (identity,
magnitude, polarity) are named explicitly.

### 7. The chatbot - LLM version

> Create `chatbot.py` with a `ClaudeEngine` that uses the Anthropic Python SDK and **tool use**.
> Expose every function in `analytics.py` as a tool (use the SDK's `@beta_tool` decorator and
> `client.beta.messages.tool_runner`). Write a system prompt that makes the bot a friendly
> "data buddy" for a first-year student, forces it to call a tool for every number, and asks
> it to explain results in plain English with one insight or caveat. Use the model
> `claude-opus-5-5` at low effort since this is chat, and handle the `refusal` stop reason.

*Why:* tool use is what stops the model from making up statistics. It does the language part
and pandas does the arithmetic.

### 8. The chatbot - offline fallback

> Not everyone grading this will have an API key. Add an `OfflineEngine` with the same
> `answer(question, history)` method that understands common questions with keyword rules:
> overview, missing values, correlation with survival, "survival rate by X (and Y)",
> "average/median/max of METRIC by X", "how many ...", and filters like "women",
> "third class", "children", "older than 50", port names. Answer with markdown tables and a
> one-line explanation. `get_engine()` should pick Claude when a key exists, otherwise offline.

*Why:* the demo works without internet or a paid key, and graders see the same interface either way.

### 9. Chat UI

> In the chatbot tab show which engine is active, five suggested-question buttons, a scrollable
> chat history with `st.chat_message`, and `st.chat_input`. Keep history in `st.session_state`.
> Add an optional password field in the sidebar for the API key. If the Claude call fails,
> show a warning and answer with the offline engine instead.

### 10. Test and fix

> Run the offline engine against these questions and show me the answers:
> "survival rate by class and sex", "which columns are correlated with survival",
> "survival rate of women older than 50", "survival rate by age group", "median age by sex".
> Fix any parsing bugs you find.

*Why:* asking for real outputs catches bugs. This step found three: "by class **and** sex"
only grouped by class, "older **than** 50" was ignored, and "which **columns** are correlated"
went to the overview answer.

### 11. Screenshots

> Write `scripts/take_screenshots.py` using Playwright that opens the running app and saves
> screenshots of: the top of the dashboard, the full dashboard, a filtered view (3rd class only),
> the chatbot welcome screen, three chatbot answers, and the data dictionary tab, into `screenshots/`.

### 12. Documentation

> Write a `README.md` (setup, run, deploy) and a `DOCUMENTATION.md` project report with:
> objective, dataset description, architecture diagram, how the chatbot works (tool calling +
> offline mode), each dashboard chart and what it shows, key findings with numbers,
> screenshots, limitations and future work. Also write this `PROMPTS.md`.

### 13. Deployment

> How can I host this for free? Give me step-by-step instructions for Streamlit Community Cloud,
> including how to store the API key as a secret.

---

## Part B - Runtime prompts (inside the app)

### System prompt (`chatbot.py → SYSTEM_PROMPT`)

```text
You are "Data Buddy", a friendly data analyst embedded in a dashboard about the
Kaggle Titanic dataset (891 passengers, 12 original columns + 5 derived ones).

Columns: {columns}          <- filled in at runtime from analytics.COLUMN_INFO

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
  the data CAN answer.
```

Design choices:

| Line | Purpose |
|---|---|
| Persona + audience | Sets tone and reading level |
| Column list injected at runtime | The model knows the exact column names, so tool calls are valid on the first try |
| "ALWAYS call a tool" | Grounding: every number comes from pandas, not from the model's memory of Titanic facts |
| "one insight or caveat" | Turns a number dump into an *explanation* (the project's goal) |
| Example filter values | Shows the filter format by example |
| Out-of-scope rule | Handles off-topic questions politely |

### Tool descriptions

The docstrings of the `@beta_tool` functions in `chatbot.py` are prompts too: the model
reads them to decide which tool to call. Each one says **what it returns** and gives an
**example question** it answers, e.g.

```text
survival_rate_by: Survival rate (%), survivors and passenger count for each group.
Use for 'who survived more', 'survival by class/sex/age group', etc.
```

---

## Part C - Test questions for the chatbot

Use these during the demo. Each answer can be checked against a dashboard chart.

| # | Question | Expected answer (check against) |
|---|---|---|
| 1 | Give me an overview of the dataset | 891 rows, 342 survived, 38.4% |
| 2 | What was the survival rate by class and sex? | 1st-class women 96.8% … 3rd-class men 13.5% (heatmap) |
| 3 | Which columns are correlated with survival? | Pclass r = -0.34, Fare r = +0.26 (correlation matrix) |
| 4 | Average fare by port | Cherbourg ≈ £60, Southampton ≈ £27, Queenstown ≈ £13 |
| 5 | How many children in third class survived? | 20 of 48 (41.7%) |
| 6 | Survival rate of women older than 50 | 16 of 17 (94.1%) |
| 7 | Describe the fare column | mean 32.2, median 14.45, right-skewed |
| 8 | Which values are missing? | Cabin 687, Age 177, Embarked 2 |
| 9 | *(Claude mode only)* Why might fare and class both look important? Are they the same thing? | Explains that Fare and Pclass are correlated (r = -0.55), so their effects overlap |
| 10 | *(Claude mode only)* Who was the oldest passenger and did they survive? | Calls `sample_rows` sorted by Age |

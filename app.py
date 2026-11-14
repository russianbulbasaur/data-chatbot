"""Titanic Data Explorer - a Streamlit dashboard with a built-in data chatbot.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as A
from chatbot import ClaudeEngine, OfflineEngine, get_engine

# ---------------------------------------------------------------------------
# Visual system (validated colour-blind-safe palette)
# ---------------------------------------------------------------------------
SURVIVED, DIED = "#2a78d6", "#eb6834"          # categorical slots 1 & 2
CLASS_COLORS = {"1st": "#2a78d6", "2nd": "#eb6834", "3rd": "#1baf7a"}
OUTCOME_COLORS = {"Survived": SURVIVED, "Died": DIED}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]   # red <- 0 -> blue

st.set_page_config(page_title="Titanic Data Explorer", page_icon="🚢", layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1400px; }}
  div[data-testid="stMetric"] {{
      background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 14px 18px;
  }}
  div[data-testid="stMetricLabel"] p {{ color: {INK_2}; font-size: 0.85rem; }}
  .chart-card {{ background: {SURFACE}; border-radius: 10px; }}
  .subtle {{ color: {INK_2}; font-size: 0.9rem; }}
</style>
""", unsafe_allow_html=True)


def style(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    """Apply one consistent, quiet chart style."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        height=height, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
        bargap=0.25, bargroupgap=0.08,
        hoverlabel=dict(bgcolor="white", font_size=12, font_color=INK),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK_2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK_2))
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    return fig


# ---------------------------------------------------------------------------
# Data + sidebar filters
# ---------------------------------------------------------------------------
df_all = A.load_data()

with st.sidebar:
    st.header("🎛️ Filters")
    st.caption("Filters apply to the dashboard charts. The chatbot always sees the full dataset.")
    classes = st.multiselect("Ticket class", ["1st", "2nd", "3rd"], default=["1st", "2nd", "3rd"])
    sexes = st.multiselect("Sex", ["female", "male"], default=["female", "male"])
    ports = st.multiselect("Port of embarkation", ["Southampton", "Cherbourg", "Queenstown"],
                           default=["Southampton", "Cherbourg", "Queenstown"])
    age_lo, age_hi = st.slider("Age range", 0, 80, (0, 80))
    include_unknown_age = st.checkbox("Include passengers with unknown age", value=True)

    st.divider()
    st.header("🤖 Chatbot engine")
    key_input = st.text_input("Anthropic API key (optional)", type="password",
                              help="Without a key the bot runs a built-in offline engine. "
                                   "With a key it uses Claude + tool calls for free-form questions.")
    api_key = key_input or os.environ.get("ANTHROPIC_API_KEY")

port_ok = df_all["Port"].isin(ports)
if len(ports) == 3:                      # 2 passengers have no recorded port - keep them when nothing is excluded
    port_ok |= df_all["Port"].isna()
df = df_all[df_all["Class"].isin(classes) & df_all["Sex"].isin(sexes) & port_ok]
age_ok = df["Age"].between(age_lo, age_hi)
if include_unknown_age:
    age_ok |= df["Age"].isna()
df = df[age_ok]

# ---------------------------------------------------------------------------
# Header + KPI row
# ---------------------------------------------------------------------------
st.title("🚢 Titanic Data Explorer")
st.markdown(
    "<span class='subtle'>Kaggle dataset: <b>Titanic - Machine Learning from Disaster</b> · "
    f"{len(df_all)} passengers · showing <b>{len(df)}</b> after filters</span>",
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No passengers match the current filters. Widen them in the sidebar.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
rate = df["Survived"].mean() * 100
k1.metric("Passengers", f"{len(df):,}")
k2.metric("Survival rate", f"{rate:.1f}%",
          None if len(df) == len(df_all) else f"{rate - df_all['Survived'].mean() * 100:+.1f} pts vs all")
k3.metric("Survivors", f"{int(df['Survived'].sum()):,}")
k4.metric("Median age", f"{df['Age'].median():.0f} yrs" if df["Age"].notna().any() else "–")
k5.metric("Median fare", f"£{df['Fare'].median():.2f}")

tab_dash, tab_chat, tab_data = st.tabs(["📊 Dashboard", "💬 Ask the Data (Chatbot)", "🗂️ Data & Dictionary"])

# ---------------------------------------------------------------------------
# Tab 1: Dashboard
# ---------------------------------------------------------------------------
with tab_dash:
    dfo = df.assign(Outcome=df["Survived"].map({1: "Survived", 0: "Died"}))

    c1, c2 = st.columns(2)
    with c1:
        sdf = df.groupby("Sex")["Survived"].agg(["mean", "sum", "count"]).reset_index()
        sdf.columns = ["Sex", "rate", "survivors", "passengers"]
        sdf["survival_rate_pct"] = sdf["rate"] * 100
        fig = px.bar(sdf, x="Sex", y="survival_rate_pct", text="survival_rate_pct",
                     custom_data=["survivors", "passengers"], color_discrete_sequence=[SURVIVED])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_color=INK,
                          hovertemplate="<b>%{x}</b><br>Survival rate: %{y:.1f}%<br>%{customdata[0]} of %{customdata[1]} survived<extra></extra>")
        fig.update_yaxes(title="Survival rate (%)", range=[0, 105])
        fig.update_xaxes(title="")
        st.plotly_chart(style(fig, "Survival rate by sex"), width="stretch")
    with c2:
        cdf = (dfo.groupby(["Class", "Outcome"]).size().reset_index(name="Passengers"))
        fig = px.bar(cdf, x="Class", y="Passengers", color="Outcome", barmode="stack",
                     color_discrete_map=OUTCOME_COLORS, category_orders={"Outcome": ["Survived", "Died"], "Class": ["1st", "2nd", "3rd"]})
        fig.update_xaxes(title="Ticket class")
        style(fig, "Passengers by class and outcome")
        # 2px surface-coloured gap between stacked segments
        fig.update_traces(marker_line_color=SURFACE, marker_line_width=2,
                          hovertemplate="<b>%{x} class</b><br>%{fullData.name}: %{y}<extra></extra>")
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(dfo.dropna(subset=["Age"]), x="Age", color="Outcome", nbins=16, barmode="stack",
                           color_discrete_map=OUTCOME_COLORS, category_orders={"Outcome": ["Survived", "Died"]})
        fig.update_traces(hovertemplate="Age %{x}<br>%{fullData.name}: %{y}<extra></extra>")
        fig.update_xaxes(title="Age (years)")
        fig.update_yaxes(title="Passengers")
        st.plotly_chart(style(fig, "Age distribution by outcome"), width="stretch")
    with c4:
        fig = px.box(df, x="Class", y="Fare", color="Class", color_discrete_map=CLASS_COLORS,
                     category_orders={"Class": ["1st", "2nd", "3rd"]}, points=False, log_y=True)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="Fare (£, log scale)", tickvals=[5, 10, 20, 50, 100, 200, 500],
                         ticktext=["5", "10", "20", "50", "100", "200", "500"])
        fig.update_xaxes(title="Ticket class")
        st.plotly_chart(style(fig, "Fare paid by class"), width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        pv = df.pivot_table(index="Sex", columns="Class", values="Survived", aggfunc="mean") * 100
        pv = pv.reindex(columns=[c for c in ["1st", "2nd", "3rd"] if c in pv.columns])
        fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale=SEQ_BLUE, zmin=0, zmax=100,
                                   xgap=2, ygap=2, texttemplate="%{z:.0f}%", textfont=dict(size=14),
                                   hovertemplate="%{y}, %{x} class<br>Survival: %{z:.1f}%<extra></extra>",
                                   colorbar=dict(title="%", thickness=10)))
        fig.update_xaxes(title="Ticket class")
        st.plotly_chart(style(fig, "Survival rate (%) - sex × class"), width="stretch")
    with c6:
        ag = (df.dropna(subset=["AgeGroup"]).groupby("AgeGroup", observed=True)["Survived"]
                .agg(["mean", "count"]).reset_index())
        ag["rate"] = ag["mean"] * 100
        ag["AgeGroup"] = ag["AgeGroup"].astype(str).str.replace(" (", "<br>(", regex=False)
        fig = px.bar(ag, x="AgeGroup", y="rate", text="rate", custom_data=["count"], color_discrete_sequence=[SURVIVED])
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside", textfont_color=INK,
                          hovertemplate="<b>%{x}</b><br>Survival: %{y:.1f}%<br>n = %{customdata[0]}<extra></extra>")
        fig.update_yaxes(title="Survival rate (%)", range=[0, max(100, ag["rate"].max() + 10)])
        fig.update_xaxes(title="", tickangle=0)
        st.plotly_chart(style(fig, "Survival rate by age group"), width="stretch")

    c7, c8 = st.columns([3, 2])
    with c7:
        num = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
        corr = df[num].corr()
        fig = go.Figure(go.Heatmap(z=corr.values, x=num, y=num, colorscale=DIVERGING, zmin=-1, zmax=1, xgap=2, ygap=2,
                                   texttemplate="%{z:.2f}", textfont=dict(size=11),
                                   hovertemplate="%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>",
                                   colorbar=dict(title="r", thickness=10)))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, "Correlation matrix (Pearson r)", height=400), width="stretch")
    with c8:
        pdf = df.dropna(subset=["Port"]).groupby("Port")["Survived"].agg(["mean", "count"]).reset_index()
        pdf["rate"] = pdf["mean"] * 100
        pdf = pdf.sort_values("rate")
        fig = px.bar(pdf, y="Port", x="rate", orientation="h", text="rate", custom_data=["count"],
                     color_discrete_sequence=[SURVIVED])
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside", textfont_color=INK,
                          hovertemplate="<b>%{y}</b><br>Survival: %{x:.1f}%<br>n = %{customdata[0]}<extra></extra>")
        fig.update_xaxes(title="Survival rate (%)", range=[0, 100], gridcolor=GRID, showgrid=True)
        fig.update_yaxes(title="", showgrid=False)
        st.plotly_chart(style(fig, "Survival rate by port of embarkation", height=400), width="stretch")

    st.info("💡 **Key insights:** women survived at ~74% vs ~19% for men; 1st-class passengers survived at 63% vs 24% "
            "in 3rd class; children (0-12) had the best odds of any age group. Ask the chatbot to explain any chart!")

# ---------------------------------------------------------------------------
# Tab 2: Chatbot
# ---------------------------------------------------------------------------
with tab_chat:
    engine = get_engine(api_key)
    mode = "🟢 Claude (AI, tool-calling)" if isinstance(engine, ClaudeEngine) else "🟡 Offline rule-based engine"
    st.markdown(f"**Ask the Data** - ask for explanations or statistics in plain English. Engine: {mode}")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": OfflineEngine._help()
                                      if not isinstance(engine, ClaudeEngine) else
                                      "Hi! I'm Data Buddy. Ask me anything about the Titanic passengers - "
                                      "survival rates, averages, comparisons, correlations or what a column means."}]

    suggestions = ["Give me an overview of the dataset", "Survival rate by class and sex",
                   "Which columns are correlated with survival?", "Average fare by port",
                   "How many children in third class survived?"]
    cols = st.columns(len(suggestions))
    clicked = None
    for col, sug in zip(cols, suggestions):
        if col.button(sug, width="stretch"):
            clicked = sug

    chat_box = st.container(height=520)
    with chat_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🤖"):
                st.markdown(m["content"])

    typed = st.chat_input("e.g. What was the median age of survivors in 2nd class?")
    question = typed or clicked
    if question:
        history = st.session_state.messages[1:]   # skip the greeting
        st.session_state.messages.append({"role": "user", "content": question})
        with chat_box:
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(question)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Crunching the numbers..."):
                    try:
                        reply = engine.answer(question, history)
                    except Exception as exc:
                        reply = (f"⚠️ Claude request failed ({type(exc).__name__}). Falling back to offline mode.\n\n"
                                 + OfflineEngine().answer(question, history))
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    if st.button("🗑️ Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Tab 3: Raw data + data dictionary
# ---------------------------------------------------------------------------
with tab_data:
    l, r = st.columns([3, 2])
    with l:
        st.subheader("Filtered rows")
        st.dataframe(df.drop(columns=["AgeGroup"]).assign(AgeGroup=df["AgeGroup"].astype("string")),
                     width="stretch", height=460, hide_index=True)
        st.download_button("⬇️ Download filtered CSV", df.to_csv(index=False), "titanic_filtered.csv", "text/csv")
    with r:
        st.subheader("Data dictionary")
        st.dataframe(pd.DataFrame({"Column": list(A.COLUMN_INFO), "Meaning": list(A.COLUMN_INFO.values())}),
                     width="stretch", height=460, hide_index=True)
        st.subheader("Numeric summary")
        st.dataframe(df[["Age", "Fare", "SibSp", "Parch", "FamilySize"]].describe().round(2), width="stretch")

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data profiler", page_icon=":material/analytics:", layout="wide")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading file...")
def load_file(data: bytes, name: str, sep: str) -> pd.DataFrame:
    from io import BytesIO

    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(data))
    return pd.read_csv(BytesIO(data), sep=None if sep == "auto" else sep, engine="python")


@st.cache_data
def load_sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, n),
            "income": rng.normal(55000, 15000, n).round(2),
            "gpa": rng.uniform(2.0, 4.0, n).round(2),
            "major": rng.choice(["Biology", "Business", "Nursing", "Psychology", "Computer Science"], n),
            "campus": rng.choice(["North", "South", "Online"], n, p=[0.5, 0.3, 0.2]),
            "enrolled": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )
    df["credits"] = (df["gpa"] * 25 + rng.normal(0, 8, n)).round(0)
    df.loc[rng.choice(n, 40, replace=False), "income"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "major"] = np.nan
    return df


@st.cache_data
def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "type": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_pct": s.isna().mean() * 100,
                "unique": int(s.nunique(dropna=True)),
                "sample": ", ".join(map(str, s.dropna().unique()[:3])),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Generating full profile report (this can take a while)...")
def build_profile_html(df: pd.DataFrame, minimal: bool, title: str) -> str:
    from ydata_profiling import ProfileReport

    return ProfileReport(df, title=title, minimal=minimal, explorative=not minimal).to_html()


def generate_insights(df: pd.DataFrame, summary: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (level, message) pairs describing notable data characteristics."""
    insights: list[tuple[str, str]] = []
    n = len(df)

    dups = int(df.duplicated().sum())
    if dups:
        insights.append(("warning", f"**{dups:,}** duplicate rows ({dups / n:.1%} of the data)."))

    for _, r in summary.iterrows():
        if r["missing_pct"] >= 50:
            insights.append(("error", f"`{r['column']}` is **{r['missing_pct']:.0f}%** missing - consider dropping it."))
        elif r["missing_pct"] >= 5:
            insights.append(("warning", f"`{r['column']}` has **{r['missing_pct']:.1f}%** missing values."))
        if r["unique"] == 1:
            insights.append(("warning", f"`{r['column']}` is constant (single value) and adds no information."))
        elif r["unique"] == n and r["type"] == "object":
            insights.append(("info", f"`{r['column']}` is unique for every row - likely an identifier."))

    num = df.select_dtypes("number")
    for col in num.columns:
        s = num[col].dropna()
        if len(s) > 2 and s.std() > 0:
            skew = s.skew()
            if abs(skew) > 1:
                insights.append(("info", f"`{col}` is strongly {'right' if skew > 0 else 'left'}-skewed (skew = {skew:.2f})."))
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            if outliers and outliers / len(s) > 0.01:
                insights.append(("info", f"`{col}` has **{outliers}** potential outliers (IQR rule)."))

    if num.shape[1] >= 2:
        corr = num.corr().abs()
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = corr.where(mask).stack().sort_values(ascending=False)
        for (a, b), v in pairs[pairs >= 0.7].head(5).items():
            insights.append(("success", f"`{a}` and `{b}` are highly correlated (|r| = {v:.2f})."))

    return insights


# ---------------------------------------------------------------------------
# Header & sidebar
# ---------------------------------------------------------------------------
st.title(":material/analytics: Data profiler")
st.caption(
    "By Dr. Alexis Vera - [Contact me](mailto:alexisv@sanjuanbautista.edu). "
    "Upload a dataset to explore it interactively, get automatic insights, and generate a full "
    "[ydata-profiling](https://docs.profiling.ydata.ai) report."
)

with st.sidebar:
    st.header("Data source")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "txt", "xlsx", "xls"])
    sep = st.selectbox(
        "CSV delimiter",
        ["auto", ",", ";", "\t", "|"],
        format_func=lambda x: {"auto": "Auto-detect", "\t": "Tab"}.get(x, x),
    )
    use_sample = st.toggle("Use sample dataset", value=False, disabled=uploaded_file is not None)

if uploaded_file is not None:
    raw_df = load_file(uploaded_file.getvalue(), uploaded_file.name, sep)
    dataset_name = uploaded_file.name
elif use_sample:
    raw_df = load_sample()
    dataset_name = "sample_dataset"
else:
    st.info("Upload a CSV or Excel file in the sidebar, or turn on the sample dataset to try the app.", icon=":material/upload_file:")
    st.stop()

# Try to parse text columns that look like dates
for col in raw_df.select_dtypes("object").columns:
    if raw_df[col].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9:
        raw_df[col] = pd.to_datetime(raw_df[col], errors="coerce")

with st.sidebar:
    st.header("Filters")
    selected_cols = st.multiselect("Columns to include", raw_df.columns.tolist(), default=raw_df.columns.tolist())
    if len(raw_df) > 1000:
        max_rows = st.slider("Max rows to analyze", 1000, len(raw_df), min(len(raw_df), 50000), step=1000)
    else:
        max_rows = len(raw_df)
    drop_dups = st.checkbox("Drop duplicate rows")
    drop_na = st.checkbox("Drop rows with missing values")

if not selected_cols:
    st.warning("Select at least one column in the sidebar.")
    st.stop()

df = raw_df[selected_cols]
if max_rows < len(df):
    df = df.sample(max_rows, random_state=0).sort_index()
if drop_dups:
    df = df.drop_duplicates()
if drop_na:
    df = df.dropna()

if df.empty:
    st.warning("No rows left after applying filters.")
    st.stop()

summary = column_summary(df)
numeric_cols = df.select_dtypes("number").columns.tolist()
categorical_cols = df.select_dtypes(["object", "category", "bool"]).columns.tolist()
datetime_cols = df.select_dtypes("datetime").columns.tolist()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
with st.container(horizontal=True):
    st.metric("Rows", f"{len(df):,}", border=True)
    st.metric("Columns", df.shape[1], border=True)
    st.metric("Missing cells", f"{df.isna().mean().mean():.1%}", border=True)
    st.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}", border=True)
    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", border=True)

overview_tab, data_tab, column_tab, corr_tab, rel_tab, report_tab = st.tabs(
    [
        ":material/dashboard: Overview",
        ":material/table: Data",
        ":material/view_column: Column explorer",
        ":material/grid_on: Correlations",
        ":material/scatter_plot: Relationships",
        ":material/description: Full report",
    ]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with overview_tab:
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.subheader("Automatic insights")
            insights = generate_insights(df, summary)
            if not insights:
                st.success("No notable data quality issues found.", icon=":material/check_circle:")
            icons = {"error": ":material/error:", "warning": ":material/warning:", "info": ":material/info:", "success": ":material/link:"}
            for level, msg in insights:
                getattr(st, level)(msg, icon=icons[level])
    with right:
        with st.container(border=True):
            st.subheader("Column types")
            kinds = pd.DataFrame(
                {
                    "kind": ["Numeric", "Categorical", "Datetime", "Other"],
                    "count": [
                        len(numeric_cols),
                        len(categorical_cols),
                        len(datetime_cols),
                        df.shape[1] - len(numeric_cols) - len(categorical_cols) - len(datetime_cols),
                    ],
                }
            ).query("count > 0")
            st.altair_chart(
                alt.Chart(kinds)
                .mark_arc(innerRadius=50)
                .encode(theta="count:Q", color=alt.Color("kind:N", title=None), tooltip=["kind", "count"]),
            )

    with st.container(border=True):
        st.subheader("Missing values by column")
        miss = summary[summary["missing"] > 0]
        if miss.empty:
            st.success("No missing values.", icon=":material/check_circle:")
        else:
            st.altair_chart(
                alt.Chart(miss)
                .mark_bar()
                .encode(
                    x=alt.X("missing_pct:Q", title="% missing"),
                    y=alt.Y("column:N", sort="-x", title=None),
                    color=alt.Color("missing_pct:Q", scale=alt.Scale(scheme="orangered"), legend=None),
                    tooltip=["column", "missing", alt.Tooltip("missing_pct:Q", format=".1f")],
                )
            )

    with st.container(border=True):
        st.subheader("Column summary")
        st.dataframe(
            summary,
            hide_index=True,
            column_config={
                "missing_pct": st.column_config.ProgressColumn("Missing %", format="%.1f%%", min_value=0, max_value=100),
                "unique": st.column_config.NumberColumn("Unique"),
            },
        )

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
with data_tab:
    query = st.text_input("Search rows", placeholder="Type to filter rows containing this text", icon=":material/search:")
    view = df
    if query:
        view = df[df.astype(str).apply(lambda c: c.str.contains(query, case=False, na=False)).any(axis=1)]
    st.caption(f"Showing {len(view):,} of {len(df):,} rows")
    st.dataframe(view)
    if numeric_cols:
        with st.expander("Descriptive statistics"):
            st.dataframe(df[numeric_cols].describe().T)
    st.download_button(
        "Download filtered data (CSV)",
        view.to_csv(index=False).encode(),
        file_name=f"{dataset_name}_filtered.csv",
        mime="text/csv",
        icon=":material/download:",
    )

# ---------------------------------------------------------------------------
# Column explorer
# ---------------------------------------------------------------------------
with column_tab:
    col = st.selectbox("Select a column", df.columns.tolist())
    s = df[col]
    stats_col, chart_col = st.columns([1, 2])

    with stats_col:
        with st.container(border=True):
            st.metric("Unique values", f"{s.nunique():,}")
            st.metric("Missing", f"{s.isna().sum():,} ({s.isna().mean():.1%})")
            if col in numeric_cols:
                st.metric("Mean", f"{s.mean():,.3g}")
                st.metric("Median", f"{s.median():,.3g}")
                st.metric("Std dev", f"{s.std():,.3g}")
                st.metric("Range", f"{s.min():,.3g} - {s.max():,.3g}")
            else:
                mode = s.mode()
                st.metric("Most frequent", str(mode.iloc[0]) if not mode.empty else "-")

    with chart_col:
        with st.container(border=True):
            if col in numeric_cols:
                bins = st.slider("Bins", 5, 100, 30)
                hist = (
                    alt.Chart(df[[col]].dropna())
                    .mark_bar()
                    .encode(x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins)), y="count()", tooltip=["count()"])
                )
                box = alt.Chart(df[[col]].dropna()).mark_boxplot(extent=1.5).encode(x=f"{col}:Q").properties(height=60)
                st.altair_chart(alt.vconcat(hist, box))
            elif col in datetime_cols:
                freq = st.segmented_control("Granularity", ["D", "W", "M", "Y"], default="M",
                                            format_func={"D": "Day", "W": "Week", "M": "Month", "Y": "Year"}.get)
                counts = s.dropna().dt.to_period(freq or "M").dt.to_timestamp().value_counts().sort_index()
                ts = counts.rename_axis("date").reset_index(name="count")
                st.altair_chart(alt.Chart(ts).mark_area(line=True, opacity=0.4).encode(x="date:T", y="count:Q", tooltip=["date:T", "count"]))
            else:
                top_n = st.slider("Top N categories", 5, 50, 15)
                vc = s.fillna("(missing)").astype(str).value_counts().head(top_n).rename_axis(col).reset_index(name="count")
                st.altair_chart(
                    alt.Chart(vc).mark_bar().encode(
                        x="count:Q", y=alt.Y(f"{col}:N", sort="-x", title=None), tooltip=[col, "count"]
                    )
                )

# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
with corr_tab:
    if len(numeric_cols) < 2:
        st.info("At least two numeric columns are needed for correlation analysis.")
    else:
        method = st.segmented_control("Method", ["pearson", "spearman", "kendall"], default="pearson",
                                      format_func=str.capitalize)
        corr = df[numeric_cols].corr(method=method or "pearson")
        long = corr.rename_axis("x").reset_index().melt("x", var_name="y", value_name="r")
        base = alt.Chart(long).encode(x=alt.X("x:N", title=None), y=alt.Y("y:N", title=None))
        heat = base.mark_rect().encode(
            color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="r"),
            tooltip=["x", "y", alt.Tooltip("r:Q", format=".2f")],
        )
        text = base.mark_text(fontSize=11).encode(text=alt.Text("r:Q", format=".2f"))
        side = max(300, 45 * len(numeric_cols))
        st.altair_chart((heat + text).properties(height=side))

        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = corr.where(mask).stack().rename("r").reset_index()
        pairs.columns = ["column A", "column B", "r"]
        pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index)
        st.subheader("Strongest relationships")
        st.dataframe(
            pairs.head(10),
            hide_index=True,
            column_config={"r": st.column_config.ProgressColumn("r", format="%.2f", min_value=-1, max_value=1)},
        )

# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------
with rel_tab:
    if not numeric_cols:
        st.info("No numeric columns available for relationship plots.")
    else:
        with st.container(horizontal=True):
            x = st.selectbox("X axis", numeric_cols, index=0)
            y = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1))
            color = st.selectbox("Color by", ["(none)"] + categorical_cols)
            trend = st.toggle("Trend line", value=True)

        plot_df = df.sample(min(len(df), 5000), random_state=0)
        enc = {"x": alt.X(f"{x}:Q", scale=alt.Scale(zero=False)), "y": alt.Y(f"{y}:Q", scale=alt.Scale(zero=False)),
               "tooltip": list(dict.fromkeys([x, y] + ([color] if color != "(none)" else [])))}
        if color != "(none)":
            enc["color"] = alt.Color(f"{color}:N")
        points = alt.Chart(plot_df).mark_circle(opacity=0.6, size=50).encode(**enc).interactive()
        chart = points
        if trend:
            chart = points + alt.Chart(plot_df).transform_regression(x, y).mark_line(color="firebrick").encode(x=f"{x}:Q", y=f"{y}:Q")
        st.altair_chart(chart)
        if len(plot_df) < len(df):
            st.caption(f"Plotting a random sample of {len(plot_df):,} rows.")

        if categorical_cols:
            st.subheader("Numeric distribution by category")
            with st.container(horizontal=True):
                num_c = st.selectbox("Numeric column", numeric_cols, key="box_num")
                cat_c = st.selectbox("Category column", categorical_cols, key="box_cat")
            top = df[cat_c].value_counts().head(20).index
            st.altair_chart(
                alt.Chart(df[df[cat_c].isin(top)])
                .mark_boxplot()
                .encode(x=alt.X(f"{cat_c}:N", title=None), y=f"{num_c}:Q", color=alt.Color(f"{cat_c}:N", legend=None))
            )

# ---------------------------------------------------------------------------
# Full profile report (generated on demand - it is slow)
# ---------------------------------------------------------------------------
with report_tab:
    st.write("Generate a complete ydata-profiling report for the filtered data.")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        minimal = st.toggle("Minimal mode (faster for large data)", value=len(df) > 20000)
        generate = st.button("Generate report", type="primary", icon=":material/play_arrow:")

    if generate:
        st.session_state["report_ready"] = True

    if st.session_state.get("report_ready"):
        try:
            report_html = build_profile_html(df, minimal, f"Profile of {dataset_name}")
        except ImportError as e:
            st.error(
                f"ydata-profiling could not be loaded ({e}). Check that `requirements.txt` "
                "pins `setuptools<81` and reboot the app.",
                icon=":material/error:",
            )
            st.stop()
        st.download_button(
            "Download report (HTML)",
            report_html,
            file_name=f"{dataset_name}_report.html",
            mime="text/html",
            icon=":material/download:",
        )
        st.iframe(report_html, height=900)

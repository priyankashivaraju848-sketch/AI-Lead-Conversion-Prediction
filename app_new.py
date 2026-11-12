

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import clone

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="LeadIQ | AI Sales Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "cleaned_leads.csv"
AI_DIR = BASE_DIR / "src" / "Member3_AI_Recommendation"

# ----------------------------- PREMIUM UI ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#14213d; --muted:#64748b; --blue:#4361ee; --cyan:#4cc9f0; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#f7f9ff 0%,#f3f7fb 52%,#f8faff 100%); color:var(--ink); }
.block-container { padding-top:1.25rem; padding-bottom:2.5rem; max-width:1500px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:#14213d !important; letter-spacing:-.5px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#111c3a 0%,#182b55 100%); }
[data-testid="stSidebar"] * { color:#eef4ff !important; }
[data-testid="stMetric"] { background:rgba(255,255,255,.92); border:1px solid #e6ecf5; padding:17px 18px; border-radius:16px; box-shadow:0 5px 18px rgba(23,42,84,.045); }
[data-testid="stMetricLabel"] { color:#64748b !important; font-weight:600; }
[data-testid="stMetricValue"] { color:#14213d !important; font-weight:800; }
div[data-testid="stTabs"] button { border-radius:12px 12px 0 0; font-weight:700; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#3155d9; border-bottom:3px solid #4361ee; }
div.stButton > button, div.stFormSubmitButton > button { border-radius:11px; font-weight:700; border:0; background:linear-gradient(90deg,#4361ee,#4895ef); color:white; padding:.55rem 1rem; }
div.stButton > button:hover, div.stFormSubmitButton > button:hover { color:white; border:0; filter:brightness(1.06); }
div[data-testid="stPlotlyChart"] { background:white; border:1px solid #e8edf5; border-radius:16px; padding:8px; box-shadow:0 4px 16px rgba(23,42,84,.035); }
.hero { background:linear-gradient(110deg,#172554 0%,#2947a8 55%,#4361ee 100%); padding:26px 30px; border-radius:22px; color:white; margin-bottom:18px; box-shadow:0 12px 28px rgba(42,70,160,.18); }
.hero h1 { color:white !important; font-size:2rem; margin:0; }
.hero p { color:#dbeafe; margin:.45rem 0 0 0; font-size:1rem; }
.pill { display:inline-block; background:#eaf0ff; color:#3155d9; border-radius:99px; padding:5px 11px; font-size:12px; font-weight:800; }
.section-note { color:#64748b; font-size:13px; margin-top:-8px; margin-bottom:12px; }
div[data-testid="stExpander"] { background:rgba(255,255,255,.8); border:1px solid #e6ecf5; border-radius:14px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- DATA & MODEL -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    data = pd.read_csv(DATA_PATH)
    if "Converted" not in data.columns:
        raise ValueError("Dataset must contain a Converted target column.")
    return data

@st.cache_resource(show_spinner="Training prediction models for this session...")
def train_models(csv_path, file_mtime):
    data = pd.read_csv(csv_path)
    X = data.drop(columns=["Converted"])
    y = data["Converted"].astype(int)
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical = X.select_dtypes(include=np.number).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numerical),
        ("cat", categorical_pipe, categorical)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )
    candidates = {
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=220, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=10, class_weight="balanced", random_state=42
        ),
        "Support Vector Machine (SVM)": SVC(
            kernel="rbf", C=1.0, probability=True,
            class_weight="balanced", random_state=42
        )
    }
    models, rows = {}, []
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })
        models[name] = pipe
    comparison = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    best_name = comparison.iloc[0]["Model"]
    return models[best_name], comparison, best_name, X_test, y_test

def lead_category(prob, high_cut, medium_cut):
    if prob >= high_cut:
        return "High"
    if prob >= medium_cut:
        return "Medium"
    return "Low"

def safe_ai_imports():
    if str(AI_DIR) not in sys.path:
        sys.path.insert(0, str(AI_DIR))
    try:
        from main_ai import run_lead_ai, ask_about_lead
        return run_lead_ai, ask_about_lead, None
    except Exception as exc:
        return None, None, str(exc)

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load cleaned_leads.csv: {exc}")
    st.stop()

predictor, model_comparison, best_model_name, X_test, y_test = train_models(
    str(DATA_PATH), DATA_PATH.stat().st_mtime
)
X_all = df.drop(columns=["Converted"])
with st.spinner("Scoring leads with the trained model..."):
    all_prob = predictor.predict_proba(X_all)[:, 1]

# ----------------------------- SIDEBAR FILTERS --------------------------
with st.sidebar:
    st.markdown("## 📈 LeadIQ")
    st.caption("AI-powered sales intelligence")
    st.divider()
    st.markdown("### Global filters")
    source_values = sorted(df["Lead Source"].fillna("Unknown").astype(str).unique()) if "Lead Source" in df else []
    origin_values = sorted(df["Lead Origin"].fillna("Unknown").astype(str).unique()) if "Lead Origin" in df else []
    selected_sources = st.multiselect("Lead source", source_values, default=source_values)
    selected_origins = st.multiselect("Lead origin", origin_values, default=origin_values)
    min_visits = float(df["TotalVisits"].min()) if "TotalVisits" in df else 0
    max_visits = float(df["TotalVisits"].max()) if "TotalVisits" in df else 0
    if max_visits > min_visits:
        visits_range = st.slider("Website visits", min_value=int(min_visits), max_value=int(max_visits),
                                 value=(int(min_visits), int(max_visits)))
    else:
        visits_range = (int(min_visits), int(max_visits))
    st.divider()
    st.markdown("### Prediction settings")
    high_cut = st.slider("High-potential threshold", .50, .95, .70, .01)
    medium_cut = st.slider("Medium-potential threshold", .20, .70, .40, .01)
    if medium_cut >= high_cut:
        st.warning("Medium threshold should be lower than High threshold.")
        medium_cut = max(.20, high_cut - .01)
    st.caption(f"Model used: {best_model_name}")
    st.caption(f"Records loaded: {len(df):,}")

analytics = df.copy()
analytics["Conversion Probability"] = all_prob
analytics["Lead Category"] = [
    lead_category(p, high_cut, medium_cut) for p in all_prob
]
analytics["Predicted Conversion"] = (all_prob >= .50).astype(int)

filtered = analytics.copy()
if selected_sources and "Lead Source" in filtered:
    filtered = filtered[filtered["Lead Source"].fillna("Unknown").astype(str).isin(selected_sources)]
if selected_origins and "Lead Origin" in filtered:
    filtered = filtered[filtered["Lead Origin"].fillna("Unknown").astype(str).isin(selected_origins)]
if "TotalVisits" in filtered:
    filtered = filtered[filtered["TotalVisits"].fillna(0).between(*visits_range)]

# ----------------------------- HEADER -----------------------------------
st.markdown("""
<div class="hero">
  <div class="pill">SALES PERFORMANCE • MACHINE LEARNING • AI RECOMMENDATIONS</div>
  <h1>Lead Conversion Intelligence</h1>
  <p>One workspace to understand your leads, predict conversion, and decide the next best sales action.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"<span class='pill'>LIVE FILTER VIEW · {len(filtered):,} LEADS</span> "
    f"<span class='pill'>BEST MODEL · {best_model_name.upper()}</span>",
    unsafe_allow_html=True
)
tabs = st.tabs([
    "01  Executive Overview",
    "02  Lead Analytics",
    "03  Lead Prediction",
    "04  AI Sales Assistant",
    "05  Model Insights"
])

# ============================= TAB 1 ====================================
with tabs[0]:
    st.subheader("Executive Overview")
    st.markdown("<div class='section-note'>A real-time snapshot based on the global filters in the sidebar.</div>", unsafe_allow_html=True)
    total = len(filtered)
    converted = int(filtered["Converted"].sum()) if total else 0
    conversion_rate = (converted / total * 100) if total else 0
    high_count = int((filtered["Leahow to push" \
    "d Category"] == "High").sum())
    avg_prob = float(filtered["Conversion Probability"].mean() * 100) if total else 0
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total leads", f"{total:,}")
    k2.metric("Converted", f"{converted:,}")
    k3.metric("Actual conversion rate", f"{conversion_rate:.1f}%")
    k4.metric("High-potential leads", f"{high_count:,}")
    k5.metric("Avg. predicted probability", f"{avg_prob:.1f}%")
    left,right = st.columns(2)
    with left:
        counts = filtered["Lead Category"].value_counts().reindex(["High","Medium","Low"], fill_value=0).rename_axis("Potential").reset_index(name="Leads")
        fig = px.pie(counts, names="Potential", values="Leads", hole=.62,
                     title="Predicted Lead Potential", color="Potential",
                     color_discrete_map={"High":"#2ec4b6","Medium":"#ffb703","Low":"#8d99ae"})
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        outcomes = pd.DataFrame({"Outcome":["Not converted","Converted"],
                                 "Leads":[int((filtered["Converted"]==0).sum()), converted]})
        fig = px.bar(outcomes, x="Outcome", y="Leads", color="Outcome", text_auto=True,
                     title="Actual Conversion Outcome",
                     color_discrete_map={"Not converted":"#8d99ae","Converted":"#4361ee"})
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of leads")
        st.plotly_chart(fig, use_container_width=True)
    if "Lead Source" in filtered:
        source = filtered.groupby("Lead Source", dropna=False)["Converted"].agg(
            Leads="count", Converted="sum", Conversion_Rate="mean"
        ).reset_index()
        source["Conversion Rate (%)"] = source["Conversion_Rate"] * 100
        source = source.sort_values("Leads", ascending=False).head(12)
        fig = px.bar(source, x="Lead Source", y="Conversion Rate (%)",
                     hover_data=["Leads","Converted"], text_auto=".1f",
                     title="Conversion Rate by Lead Source", color="Conversion Rate (%)",
                     color_continuous_scale="Blues")
        fig.update_layout(xaxis_tickangle=-25, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Dataset has no date/time field, so a true month-wise trend is not available. Source performance is shown instead.")

# ============================= TAB 2 ====================================
with tabs[1]:
    st.subheader("Lead Analytics & Segmentation")
    st.markdown("<div class='section-note'>Explore lead quality, engagement, and conversion patterns.</div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    category_options = ["All","High","Medium","Low"]
    category_filter = c1.selectbox("Potential segment", category_options, key="analytics_cat")
    activity_values = sorted(filtered["Last Activity"].fillna("Unknown").astype(str).unique()) if "Last Activity" in filtered else []
    activity_filter = c2.selectbox("Last activity", ["All"]+activity_values, key="analytics_activity")
    probability_floor = c3.slider("Minimum probability (%)", 0, 100, 0, 5, key="analytics_prob")
    view = filtered.copy()
    if category_filter != "All":
        view = view[view["Lead Category"] == category_filter]
    if activity_filter != "All" and "Last Activity" in view:
        view = view[view["Last Activity"].fillna("Unknown").astype(str) == activity_filter]
    view = view[view["Conversion Probability"] >= probability_floor/100]
    a,b,c = st.columns(3)
    a.metric("Matching leads", f"{len(view):,}")
    b.metric("Actual conversions", f"{int(view['Converted'].sum()):,}" if len(view) else "0")
    b_rate = float(view["Converted"].mean()*100) if len(view) else 0
    c.metric("Actual conversion rate", f"{b_rate:.1f}%")
    left,right = st.columns(2)
    with left:
        fig = px.histogram(view, x="Conversion Probability", color="Lead Category", nbins=24,
                           title="Conversion Probability Distribution",
                           color_discrete_map={"High":"#2ec4b6","Medium":"#ffb703","Low":"#8d99ae"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if "Lead Origin" in view:
            origin_perf = view.groupby("Lead Origin", dropna=False)["Converted"].mean().mul(100).reset_index(name="Conversion Rate (%)")
            fig = px.bar(origin_perf, x="Lead Origin", y="Conversion Rate (%)",
                         title="Conversion Rate by Lead Origin", text_auto=".1f")
            fig.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
    show_cols = [c for c in ["Lead Origin","Lead Source","TotalVisits","Total Time Spent on Website",
                              "Page Views Per Visit","Last Activity","Conversion Probability",
                              "Lead Category","Converted"] if c in view.columns]
    st.markdown("#### Lead records")
    st.dataframe(view[show_cols].sort_values("Conversion Probability", ascending=False),
                 use_container_width=True, hide_index=True, height=360)
    st.download_button("⬇ Download filtered leads (CSV)", view.to_csv(index=False).encode("utf-8"),
                       "filtered_leads.csv", "text/csv")

# ============================= TAB 3 ====================================
with tabs[2]:
    st.subheader("Individual Lead Prediction")
    st.markdown("<div class='section-note'>Change the lead attributes and run the trained model to estimate conversion likelihood.</div>", unsafe_allow_html=True)
    st.info("The prediction uses the best-performing model selected from XGBoost, Random Forest, Decision Tree and Support Vector Machine (SVM) on a held-out test split.")
    with st.form("prediction_form", border=True):
        inputs = {}
        cols = st.columns(2)
        for i, col in enumerate(X_all.columns):
            with cols[i % 2]:
                series = X_all[col]
                if pd.api.types.is_numeric_dtype(series):
                    med = float(series.median()) if series.notna().any() else 0.0
                    mn = float(series.min()) if series.notna().any() else 0.0
                    mx = float(series.max()) if series.notna().any() else 100.0
                    if mn == mx: mx = mn + 1.0
                    inputs[col] = st.number_input(col, min_value=mn, max_value=mx,
                                                  value=min(max(med,mn),mx), key=f"input_{col}")
                else:
                    opts = sorted(series.dropna().astype(str).unique().tolist()) or ["Unknown"]
                    inputs[col] = st.selectbox(col, opts, key=f"input_{col}")
        submitted = st.form_submit_button("✨ Predict conversion probability", use_container_width=True)
    if submitted:
        row = pd.DataFrame([inputs], columns=X_all.columns)
        try:
            probability = float(predictor.predict_proba(row)[0][1])
            cat = lead_category(probability, high_cut, medium_cut)
            st.session_state["predicted_lead"] = inputs
            st.session_state["predicted_probability"] = probability
            st.session_state["predicted_category"] = cat
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    if "predicted_probability" in st.session_state:
        probability = st.session_state["predicted_probability"]
        cat = st.session_state["predicted_category"]
        p1,p2,p3 = st.columns(3)
        p1.metric("Conversion probability", f"{probability*100:.2f}%")
        p2.metric("Lead potential", cat)
        p3.metric("Suggested priority", {"High":"Contact immediately","Medium":"Nurture soon","Low":"Long-term nurture"}[cat])
        st.progress(probability)
        st.caption("This is a model-estimated probability, not a guarantee that the lead will convert.")
        st.markdown("#### AI explanation & recommendation")
        run_ai, _, import_error = safe_ai_imports()
        if run_ai is None:
            st.warning(f"AI module unavailable: {import_error}")
        elif st.button("Generate AI sales recommendation", key="generate_recommendation"):
            try:
                with st.spinner("Preparing personalized sales recommendations..."):
                    result = run_ai(st.session_state["predicted_lead"], probability)
                st.markdown("**Prediction explanation**")
                st.write(result.get("explanation","Not returned"))
                st.markdown("**Recommended sales action**")
                st.write(result.get("recommendation","Not returned"))
                st.markdown("**Follow-up strategy**")
                st.write(result.get("follow_up","Not returned"))
            except Exception as exc:
                st.error(f"AI request failed. Set OPENROUTER_API_KEY and check API access. Details: {exc}")

# ============================= TAB 4 ====================================
with tabs[3]:
    st.subheader("AI Sales Assistant")
    st.markdown("<div class='section-note'>Ask practical questions about a lead and receive next-step guidance.</div>", unsafe_allow_html=True)
    row_number = st.number_input("Choose lead row", min_value=1, max_value=len(df), value=1, step=1, key="assistant_row")
    idx = int(row_number)-1
    lead = X_all.iloc[idx].to_dict()
    prob = float(all_prob[idx])
    cat = lead_category(prob, high_cut, medium_cut)
    x,y,z = st.columns(3)
    x.metric("Conversion probability", f"{prob*100:.1f}%")
    y.metric("Potential", cat)
    z.metric("Lead source", str(lead.get("Lead Source","Not available")))
    with st.expander("View selected lead details"):
        st.json(lead)
    question = st.text_area("Ask your sales question",
        placeholder="Example: Why is this lead high potential? What should my next action be?",
        key="assistant_question")
    if st.button("Ask AI Sales Assistant", key="ask_assistant", disabled=not question.strip()):
        _, ask_ai, import_error = safe_ai_imports()
        if ask_ai is None:
            st.error(f"AI module unavailable: {import_error}")
        else:
            try:
                with st.spinner("AI assistant is thinking..."):
                    answer = ask_ai(question, lead, prob, cat)
                st.markdown("#### AI response")
                st.write(answer)
            except Exception as exc:
                st.error(f"AI request failed. Set OPENROUTER_API_KEY and check API access. Details: {exc}")

# ============================= TAB 5 ====================================
with tabs[4]:
    st.subheader("Machine Learning Model Insights")
    st.markdown("<div class='section-note'>Compare candidate classifiers and inspect the selected model's test performance.</div>", unsafe_allow_html=True)
    best = model_comparison.iloc[0]
    m1,m2,m3 = st.columns(3)
    m1.metric("Selected model", best_model_name)
    m2.metric("Test F1-score", f"{best['F1 Score']:.3f}")
    m3.metric("Test ROC-AUC", f"{best['ROC-AUC']:.3f}")
    st.dataframe(model_comparison.style.format({
        "Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}",
        "F1 Score":"{:.3f}","ROC-AUC":"{:.3f}"
    }), use_container_width=True, hide_index=True)
    metric_choice = st.multiselect("Metrics to compare",
        ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
        default=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"])
    if metric_choice:
        melted = model_comparison.melt(id_vars="Model", value_vars=metric_choice,
                                       var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group",
                     title="Candidate Model Performance", text_auto=".2f",
                     range_y=[0,1])
        st.plotly_chart(fig, use_container_width=True)
    threshold = st.slider("Classification threshold for confusion matrix", .10, .90, .50, .05)
    test_prob = predictor.predict_proba(X_test)[:,1]
    test_pred = (test_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, test_pred, labels=[0,1])
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                    x=["Predicted: Not converted","Predicted: Converted"],
                    y=["Actual: Not converted","Actual: Converted"],
                    color_continuous_scale="Blues", title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Model metrics are measured on a held-out 20% test split. The classification threshold above changes the confusion matrix interactively; it does not retrain the model.")
    st.markdown("#### Lead potential thresholds")
    st.write(f"- High potential: probability ≥ {high_cut:.0%}")
    st.write(f"- Medium potential: {medium_cut:.0%} to below {high_cut:.0%}")
    st.write(f"- Low potential: below {medium_cut:.0%}")

st.divider()
st.caption("LeadIQ • AI-Based Lead Conversion Prediction and Intelligent Sales Recommendation System | Built with Streamlit, Plotly, Scikit-learn and OpenRouter")

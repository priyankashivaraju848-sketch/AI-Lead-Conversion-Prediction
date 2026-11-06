import os
import sys
import warnings
from pathlib import Path
from xgboost import XGBClassifier

import joblib
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

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR.parent / "dataset" / "cleaned_leads.csv"
MODEL_PATH = BASE_DIR.parent / "models" / "best_lead_model.json"
PREPROCESSOR_PATH = BASE_DIR.parent / "models" / "preprocessor.pkl"

AI_DIR = BASE_DIR.parent / "src" / "Member3_AI_Recommendation"

st.set_page_config(
    page_title="AI Lead Conversion Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    [data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        border: 1px solid #e7ebf2;
    }
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    h1, h2, h3 {color: #172554;}
    div[data-testid="stForm"] {background: white; padding: 18px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("📊 AI-Based Lead Conversion & Intelligent Sales Recommendation")
st.caption("Business Analytics Dashboard | Member 4")

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "cleaned_leads.csv is missing. Place Member 1's cleaned CSV beside app.py."
        )
    data = pd.read_csv(DATA_PATH)
    if "Converted" not in data.columns:
        raise ValueError("The dataset must contain the target column 'Converted'.")
    return data

@st.cache_resource
def train_dashboard_model(csv_path, file_mtime):
    """Fallback model for dashboard execution when Member 2's model artifact is absent/incompatible.
    It trains on Member 1's raw cleaned CSV using a fresh preprocessing pipeline.
    Replace with Member 2's compatible model + preprocessor for final integration.
    """
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
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numerical),
        ("cat", categorical_pipe, categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=10, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    }
    fitted = {}
    rows = []
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", preprocess), ("model", estimator)])
        # Build an independent preprocessing instance for each estimator.
        from sklearn.base import clone
        pipe = Pipeline([("preprocessor", clone(preprocess)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, prob)
        except ValueError:
            auc = 0.0
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": auc
        })
        fitted[name] = pipe

    comparison = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    best_name = comparison.iloc[0]["Model"]
    return fitted[best_name], comparison, best_name, X_test, y_test

def load_predictor(data):
    """Load Member 2's verified XGBoost model and preprocessor."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    class ExternalPredictor:
        def predict_proba(self, raw_df):
            processed = preprocessor.transform(raw_df)
            return model.predict_proba(processed)

        def predict(self, raw_df):
            processed = preprocessor.transform(raw_df)
            return model.predict(processed)

    return ExternalPredictor(), "Member 2 XGBoost model"

def category(probability):
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"

def probability_for(predictor, row):
    prediction = predictor.predict_proba(row)
    probability = float(prediction[0][1])

    print("\n========== NEW LEAD PREDICTION ==========")
    print("Input values:")
    print(row.to_string(index=False))
    print(f"Predicted probability: {probability * 100:.2f}%")
    print("=========================================\n")

    return probability

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
    st.error(str(exc))
    st.stop()

predictor, predictor_source = load_predictor(df)
X_all = df.drop(columns=["Converted"])
with st.spinner("Calculating lead conversion probabilities..."):
    all_prob = predictor.predict_proba(X_all)[:, 1]
    analytics = df.copy()
    analytics["Conversion Probability"] = all_prob
    analytics["Lead Category"] = [category(p) for p in all_prob]
    analytics["Predicted Conversion"] = (all_prob >= 0.5).astype(int)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=62)
    st.header("Navigation")
    page = st.radio(
        "Select dashboard page",
        ["Overview", "Lead Analytics", "Lead Prediction", "AI Sales Assistant", "Model Insights"]
    )
    st.divider()
    st.caption(f"Prediction source: {predictor_source}")
    st.caption(f"Dataset: {len(df):,} leads")

if page == "Overview":
    st.subheader("Business Overview")
    total = len(df)
    converted = int(df["Converted"].sum())
    rate = (converted / total * 100) if total else 0
    high = int((analytics["Lead Category"] == "High").sum())
    medium = int((analytics["Lead Category"] == "Medium").sum())
    low = int((analytics["Lead Category"] == "Low").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Leads", f"{total:,}")
    c2.metric("Converted Leads", f"{converted:,}")
    c3.metric("Overall Conversion Rate", f"{rate:.2f}%")
    c4.metric("High-Potential Leads", f"{high:,}")

    st.markdown("### Lead Potential Distribution")
    left, right = st.columns(2)
    with left:
        counts = analytics["Lead Category"].value_counts().reindex(
            ["High", "Medium", "Low"], fill_value=0
        ).rename_axis("Category").reset_index(name="Leads")
        fig = px.pie(
            counts, names="Category", values="Leads", hole=0.55,
            title="Predicted Lead Categories"
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        actual = pd.DataFrame({
            "Outcome": ["Not Converted", "Converted"],
            "Leads": [(df["Converted"] == 0).sum(), (df["Converted"] == 1).sum()]
        })
        fig = px.bar(actual, x="Outcome", y="Leads", color="Outcome",
                     title="Historical Conversion Outcome", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Lead Source Performance")
    if "Lead Source" in df.columns:
        source = df.groupby("Lead Source", dropna=False)["Converted"].agg(
            Leads="count", Converted="sum", Conversion_Rate="mean"
        ).reset_index()
        source["Conversion Rate (%)"] = source["Conversion_Rate"] * 100
        source = source.sort_values("Leads", ascending=False).head(12)
        fig = px.bar(
            source, x="Lead Source", y="Conversion Rate (%)",
            hover_data=["Leads", "Converted"], title="Conversion Rate by Lead Source"
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The supplied dataset has no date/time column. Therefore, a true month-wise "
        "conversion trend cannot be calculated. The dashboard shows historical outcomes "
        "and conversion performance by lead source instead."
    )

elif page == "Lead Analytics":
    st.subheader("Lead Analytics & Segmentation")
    a, b = st.columns(2)
    with a:
        if "Lead Source" in analytics.columns:
            sources = ["All"] + sorted(analytics["Lead Source"].fillna("Unknown").astype(str).unique())
            selected_source = st.selectbox("Filter by Lead Source", sources)
        else:
            selected_source = "All"
    with b:
        selected_category = st.selectbox("Filter by Potential", ["All", "High", "Medium", "Low"])

    view = analytics.copy()
    if selected_source != "All":
        view = view[view["Lead Source"].fillna("Unknown").astype(str) == selected_source]
    if selected_category != "All":
        view = view[view["Lead Category"] == selected_category]

    st.metric("Leads matching filters", f"{len(view):,}")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(view, x="Conversion Probability", nbins=20,
                           title="Distribution of Conversion Probability")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "Lead Origin" in view.columns:
            fig = px.histogram(view, x="Lead Origin", color="Lead Category",
                               barmode="group", title="Lead Origin by Potential")
            fig.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

    display_cols = [c for c in [
        "Lead Source", "Lead Origin", "TotalVisits",
        "Total Time Spent on Website", "Conversion Probability",
        "Lead Category", "Converted"
    ] if c in view.columns]
    st.markdown("### Lead Records")
    st.dataframe(
        view[display_cols].sort_values("Conversion Probability", ascending=False),
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "Download filtered leads as CSV",
        view.to_csv(index=False).encode("utf-8"),
        "filtered_leads.csv", "text/csv"
    )

elif page == "Lead Prediction":
    st.subheader("Individual Lead Prediction")
    st.write("Enter the lead attributes below to estimate conversion probability.")
    with st.form("lead_prediction_form"):
        inputs = {}
        columns = st.columns(2)
        for i, col in enumerate(X_all.columns):
            with columns[i % 2]:
                if pd.api.types.is_numeric_dtype(X_all[col]):
                    median = float(X_all[col].median()) if X_all[col].notna().any() else 0.0
                    minimum = float(X_all[col].min()) if X_all[col].notna().any() else 0.0
                    maximum = float(X_all[col].max()) if X_all[col].notna().any() else 100.0
                    if minimum == maximum:
                        maximum = minimum + 1.0
                    inputs[col] = st.number_input(
                        col, min_value=minimum, max_value=maximum,
                        value=min(max(median, minimum), maximum), key=f"pred_{col}"
                    )
                else:
                    choices = sorted(X_all[col].dropna().astype(str).unique().tolist())
                    if not choices:
                        choices = ["Unknown"]
                    inputs[col] = st.selectbox(col, choices, key=f"pred_{col}")
        submitted = st.form_submit_button("Predict Conversion Probability", type="primary")

    if submitted:
        row = pd.DataFrame([inputs], columns=X_all.columns)
        try:
            prob = probability_for(predictor, row)
            lead_category = category(prob)
            st.markdown("### Prediction Result")
            c1, c2 = st.columns(2)
            c1.metric("Conversion Probability", f"{prob*100:.2f}%")
            c2.metric("Lead Potential", lead_category)
            st.progress(prob)
            st.caption("Probability is the model-estimated likelihood of Converted = 1.")
            st.session_state["selected_lead"] = inputs
            st.session_state["selected_probability"] = prob
            st.session_state["selected_category"] = lead_category
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if "selected_lead" in st.session_state:
        st.markdown("### AI Explanation & Sales Recommendation")
        run_ai, ask_ai, import_error = safe_ai_imports()
        if run_ai is None:
            st.warning(f"Member 3 AI module could not be imported: {import_error}")
        elif st.button("Generate AI Recommendation", type="primary"):
            try:
                with st.spinner("Generating AI explanation and recommendations..."):
                    ai_result = run_ai(
                        st.session_state["selected_lead"],
                        st.session_state["selected_probability"]
                    )
                st.markdown("**Prediction Explanation**")
                st.write(ai_result.get("explanation", "Not returned"))
                st.markdown("**Recommended Sales Action**")
                st.write(ai_result.get("recommendation", "Not returned"))
                st.markdown("**Follow-up Strategy**")
                st.write(ai_result.get("follow_up", "Not returned"))
            except Exception as exc:
                st.error(
                    "AI request failed. Check OPENROUTER_API_KEY and internet/API availability. "
                    f"Details: {exc}"
                )

elif page == "AI Sales Assistant":
    st.subheader("🤖 AI Sales Assistant")
    st.write("Ask questions about a selected lead's prediction and next sales actions.")
    selected_index = st.number_input(
        "Lead row number (1-based)", min_value=1, max_value=len(df), value=1, step=1
    )
    idx = int(selected_index) - 1
    lead = X_all.iloc[idx].to_dict()
    prob = float(all_prob[idx])
    cat = category(prob)
    c1, c2 = st.columns(2)
    c1.metric("Conversion Probability", f"{prob*100:.2f}%")
    c2.metric("Lead Category", cat)
    with st.expander("Selected lead details"):
        st.json(lead)
    question = st.text_area(
        "Your question",
        placeholder="Why is this lead classified as high potential? What should I do next?"
    )
    if st.button("Ask AI Assistant", type="primary", disabled=not question.strip()):
        run_ai, ask_ai, import_error = safe_ai_imports()
        if ask_ai is None:
            st.error(f"Member 3 AI module could not be imported: {import_error}")
        else:
            try:
                with st.spinner("AI Sales Assistant is preparing an answer..."):
                    answer = ask_ai(question, lead, prob, cat)
                st.markdown("### AI Answer")
                st.write(answer)
            except Exception as exc:
                st.error(
                    "AI request failed. Check OPENROUTER_API_KEY and API availability. "
                    f"Details: {exc}"
                )

elif page == "Model Insights":
    st.subheader("Machine Learning Model Insights")
    if predictor_source == "Dashboard fallback model":
        _, comparison, best_name, X_test, y_test = train_dashboard_model(
            str(DATA_PATH), DATA_PATH.stat().st_mtime
        )
        st.caption("These metrics are generated by the dashboard fallback training run, not Member 2's missing model file.")
        st.dataframe(comparison.style.format({
            "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
            "F1 Score": "{:.3f}", "ROC-AUC": "{:.3f}"
        }), use_container_width=True, hide_index=True)
        fig = px.bar(
            comparison.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric", barmode="group",
            title="Model Performance Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Best fallback model by F1-score: {best_name}")
    else:
        st.success("Member 2's model and preprocessor artifacts are loaded.")
        st.info(
            "To show Member 2's exact Accuracy, Precision, Recall, F1-score, ROC-AUC "
            "and confusion matrix here, provide the model comparison results or evaluation file."
        )

    st.markdown("### Prediction Thresholds")
    st.write("- High potential: probability ≥ 70%")
    st.write("- Medium potential: probability ≥ 40% and < 70%")
    st.write("- Low potential: probability < 40%")
    st.caption(
        "These thresholds follow Member 3's categorize_lead() implementation. "
        "A probability is a model estimate, not a guarantee of conversion."
    )

st.divider()
st.caption(
    "AI-Based Lead Conversion Prediction and Intelligent Sales Recommendation System | "
    "Dashboard developed by Member 4"
)

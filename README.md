# AI-Based Lead Conversion Prediction and Intelligent Sales Recommendation System

## 1. Project Overview

The **AI-Based Lead Conversion Prediction and Intelligent Sales Recommendation System** is a machine learning-based business analytics application designed to predict the likelihood of a lead converting into a customer.

The system uses an XGBoost classification model to estimate lead conversion probability and categorizes leads into High, Medium, and Low potential. It also provides explanations, sales recommendations, and follow-up strategies through an AI recommendation module.

A Streamlit dashboard provides an interactive interface for lead analysis, prediction, and business insights.

## 2. Objectives

- Predict the probability of lead conversion using machine learning.
- Classify leads based on their conversion potential.
- Provide explanations for individual lead predictions.
- Generate intelligent sales recommendations and follow-up strategies.
- Present lead analytics through an interactive dashboard.
- Maintain functionality through a rule-based fallback when the external LLM service is unavailable.

## 3. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Dashboard | Streamlit |
| AI Integration | OpenRouter API |
| Model Storage | JSON, Pickle |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## 4. Dataset

**Dataset:** X Education Lead Scoring Dataset

- Total records: 9,240 leads
- Target variable: `Converted`
- The dataset contains lead-related attributes such as website visits, website time, lead source, specialization, and other engagement information.

The dataset was cleaned and preprocessed before model training.

## 5. Machine Learning Model

The project uses an **XGBoost Classifier** to predict lead conversion probability.

The model processes the lead attributes using a saved preprocessing pipeline and generates a conversion probability.

### Lead Classification

| Conversion Probability | Category |
|---|---|
| 70% and above | High |
| 40% to below 70% | Medium |
| Below 40% | Low |

The machine learning model is the authoritative source of the conversion probability and category.

## 6. AI Recommendation Module

The AI module uses the ML prediction and supplied lead information to generate:

- Prediction explanations
- Recommended sales actions
- Follow-up strategies
- Personalized sales messages
- Lead-related question answering through the AI Sales Assistant

The AI module does not independently determine the conversion probability.

**Fallback mechanism:** When OpenRouter is unavailable or its free quota is exhausted, the system uses locally implemented rule-based responses to keep the dashboard functional. These fallback responses are not LLM-generated.

## 7. Dashboard Features

The Streamlit dashboard includes five pages:

1. **Overview:** Displays an overall summary of the lead dataset.
2. **Lead Analytics:** Provides visual analysis of lead information.
3. **Lead Prediction:** Allows users to enter lead attributes and obtain conversion predictions.
4. **AI Sales Assistant:** Provides lead-related explanations and sales guidance.
5. **Model Insights:** Displays information about the machine learning model.

## 8. Project Structure

```text
AI-Lead-Conversion-Prediction/
│
├── dashboard/
│   └── app.py
│
├── dataset/
│   ├── cleaned_leads.csv
│   └── README.md
│
├── docs/
│
├── models/
│   ├── best_lead_model.json
│   ├── preprocessor.pkl
│   └── README.md
│
├── notebooks/
│   ├── ai_lead_project.ipynb
│   └── README.md
│
├── reports/
│
├── screenshots/
│
├── src/
│   └── Member3_AI_Recommendation/
│       ├── ai_client.py
│       ├── ai_explanation.py
│       ├── ai_sales_assistant.py
│       ├── ai_utils.py
│       ├── follow_up_strategy.py
│       ├── lead_utils.py
│       ├── main_ai.py
│       └── recommendation.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 9. Installation and Execution

### Step 1: Clone the repository

```bash
git clone <https://github.com/priyankashivaraju848-sketch/AI-Lead-Conversion-Prediction.git>
cd AI-Lead-Conversion-Prediction
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure the OpenRouter API (Optional)

Set your API key in PowerShell:

```powershell
$env:OPENROUTER_API_KEY="YOUR_API_KEY"
```

The dashboard can use its local rule-based fallback when the external AI service is unavailable.

### Step 5: Run the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

## 10. Team Contributions

| Member | Responsibility |
|---|---|
| Member 2 | Machine learning model development and preprocessing |
| Member 3 | AI explanation and sales recommendation module |
| Member 4 | Streamlit dashboard development |
| Member 5 | Module integration, testing, GitHub management, documentation, and demonstration |

## 11. Screenshots

Dashboard screenshots are available in the `screenshots/` directory.

- Overview
- Lead Analytics
- Lead Prediction
- AI Sales Assistant
- Model Insights

## 12. Expected Outcome

The system helps sales teams prioritize leads using machine learning predictions and supports decision-making through explanations, intelligent recommendations, and interactive business analytics.

## 13. Future Enhancements

- Integrate additional lead engagement data.
- Improve AI recommendation personalization.
- Add real-time lead monitoring.
- Introduce advanced model evaluation and comparison.
- Improve dashboard reporting and export capabilities.

---

**Project Title:** AI-Based Lead Conversion Prediction and Intelligent Sales Recommendation System

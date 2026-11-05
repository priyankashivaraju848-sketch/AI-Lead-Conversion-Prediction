# AI-Based Lead Conversion Prediction and Intelligent Sales Recommendation System
## Member 4 – Business Analytics Dashboard & Visualization

### Files
- `app.py` – Streamlit dashboard.
- `cleaned_leads.csv` – cleaned dataset supplied by Member 1.
- `Member3_AI_Recommendation/` – AI functions supplied by Member 3.
- `requirements.txt` – Python dependencies.

### Run locally
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

### AI setup
Member 3's module uses OpenRouter. Set the API key as an environment variable:
```powershell
# Windows PowerShell
$env:OPENROUTER_API_KEY="your_key_here"
```
Do not hard-code the key or upload it to GitHub.

### Dashboard pages
1. Overview – KPI cards, lead potential distribution, historical conversion outcome and lead-source performance.
2. Lead Analytics – filters, charts, lead table and CSV export.
3. Lead Prediction – individual lead form, probability, category and AI recommendations.
4. AI Sales Assistant – question answering for a selected lead.
5. Model Insights – performance comparison for the fallback model and lead category thresholds.

### Important integration note
The uploaded Member 1/2 ZIPs contain the cleaned CSV, preprocessing pipeline, processed train/test arrays and feature names. They do **not** contain `best_lead_model.pkl`. Also, the supplied `preprocessor.pkl` was serialized with a different scikit-learn version and may not load in every environment.

For immediate dashboard execution, `app.py` trains a dashboard fallback pipeline on the cleaned CSV and labels it as such. This is not a replacement for Member 2's final model. For final team integration, Member 2/5 should provide a compatible `best_lead_model.pkl` and `preprocessor.pkl` generated in the same Python/scikit-learn environment, or preferably one complete sklearn Pipeline artifact. Place the compatible files beside `app.py`.

### Dataset limitation
The supplied cleaned dataset has 9,240 rows and 11 columns, with `Converted` as the target. It has no date/time field, so a genuine time-based conversion trend cannot be plotted. The dashboard transparently displays historical outcomes and conversion rate by lead source instead.

### Lead category thresholds
- High: probability >= 0.70
- Medium: probability >= 0.40 and < 0.70
- Low: probability < 0.40

These match Member 3's categorization function.

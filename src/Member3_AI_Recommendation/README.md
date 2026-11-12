# Member 3 – AI-Based Lead Recommendation Module

## Files
- `ai_client.py`: Creates the OpenRouter client from an environment variable.
- `ai_utils.py`: Shared API request and retry helper.
- `lead_utils.py`: High/Medium/Low lead categorization.
- `ai_explanation.py`: Explains an ML prediction cautiously.
- `recommendation.py`: Generates a sales recommendation report.
- `follow_up_strategy.py`: Generates a follow-up plan and message.
- `ai_sales_assistant.py`: Answers salesperson questions about a lead.
- `main_ai.py`: Main integration functions.

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

Set your OpenRouter key securely as `OPENROUTER_API_KEY`.

Google Colab example:
```python
import os
from google.colab import userdata
os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")
```

Never hard-code the key or commit it to GitHub.

## Example integration
```python
from main_ai import run_lead_ai, ask_about_lead

lead_details = {
    "Lead Source": "olark chat",
    "TotalVisits": 0,
    "Total Time Spent on Website": 0,
    "Page Views Per Visit": 0,
    "Last Activity": "Page Visited on Website"
}
probability = 0.1518  # Replace with Member 2's predict_proba output for Converted=1.

result = run_lead_ai(lead_details, probability)
print(result["category"])
print(result["explanation"])
print(result["recommendation"])
print(result["follow_up"])

answer = ask_about_lead(
    "What sales action should I take?",
    lead_details,
    probability,
    result["category"]
)
print(answer)
```

## Probability thresholds
- High: >= 0.70
- Medium: >= 0.40 and < 0.70
- Low: < 0.40

## Integration notes
- This package does not load or retrain Member 2's model. Member 5 must run
  Member 2's preprocessing and prediction steps, then pass the lead details
  and positive-class conversion probability to `run_lead_ai`.
- The LLM output is generated text, not verified model feature attribution.
  Use SHAP or another explainability method to substantiate claims about
  which features drove a prediction.
- A `Do Not Email = No` value alone does not prove marketing consent.
- The `openrouter/free` router may choose different free models and can have
  temporary failures. The shared helper retries empty responses and errors.

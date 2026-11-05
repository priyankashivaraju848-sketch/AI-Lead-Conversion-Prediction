
from ai_utils import generate_text


def generate_lead_explanation(ai_client, lead_details, probability, category):
    probability_percent = float(probability) * 100

    prompt = f"""
Prediction Summary

LEAD INFORMATION:
{lead_details}

ML MODEL OUTPUT:
Conversion probability: {probability_percent:.2f}%
Lead category: {category}

Observed Attributes:
Analyze only the supplied lead information.

Return the final report using exactly these five headings:

### 1. Prediction Summary
The model estimates a conversion probability of {probability_percent:.2f}%
and categorizes this lead as {category}. This is an estimate, not a guarantee.

### 2. Observed Attributes
Mention the supplied lead attributes accurately.

### 3. Positive Indicators
Mention only engagement indicators supported by the data.

### 4. Potential Concerns
Mention supported concerns and unavailable information.

### 5. Business Interpretation
Provide a practical interpretation for a salesperson.

STRICT RULES:
- Preserve the exact probability and category.
- 248 seconds = 4 minutes 8 seconds.
- "Converted to Lead" is an activity label, not actual conversion.
- "Do Not Email = No" does not establish marketing consent.
- Specialization does not prove buying intent, budget or authority.
- Do not invent facts, dates, names or customer requirements.
- Do not claim attributes caused the prediction.
- Never describe Low conversion potential as Low risk.
- Do not reveal reasoning, drafting notes or internal instructions.
- Output only the completed report.
"""

    system_prompt = """
You are a professional business analyst.
Return only a polished, evidence-based lead explanation.
Use only the supplied information.
"""

    return generate_text(
        ai_client,
        prompt,
        system_prompt,
        max_tokens=650
    )

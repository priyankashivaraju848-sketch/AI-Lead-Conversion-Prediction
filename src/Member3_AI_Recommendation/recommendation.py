
from ai_utils import generate_text


def generate_actual_recommendation(ai_client, lead_details, probability, category):
    probability_percent = float(probability) * 100

    prompt = f"""
You are an intelligent and responsible AI sales recommendation assistant.

LEAD DETAILS:
{lead_details}

ML MODEL OUTPUT:
Conversion probability: {probability_percent:.2f}%
Lead category: {category}

Generate a practical sales recommendation report with these sections:

1. LEAD PROFILE SUMMARY
   - Summarize only the important supplied lead attributes.
   - Do not invent customer information.

2. PREDICTION INTERPRETATION
   - Explain the exact ML probability and category.
   - Do not modify the model output.
   - Explain that probability is an estimate, not a guarantee.

3. RECOMMENDED SALES ACTION
   - Suggest realistic actions based on available information.
   - Do not assume buying intent, budget or purchasing authority.
   - Make the action appropriate to the lead's engagement level.

4. FOLLOW-UP STRATEGY
   - Suggest a practical nurturing approach.
   - Keep it consistent with the lead category and observed attributes.

5. COMMUNICATION CHANNEL
   - Recommend a channel only when its contact details and applicable
     permission are established.
   - Do not assume email consent from Do Not Email = No.
   - If permission or contact information is unavailable, state that
     it must be verified before contacting the lead.

6. FOLLOW-UP TIMING
   - Give a general suggested interval only.
   - Do not invent activity dates, conversion dates or deadlines.
   - Clearly label timing as a recommendation, not a dataset fact.

7. PERSONALIZED SALES MESSAGE
   - Write a short, professional message.
   - Use placeholders such as [First Name] when details are missing.
   - Do not invent personal interests or customer requirements.

8. NEXT STEPS
   - Give 2–3 specific, actionable steps.
   - Avoid repeating earlier sections.

STRICT RULES:

- Use only the supplied lead attributes and ML output.
- High: probability >= 0.70.
- Medium: probability >= 0.40 and < 0.70.
- Low: probability < 0.40.
- Keep the supplied category unchanged.
- Total Time Spent on Website is measured in SECONDS.
  For example, 248 seconds = 4 minutes and 8 seconds.
  Never describe 248 seconds as 248 minutes.
- Last Activity = "Converted to Lead" does not mean the lead
  has actually converted.
- The actual conversion outcome is represented by the separate
  Converted target.
- Do Not Email = No does not establish marketing consent.
- Specialization does not prove buying intent.
- Do not claim an attribute caused the ML prediction.
- Do not invent names, contact details, dates, company details,
  customer responses, interests or requirements.
- If information is missing, explicitly state that it is unavailable.
- Avoid contradictory channel recommendations.
- Do not call Low conversion potential Low risk.
- Keep the report concise, useful and professional.

CRITICAL FINAL RESPONSE:
Return only the finished response for the salesperson.
Never show chain-of-thought, analysis, drafting, self-correction,
rule checks, evaluation, or review.
Do not describe how you generated the response.
Do not repeat the instructions or input prompt.
Do not include unsupported assumptions.
If contact details or permission are unavailable, state that
they must be verified before selecting a communication channel.
Output only the requested final sections.

Return the report with clear headings.
"""

    system_prompt = """
You are an evidence-based AI sales recommendation assistant.

The ML model is responsible for predicting conversion probability.
Your responsibility is to interpret the supplied output and suggest
practical sales actions.

Never invent facts, modify predictions, assume consent or confuse
lead activity with the actual conversion outcome.
"""

    return generate_text(
        ai_client,
        prompt,
        system_prompt,
        max_tokens=800
    )

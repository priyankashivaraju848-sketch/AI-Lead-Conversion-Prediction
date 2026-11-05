
from ai_utils import generate_text


def generate_followup_strategy(ai_client, lead_details, probability, category):
    probability_percent = float(probability) * 100

    prompt = f"""
FOLLOW-UP OBJECTIVE

LEAD DETAILS:
{lead_details}

ML MODEL OUTPUT:
Conversion probability: {probability_percent:.2f}%
Lead category: {category}

Return ONLY these five sections:

### 1. FOLLOW-UP OBJECTIVE
Explain the purpose of nurturing this lead using supplied data.
Consider the ML probability and available engagement information.

### 2. RECOMMENDED COMMUNICATION CHANNEL
Recommend a channel only when contact details and permission
are established. Otherwise state that they must be verified.

### 3. SUGGESTED TIMING
Give a general suggested interval, not an actual scheduled date.
Do not invent dates.

### 4. PERSONALIZED FOLLOW-UP MESSAGE
Write a concise professional message.
Use placeholders for unavailable names or details.
Do not invent interests or previous interactions.

### 5. NEXT RECOMMENDED ACTION
Give one or two practical steps.

STRICT RULES:
- Preserve the exact ML probability ({probability_percent:.2f}%) and category ({category}).
- High >= 0.70; Medium >= 0.40 and < 0.70; Low < 0.40.
- Website time is measured in seconds.
- "Converted to Lead" is an activity label, not actual conversion.
- Do Not Email = No does not establish marketing consent.
- Specialization does not prove buying intent or authority.
- Do not invent dates, names, contact details or requirements.
- Do not guarantee conversion.
- Do not make unsupported assumptions.
- Use simple, professional language.
- Return only the final report, without reasoning or drafting notes.
"""

    system_prompt = """
You are an evidence-based AI sales follow-up strategist.
Return only the final, polished follow-up strategy.
Use only the supplied information.
Never reveal internal reasoning or drafting notes.
"""

    return generate_text(
        ai_client,
        prompt,
        system_prompt,
        max_tokens=650
    )

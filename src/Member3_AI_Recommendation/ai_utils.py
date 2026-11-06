
import re
import time


def local_fallback(prompt):
    """Generate a local rule-based report using supplied lead information."""

    def get_value(pattern, default="Not available"):
        match = re.search(pattern, prompt, re.IGNORECASE)
        return match.group(1).strip().strip("'\"") if match else default

    # Extract lead information
    probability = get_value(
        r"Conversion probability:\s*([\d.]+)\s*%"
    )

    category = get_value(
        r"Lead category:\s*(High|Medium|Low)\b"
    ).title()

    visits = get_value(
        r"TotalVisits['\"]?\s*:\s*([\d.]+)"
    )

    website_time = get_value(
        r"Total Time Spent on Website['\"]?\s*:\s*([\d.]+)"
    )

    page_views = get_value(
        r"Page Views Per Visit['\"]?\s*:\s*([\d.]+)"
    )

    last_activity = get_value(
        r"Last Activity['\"]?\s*:\s*([^,\n}]+)"
    )

    specialization = get_value(
        r"Specialization['\"]?\s*:\s*([^}\n]+)"
    )

    specialization = specialization.split(", '")[0].split(', "')[0]
    specialization = specialization.strip().strip("'\"")

    # AI Explanation fallback
    if "Prediction Summary" in prompt and "Observed Attributes" in prompt:

        report = f"""
### 1. Prediction Summary
The ML model estimates a conversion probability of {probability}%.
The lead category is {category}. This is an estimate, not a guarantee.

### 2. Observed Attributes
- Website visits: {visits}
- Total website time: {website_time} seconds
- Page views per visit: {page_views}
- Last activity: {last_activity}
- Specialization: {specialization}

### 3. Positive Indicators
Recorded website visits indicate website engagement.

### 4. Potential Concerns
Budget, purchasing authority and specific buying requirements
are not established by the available information.

### 5. Business Interpretation
The lead falls into the {category} conversion category according
to the ML model. Consider a measured nurturing approach.
"""

    # Follow-up strategy fallback
    elif "FOLLOW-UP OBJECTIVE" in prompt:

        report = f"""
### 1. FOLLOW-UP OBJECTIVE
Nurture the lead using an informative approach, considering
the ML probability of {probability}% and category {category}.

### 2. RECOMMENDED COMMUNICATION CHANNEL
Verify contact details and applicable communication permission
before selecting a communication channel.

### 3. SUGGESTED TIMING
Consider a general interval of 3–5 business days, subject to
the sales team's process. This is not a scheduled date.

### 4. PERSONALIZED FOLLOW-UP MESSAGE
Hello [First Name],

Thank you for visiting our website. Please let us know if you
would like further information about our offerings.

Best regards,
[Sales Team]

### 5. NEXT RECOMMENDED ACTION
1. Verify contact details and applicable communication permission.
2. Review new engagement information before further outreach.
"""

    # Sales recommendation fallback
    elif (
        "LEAD PROFILE SUMMARY" in prompt.upper()
        or "PREDICTION INTERPRETATION" in prompt.upper()
        or "NEXT STEPS" in prompt.upper()
    ):

        report = f"""
### 1. Lead Profile Summary
The lead has {visits} recorded website visits and {website_time}
seconds of total website time.

### 2. Prediction Interpretation
The ML model estimates a conversion probability of {probability}%
and assigns the category {category}.

### 3. Recommended Sales Action
Use a measured, information-focused nurturing approach.
Do not assume buying intent, budget or purchasing authority.

### 4. Follow-up Strategy
Review new engagement information before deciding on further
sales activity.

### 5. Communication Channel
Verify contact details and applicable communication permission
before outreach.

### 6. Follow-up Timing
Consider a general interval of 3–5 business days, subject to
the sales team's process.

### 7. Personalized Sales Message
Hello [First Name],

Thank you for visiting our website. Please let us know if you
would like further information about our offerings.

Best regards,
[Sales Team]

### 8. Next Steps
1. Verify contact details and permission.
2. Review the lead's engagement.
3. Reassess when new information becomes available.
"""

    # Generic AI assistant fallback
    else:

        report = f"""
The ML model estimates a conversion probability of {probability}%
and assigns the category {category}.

The lead has {visits} recorded website visits and {website_time}
seconds of total website time.

Review the available lead information before making further
sales decisions. Verify contact details and communication
permission before outreach.
"""

    return (
        "**Local Fallback Mode**\n\n"
        "*OpenRouter is unavailable. This report uses local rules, "
        "not an LLM.*\n\n"
        + report
    )


def generate_text(ai_client, prompt, system_message,
                  max_tokens=650, retries=3):
    """Try OpenRouter and use local fallback if unavailable."""

    # Immediately use local fallback if API key/client is missing
    if ai_client is None:
        print("OpenRouter unavailable. Using local fallback.")
        return local_fallback(prompt)

    last_error = None

    for attempt in range(retries):
        try:
            response = ai_client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max(max_tokens, 1000),
                temperature=0.3
            )

            if response.choices:
                content = response.choices[0].message.content

                if content and content.strip():
                    content = content.strip()

                    unwanted_phrases = [
                        "Analyze the Request",
                        "Drafting Section",
                        "Drafting Content",
                        "Rule Check",
                        "Data Check",
                        "Output Requirements"
                    ]

                    if not any(
                        phrase.lower() in content.lower()
                        for phrase in unwanted_phrases
                    ):
                        return content

                    last_error = (
                        "OpenRouter returned internal drafting text."
                    )

            else:
                last_error = "OpenRouter returned an empty response."

        except Exception as exc:
            last_error = str(exc)

            if "429" in last_error or "rate limit" in last_error.lower():
                break

        if attempt < retries - 1:
            time.sleep(2)

    print(f"Using local fallback. Reason: {last_error}")
    return local_fallback(prompt)

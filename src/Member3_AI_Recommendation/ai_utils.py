
import time


def generate_text(
    ai_client,
    prompt,
    system_message,
    max_tokens=650,
    retries=3,
    fallback_text=None
):
    """Try OpenRouter first, then use local Python fallback."""

    last_error = "OpenRouter client is unavailable."

    if ai_client is not None:
        for attempt in range(retries):
            try:
                response = ai_client.chat.completions.create(
                    model="openrouter/free",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens
                )

                if response.choices:
                    result = response.choices[0].message.content
                    if result and result.strip():
                        return result.strip()

                last_error = "The model returned an empty response."

            except Exception as exc:
                last_error = str(exc)

            if attempt < retries - 1:
                time.sleep(2)

    if fallback_text:
        return (
            "LOCAL FALLBACK MODE\n\n"
            "OpenRouter is unavailable or its quota is exhausted. "
            "This response was generated using local Python rules, not an LLM.\n\n"
            + fallback_text
        )

    return (
        "OpenRouter is unavailable, and no local fallback "
        "was supplied. Reason: " + last_error
    )


def build_local_fallback(kind, lead_details, probability, category):
    details = dict(lead_details)
    probability_percent = float(probability) * 100

    website_seconds = details.get("Total Time Spent on Website")

    if website_seconds is not None:
        try:
            seconds = int(float(website_seconds))
            minutes, remaining = divmod(seconds, 60)
            details["Website Time"] = (
                f"{minutes} minutes {remaining} seconds"
            )
        except (ValueError, TypeError):
            details["Website Time"] = "Not available"

    details.pop("Total Time Spent on Website", None)

    profile = "\n".join(
        f"- {key}: {value}" for key, value in details.items()
    )

    if kind == "explanation":
        return f"""PREDICTION EXPLANATION

Estimated conversion probability: {probability_percent:.2f}%

Observed Lead Attributes:
{profile}

The model assigns this lead to the {category} category.
This is a statistical estimate, not a guarantee.
No individual feature attribution values are available here.
The sales team should verify the lead's actual requirements."""

    if kind == "recommendation":
        return f"""SALES RECOMMENDATION

Lead Profile:
{profile}

Estimated conversion probability: {probability_percent:.2f}%
Lead Category: {category}

Recommended Action:
Review the available lead information and qualify the lead
before deciding whether direct sales engagement is appropriate.

Follow-up:
Verify contact information, communication preferences
and the lead's actual requirements before contacting them."""

    if kind == "followup":
        return """FOLLOW-UP STRATEGY

Objective:
Understand the lead's requirements.

Communication:
Use a channel only after verifying contact details
and the required communication permission.

Suggested Message:
Hello [Name],

Thank you for your interest. We would be happy to
understand your requirements and share relevant information.

Regards,
[Your Name]

Next Action:
Verify contact information and record the follow-up outcome."""

    return "Local fallback is not configured for this request."

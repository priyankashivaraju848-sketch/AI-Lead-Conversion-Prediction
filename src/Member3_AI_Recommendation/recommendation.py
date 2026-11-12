
from ai_utils import generate_text, build_local_fallback


def generate_actual_recommendation(ai_client, lead_details, probability, category):

    prompt = f"""
You are an expert AI sales recommendation assistant.

Actual lead details: {lead_details}
Conversion probability: {float(probability) * 100:.2f}%
Lead category: {category}

Generate a report containing:
1. Lead profile summary
2. Prediction interpretation
3. Recommended sales action
4. Personalized follow-up strategy
5. Suggested communication channel
6. Suggested follow-up timing
7. Personalized sales message
8. Next steps

Use only supplied attributes. Distinguish facts from suggestions. Do not claim
feature causality without model explainability evidence. Do not guarantee
conversion. Do not assume marketing consent from a Do Not Email field alone.
Do not invent resources, URLs, names, or company details; use placeholders
where information is unavailable.
"""

    fallback_text = build_local_fallback(
        "recommendation", lead_details, probability, category
    )

    return generate_text(
        ai_client,
        prompt,
        "You are an expert AI sales analyst. Be practical and evidence-aware.",
        max_tokens=800,
        fallback_text=fallback_text
    )

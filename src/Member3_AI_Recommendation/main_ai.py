"""Entry points for Member 3's AI recommendation module."""
from ai_client import create_ai_client
from lead_utils import categorize_lead
from ai_explanation import generate_lead_explanation
from recommendation import generate_actual_recommendation
from follow_up_strategy import generate_followup_strategy
from ai_sales_assistant import ai_sales_assistant

def run_lead_ai(lead_details, probability, ai_client=None):
    """
    Generate category, explanation, recommendation, and follow-up strategy.
    probability must be the ML probability for Converted=1, from 0 to 1.
    """
    if ai_client is None:
        ai_client = create_ai_client()
    category = categorize_lead(probability)
    return {
        "probability": float(probability),
        "category": category,
        "explanation": generate_lead_explanation(
            ai_client, lead_details, probability, category
        ),
        "recommendation": generate_actual_recommendation(
            ai_client, lead_details, probability, category
        ),
        "follow_up": generate_followup_strategy(
            ai_client, lead_details, probability, category
        )
    }

def ask_about_lead(question, lead_details, probability, category, ai_client=None):
    """Answer a salesperson's question about one lead."""
    if ai_client is None:
        ai_client = create_ai_client()
    return ai_sales_assistant(
        ai_client, question, lead_details, probability, category
    )

if __name__ == "__main__":
    print("Member 3 AI module ready. Import run_lead_ai and ask_about_lead.")

import sys
from pathlib import Path

AI_DIR = Path("src/Member3_AI_Recommendation")
sys.path.insert(0, str(AI_DIR.resolve()))

from main_ai import run_lead_ai, ask_about_lead
from lead_utils import categorize_lead

print("AI modules imported successfully!")

probability = 0.4406

category = categorize_lead(probability)

print("Conversion Probability:", probability * 100, "%")
print("Lead Category:", category)
"""
check_models.py - lists the model IDs your Groq key can actually use.

Usage:
    python check_models.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys

from groq import Groq

key = os.environ.get("GROQ_API_KEY")
if not key:
    sys.exit("GROQ_API_KEY is not set (check your .env file).")

print(f"Key starts with: {key[:4]}  (a Groq key starts with 'gsk_')")

try:
    models = Groq(api_key=key).models.list().data
except Exception as e:
    sys.exit(f"Could not list models: {e}")

print("\nModels available to your key:")
for m in sorted(models, key=lambda m: m.id):
    print(f"  {m.id}")

print("\nPick a general chat model from the list and add it to .env:")
print("  GROQ_MODEL=<model id>")
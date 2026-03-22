import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from pyairtable import Api

load_dotenv()

app = Flask(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TODAY_LONG = datetime.now().strftime("%B %d, %Y").replace(" 0", " ")
TODAY_SHORT = datetime.now().strftime("%m/%d/%Y")

SYSTEM_PROMPT = f"""
You are the engine for a political claim analysis demo.

Return ONLY valid JSON.
Do not use markdown fences.
Do not include commentary outside the JSON.

You must return this exact schema and fill every field as best as possible:

{{
  "Stripped Claim": "",
  "Speaker": "",
  "Date": "",
  "Source URLs": "",
  "Sub-Claims": "",
  "Sub-Claim 1": "",
  "Sub-Claim 2": "",
  "Sub-Claim 3": "",
  "Verdict: Sub-Claim1": "",
  "Verdict: Sub-Claim 2": "",
  "Verdict: Sub-Claim 3": "",
  "Direct Facts": "",
  "Adjacent Facts": "",
  "Root Concern": "",
  "Values Divergence": "",
  "Left Perspective": "",
  "Right Perspective": "",
  "Constitutional Framework": "",
  "Overall Verdict": "",
  "Strip Mode Summary": "",
  "Scenario Map": "",
  "Topic": "",
  "Status": "",
  "Human Reviewed": "",
  "Published": "",
  "Date Added": "",
  "Last Updated": ""
}}

Rules:
- "Stripped Claim" = remove rhetoric, keep the core literal claim
- "Speaker" = best estimate of who is making the claim
- "Date" = use exactly this date: {TODAY_LONG}
- "Source URLs" = "Sources not yet attached in demo mode."
- "Sub-Claims" = one paragraph summary
- "Sub-Claim 1/2/3" = break into separate claims if possible
- "Verdict: Sub-Claim..." = choose from:
  True, Mostly True, Mixed, Exaggerated, Unproven, Misleading, False, Contested, Not Enough Information, Substantially True, Plausible/Mixed
- "Direct Facts" = concise factual points directly tied to the claim
- "Adjacent Facts" = context or complicating facts
- "Root Concern" = what legitimate issue this claim is pointing at underneath the rhetoric
- "Values Divergence" = where the disagreement becomes about values, priorities, tradeoffs, or philosophy
- "Left Perspective" = strongest fair progressive interpretation
- "Right Perspective" = strongest fair conservative interpretation
- "Constitutional Framework" = relevant only if applicable, otherwise say "Not primarily a constitutional claim."
- "Overall Verdict" = same verdict options as above
- "Strip Mode Summary" = short plain English summary for a quick-read card
- "Scenario Map" = plausible future consequences if the claim's underlying issue continues or policy moves forward
- "Topic" = short topic label like "Energy", "Immigration", "Iran War", "Military", "Healthcare", "Social Security"
- "Status" = default "Active"
- "Human Reviewed" = default "No"
- "Published" = default "No"
- "Date Added" = use exactly this date: {TODAY_SHORT}
- "Last Updated" = use exactly this date: {TODAY_SHORT}
- Fill all fields even if partially uncertain
- If there is uncertainty, state it clearly rather than pretending certainty
""".strip()


def parse_json_response(text):
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1 and end > start:
            possible_json = cleaned[start:end + 1]
            try:
                return json.loads(possible_json)
            except Exception:
                pass

        return {
            "error": "JSON parsing failed",
            "raw": text
        }


def ask_claude(claim):
    if not anthropic_client:
        return {"error": "Claude not configured"}

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2200,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": claim}
            ]
        )

        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        return parse_json_response("".join(text_parts))

    except Exception as e:
        return {"error": str(e)}


def ask_openai(claim):
    if not openai_client:
        return {"error": "OpenAI not configured"}

    try:
        response = openai_client.responses.create(
            model="gpt-5",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": claim}
            ]
        )

        return parse_json_response(response.output_text)

    except Exception as e:
        return {"error": str(e)}


def normalize_for_airtable(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, indent=2)
    return str(value)


def normalize_multi_select(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def normalize_checkbox(value):
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in ["yes", "true", "1", "checked"]


def normalize_verdict(value):
    if value is None:
        return "Unproven"

    text = str(value).strip().lower()

    mapping = {
        "true": "True",
        "mostly true": "Mostly True",
        "substantially true": "Substantially True",
        "mixed": "Plausible/Mixed",
        "plausible/mixed": "Plausible/Mixed",
        "plausible": "Plausible/Mixed",
        "misleading": "Misleading",
        "exaggerated": "Exaggerated",
        "unproven": "Unproven",
        "false": "False",
        "contested": "Contested",
        "not enough information": "Unproven"
    }

    return mapping.get(text, "Unproven")


def save_to_airtable(original_quote, claude_result, openai_result):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        return {"saved": False, "error": "Airtable is not configured in .env"}

    if isinstance(openai_result, dict) and "error" not in openai_result:
        primary = openai_result
    elif isinstance(claude_result, dict) and "error" not in claude_result:
        primary = claude_result
    else:
        return {"saved": False, "error": "No valid model output to save"}

    try:
        api = Api(AIRTABLE_TOKEN)
        table = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)

        fields = {
            "Original Quote": original_quote,
            "Stripped Claim": normalize_for_airtable(primary.get("Stripped Claim")),
            "Speaker": normalize_for_airtable(primary.get("Speaker")),
            "Date": normalize_for_airtable(primary.get("Date")),
            "Source URLs": normalize_for_airtable(primary.get("Source URLs")),
            "Sub-Claims": normalize_for_airtable(primary.get("Sub-Claims")),
            "Sub-Claim 1": normalize_for_airtable(primary.get("Sub-Claim 1")),
            "Sub-Claim 2": normalize_for_airtable(primary.get("Sub-Claim 2")),
            "Sub-Claim 3": normalize_for_airtable(primary.get("Sub-Claim 3")),
            "Verdict: Sub-Claim1": normalize_verdict(primary.get("Verdict: Sub-Claim1")),
            "Verdict: Sub-Claim 2": normalize_verdict(primary.get("Verdict: Sub-Claim 2")),
            "Verdict: Sub-Claim 3": normalize_verdict(primary.get("Verdict: Sub-Claim 3")),
            "Direct Facts": normalize_for_airtable(primary.get("Direct Facts")),
            "Adjacent Facts": normalize_for_airtable(primary.get("Adjacent Facts")),
            "Root Concern": normalize_for_airtable(primary.get("Root Concern")),
            "Values Divergence": normalize_for_airtable(primary.get("Values Divergence")),
            "Left Perspective": normalize_for_airtable(primary.get("Left Perspective")),
            "Right Perspective": normalize_for_airtable(primary.get("Right Perspective")),
            "Constitutional Framework": normalize_for_airtable(primary.get("Constitutional Framework")),
            "Overall Verdict": normalize_verdict(primary.get("Overall Verdict")),
            "Strip Mode Summary": normalize_for_airtable(primary.get("Strip Mode Summary")),
            "Scenario Map": normalize_for_airtable(primary.get("Scenario Map")),
            "Topic": normalize_multi_select(primary.get("Topic")),
            "Status": normalize_for_airtable(primary.get("Status")),
            "Human Reviewed": normalize_checkbox(primary.get("Human Reviewed")),
            "Published": normalize_checkbox(primary.get("Published")),
            "Date Added": normalize_for_airtable(primary.get("Date Added")),
            "Last Updated": normalize_for_airtable(primary.get("Last Updated")),
        }

        record = table.create(fields)

        return {
            "saved": True,
            "record_id": record.get("id")
        }

    except Exception as e:
        return {
            "saved": False,
            "error": str(e)
        }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    claim = (data.get("claim") or "").strip()

    if not claim:
        return jsonify({"error": "Claim is required."}), 400

    claude_result = ask_claude(claim)
    openai_result = ask_openai(claim)
    airtable_result = save_to_airtable(claim, claude_result, openai_result)

    return jsonify({
        "claude": claude_result,
        "openai": openai_result,
        "airtable": airtable_result
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
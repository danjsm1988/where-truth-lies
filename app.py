import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from pyairtable import Api

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-secret")

APP_PASSWORD = os.getenv("APP_PASSWORD")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TODAY_LONG = datetime.now().strftime("%B %d, %Y").replace(" 0", " ")
TODAY_SHORT = datetime.now().strftime("%m/%d/%Y")


# -------- PASSWORD GATE --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password")

        if password == APP_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("home"))

        return "Wrong password", 401

    return """
    <html style="background:black;color:white;font-family:sans-serif;text-align:center;padding-top:100px;">
        <h1>Where Truth Lies</h1>
        <p>Enter access password</p>
        <form method="POST">
            <input type="password" name="password" style="padding:10px;font-size:16px;">
            <br><br>
            <button type="submit" style="padding:10px 20px;">Enter</button>
        </form>
    </html>
    """


def require_auth():
    if not session.get("authenticated"):
        return False
    return True


# -------- SYSTEM PROMPT --------
SYSTEM_PROMPT = f"""
Return ONLY valid JSON.

Use today's date:
{TODAY_LONG}

Fill all fields.
""".strip()


# -------- HELPERS --------
def parse_json_response(text):
    try:
        return json.loads(text)
    except:
        return {"error": "parse failed", "raw": text}


def normalize_verdict(value):
    if not value:
        return "Unproven"

    text = str(value).lower()

    if "true" in text:
        return "True"
    if "misleading" in text:
        return "Misleading"
    if "false" in text:
        return "False"
    if "mixed" in text:
        return "Plausible/Mixed"

    return "Unproven"


def normalize_checkbox(value):
    return str(value).lower() in ["yes", "true"]


def normalize_topic(value):
    allowed = [
        "Healthcare",
        "Energy",
        "Defense",
        "Iran War",
        "Military",
        "Social Security",
        "Other"
    ]

    if not value:
        return ["Other"]

    text = str(value)

    for t in allowed:
        if t.lower() in text.lower():
            return [t]

    return ["Other"]


# -------- AI CALLS --------
def ask_claude(claim):
    if not anthropic_client:
        return {"error": "Claude not configured"}

    try:
        res = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": claim}]
        )

        return parse_json_response(res.content[0].text)

    except Exception as e:
        return {"error": str(e)}


def ask_openai(claim):
    if not openai_client:
        return {"error": "OpenAI not configured"}

    try:
        res = openai_client.responses.create(
            model="gpt-5",
            input=claim
        )

        return parse_json_response(res.output_text)

    except Exception as e:
        return {"error": str(e)}


# -------- AIRTABLE --------
def save_to_airtable(original_quote, result):
    try:
        api = Api(AIRTABLE_TOKEN)
        table = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)

        fields = {
            "Original Quote": original_quote,
            "Stripped Claim": result.get("Stripped Claim"),
            "Speaker": result.get("Speaker"),
            "Date": result.get("Date"),
            "Direct Facts": result.get("Direct Facts"),
            "Overall Verdict": normalize_verdict(result.get("Overall Verdict")),
            "Topic": normalize_topic(result.get("Topic")),
            "Human Reviewed": False,
            "Published": False,
            "Date Added": TODAY_SHORT,
            "Last Updated": TODAY_SHORT,
        }

        table.create(fields)

        return {"saved": True}

    except Exception as e:
        return {"error": str(e)}


# -------- ROUTES --------
@app.route("/")
def home():
    if not require_auth():
        return redirect("/login")

    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if not require_auth():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    claim = data.get("claim")

    claude = ask_claude(claim)
    openai = ask_openai(claim)

    primary = openai if "error" not in openai else claude

    airtable = save_to_airtable(claim, primary)

    return jsonify({
        "claude": claude,
        "openai": openai,
        "airtable": airtable
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
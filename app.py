import os
import json
import re
import unicodedata
import requests
from flask import Flask, request, jsonify, render_template, redirect, session
from openai import OpenAI
import anthropic

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-secret")

APP_PASSWORD = os.getenv("APP_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


@app.route("/health")
def health():
    return "ok", 200


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password")
        if password == APP_PASSWORD:
            session["logged_in"] = True
            return redirect("/")
        return "Wrong password", 401

    return """
    <html>
    <head>
        <title>Where the Truth Lies</title>
        <style>
            body {
                background:#0d1b2a;
                color:#fff;
                font-family:Arial;
                display:flex;
                align-items:center;
                justify-content:center;
                height:100vh;
            }
            .box {
                background:#1b263b;
                padding:30px;
                border-radius:8px;
                text-align:center;
            }
            input {
                padding:10px;
                margin-top:10px;
                width:220px;
            }
            button {
                margin-top:10px;
                padding:10px 20px;
                background:#c9a84c;
                border:none;
                cursor:pointer;
            }
        </style>
    </head>
    <body>
        <div class="box">
            <h2>Where the Truth Lies</h2>
            <form method="POST">
                <input type="password" name="password" placeholder="Enter Password" required />
                <br/>
                <button type="submit">Enter</button>
            </form>
        </div>
    </body>
    </html>
    """


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
def home():
    if not session.get("logged_in"):
        return redirect("/login")
    return render_template("index.html")


def safe_json_parse(text):
    if not text:
        return {}

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
        return {"raw": text}


def normalize_topic(raw_topic):
    if not raw_topic:
        return "Other"

    text = str(raw_topic).lower()
    if "social security" in text:
        return "Social Security"
    if "health" in text:
        return "Healthcare"
    if "energy" in text or "wind" in text or "power" in text or "electric" in text:
        return "Energy"
    if "iran" in text:
        return "Iran War"
    if "defense" in text:
        return "Defense"
    if "military" in text or "war" in text:
        return "Military"
    return "Other"


def slugify(text):
    if not text:
        return "untitled-claim"
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:120] if text else "untitled-claim"


def build_url_slug(parsed, claim):
    title_source = parsed.get("Original Quote") or parsed.get("Stripped Claim") or claim
    return slugify(title_source)


def extract_primary_record_fields(claim, parsed, mode):
    return {
        "Original Quote": claim,
        "Stripped Claim": parsed.get("Stripped Claim", claim),
        "Speaker": parsed.get("Speaker", "User Submission"),
        "Topic": normalize_topic(parsed.get("Topic")),
        "Human Reviewed": False,
        "Published": False,
        "Mode": mode,
        "URL Slug": build_url_slug(parsed, claim)
    }


@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    claim = (data.get("claim") or "").strip()
    mode = data.get("mode", "strip")

    if not claim:
        return jsonify({"error": "Claim is required"}), 400

    claude_json = {}
    openai_json = {}

    try:
        if anthropic_client:
            claude_response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1800,
                messages=[{
                    "role": "user",
                    "content": f"Analyze this claim in {mode} mode and return JSON:\n\n{claim}"
                }]
            )
            claude_text = claude_response.content[0].text
            claude_json = safe_json_parse(claude_text)
        else:
            claude_json = {"error": "Anthropic not configured"}
    except Exception as e:
        claude_json = {"error": str(e)}

    try:
        if openai_client:
            openai_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a political claim analysis engine. Return valid JSON."},
                    {"role": "user", "content": f"Analyze this claim in {mode} mode and return JSON:\n\n{claim}"}
                ]
            )
            openai_text = openai_response.choices[0].message.content
            openai_json = safe_json_parse(openai_text)
        else:
            openai_json = {"error": "OpenAI not configured"}
    except Exception as e:
        openai_json = {"error": str(e)}

    airtable_result = {}

    try:
        if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
            airtable_result = {"saved": False, "error": "Airtable not configured"}
        else:
            url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
            headers = {
                "Authorization": f"Bearer {AIRTABLE_TOKEN}",
                "Content-Type": "application/json"
            }

            primary = openai_json if "error" not in openai_json else claude_json
            fields = extract_primary_record_fields(claim, primary, mode)
            fields["Claude Raw JSON"] = json.dumps(claude_json, indent=2)
            fields["OpenAI Raw JSON"] = json.dumps(openai_json, indent=2)

            response = requests.post(url, headers=headers, json={"fields": fields}, timeout=30)

            if response.status_code == 200:
                record = response.json()
                airtable_result = {
                    "saved": True,
                    "record_id": record.get("id"),
                    "url_slug": fields["URL Slug"]
                }
            else:
                airtable_result = {"saved": False, "error": response.text}
    except Exception as e:
        airtable_result = {"saved": False, "error": str(e)}

    return jsonify({
        "claude": claude_json,
        "openai": openai_json,
        "airtable": airtable_result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
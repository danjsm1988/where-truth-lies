import os
import json
import requests
from flask import Flask, request, jsonify, render_template, redirect, session

from openai import OpenAI
import anthropic

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-secret")

# ENV VARIABLES
APP_PASSWORD = os.getenv("APP_PASSWORD")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

# CLIENTS
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# -------------------------
# LOGIN / LOGOUT
# -------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password")

        if password == APP_PASSWORD:
            session["logged_in"] = True
            return redirect("/")
        else:
            return "Wrong password", 401

    return """
    <html>
    <head>
        <title>Where Truth Lies</title>
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
                width:200px;
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


# -------------------------
# MAIN PAGE
# -------------------------

@app.route("/")
def home():
    if not session.get("logged_in"):
        return redirect("/login")

    return render_template("index.html")


# -------------------------
# ANALYZE
# -------------------------

@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    claim = data.get("claim")
    mode = data.get("mode", "strip")

    try:
        # -------------------------
        # CLAUDE
        # -------------------------
        claude_response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"Analyze this claim in {mode} mode:\n\n{claim}"
            }]
        )

        claude_text = claude_response.content[0].text

        try:
            claude_json = json.loads(claude_text)
        except:
            claude_json = {"raw": claude_text}

    except Exception as e:
        claude_json = {"error": str(e)}

    # -------------------------
    # OPENAI
    # -------------------------
    try:
        openai_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a political claim analysis engine."},
                {"role": "user", "content": f"Analyze this claim in {mode} mode:\n\n{claim}"}
            ]
        )

        openai_text = openai_response.choices[0].message.content

        try:
            openai_json = json.loads(openai_text)
        except:
            openai_json = {"raw": openai_text}

    except Exception as e:
        openai_json = {"error": str(e)}

    # -------------------------
    # AIRTABLE SAVE
    # -------------------------
    airtable_result = {}

    try:
        url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"

        headers = {
            "Authorization": f"Bearer {AIRTABLE_TOKEN}",
            "Content-Type": "application/json"
        }

        fields = {
            "Original Quote": claim,
            "Stripped Claim": claude_json.get("Stripped Claim", claim),
            "Speaker": "User Submission",
            "Topic": "Other",  # IMPORTANT: must match your Airtable options
            "Human Reviewed": False,
            "Published": False
        }

        response = requests.post(
            url,
            headers=headers,
            json={"fields": fields}
        )

        if response.status_code == 200:
            record = response.json()
            airtable_result = {
                "saved": True,
                "record_id": record.get("id")
            }
        else:
            airtable_result = {
                "saved": False,
                "error": response.text
            }

    except Exception as e:
        airtable_result = {
            "saved": False,
            "error": str(e)
        }

    return jsonify({
        "claude": claude_json,
        "openai": openai_json,
        "airtable": airtable_result
    })


# -------------------------
# RUN
# -------------------------

if __name__ == "__main__":
    app.run(debug=True)
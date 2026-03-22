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
 
# ── SYSTEM PROMPTS ──────────────────────────────────────────────────────────
 
STRIP_SYSTEM = """You are ClaimLab, the analytical engine of Where the Truth Lies — a political intelligence platform built on an excavation methodology. Motto: Beyond the Argument. Latin seal: Ubi Veritas Latet.
 
You do not fact-check. You excavate.
 
Return ONLY valid JSON. No markdown fences. No preamble. No explanation outside the JSON.
 
For Strip Mode return this exact structure:
{
  "Stripped Claim": "plain language version of what is actually being claimed",
  "Speaker": "who made the claim or Unknown",
  "Topic": "one of: Iran War, Energy, Healthcare, Social Security, Defense, Military, Elections, Economy, Immigration, Other",
  "Direct Facts": "what the data actually shows about this claim in 2-3 sentences",
  "Adjacent Facts": "the most important thing the claim ignores or omits in 1-2 sentences",
  "Root Concern": "the legitimate underlying concern in 1 sentence",
  "Verdict": "one of: True, Mostly True, Plausible/Mixed, Exaggerated, Unproven, Misleading, False, Contested, Substantially True",
  "Strip Mode Summary": "a 2-3 sentence bottom line that states what is true, what is contested, and what the real question is. Write in prose. No bullet points. No dashes."
}"""
 
FULL_SYSTEM = """You are ClaimLab, the analytical engine of Where the Truth Lies — a political intelligence platform built on an excavation methodology. Motto: Beyond the Argument. Latin seal: Ubi Veritas Latet.
 
You do not fact-check. You excavate.
 
Return ONLY valid JSON. No markdown fences. No preamble. No explanation outside the JSON.
 
For Full Excavation return this exact structure:
{
  "Stripped Claim": "plain language version of what is actually being claimed",
  "Speaker": "who made the claim or Unknown",
  "Topic": "one of: Iran War, Energy, Healthcare, Social Security, Defense, Military, Elections, Economy, Immigration, Other",
  "Sub Claims": [
    {"claim": "first distinct factual claim", "verdict": "True/False/Contested/etc"},
    {"claim": "second distinct factual claim", "verdict": "True/False/Contested/etc"}
  ],
  "Direct Facts": "what the documented data actually shows. 3-4 sentences. No bullet points.",
  "Adjacent Facts": "what the claim ignores or omits on both sides equally. 2-3 sentences. No bullet points.",
  "Root Concern": "the legitimate issue underneath even a false or misleading claim. 1-2 sentences.",
  "Values Divergence": "where real disagreement lives. Usually not in facts but in values or priorities. 2-3 sentences.",
  "Left Perspective": "how the left frames this claim and what they get right and wrong. 2-3 sentences.",
  "Right Perspective": "how the right frames this claim and what they get right and wrong. 2-3 sentences.",
  "Overall Verdict": "one of: True, Mostly True, Plausible/Mixed, Exaggerated, Unproven, Misleading, False, Contested, Substantially True",
  "Strip Mode Summary": "a 3-4 sentence bottom line. What is documented. What is contested. What the real question is. Prose only. No bullet points. No dashes."
}
 
Critical rules: Never use bullet points or dashes in any text field. Write all text fields as flowing prose. Apply the same standard regardless of political party."""
 
 
# ── HELPERS ─────────────────────────────────────────────────────────────────
 
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
            try:
                return json.loads(cleaned[start:end + 1])
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
    if "election" in text or "vote" in text or "ballot" in text:
        return "Elections"
    if "econom" in text or "job" in text or "inflation" in text or "paycheck" in text:
        return "Economy"
    if "immigr" in text or "border" in text:
        return "Immigration"
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
    title_source = parsed.get("Stripped Claim") or parsed.get("Original Quote") or claim
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
 
 
# ── ROUTES ───────────────────────────────────────────────────────────────────
 
@app.route("/health")
def health():
    return "ok", 200
 
 
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
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
                color:#f5f0e8;
                font-family:'Georgia', serif;
                display:flex;
                align-items:center;
                justify-content:center;
                height:100vh;
                margin:0;
            }
            .box {
                background:#162336;
                padding:40px;
                border:1px solid #8a6f2e;
                text-align:center;
                max-width:360px;
                width:100%;
            }
            h2 {
                font-size:22px;
                margin-bottom:6px;
                letter-spacing:0.02em;
            }
            .sub {
                font-size:12px;
                color:#c9a84c;
                letter-spacing:0.18em;
                text-transform:uppercase;
                font-style:italic;
                margin-bottom:28px;
            }
            input {
                padding:12px 14px;
                margin-top:10px;
                width:100%;
                background:rgba(255,255,255,0.04);
                border:1px solid rgba(255,255,255,0.12);
                color:#f5f0e8;
                font-size:16px;
                font-family:Georgia,serif;
                box-sizing:border-box;
            }
            button {
                margin-top:14px;
                padding:12px;
                width:100%;
                background:#c9a84c;
                border:none;
                cursor:pointer;
                font-family:Georgia,serif;
                font-size:12px;
                letter-spacing:0.18em;
                text-transform:uppercase;
                font-weight:700;
                color:#0d1b2a;
            }
            button:hover { background:#e8c97a; }
            .latin {
                margin-top:24px;
                font-size:11px;
                color:#8a6f2e;
                font-style:italic;
                letter-spacing:0.12em;
            }
        </style>
    </head>
    <body>
        <div class="box">
            <h2>Where the Truth Lies</h2>
            <div class="sub">Beyond the Argument</div>
            <form method="POST">
                <input type="password" name="password" placeholder="Enter Password" required autofocus />
                <button type="submit">Enter</button>
            </form>
            <div class="latin">Ubi Veritas Latet</div>
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
 
 
@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
 
    data = request.get_json() or {}
    claim = (data.get("claim") or "").strip()
    mode = data.get("mode", "strip")
 
    if not claim:
        return jsonify({"error": "Claim is required"}), 400
 
    system_prompt = STRIP_SYSTEM if mode == "strip" else FULL_SYSTEM
 
    claude_json = {}
    openai_json = {}
 
    # ── Claude ──
    try:
        if anthropic_client:
            claude_response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1800,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Excavate this claim: \"{claim}\""
                }]
            )
            claude_text = claude_response.content[0].text
            claude_json = safe_json_parse(claude_text)
        else:
            claude_json = {"error": "Anthropic not configured"}
    except Exception as e:
        claude_json = {"error": str(e)}
 
    # ── OpenAI ──
    try:
        if openai_client:
            openai_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Excavate this claim: \"{claim}\""}
                ]
            )
            openai_text = openai_response.choices[0].message.content
            openai_json = safe_json_parse(openai_text)
        else:
            openai_json = {"error": "OpenAI not configured"}
    except Exception as e:
        openai_json = {"error": str(e)}
 
    # ── Airtable ──
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
 
            primary = claude_json if "error" not in claude_json else openai_json
            fields = extract_primary_record_fields(claim, primary, mode)
            fields["Claude Raw JSON"] = json.dumps(claude_json, indent=2)
            fields["OpenAI Raw JSON"] = json.dumps(openai_json, indent=2)
 
            # Add rendered fields if available
            if "Strip Mode Summary" in claude_json:
                fields["Strip Mode Summary"] = claude_json["Strip Mode Summary"]
            if "Overall Verdict" in claude_json:
                fields["Overall Verdict"] = claude_json["Overall Verdict"]
            if "Verdict" in claude_json:
                fields["Overall Verdict"] = claude_json["Verdict"]
            if "Direct Facts" in claude_json:
                fields["Direct Facts"] = claude_json["Direct Facts"]
            if "Adjacent Facts" in claude_json:
                fields["Adjacent Facts"] = claude_json["Adjacent Facts"]
            if "Root Concern" in claude_json:
                fields["Root Concern"] = claude_json["Root Concern"]
            if "Values Divergence" in claude_json:
                fields["Values Divergence"] = claude_json["Values Divergence"]
 
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
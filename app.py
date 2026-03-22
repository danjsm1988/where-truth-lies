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
SUPER_PASSWORD = os.getenv("SUPER_PASSWORD", APP_PASSWORD)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

CLAIMLAB_SYSTEM = """You are ClaimLab, the analytical engine of Where the Truth Lies — a political intelligence platform built on an excavation methodology. Motto: Beyond the Argument. Latin seal: Ubi Veritas Latet.

You do not fact-check. You excavate.

Return ONLY valid JSON. No markdown fences. No preamble. No explanation outside the JSON.

Always return this exact structure:
{
  "Stripped Claim": "plain language version of what is actually being claimed, removing emotional language and rhetorical framing",
  "Speaker": "who made the claim, or Unknown if not specified",
  "Topic": "one of: Iran War, Energy, Healthcare, Social Security, Defense, Military, Elections, Economy, Immigration, Other",
  "Sub Claims": [
    {"claim": "first distinct factual claim within the statement", "verdict": "True"},
    {"claim": "second distinct factual claim within the statement", "verdict": "Contested"}
  ],
  "Direct Facts": "what the documented data actually shows. 3-4 sentences. Prose only. No bullet points. No dashes. Use reportedly or estimated for unconfirmed claims.",
  "Adjacent Facts": "what the claim ignores or omits on both sides equally. 2-3 sentences. Prose only. No bullet points. No dashes.",
  "Root Concern": "the legitimate issue underneath even a false or misleading claim. 1-2 sentences. Prose only.",
  "Values Divergence": "where real disagreement lives. Usually not in facts but in values or priorities. 2-3 sentences. Prose only. No bullet points. No dashes.",
  "Left Perspective": "how the left frames this claim, what they get right, and where their framing fails. 2-3 sentences. Prose only.",
  "Right Perspective": "how the right frames this claim, what they get right, and where their framing fails. 2-3 sentences. Prose only.",
  "Constitutional Framework": "if this claim touches on rights, government power, legislative authority, or executive action — what the Constitution actually says and what the founders wrote about intent. Return empty string if not applicable.",
  "Overall Verdict": "one of: True, Mostly True, Plausible/Mixed, Exaggerated, Unproven, Misleading, False, Contested, Substantially True",
  "Strip Mode Summary": "a 3-4 sentence bottom line. What is documented. What is contested. What the real question is. Prose only. No bullet points. No dashes. No hyphens."
}

Critical rules: Never use bullet points, dashes, or hyphens in any text field. Write all text fields as flowing prose. Never tell readers what to think. Apply the same analytical standard regardless of political party or speaker. Separate facts from interpretation."""


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

    if "medicaid" in text:
        return "Medicaid"
    if "medicare" in text:
        return "Medicare"
    if "social security" in text:
        return "Social Security"
    if "health" in text:
        return "Healthcare"
    if "energy" in text or "fossil" in text or "renewable" in text or "climate" in text:
        return "Energy"
    if "iran" in text:
        return "Iran War"
    if "foreign policy" in text:
        return "Foreign Policy"
    if "crime" in text:
        return "Crime"
    if "election" in text or "vote" in text or "ballot" in text or "save act" in text:
        return "Elections"
    if "econom" in text or "job" in text or "inflation" in text or "paycheck" in text or "tax" in text or "wage" in text:
        return "Economy"
    if "immigr" in text or "border" in text:
        return "Immigration"
    if "defense" in text:
        return "Defense"
    if "military" in text or "war" in text:
        return "Military"
    if "school" in text or "educat" in text or "teach" in text or "curriculum" in text:
        return "Education"
    if "gender" in text or "trans" in text or "lgbtq" in text or "pronouns" in text:
        return "Gender Issues"
    if "constitution" in text or "amendment" in text or "rights" in text or "court" in text:
        return "Constitutional Rights"

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
    title_source = parsed.get("Stripped Claim") or claim
    return slugify(title_source)


def extract_primary_record_fields(claim, parsed, mode):
    fields = {
        "Original Quote": claim,
        "Stripped Claim": parsed.get("Stripped Claim", claim),
        "Speaker": parsed.get("Speaker", "User Submission"),
        "Topic": [normalize_topic(parsed.get("Topic"))],
        "Human Reviewed": False,
        "Published": False,
        "Mode": mode if mode in ["strip", "full"] else "strip",
        "URL Slug": build_url_slug(parsed, claim)
    }

    for field in [
        "Strip Mode Summary",
        "Direct Facts",
        "Adjacent Facts",
        "Root Concern",
        "Values Divergence",
        "Left Perspective",
        "Right Perspective",
        "Constitutional Framework"
    ]:
        if parsed.get(field):
            fields[field] = parsed[field]

    verdict = parsed.get("Overall Verdict") or parsed.get("Verdict")
    if verdict:
        fields["Overall Verdict"] = verdict

    sub_claims = parsed.get("Sub Claims")
    if isinstance(sub_claims, list):
        if len(sub_claims) > 0 and sub_claims[0].get("claim"):
            fields["Sub-Claim 1"] = sub_claims[0]["claim"]
            if sub_claims[0].get("verdict"):
                fields["Verdict: Sub-Claim1"] = sub_claims[0]["verdict"]

        if len(sub_claims) > 1 and sub_claims[1].get("claim"):
            fields["Sub-Claim 2"] = sub_claims[1]["claim"]
            if sub_claims[1].get("verdict"):
                fields["Verdict: Sub-Claim 2"] = sub_claims[1]["verdict"]

        if len(sub_claims) > 2 and sub_claims[2].get("claim"):
            fields["Sub-Claim 3"] = sub_claims[2]["claim"]
            if sub_claims[2].get("verdict"):
                fields["Verdict: Sub-Claim 3"] = sub_claims[2]["verdict"]

        if sub_claims:
            fields["Sub-Claims"] = " ".join(
                [sc.get("claim", "") for sc in sub_claims if sc.get("claim")]
            ).strip()

    return fields


def airtable_headers():
    return {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json"
    }


def airtable_url():
    return f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"


def parse_topics(topic_value):
    if isinstance(topic_value, list):
        return topic_value
    if isinstance(topic_value, str) and topic_value.strip():
        return [topic_value.strip()]
    return []


def try_parse_raw_json(fields):
    for key in ["Claude Raw JSON", "OpenAI Raw JSON"]:
        raw = fields.get(key)
        if raw:
            parsed = safe_json_parse(raw)
            if isinstance(parsed, dict) and parsed and "raw" not in parsed:
                return parsed
    return {}


def build_subclaims(fields, parsed_json):
    subclaims = []

    parsed_subclaims = parsed_json.get("Sub Claims")
    if isinstance(parsed_subclaims, list) and parsed_subclaims:
        for item in parsed_subclaims[:3]:
            if item.get("claim"):
                subclaims.append({
                    "claim": item.get("claim", ""),
                    "verdict": item.get("verdict", "Unproven")
                })
        return subclaims

    sc1 = fields.get("Sub-Claim 1")
    sc2 = fields.get("Sub-Claim 2")
    sc3 = fields.get("Sub-Claim 3")

    if sc1:
        subclaims.append({
            "claim": sc1,
            "verdict": fields.get("Verdict: Sub-Claim1", "Unproven")
        })
    if sc2:
        subclaims.append({
            "claim": sc2,
            "verdict": fields.get("Verdict: Sub-Claim 2", "Unproven")
        })
    if sc3:
        subclaims.append({
            "claim": sc3,
            "verdict": fields.get("Verdict: Sub-Claim 3", "Unproven")
        })

    return subclaims


def build_claim_context(record):
    if not record:
        return None

    fields = record.get("fields", {})
    parsed_json = try_parse_raw_json(fields)

    title = fields.get("Original Quote") or fields.get("Stripped Claim") or "Untitled Claim"

    current_claim = {
        "record_id": record.get("id"),
        "slug": fields.get("URL Slug", ""),
        "title": title,
        "stripped_claim": fields.get("Stripped Claim", ""),
        "speaker": fields.get("Speaker", "Unknown"),
        "topics": parse_topics(fields.get("Topic")),
        "date": fields.get("Date") or fields.get("Date Added", ""),
        "overall_verdict": fields.get("Overall Verdict", "Unproven"),
        "strip_mode_summary": fields.get("Strip Mode Summary", ""),
        "direct_facts": fields.get("Direct Facts", ""),
        "adjacent_facts": fields.get("Adjacent Facts", ""),
        "root_concern": fields.get("Root Concern", ""),
        "values_divergence": fields.get("Values Divergence", ""),
        "left_perspective": fields.get("Left Perspective", ""),
        "right_perspective": fields.get("Right Perspective", ""),
        "constitutional_framework": fields.get("Constitutional Framework", ""),
        "scenario_map": fields.get("Scenario Map", ""),
        "subclaims": build_subclaims(fields, parsed_json),
        "source_urls": fields.get("Source URLs", ""),
        "status": fields.get("Status", "Active"),
        "mode": fields.get("Mode", ""),
        "published": fields.get("Published", False),
        "human_reviewed": fields.get("Human Reviewed", False)
    }

    return current_claim


def get_recent_claims(limit=5):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        return []

    try:
        params = {
            "maxRecords": limit,
            "sort[0][field]": "Date Added",
            "sort[0][direction]": "desc"
        }

        response = requests.get(
            airtable_url(),
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            params=params,
            timeout=20
        )
        response.raise_for_status()

        records = response.json().get("records", [])
        recent = []

        for record in records:
            fields = record.get("fields", {})
            recent.append({
                "title": fields.get("Original Quote") or fields.get("Stripped Claim") or "Untitled Claim",
                "slug": fields.get("URL Slug", ""),
                "date": fields.get("Date") or fields.get("Date Added", ""),
                "verdict": fields.get("Overall Verdict", "Unproven")
            })

        return recent

    except Exception as e:
        print("RECENT CLAIMS ERROR:", str(e), flush=True)
        return []


def get_claim_by_slug(slug):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME or not slug:
        return None

    try:
        params = {
            "filterByFormula": f"{{URL Slug}}='{slug}'",
            "maxRecords": 1
        }

        response = requests.get(
            airtable_url(),
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            params=params,
            timeout=20
        )
        response.raise_for_status()

        records = response.json().get("records", [])
        if not records:
            return None

        return records[0]

    except Exception as e:
        print("GET CLAIM BY SLUG ERROR:", str(e), flush=True)
        return None


def get_latest_claim():
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        return None

    try:
        params = {
            "maxRecords": 1,
            "sort[0][field]": "Date Added",
            "sort[0][direction]": "desc"
        }

        response = requests.get(
            airtable_url(),
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            params=params,
            timeout=20
        )
        response.raise_for_status()

        records = response.json().get("records", [])
        if not records:
            return None

        return records[0]

    except Exception as e:
        print("GET LATEST CLAIM ERROR:", str(e), flush=True)
        return None


@app.route("/health")
def health():
    return "ok", 200


@app.route("/bootcheck")
def bootcheck():
    return jsonify({
        "status": "booted",
        "openai_key_present": bool(OPENAI_API_KEY),
        "anthropic_key_present": bool(ANTHROPIC_API_KEY),
        "airtable_present": bool(AIRTABLE_TOKEN and AIRTABLE_BASE_ID and AIRTABLE_TABLE_NAME)
    }), 200


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()

        if password == SUPER_PASSWORD:
            session["logged_in"] = True
            session["superuser"] = True
            return redirect("/")

        if password == APP_PASSWORD:
            session["logged_in"] = True
            session["superuser"] = False
            return redirect("/")

        return "Wrong password", 401

    return """
    <html>
    <head>
        <title>Where the Truth Lies</title>
        <style>
            body{background:#0d1b2a;color:#f5f0e8;font-family:Georgia,serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
            .box{background:#162336;padding:40px;border:1px solid #8a6f2e;text-align:center;max-width:360px;width:90%}
            h2{font-size:22px;margin-bottom:6px}
            .sub{font-size:12px;color:#c9a84c;letter-spacing:.18em;text-transform:uppercase;font-style:italic;margin-bottom:28px}
            input{padding:12px 14px;margin-top:10px;width:100%;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.12);color:#f5f0e8;font-size:16px;font-family:Georgia,serif;box-sizing:border-box;outline:none}
            input:focus{border-color:rgba(201,168,76,.4)}
            button{margin-top:14px;padding:12px;width:100%;background:#c9a84c;border:none;cursor:pointer;font-size:12px;letter-spacing:.18em;text-transform:uppercase;font-weight:700;color:#0d1b2a}
            button:hover{background:#e8c97a}
            .latin{margin-top:24px;font-size:11px;color:#8a6f2e;font-style:italic;letter-spacing:.12em}
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

    recent_claims = get_recent_claims(limit=10)
    latest_record = get_latest_claim()
    current_claim = build_claim_context(latest_record) if latest_record else None

    return render_template(
        "index.html",
        superuser=session.get("superuser", False),
        recent_claims=recent_claims,
        current_claim=current_claim
    )


@app.route("/claim/<slug>")
def claim_detail(slug):
    if not session.get("logged_in"):
        return redirect("/login")

    recent_claims = get_recent_claims(limit=10)
    record = get_claim_by_slug(slug)

    if not record:
        latest_record = get_latest_claim()
        current_claim = build_claim_context(latest_record) if latest_record else None
    else:
        current_claim = build_claim_context(record)

    return render_template(
        "index.html",
        superuser=session.get("superuser", False),
        recent_claims=recent_claims,
        current_claim=current_claim
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

    data = request.get_json() or {}
    claim = (data.get("claim") or "").strip()
    mode = data.get("mode", "strip")

    if not claim:
        return jsonify({"error": "Claim is required"}), 400

    is_super = session.get("superuser", False)

    claude_json = {}
    openai_json = {}

    try:
        if anthropic_client:
            claude_response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system=CLAIMLAB_SYSTEM,
                messages=[{"role": "user", "content": f"Excavate this claim: \"{claim}\""}]
            )
            claude_json = safe_json_parse(claude_response.content[0].text)
        else:
            claude_json = {"error": "Anthropic not configured"}
    except Exception as e:
        claude_json = {"error": str(e)}

    try:
        if openai_client:
            openai_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CLAIMLAB_SYSTEM},
                    {"role": "user", "content": f"Excavate this claim: \"{claim}\""}
                ]
            )
            openai_json = safe_json_parse(openai_response.choices[0].message.content)
        else:
            openai_json = {"error": "OpenAI not configured"}
    except Exception as e:
        openai_json = {"error": str(e)}

    airtable_result = {}
    try:
        if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
            airtable_result = {"saved": False, "error": "Airtable not configured"}
        else:
            primary = claude_json if "error" not in claude_json else openai_json
            fields = extract_primary_record_fields(claim, primary, mode)
            fields["Claude Raw JSON"] = json.dumps(claude_json)[:100000]
            fields["OpenAI Raw JSON"] = json.dumps(openai_json)[:100000]

            response = requests.post(
                airtable_url(),
                headers=airtable_headers(),
                json={"fields": fields},
                timeout=30
            )

            if response.status_code == 200:
                record = response.json()
                airtable_result = {
                    "saved": True,
                    "record_id": record.get("id"),
                    "url_slug": fields["URL Slug"],
                    "redirect_to": f"/claim/{fields['URL Slug']}"
                }
            else:
                try:
                    airtable_error = response.json()
                except Exception:
                    airtable_error = response.text

                print("AIRTABLE STATUS:", response.status_code, flush=True)
                print("AIRTABLE ERROR:", airtable_error, flush=True)
                print("AIRTABLE FIELDS SENT:", json.dumps(fields, indent=2), flush=True)

                airtable_result = {
                    "saved": False,
                    "status_code": response.status_code,
                    "error": airtable_error
                }

    except Exception as e:
        airtable_result = {"saved": False, "error": str(e)}

    return jsonify({
        "claude": claude_json,
        "openai": openai_json,
        "airtable": airtable_result,
        "superuser": is_super
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
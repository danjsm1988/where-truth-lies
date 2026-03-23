import os
import json
import re
import unicodedata
from datetime import datetime

import requests
from flask import Flask, request, jsonify, render_template, redirect, session
from openai import OpenAI
import anthropic

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-secret")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Claims")
AIRTABLE_USERS_TABLE_NAME = os.getenv("AIRTABLE_USERS_TABLE_NAME", "Users")

CLAIMLAB_SYSTEM = """You are ClaimLab, the analytical engine of Where the Truth Lies — a political intelligence platform built on an excavation methodology. Motto: Beyond the Argument. Latin seal: Ubi Veritas Latet.

You do not fact-check. You excavate. Your job is to remove what a claim is NOT so that what it actually IS becomes visible.

Return ONLY valid JSON. No markdown fences. No preamble. No explanation outside the JSON.

CRITICAL FORMATTING RULES that cannot be violated under any circumstance:
Never use bullet points, dashes, or hyphens in ANY text field
Write ALL text fields as flowing prose with complete sentences
Never tell readers what to think
Apply the same analytical standard regardless of political party or speaker
Use reportedly or estimated for unconfirmed claims
ALL fields must be populated unless truly not applicable
Reality anchor rule:
Before determining that a claim is false, fabricated, or based on a nonexistent event, first verify whether the underlying event actually occurred.
If the claim depends on a real event, analyze the claim on that real foundation even if the wording resembles misinformation, satire, or fabricated narrative formats.
If the claim depends on an event that did not occur, the entire claim must be treated as false or structurally invalid rather than merely unproven.
When a claim contains both a real event and an unverified accusation layered on top of it, treat the real event as established and evaluate the accusation separately.

Speaker rule:
If the claim explicitly names a speaker, use that person or entity
If the claim does not explicitly name a speaker but is strongly associated with a dominant public figure or institution, infer that speaker
If there is no clearly dominant speaker, return Unknown

Always return this exact JSON structure:

{
  "Stripped Claim": "The claim in plain language, with emotional framing and rhetorical decoration removed. One sentence if possible.",
  "Speaker": "Who made the claim, or Unknown if not specified.",
  "Topic": "Exactly one of: Iran War, Energy, Healthcare, Social Security, Medicare, Medicaid, Defense, Military, Elections, Economy, Immigration, Foreign Policy, Crime, Gender Issues, Constitutional Rights, Education, Other",
  "Sub Claims": [
    {"claim": "First distinct falsifiable claim within the statement", "verdict": "True"},
    {"claim": "Second distinct falsifiable claim", "verdict": "Contested"},
    {"claim": "Third distinct falsifiable claim if present", "verdict": "Unproven"}
  ],
  "Direct Facts": "What the documented record actually shows. 3 to 4 sentences of prose. Cite specific figures, dates, and institutional confirmations where available. Use reportedly or estimated for unconfirmed items.",
  "Adjacent Facts": "What the claim omits or ignores on BOTH sides equally. 2 to 3 sentences of prose. Complicate the primary narrative from the left and from the right.",
  "Root Concern": "The legitimate underlying concern that exists even beneath a false or misleading claim. 1 to 2 sentences of prose.",
  "Values Divergence": "Where the real disagreement lives. Usually not in the facts themselves but in what people prioritize. 2 to 3 sentences of prose identifying the competing values.",
  "Constitutional Framework": "If the claim touches government action, rights, authority, public funds, war, law enforcement, elections, or institutional power, identify the specific Article, Section, or Amendment that applies and explain relevant founder intent. If not applicable, explain briefly why not.",
  "Common Ground": "Layer 06. Identify the narrow but genuine overlap between opposing sides. What are both sides actually trying to protect, prevent, preserve, or solve even if they disagree sharply on means. 2 to 3 sentences of prose.",
  "Left Perspective": "How the political left frames this claim, what their framing gets right, and where it fails or overstates. 2 to 3 sentences of prose.",
  "Right Perspective": "How the political right frames this claim, what their framing gets right, and where it fails or overstates. 2 to 3 sentences of prose.",
  "Founders Perspectives": {
    "George Washington": "What Washington would say, grounded in documented writings. 2 sentences of prose.",
    "Thomas Jefferson": "What Jefferson would say, grounded in documented writings. 2 sentences of prose.",
    "James Madison": "What Madison would say, grounded in documented writings. 2 sentences of prose.",
    "Alexander Hamilton": "What Hamilton would say, grounded in documented writings. 2 sentences of prose.",
    "Benjamin Franklin": "What Franklin would say, grounded in documented writings. 2 sentences of prose.",
    "John Adams": "What Adams would say, grounded in documented writings. 2 sentences of prose.",
    "John Jay": "What Jay would say, grounded in documented writings. 2 sentences of prose.",
    "John Marshall": "What Marshall would say, grounded in Marbury v. Madison, McCulloch v. Maryland, and early Supreme Court constitutional reasoning. 2 sentences of prose."
  },
  "Scenario Map": "MANDATORY — always populate with exactly five scenarios in this exact format: SCENARIO A — [Short Name]: Confidence: [Documented/Mixed/Speculative]. [2 to 3 sentences.] Analyst Divergence: [Low/Moderate/High]. Repeat through Scenario E. End with NOTE: These are plausible trajectories only. Not predictions. Only actions and time will determine the actual path.",
  "Glossary": [
    {"term": "A term, name, or concept that general readers may not recognize", "definition": "Plain language definition in 1 to 2 sentences."},
    {"term": "Another term", "definition": "Plain language definition."},
    {"term": "A third term", "definition": "Plain language definition."}
  ],
  "Sources": "Primary sources:\\nSource description one: https://url-one.com\\nSource description two: https://url-two.com\\nSource description three: https://url-three.com\\nSource description four: https://url-four.com\\nSource description five: https://url-five.com\\n\\nInclude 6 to 10 real, verifiable URLs from major news outlets, government sites, institutional bodies, or authoritative sources. Format each line exactly as: Label: URL",
  "Overall Verdict": "Exactly one of: True, Mostly True, Substantially True, Plausible/Mixed, Contested, Exaggerated, Misleading, Unproven, False",
  "Strip Mode Summary": "Bottom line in 3 to 4 sentences of prose. What is documented. What remains contested. What the real question underneath this claim actually is."
}"""


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
    if "foreign policy" in text or "foreign" in text:
        return "Foreign Policy"
    if "crime" in text or "criminal" in text:
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


def now_dates():
    now = datetime.utcnow()
    try:
        display_date = now.strftime("%B %-d, %Y")
    except Exception:
        display_date = now.strftime("%B %d, %Y").replace(" 0", " ")
    short_date = now.strftime("%m/%d/%Y")
    return {
        "display_date": display_date,
        "short_date": short_date
    }


def escape_airtable_formula_value(value):
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def airtable_headers():
    return {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json"
    }


def airtable_url(table_name=None):
    table = table_name or AIRTABLE_TABLE_NAME
    return f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table}"


def parse_topics(topic_value):
    if isinstance(topic_value, list):
        return topic_value
    if isinstance(topic_value, str) and topic_value.strip():
        return [topic_value.strip()]
    return []


def parse_founders(raw):
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def parse_glossary(raw):
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except Exception:
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
        subclaims.append({"claim": sc1, "verdict": fields.get("Verdict: Sub-Claim1", "Unproven")})
    if sc2:
        subclaims.append({"claim": sc2, "verdict": fields.get("Verdict: Sub-Claim 2", "Unproven")})
    if sc3:
        subclaims.append({"claim": sc3, "verdict": fields.get("Verdict: Sub-Claim 3", "Unproven")})
    return subclaims


def build_claim_context(record):
    if not record:
        return None
    fields = record.get("fields", {})
    parsed_json = try_parse_raw_json(fields)
    title = fields.get("Original Quote") or fields.get("Stripped Claim") or "Untitled Claim"

    return {
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
        "constitutional_framework": fields.get("Constitutional Framework", ""),
        "common_ground": fields.get("Common Ground", ""),
        "left_perspective": fields.get("Left Perspective", ""),
        "right_perspective": fields.get("Right Perspective", ""),
        "scenario_map": fields.get("Scenario Map", ""),
        "source_urls": fields.get("Source URLs", ""),
        "founders_perspectives": parse_founders(fields.get("Founders Perspectives", "")),
        "glossary": parse_glossary(fields.get("Glossary", "")),
        "subclaims": build_subclaims(fields, parsed_json),
        "status": fields.get("Status", "Active"),
        "mode": fields.get("Mode", ""),
        "published": fields.get("Published", False),
        "human_reviewed": fields.get("Human Reviewed", False)
    }


def extract_primary_record_fields(claim, parsed, mode, existing_fields=None):
    dates = now_dates()
    existing_fields = existing_fields or {}
    is_full_reexcavate = mode == "full"

    if existing_fields and not is_full_reexcavate:
        human_reviewed_value = existing_fields.get("Human Reviewed", False)
        published_value = existing_fields.get("Published", False)
    else:
        human_reviewed_value = False
        published_value = False

    fields = {
        "Original Quote": claim,
        "Stripped Claim": parsed.get("Stripped Claim", claim),
        "Speaker": parsed.get("Speaker") or "Unknown",
        "Topic": [normalize_topic(parsed.get("Topic"))],
        "Human Reviewed": human_reviewed_value,
        "Published": published_value,
        "Status": "Active",
        "Mode": mode if mode in ["strip", "full"] else "strip",
        "URL Slug": build_url_slug(parsed, claim),
        "Date": dates["display_date"],
        "Date Added": dates["short_date"],
        "Last Updated": dates["short_date"]
    }

    for field in [
        "Strip Mode Summary",
        "Direct Facts",
        "Adjacent Facts",
        "Root Concern",
        "Values Divergence",
        "Constitutional Framework",
        "Common Ground",
        "Left Perspective",
        "Right Perspective",
        "Scenario Map"
    ]:
        if parsed.get(field):
            fields[field] = parsed[field]

    if parsed.get("Sources"):
        fields["Source URLs"] = parsed["Sources"]

    verdict = parsed.get("Overall Verdict") or parsed.get("Verdict")
    if verdict:
        fields["Overall Verdict"] = verdict

    founders = parsed.get("Founders Perspectives")
    if founders and isinstance(founders, dict):
        fields["Founders Perspectives"] = json.dumps(founders)

    glossary = parsed.get("Glossary")
    if glossary and isinstance(glossary, list):
        fields["Glossary"] = json.dumps(glossary)

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
            fields["Sub-Claims"] = " | ".join(
                [sc.get("claim", "") for sc in sub_claims if sc.get("claim")]
            ).strip()

    return fields


def get_recent_claims(limit=10):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        return []
    try:
        params = {
            "maxRecords": limit,
            "sort[0][field]": "Date Added",
            "sort[0][direction]": "desc"
        }
        response = requests.get(
            airtable_url(AIRTABLE_TABLE_NAME),
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            params=params,
            timeout=20
        )
        response.raise_for_status()
        records = response.json().get("records", [])
        recent = []
        for record in records:
            f = record.get("fields", {})
            recent.append({
                "title": f.get("Original Quote") or f.get("Stripped Claim") or "Untitled Claim",
                "slug": f.get("URL Slug", ""),
                "date": f.get("Date") or f.get("Date Added", ""),
                "verdict": f.get("Overall Verdict", "Unproven")
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
            "filterByFormula": f"{{URL Slug}}='{escape_airtable_formula_value(slug)}'",
            "maxRecords": 1
        }
        response = requests.get(
            airtable_url(AIRTABLE_TABLE_NAME),
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            params=params,
            timeout=20
        )
        response.raise_for_status()
        records = response.json().get("records", [])
        return records[0] if records else None
    except Exception as e:
        print("GET CLAIM BY SLUG ERROR:", str(e), flush=True)
        return None


def get_claim_by_original_quote(claim):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME or not claim:
        return None
    try:
        safe_claim = escape_airtable_formula_value(claim.strip())
        params = {
            "filterByFormula": f"{{Original Quote}}='{safe_claim}'",
            "maxRecords": 1
        }
        response = requests.get(
            airtable_url(AIRTABLE_TABLE_NAME),
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            params=params,
            timeout=20
        )
        response.raise_for_status()
        records = response.json().get("records", [])
        return records[0] if records else None
    except Exception as e:
        print("GET CLAIM BY ORIGINAL QUOTE ERROR:", str(e), flush=True)
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
            airtable_url(AIRTABLE_TABLE_NAME),
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            params=params,
            timeout=20
        )
        response.raise_for_status()
        records = response.json().get("records", [])
        return records[0] if records else None
    except Exception as e:
        print("GET LATEST CLAIM ERROR:", str(e), flush=True)
        return None


def get_user_by_username(username):
    if not username:
        return None
    try:
        safe_username = escape_airtable_formula_value(username.strip())
        params = {
            "filterByFormula": f"{{Username}}='{safe_username}'",
            "maxRecords": 1
        }
        response = requests.get(
            airtable_url(AIRTABLE_USERS_TABLE_NAME),
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            params=params,
            timeout=20
        )
        response.raise_for_status()
        records = response.json().get("records", [])
        return records[0] if records else None
    except Exception as e:
        print("GET USER ERROR:", str(e), flush=True)
        return None


def update_user_claims_remaining(user_record_id, remaining):
    return requests.patch(
        f"{airtable_url(AIRTABLE_USERS_TABLE_NAME)}/{user_record_id}",
        headers=airtable_headers(),
        json={"fields": {"Claims Remaining": remaining}},
        timeout=20
    )


def update_airtable_record(record_id, fields):
    return requests.patch(
        f"{airtable_url(AIRTABLE_TABLE_NAME)}/{record_id}",
        headers=airtable_headers(),
        json={"fields": fields},
        timeout=30
    )


def create_airtable_record(fields):
    return requests.post(
        airtable_url(AIRTABLE_TABLE_NAME),
        headers=airtable_headers(),
        json={"fields": fields},
        timeout=30
    )


@app.route("/health")
def health():
    return "ok", 200


@app.route("/bootcheck")
def bootcheck():
    return jsonify({
        "status": "booted",
        "openai_key_present": bool(OPENAI_API_KEY),
        "anthropic_key_present": bool(ANTHROPIC_API_KEY),
        "airtable_present": bool(AIRTABLE_TOKEN and AIRTABLE_BASE_ID and AIRTABLE_TABLE_NAME),
        "users_table_present": bool(AIRTABLE_USERS_TABLE_NAME)
    }), 200


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        user = get_user_by_username(username)
        if not user:
            return "Invalid login", 401

        fields = user.get("fields", {})
        if not fields.get("Active", False):
            return "Account disabled", 403

        if password != fields.get("Password", ""):
            return "Invalid login", 401

        role = fields.get("Role", "standard")
        claims_remaining = int(fields.get("Claims Remaining", 0) or 0)

        session["logged_in"] = True
        session["username"] = username
        session["user_id"] = user.get("id")
        session["role"] = role
        session["superuser"] = role in ["superuser", "limited_superuser"]
        session["claims_remaining"] = claims_remaining

        return redirect("/")

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
                <input type="text" name="username" placeholder="Username" required autofocus />
                <input type="password" name="password" placeholder="Password" required />
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

    role = session.get("role", "standard")
    claims_remaining = int(session.get("claims_remaining", 0) or 0)

    if role == "standard":
        return jsonify({
            "error": "Your account can browse existing claim files, but cannot run new excavations."
        }), 403

    if role == "limited_superuser" and claims_remaining <= 0:
        return jsonify({
            "error": "Claim limit reached. You can still browse existing claim files, but new excavations are blocked."
        }), 403

    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

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
                max_tokens=4000,
                temperature=0,
                system=CLAIMLAB_SYSTEM,
                messages=[{"role": "user", "content": f'Excavate this claim: "{claim}"'}]
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
                    {"role": "user", "content": f'Excavate this claim: "{claim}"'}
                ],
                max_tokens=4000,
                temperature=0
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
            existing_record = get_claim_by_original_quote(claim)
            existing_fields = existing_record.get("fields", {}) if existing_record else {}

            fields = extract_primary_record_fields(
                claim=claim,
                parsed=primary,
                mode=mode,
                existing_fields=existing_fields
            )
            fields["Claude Raw JSON"] = json.dumps(claude_json)[:100000]
            fields["OpenAI Raw JSON"] = json.dumps(openai_json)[:100000]

            if existing_record:
                response = update_airtable_record(existing_record["id"], fields)
                if response.status_code == 200:
                    record = response.json()
                    airtable_result = {
                        "saved": True,
                        "record_id": record.get("id"),
                        "url_slug": fields["URL Slug"],
                        "redirect_to": f"/claim/{fields['URL Slug']}",
                        "updated_existing": True
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
            else:
                response = create_airtable_record(fields)
                if response.status_code == 200:
                    record = response.json()
                    airtable_result = {
                        "saved": True,
                        "record_id": record.get("id"),
                        "url_slug": fields["URL Slug"],
                        "redirect_to": f"/claim/{fields['URL Slug']}",
                        "created_new": True
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

            if airtable_result.get("saved") and role == "limited_superuser":
                new_count = max(0, claims_remaining - 1)
                update_resp = update_user_claims_remaining(session["user_id"], new_count)
                if update_resp.status_code == 200:
                    session["claims_remaining"] = new_count
                else:
                    print("USER CLAIM COUNT UPDATE FAILED:", update_resp.text, flush=True)

    except Exception as e:
        airtable_result = {"saved": False, "error": str(e)}

    return jsonify({
        "claude": claude_json,
        "openai": openai_json,
        "airtable": airtable_result,
        "superuser": session.get("superuser", False),
        "claims_remaining": session.get("claims_remaining")
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
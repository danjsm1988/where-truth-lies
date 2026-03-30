import os
import json
import re
import unicodedata
import threading
from datetime import datetime

import requests
from flask import Flask, request, jsonify, render_template, redirect, session
from openai import OpenAI
import anthropic

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-secret")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Claims")
AIRTABLE_USERS_TABLE_NAME = os.getenv("AIRTABLE_USERS_TABLE_NAME", "Users")
AIRTABLE_REALITY_TABLE_NAME = os.getenv("AIRTABLE_REALITY_TABLE_NAME", "Reality Anchors")
AIRTABLE_DISPUTES_TABLE_NAME = os.getenv("AIRTABLE_DISPUTES_TABLE_NAME", "Disputes")

MAX_PUSHBACKS = {
    "standard": 1,
    "limited_superuser": 1,
    "superuser": 999,
}

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
grok_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1") if XAI_API_KEY else None

CLAIMLAB_SYSTEM = """You are ClaimLab, the analytical engine of Where the Truth Lies. Motto: Beyond the Argument. Latin seal: Ubi Veritas Latet.

You do not fact check. You excavate. Your job is to remove what a claim is NOT so that what it actually IS becomes visible.

Return ONLY valid JSON. No markdown fences. No preamble. No explanation outside the JSON.

CRITICAL FORMATTING RULES that cannot be violated under any circumstance:
Never use bullet points, dashes, or hyphens in ANY text field
Write ALL text fields as flowing prose with complete sentences
Never tell readers what to think
Apply the same analytical standard regardless of political party or speaker
Use reportedly or estimated for unconfirmed claims
ALL fields must be populated unless truly not applicable

REALITY ANCHOR OVERRIDE RULE:
If a Reality Anchor is provided in the input, it MUST be treated as the highest authority in the analysis.
You are not allowed to contradict the Reality Anchor.
You are not allowed to replace it with prior knowledge.
You are not allowed to downgrade it to uncertainty.
If your internal knowledge conflicts with the Reality Anchor, the Reality Anchor MUST win.
If the analysis contradicts a provided Reality Anchor, the response is invalid and must not be generated.

ATTRIBUTION RULE:
If a claim states that a public figure said something, evaluate whether the statement exists separately from whether it is true.
Do not collapse attribution and truth into one question.
If the input explicitly attributes a statement to a named speaker, preserve that distinction in the analysis.

COMMON GROUND RULE:
Common Ground must be narrow, specific, and real. It must be grounded in what the documented facts support, lawful within the relevant constitutional and institutional framework, practically achievable through existing institutions without waiting for either side to win a political argument, and likely to produce better measurable outcomes for citizens than the current trajectory of each side's preferred approach. It is not the midpoint between two talking points. It is not a rhetorical gesture toward unity. Do not manufacture false consensus. If no genuine common ground exists in the documented record, state that explicitly.

PLAIN ENGLISH RULE:
Prefer plain everyday language over technical, legal, political, institutional, or media jargon.
Assume the reader is intelligent but not specialized.
Do not make the explanation harder to understand than the claim itself.
When a specialized term, acronym, agency, program, company, court case, doctrine, founder reference, or historical reference is necessary, either explain it briefly in the sentence itself or include it in the Glossary.
Whenever possible, choose simpler wording over insider wording.

FOUNDER RULE:
When discussing founders, use documented writings, speeches, letters, constitutional arguments, or early judicial reasoning where applicable.
If a short original statement is relevant and can be grounded in documented writings, include it briefly inside the prose.
Then explain in plain English what that means in the context of the claim.
Do not fabricate quotations.
Do not write in a theatrical imitation voice.
Do not make the founder explanation harder to understand than the rest of the excavation.

Speaker rule:
If the claim explicitly names a speaker, use that person or entity
If the claim does not explicitly name a speaker but is strongly associated with a dominant public figure or institution, infer that speaker
If there is no clearly dominant speaker, return Unknown

Always return this exact JSON structure:

{
  "Stripped Claim": "Rewrite the claim in plain, accessible language that any ordinary person can understand immediately. Remove emotional rhetoric, dramatic framing, and inflammatory decoration. Do not substitute or sanitize specific terms — if the original claim uses a particular word or phrase, preserve it unless it is purely emotional amplification with no factual content. One sentence only.",
  "Quick Explanation": "Write exactly four labeled lines. Each line starts with its label followed by a colon. Line 1 — ONE-LINE READ: One sentence with deliberate tension between two truths. Do not summarize. Make the reader feel they almost understand but need to go deeper. Example shape: The claim is grounded in real evidence, but its strongest version goes further than the current record supports. Line 2 — WHAT HOLDS UP: One sentence on what the record actually supports. Line 3 — WHAT IS DISPUTED: One sentence on what remains contested or unsupported. Line 4 — WHERE AGREEMENT EXISTS: One sentence identifying narrow genuine common ground, or state plainly that none exists. No bullet points. No dashes. Plain language only. No preamble before the first label.",
  "Speaker": "Who made the claim, or Unknown if not specified.",
  "Topic": "Exactly one of: Iran War, Energy, Healthcare, Social Security, Medicare, Medicaid, Defense, Military, Elections, Economy, Immigration, Foreign Policy, Crime, Gender Issues, Constitutional Rights, Education, Other",
  "Sub Claims": [
    {"claim": "First distinct falsifiable claim within the statement", "verdict": "True"},
    {"claim": "Second distinct falsifiable claim", "verdict": "Contested"},
    {"claim": "Third distinct falsifiable claim if present", "verdict": "Unproven"}
  ],
  "Direct Facts": "What the documented record actually shows. 3 to 4 sentences of prose in plain English.",
  "Adjacent Facts": "What the claim omits or ignores on BOTH sides equally. 2 to 3 sentences of prose in plain English.",
  "Root Concern": "The legitimate underlying concern that exists even beneath a false or misleading claim. 1 to 2 sentences of prose in plain English.",
  "Values Divergence": "Where the real disagreement lives. Usually not in the facts themselves but in what people prioritize. 2 to 3 sentences of prose identifying the competing values in plain English.",
  "Constitutional Framework": "If the claim touches government action, rights, authority, public funds, war, law enforcement, elections, or institutional power, identify the specific Article, Section, or Amendment that applies and explain relevant founder intent in plain English. If not applicable, explain briefly why not.",
  "Common Ground": "Layer 06. Identify the narrow but genuine overlap between opposing sides. 2 to 3 sentences of prose in plain English.",
  "Left Perspective": "How the political left frames this claim, what their framing gets right, and where it fails or overstates. 2 to 3 sentences of prose in plain English.",
  "Right Perspective": "How the political right frames this claim, what their framing gets right, and where it fails or overstates. 2 to 3 sentences of prose in plain English.",
  "Founders Perspectives": {
    "George Washington": "Use source grounded prose. If applicable, briefly include a documented Washington line or idea and then explain in plain English what it means here. 2 sentences.",
    "Thomas Jefferson": "Use source grounded prose. If applicable, briefly include a documented Jefferson line or idea and then explain in plain English what it means here. 2 sentences.",
    "James Madison": "Use source grounded prose. If applicable, briefly include a documented Madison line or idea and then explain in plain English what it means here. 2 sentences.",
    "Alexander Hamilton": "Use source grounded prose. If applicable, briefly include a documented Hamilton line or idea and then explain in plain English what it means here. 2 sentences.",
    "Benjamin Franklin": "Use source grounded prose. If applicable, briefly include a documented Franklin line or idea and then explain in plain English what it means here. 2 sentences.",
    "John Adams": "Use source grounded prose. If applicable, briefly include a documented Adams line or idea and then explain in plain English what it means here. 2 sentences.",
    "John Jay": "Use source grounded prose. If applicable, briefly include a documented Jay line or idea and then explain in plain English what it means here. 2 sentences.",
    "John Marshall": "Use source grounded prose from Marbury v. Madison, McCulloch v. Maryland, and early Supreme Court constitutional reasoning where applicable, then explain in plain English what it means here. 2 sentences."
  },
  "Scenario Map": "MANDATORY. Always populate with exactly five scenarios in this exact format: SCENARIO A — [Short Name]: Confidence: [Documented/Mixed/Speculative]. [2 to 3 sentences.] Analyst Divergence: [Low/Moderate/High]. Repeat through Scenario E. End with NOTE: These are plausible trajectories only. Not predictions. Only actions and time will determine the actual path.",
  "Glossary": [
    {"term": "A term, acronym, agency, company, program, court case, founder reference, or concept that general readers may not recognize", "definition": "Plain language definition in 1 to 2 sentences."},
    {"term": "Another term", "definition": "Plain language definition."},
    {"term": "A third term", "definition": "Plain language definition."}
  ],
  "Sources": "Primary sources:\\nSource description one: https://url-one.com\\nSource description two: https://url-two.com\\nSource description three: https://url-three.com\\nSource description four: https://url-four.com\\nSource description five: https://url-five.com\\n\\nInclude 6 to 10 real, verifiable URLs from major news outlets, government sites, institutional bodies, or authoritative sources. Format each line exactly as: Label: URL",
  "Overall Verdict": "Exactly one of: True, Mostly True, Substantially True, Plausible/Mixed, Contested, Exaggerated, Misleading, Unproven, False",
  "Strip Mode Summary": "This is the full paid analytical paragraph, not the short Quick Explanation. Write this in the spirit of Thomas Paine's Common Sense — plain language, no jargon, no hedging, accessible to anyone regardless of political background or education level. Answer three things clearly: what is actually happening beneath the claim, why it matters in real terms, and what someone should be paying attention to next. This should reflect the full excavation without referencing the layers directly. 3 to 4 sentences. No bullet points. No dashes. Do not be condescending. Do not be sarcastic. Avoid phrases like obviously or clearly. Do not over-explain uncertainty, but do not present future outcomes as guaranteed. Let the tone reflect that situations evolve. Write with calm, grounded clarity."
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


def extract_verdict_from_parsed(parsed):
    if not isinstance(parsed, dict):
        return ""
    return (parsed.get("Overall Verdict") or parsed.get("Verdict") or "").strip()


def detect_attribution_metadata(original_quote, speaker):
    text = (original_quote or "").strip()
    speaker = (speaker or "").strip()

    if not text:
        return {"status": "Unknown", "detail": "No input text available."}

    lowered = text.lower()
    patterns = [
        r"\bsaid\b", r"\bsays\b", r"\bclaims\b", r"\bclaimed\b", r"\bstated\b",
        r"\bposted\b", r"\bwrote\b", r"\bargued\b", r"\bannounced\b", r"\btold\b",
        r"\baccording to\b"
    ]
    explicitly_attributed = any(re.search(p, lowered) for p in patterns)

    if explicitly_attributed and speaker and speaker != "Unknown":
        return {
            "status": "Attributed",
            "detail": "The input explicitly attributes the statement to the named speaker."
        }

    if speaker and speaker != "Unknown":
        return {
            "status": "Inferred",
            "detail": "The system assigned a speaker, but the input did not explicitly attribute the statement."
        }

    return {
        "status": "Unknown",
        "detail": "No clear speaker attribution was identified."
    }


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
    if "crime" in text or "criminal" in text or "murder" in text or "shoot" in text or "assass" in text:
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
    return {"display_date": display_date, "short_date": short_date}


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


def airtable_get_all(table_name, params=None):
    all_records = []
    offset = None
    while True:
        use_params = dict(params or {})
        if offset:
            use_params["offset"] = offset
        response = requests.get(
            airtable_url(table_name),
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            params=use_params,
            timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        all_records.extend(payload.get("records", []))
        offset = payload.get("offset")
        if not offset:
            break
    return all_records


def clean_display_title(raw_title):
    """Strip appended user context from display titles."""
    if not raw_title:
        return raw_title
    marker = '\n\nAdditional context from user:'
    idx = raw_title.find(marker)
    if idx != -1:
        return raw_title[:idx].strip()
    return raw_title


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


def _safe_get_breakout_grouped(record_id, fields):
    """Load breakout claims for this record. Fails silently."""
    try:
        breakouts = get_breakout_claims_for_parent(record_id)
        return group_breakout_claims(breakouts)
    except Exception as e:
        print(f"BREAKOUT LOAD ERROR (non-fatal): {e}", flush=True)
        return []


def build_claim_context(record):
    if not record:
        return None

    fields = record.get("fields", {})
    claim_slug = fields.get("URL Slug", "")
    claim_disputes = get_disputes_for_claim(claim_slug)
    dispute_threads = group_disputes_into_threads(claim_disputes)
    title = clean_display_title(fields.get("Original Quote") or fields.get("Stripped Claim") or "Untitled Claim")

    claude_parsed = safe_json_parse(fields.get("Claude Raw JSON", ""))
    openai_parsed = safe_json_parse(fields.get("OpenAI Raw JSON", ""))
    grok_adjudication = safe_json_parse(fields.get("Grok Raw JSON", ""))

    claude_verdict = extract_verdict_from_parsed(claude_parsed)
    openai_verdict = extract_verdict_from_parsed(openai_parsed)

    models_diverged = (
        bool(claude_verdict)
        and bool(openai_verdict)
        and claude_verdict.strip().lower() != openai_verdict.strip().lower()
    )
    # Also check stored divergence flag and note
    stored_diverged = fields.get("Models Diverged", False)
    divergence_note = (fields.get("Model Divergence Note") or "").strip()
    # Use stored flag if available (more reliable than re-comparing)
    if stored_diverged:
        models_diverged = True

    # OpenAI challenge layer data
    openai_challenge = safe_json_parse(fields.get("OpenAI Challenge JSON", "{}") or "{}")
    where_claude_wrong = (openai_challenge.get("where_claude_is_most_likely_wrong") or "").strip()
    what_claude_missed = (openai_challenge.get("what_claude_missed") or "").strip()
    strongest_alternative = (openai_challenge.get("strongest_alternative_interpretation") or "").strip()

    grok_grounded = False
    grok_status = "not_run"
    if isinstance(grok_adjudication, dict) and grok_adjudication:
        event_status = grok_adjudication.get("event_status", "unclear")
        requires_grounding = grok_adjudication.get("requires_live_grounding", False)
        if event_status in ("established", "false", "unverified") or requires_grounding:
            grok_grounded = True
            grok_status = "grounded"
        else:
            grok_status = "ran_no_anchor"

    speaker = fields.get("Speaker", "Unknown")
    attribution = detect_attribution_metadata(fields.get("Original Quote", ""), speaker)
    parsed_json = try_parse_raw_json(fields)

    quick_explanation = fields.get("Quick Explanation", "") or parsed_json.get("Quick Explanation", "")
    stripped_claim = fields.get("Stripped Claim", "") or parsed_json.get("Stripped Claim", "")
    overall_verdict = fields.get("Overall Verdict", "Unproven") or parsed_json.get("Overall Verdict", "Unproven")
    subclaims = build_subclaims(fields, parsed_json)

    direct_facts = fields.get("Direct Facts", "") or parsed_json.get("Direct Facts", "")
    adjacent_facts = fields.get("Adjacent Facts", "") or parsed_json.get("Adjacent Facts", "")
    root_concern = fields.get("Root Concern", "") or parsed_json.get("Root Concern", "")
    values_divergence = fields.get("Values Divergence", "") or parsed_json.get("Values Divergence", "")
    constitutional_framework = fields.get("Constitutional Framework", "") or parsed_json.get("Constitutional Framework", "")
    common_ground = fields.get("Common Ground", "") or parsed_json.get("Common Ground", "")
    left_perspective = fields.get("Left Perspective", "") or parsed_json.get("Left Perspective", "")
    right_perspective = fields.get("Right Perspective", "") or parsed_json.get("Right Perspective", "")
    scenario_map = fields.get("Scenario Map", "") or parsed_json.get("Scenario Map", "")
    strip_mode_summary = fields.get("Strip Mode Summary", "") or parsed_json.get("Strip Mode Summary", "")
    source_urls = fields.get("Source URLs", "") or parsed_json.get("Sources", "")
    founders_perspectives = parse_founders(fields.get("Founders Perspectives", "") or parsed_json.get("Founders Perspectives", {}))
    glossary = parse_glossary(fields.get("Glossary", "") or parsed_json.get("Glossary", []))

    quick_view = {
        "stripped_claim": stripped_claim,
        "overall_verdict": overall_verdict,
        "quick_explanation": quick_explanation,
        "subclaims": subclaims
    }

    full_view = {
        "bottom_line": strip_mode_summary,
        "direct_facts": direct_facts,
        "adjacent_facts": adjacent_facts,
        "root_concern": root_concern,
        "values_divergence": values_divergence,
        "constitutional_framework": constitutional_framework,
        "common_ground": common_ground,
        "left_perspective": left_perspective,
        "right_perspective": right_perspective,
        "founders_perspectives": founders_perspectives,
        "scenario_map": scenario_map,
        "source_urls": source_urls,
        "glossary": glossary,
        "subclaims": subclaims
    }

    return {
        "record_id": record.get("id"),
        "slug": fields.get("URL Slug", ""),
        "title": title,
        "stripped_claim": stripped_claim,
        "quick_explanation": quick_explanation,
        "speaker": speaker,
        "topics": parse_topics(fields.get("Topic")),
        "date": fields.get("Date") or fields.get("Date Added", ""),
        "overall_verdict": overall_verdict,
        "strip_mode_summary": strip_mode_summary,
        "direct_facts": direct_facts,
        "adjacent_facts": adjacent_facts,
        "root_concern": root_concern,
        "values_divergence": values_divergence,
        "constitutional_framework": constitutional_framework,
        "common_ground": common_ground,
        "left_perspective": left_perspective,
        "right_perspective": right_perspective,
        "scenario_map": scenario_map,
        "source_urls": source_urls,
        "founders_perspectives": founders_perspectives,
        "glossary": glossary,
        "subclaims": subclaims,
        "status": fields.get("Status", "Active"),
        "mode": fields.get("Mode", ""),
        "published": fields.get("Published", False),
        "human_reviewed": fields.get("Human Reviewed", False),
        "entered_by": fields.get("Entered By", ""),
        "last_reanalyzed": (fields.get("Last Reanalyzed") or "")[:10],
        "reanalyzed_by": fields.get("Reanalyzed By", ""),
        "dispute_threads": dispute_threads,
        "disputes": claim_disputes,
        "quick_view": quick_view,
        "full_view": full_view,

        "attribution_status": attribution["status"],
        "attribution_detail": attribution["detail"],
        "claude_verdict": claude_verdict,
        "openai_verdict": openai_verdict,
        "models_diverged": models_diverged,
        "divergence_note": divergence_note,
        "where_claude_wrong": where_claude_wrong,
        "what_claude_missed": what_claude_missed,
        "strongest_alternative": strongest_alternative,
        "grok_adjudication": grok_adjudication,
        "grok_grounded": grok_grounded,
        "grok_status": grok_status,
        "confirmed_current_facts": (grok_adjudication or {}).get("confirmed_current_facts") or [],
        "contested_current_facts": (grok_adjudication or {}).get("contested_current_facts") or [],
        "current_narratives": (grok_adjudication or {}).get("current_narratives") or [],
        "view_count": int(fields.get("View Count", 0) or 0),
        "breakout_claims_grouped": _safe_get_breakout_grouped(record.get("id"), fields),
        "has_breakout_children": fields.get("Has Breakout Children", False),
        "claim_identifier": fields.get("Claim Identifier", ""),
        "claim_depth": int(fields.get("Claim Depth", 0) or 0),
        "origin_type": fields.get("Origin Type", "Original"),
        "breakout_source_section": fields.get("Breakout Source Section", ""),
    }


def extract_primary_record_fields(claim, parsed, mode, username, existing_fields=None):
    dates = now_dates()
    existing_fields = existing_fields or {}
    is_full_reexcavate = mode == "full"

    if existing_fields and not is_full_reexcavate:
        human_reviewed_value = existing_fields.get("Human Reviewed", False)
        published_value = existing_fields.get("Published", False)
    else:
        human_reviewed_value = False
        published_value = False

    entered_by = existing_fields.get("Entered By") or username or "Unknown"

    fields = {
        "Original Quote": claim,
        "Stripped Claim": parsed.get("Stripped Claim", claim),
        "Quick Explanation": parsed.get("Quick Explanation", ""),
        "Speaker": parsed.get("Speaker") or "Unknown",
        "Topic": [normalize_topic(parsed.get("Topic"))],
        "Human Reviewed": human_reviewed_value,
        "Published": published_value,
        "Status": "Active",
        "Mode": mode if mode in ["strip", "full"] else "strip",
        "URL Slug": build_url_slug(parsed, claim),
        "Date": dates["display_date"],
        "Date Added": existing_fields.get("Date Added", dates["short_date"]),
        "Last Updated": dates["short_date"],
        "Last Reanalyzed": dates["short_date"],
        "Entered By": entered_by
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
            fields["Sub-Claims"] = " | ".join([sc.get("claim", "") for sc in sub_claims if sc.get("claim")]).strip()

    return fields


def get_recent_claims(limit=10):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        return []
    try:
        params = {
            "maxRecords": limit,
            "filterByFormula": "OR(NOT({Breakout User Excavated}), {Breakout User Excavated}!='No')",
            "sort[0][field]": "Date Added",
            "sort[0][direction]": "desc"
        }
        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        recent = []
        for record in records[:limit]:
            f = record.get("fields", {})
            recent.append({
    "title": clean_display_title(f.get("Original Quote") or f.get("Stripped Claim") or "Untitled Claim"),
    "slug": f.get("URL Slug", ""),
    "date": f.get("Date") or f.get("Date Added", ""),
    "verdict": f.get("Overall Verdict", "Unproven"),
    "topics": parse_topics(f.get("Topic")),
    "speaker": f.get("Speaker", "Unknown"),
    "entered_by": f.get("Entered By", "")
})
        return recent
    except Exception as e:
        print("RECENT CLAIMS ERROR:", str(e), flush=True)
        return []


def get_all_claims():
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        return []
    try:
        params = {
            "filterByFormula": "OR(NOT({Breakout User Excavated}), {Breakout User Excavated}!='No')",
            "sort[0][field]": "Date Added",
            "sort[0][direction]": "desc"
        }
        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        claims = []
        for record in records:
            f = record.get("fields", {})
            claims.append({
                "record_id": record.get("id"),
                "title": clean_display_title(f.get("Original Quote") or f.get("Stripped Claim") or "Untitled Claim"),
                "slug": f.get("URL Slug", ""),
                "date": f.get("Date") or f.get("Date Added", ""),
                "verdict": f.get("Overall Verdict", "Unproven"),
                "topics": parse_topics(f.get("Topic")),
                "speaker": f.get("Speaker", "Unknown"),
                "entered_by": f.get("Entered By", ""),
                "stripped_claim": f.get("Stripped Claim", ""),
                "human_reviewed": f.get("Human Reviewed", False)
            })
        return claims
    except Exception as e:
        print("ALL CLAIMS ERROR:", str(e), flush=True)
        return []


def get_topic_archives():
    grouped = {}
    all_claims = get_all_claims()
    for claim in all_claims:
        topics = claim.get("topics") or ["Other"]
        topic = topics[0] if topics else "Other"
        grouped.setdefault(topic, []).append(claim)
    return dict(sorted(grouped.items(), key=lambda x: x[0].lower()))

def get_disputes_for_user(username):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_DISPUTES_TABLE_NAME or not username:
        return []
    try:
        safe_username = escape_airtable_formula_value(username.strip())
        params = {
            "filterByFormula": f"{{Username}}='{safe_username}'",
            "sort[0][field]": "Last Updated",
            "sort[0][direction]": "desc",
            "maxRecords": 50
        }
        records = airtable_get_all(AIRTABLE_DISPUTES_TABLE_NAME, params=params)
        disputes = []
        for record in records:
            f = record.get("fields", {})
            disputes.append({
                "record_id": record.get("id"),
                "title": f.get("Original Claim Title", "Untitled Claim"),
                "claim_slug": f.get("Claim Slug", ""),
                "sections_disputed": f.get("Sections Disputed", []),
                "dispute_text": f.get("Dispute Text", ""),
                "status": f.get("Status", "Open"),
                "date_submitted": (f.get("Date Submitted", "") or "")[:10],
                "ai_response": f.get("AI Response", ""),
                "editor_resolution": f.get("Editor Resolution", ""),
                "escalated_to_human": f.get("Escalated To Human", False),
            })
        return disputes
    except Exception as e:
        print("GET DISPUTES FOR USER ERROR:", str(e), flush=True)
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

def get_disputes_for_claim(claim_slug):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_DISPUTES_TABLE_NAME or not claim_slug:
        return []

    try:
        safe_slug = escape_airtable_formula_value(claim_slug)
        params = {
            "filterByFormula": f"{{Claim Slug}}='{safe_slug}'"
        }

        records = airtable_get_all(AIRTABLE_DISPUTES_TABLE_NAME, params=params)

        disputes = []
        for record in records:
            f = record.get("fields", {})

            parent_dispute = f.get("Parent Dispute", [])
            if isinstance(parent_dispute, list):
                parent_dispute_id = parent_dispute[0] if parent_dispute else ""
            else:
                parent_dispute_id = parent_dispute or ""

            disputes.append({
                "record_id": record.get("id"),
                "thread_root_id": (f.get("Thread Root ID") or "").strip() or record.get("id"),
                "thread_sequence": int(f.get("Thread Sequence", 0) or 0),
                "entry_label": f.get("Entry Label", f.get("Dispute Type", "Dispute")),
                "entered_by": f.get("Entered By", f.get("Username", "User")),
                "entered_by_type": f.get("Entered By Type", "User"),
                "thread_title": f.get("Thread Title", f.get("Dispute Title", "Dispute Thread")),

                "title": f.get("Original Claim Title", "Untitled Claim"),
                "claim_slug": f.get("Claim Slug", ""),
                "sections_disputed": f.get("Sections Disputed", []),
                "dispute_text": f.get("Dispute Text", ""),
                "response_text": f.get("Response Text", ""),
                "user_source_url": f.get("User Source URL", ""),
                "pushback_round_count": int(f.get("Pushback Round Count", 0) or 0),
                "status": f.get("Status", "Open"),
                "date_submitted": (f.get("Date Submitted", "") or "")[:10],
                "last_updated": f.get("Last Updated", ""),
                "ai_response": f.get("AI Response", ""),
                "human_response": f.get("Editor Resolution", ""),
                "escalated_to_human": f.get("Escalated To Human", False),
                "editor_notes": f.get("Editor Notes", ""),
                "editor_resolution": f.get("Editor Resolution", ""),
                "editor_approved_changes": f.get("Editor Approved Changes", ""),
                "resolution_type": f.get("Resolution Type", ""),
                "dispute_type": f.get("Dispute Type", ""),
                "parent_dispute": parent_dispute_id,
                "update_scope": f.get("Update Scope", []),
                "applied_update_scope": f.get("Applied Update Scope", []),
                "quick_view_outcome": f.get("Quick View Outcome", ""),
                "full_excavation_outcome": f.get("Full Excavation Outcome", "")
            })

        disputes.sort(
            key=lambda d: (
                (d.get("thread_title") or "").lower(),
                d.get("thread_root_id") or "",
                d.get("thread_sequence", 0),
                d.get("date_submitted") or ""
            )
        )

        return disputes

    except Exception as e:
        print("GET DISPUTES FOR CLAIM ERROR:", str(e), flush=True)
        return []

def group_disputes_into_threads(disputes):
    threads_by_root = {}

    for dispute in disputes:
        root_id = dispute.get("thread_root_id") or dispute.get("record_id")
        threads_by_root.setdefault(root_id, []).append(dispute)

    threads = []
    for root_id, entries in threads_by_root.items():
        entries.sort(key=lambda d: d.get("thread_sequence", 0))

        root_entry = entries[0]
        latest_entry = entries[-1]

        threads.append({
            "thread_root_id": root_id,
            "thread_title": root_entry.get("thread_title") or root_entry.get("title") or "Dispute Thread",
            "current_status": latest_entry.get("status", "Open"),
            "current_entry_label": latest_entry.get("entry_label", latest_entry.get("dispute_type", "Dispute")),
            "entries": entries
        })

    threads.sort(
        key=lambda t: (
            t["entries"][-1].get("thread_sequence", 0),
            t["entries"][-1].get("date_submitted") or ""
        ),
        reverse=True
    )

    return threads

def get_disputes_for_claim_record_id(claim_record_id):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_DISPUTES_TABLE_NAME or not claim_record_id:
        return []

    try:
        safe_claim_id = escape_airtable_formula_value(claim_record_id)
        params = {
            "filterByFormula": f"FIND('{safe_claim_id}', ARRAYJOIN({{Claim Record ID}}))",
            "sort[0][field]": "Thread Sequence",
            "sort[0][direction]": "asc"
        }

        records = airtable_get_all(AIRTABLE_DISPUTES_TABLE_NAME, params=params)
        return records

    except Exception as e:
        print("GET DISPUTES FOR CLAIM RECORD ID ERROR:", str(e), flush=True)
        return []

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
            "filterByFormula": "OR(NOT({Breakout User Excavated}), {Breakout User Excavated}!='No')",
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


def get_fresh_user_session_info():
    username = session.get("username")
    user = get_user_by_username(username) if username else None
    if not user:
        return None
    fields = user.get("fields", {})
    return {
        "record_id": user.get("id"),
        "role": fields.get("Role", "standard"),
        "claims_remaining": int(fields.get("Claims Remaining", 0) or 0),
        "active": fields.get("Active", False)
    }


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

def get_claim_by_record_id(record_id):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME or not record_id:
        return None
    try:
        response = requests.get(
            f"{airtable_url(AIRTABLE_TABLE_NAME)}/{record_id}",
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            timeout=20
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print("GET CLAIM BY RECORD ID ERROR:", str(e), flush=True)
        return None


def update_dispute_record(record_id, fields):
    return requests.patch(
        f"{airtable_url(AIRTABLE_DISPUTES_TABLE_NAME)}/{record_id}",
        headers=airtable_headers(),
        json={"fields": fields},
        timeout=30
    )

def get_dispute_by_id(record_id):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_DISPUTES_TABLE_NAME or not record_id:
        return None
    try:
        response = requests.get(
            f"{airtable_url(AIRTABLE_DISPUTES_TABLE_NAME)}/{record_id}",
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            timeout=20
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print("GET DISPUTE BY ID ERROR:", str(e), flush=True)
        return None


DISPUTE_REVIEW_SYSTEM = """You are the dispute review layer for Where the Truth Lies.

Your job is to review a user dispute against an existing claim excavation.

You are not rewriting the claim. You are deciding whether the dispute appears strong enough to:
1. uphold the existing claim as written
2. recommend correction
3. escalate for human review

Return ONLY valid JSON.

Use this exact structure:

{
  "decision": "uphold" or "recommend_correction" or "needs_human_review",
  "ai_response": "A direct 2 to 4 sentence response to the user explaining what the review found.",
  "quick_view_outcome": "No change to Quick View." or "Quick View → Recommend update" or "Quick View should be reviewed by a human editor.",
  "full_excavation_outcome": "No change to Full Excavation." or "Full Excavation → Recommend update" or "Full Excavation should be reviewed by a human editor.",
  "editor_notes": "Short internal note summarizing why the AI reached this conclusion.",
  "escalate": true or false
}

Rules:
Be plainspoken and specific.
Do not claim certainty if the underlying issue is unresolved.
If the user is clearly pointing out a real problem in wording, verdict, or subclaims, prefer recommend_correction.
If the issue is high risk, ambiguous, or could materially alter the claim record, prefer needs_human_review.
If the dispute is weak or unsupported by the current claim context, prefer uphold.
"""

PUSHBACK_REVIEW_SYSTEM = """You are the pushback review layer for Where the Truth Lies.

A user is responding to an earlier AI dispute review. Your job is to reconsider the dispute in light of the user's pushback and the prior AI response.

Return ONLY valid JSON.

Use this exact structure:

{
  "decision": "uphold" or "recommend_correction" or "needs_human_review",
  "ai_response": "A direct 2 to 4 sentence response to the user's pushback.",
  "quick_view_outcome": "No change to Quick View." or "Recommend updating Quick View." or "Quick View should be reviewed by a human editor.",
  "full_excavation_outcome": "No change to Full Excavation." or "Recommend updating Full Excavation." or "Full Excavation should be reviewed by a human editor.",
  "editor_notes": "Short internal note summarizing why the AI reached this conclusion after pushback.",
  "escalate": true or false
}

Rules:
Take the user's pushback seriously.
If the pushback exposes a likely factual or framing problem, prefer recommend_correction.
If the issue remains ambiguous, high risk, or likely requires human judgment, prefer needs_human_review.
If the pushback is weak or repetitive and does not materially change the case, prefer uphold.
"""

def review_dispute_with_ai(claim_context, dispute_payload):
    if not openai_client:
        return None

    try:
        prompt = f"""
Claim title:
{claim_context.get('title', '')}

Stripped claim:
{claim_context.get('stripped_claim', '')}

Overall verdict:
{claim_context.get('overall_verdict', '')}

Quick explanation:
{claim_context.get('quick_explanation', '')}

Subclaims:
{json.dumps(claim_context.get('subclaims', []), ensure_ascii=False)}

Direct facts:
{claim_context.get('direct_facts', '')}

Adjacent facts:
{claim_context.get('adjacent_facts', '')}

Root concern:
{claim_context.get('root_concern', '')}

Values divergence:
{claim_context.get('values_divergence', '')}

Constitutional framework:
{claim_context.get('constitutional_framework', '')}

Common ground:
{claim_context.get('common_ground', '')}

User dispute sections:
{json.dumps(dispute_payload.get('sections_disputed', []), ensure_ascii=False)}

User dispute text:
{dispute_payload.get('dispute_text', '')}

User source URL:
{dispute_payload.get('source_url', '')}
""".strip()

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": DISPUTE_REVIEW_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=900
        )

        parsed = safe_json_parse(response.choices[0].message.content)
        if isinstance(parsed, dict) and parsed.get("decision"):
            return parsed
        return None

    except Exception as e:
        print("DISPUTE REVIEW AI ERROR:", str(e), flush=True)
        return None

def review_pushback_with_ai(claim_context, dispute_record, pushback_text):
    if not openai_client:
        return None

    try:
        fields = dispute_record.get("fields", {})

        prompt = f"""
Claim title:
{claim_context.get('title', '')}

Stripped claim:
{claim_context.get('stripped_claim', '')}

Overall verdict:
{claim_context.get('overall_verdict', '')}

Quick explanation:
{claim_context.get('quick_explanation', '')}

Subclaims:
{json.dumps(claim_context.get('subclaims', []), ensure_ascii=False)}

Direct facts:
{claim_context.get('direct_facts', '')}

Adjacent facts:
{claim_context.get('adjacent_facts', '')}

Root concern:
{claim_context.get('root_concern', '')}

Values divergence:
{claim_context.get('values_divergence', '')}

Constitutional framework:
{claim_context.get('constitutional_framework', '')}

Common ground:
{claim_context.get('common_ground', '')}

Original disputed sections:
{json.dumps(fields.get('Sections Disputed', []), ensure_ascii=False)}

Original dispute text:
{fields.get('Dispute Text', '')}

Prior AI response:
{fields.get('AI Response', '')}

Pushback round count:
{fields.get('Pushback Round Count', 0)}

New user pushback:
{pushback_text}
""".strip()

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PUSHBACK_REVIEW_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=900
        )

        parsed = safe_json_parse(response.choices[0].message.content)
        if isinstance(parsed, dict) and parsed.get("decision"):
            return parsed
        return None

    except Exception as e:
        print("PUSHBACK REVIEW AI ERROR:", str(e), flush=True)
        return None

def get_active_reality_anchors():
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_REALITY_TABLE_NAME:
        return []
    try:
        params = {
            "filterByFormula": "{Status}='Active'",
            "sort[0][field]": "Last Updated",
            "sort[0][direction]": "desc",
            "maxRecords": 100
        }
        return airtable_get_all(AIRTABLE_REALITY_TABLE_NAME, params=params)
    except Exception as e:
        print("REALITY ANCHORS ERROR:", str(e), flush=True)
        return []


def split_phrases(raw_value):
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        return [str(x).strip().lower() for x in raw_value if str(x).strip()]
    return [line.strip().lower() for line in str(raw_value).splitlines() if line.strip()]


def get_reality_anchor_for_claim(claim):
    claim_text = (claim or "").lower()
    anchors = get_active_reality_anchors()
    best_match = None
    best_score = 0

    for record in anchors:
        fields = record.get("fields", {})
        phrases = split_phrases(fields.get("Match Any Phrases", ""))
        if not phrases:
            continue

        matches = sum(1 for phrase in phrases if phrase in claim_text)
        if matches > best_score:
            best_score = matches
            best_match = fields

    if best_match and best_score > 0:
        anchor_text = (best_match.get("Anchor Text") or "").strip()
        source_urls = (best_match.get("Source URLs") or "").strip()
        anchor_key = (best_match.get("Anchor Key") or "").strip()

        if anchor_text:
            source_block = f"\nSupporting sources:\n{source_urls}" if source_urls else ""
            label = f"\nReality Anchor Key: {anchor_key}" if anchor_key else ""
            return f"""REALITY ANCHOR (HIGHEST PRIORITY FACT — DO NOT OVERRIDE):
{anchor_text}{label}{source_block}

You MUST treat the above as confirmed reality.
"""
    return ""


def hardcoded_reality_fallback(claim):
    text = (claim or "").lower()
    if ("charlie kirk" in text or "erika kirk" in text) and any(k in text for k in ["murder", "killed", "assassinated", "death"]):
        return """REALITY ANCHOR (HIGHEST PRIORITY FACT — DO NOT OVERRIDE):
Charlie Kirk was shot and killed at Utah Valley University on September 10, 2025. This event is established and must be treated as real background fact when analyzing related claims. Any analysis should evaluate whether a person said something about the event or was involved in the event, not whether the event itself occurred.
Supporting sources:
FBI update: https://www.fbi.gov/news/press-releases/utah-valley-shooting-updates

You MUST treat the above as confirmed reality.
"""
    return ""


# ── GROK ADJUDICATION LAYER ──

GROK_ADJUDICATION_PROMPT = """You are a live adjudication layer for a political intelligence platform called Where the Truth Lies.

Your job is to determine the current factual status of a claim before deeper analysis begins. You have access to X Search and Web Search. Use them.

You do not write the analysis. You establish what is real, what is attributed, and what is contested so that the analysis engine can operate on grounded facts.

Return ONLY valid JSON. No markdown fences. No preamble.

Search X and the web for the claim. Then return this exact structure:

{
  "requires_live_grounding": true or false,
  "event_status": "established" or "unverified" or "false" or "unclear" or "not_applicable",
  "attribution_status": "verified" or "unverified" or "misattributed" or "no_attribution" or "not_applicable",
  "risk_level": "high" or "medium" or "low",
  "ground_truth_summary": "1-2 sentences. What did you actually find? Be direct. If you found nothing, say so.",
  "confirmed_current_facts": ["A specific fact confirmed by live sources right now.", "Another confirmed fact."],
  "contested_current_facts": ["A specific point that live sources show is disputed or unclear right now.", "Another contested point."],
  "current_narratives": ["The dominant narrative currently circulating around this claim.", "An opposing or alternative narrative currently in circulation."],
  "recent_developments": ["A specific recent development relevant to this claim.", "Another recent development if applicable."],
  "established_facts": ["Fact 1 confirmed by sources.", "Fact 2 confirmed by sources."],
  "contested_points": ["Point 1 that remains unverified or disputed.", "Point 2 still unclear."],
  "recommended_anchor_text": "The text that should be injected as a Reality Anchor into the excavation prompt. Write this as a direct factual briefing. If nothing was found, write: No live grounding found. Proceed with model knowledge and standard uncertainty.",
  "sources_found": ["url1", "url2"]
}

Rules:
If you find clear evidence the event happened, set event_status to established.
If you find clear evidence the event did not happen or the quote is fabricated, set event_status to false.
If you searched and found nothing relevant, set event_status to unclear and say so in ground_truth_summary.
If the claim quotes a specific person, always check whether that quote is verifiable.
Never fabricate sources. Only include URLs you actually retrieved.
recommended_anchor_text is what gets injected into the analysis engine. Make it factual, direct, and useful."""


GROK_TRIGGER_PATTERNS = [
    r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
    r'\b(today|yesterday|just|recently|breaking|last night|this week|this morning|hours ago|minutes ago)\b',
    r'\b(killed|dead|died|murder|shot|arrested|indicted|charged|convicted|resigned|fired|quit|announced|signed|passed|vetoed|invaded|attacked|bombed|launched|declared)\b',
    r'["\u201c\u201d]',
    r'\b(said|says|stated|claimed|posted|tweeted|wrote|announced|told|according to)\b',
    r'\b(viral|trending|went viral|circulating|spreading|posted on|tweet|x post)\b',
]


def should_trigger_grok(claim):
    if not grok_client:
        return False
    for pattern in GROK_TRIGGER_PATTERNS:
        if re.search(pattern, claim or "", re.IGNORECASE):
            return True
    return False


def get_grok_adjudication(claim):
    if not grok_client:
        return None
    try:
        response = grok_client.responses.create(
            model="grok-4-1-fast",
            input=[
                {"role": "system", "content": GROK_ADJUDICATION_PROMPT},
                {"role": "user", "content": f"Adjudicate this claim: \"{claim}\""}
            ],
            tools=[
                {"type": "web_search"},
                {"type": "x_search"}
            ],
            max_output_tokens=1500,
            temperature=0
        )
        raw = ""
        for item in (response.output or []):
            if hasattr(item, "content"):
                for block in (item.content or []):
                    if hasattr(block, "text"):
                        raw += block.text
            elif hasattr(item, "text"):
                raw += item.text
        if not raw and hasattr(response, "output_text"):
            raw = response.output_text or ""
        parsed = safe_json_parse(raw)
        if isinstance(parsed, dict) and "event_status" in parsed:
            return parsed
        return None
    except Exception as e:
        print(f"GROK ADJUDICATION ERROR: {e}", flush=True)
        return None


def format_grok_anchor(adjudication):
    if not adjudication:
        return ""
    anchor_text = (adjudication.get("recommended_anchor_text") or "").strip()
    if not anchor_text or "no live grounding found" in anchor_text.lower():
        return ""
    risk = adjudication.get("risk_level", "low")
    event_status = adjudication.get("event_status", "unclear")
    attribution_status = adjudication.get("attribution_status", "not_applicable")

    # Structured fields (new)
    confirmed = adjudication.get("confirmed_current_facts") or adjudication.get("established_facts") or []
    contested = adjudication.get("contested_current_facts") or adjudication.get("contested_points") or []
    narratives = adjudication.get("current_narratives") or []
    developments = adjudication.get("recent_developments") or []
    sources = adjudication.get("sources_found") or []

    confirmed_block = ""
    if confirmed:
        confirmed_block = "\nCONFIRMED CURRENT FACTS:\n" + "\n".join(f"- {f}" for f in confirmed)

    contested_block = ""
    if contested:
        contested_block = "\nCONTESTED CURRENT FACTS:\n" + "\n".join(f"- {f}" for f in contested)

    narratives_block = ""
    if narratives:
        narratives_block = "\nCURRENT NARRATIVES IN CIRCULATION:\n" + "\n".join(f"- {n}" for n in narratives)

    developments_block = ""
    if developments:
        developments_block = "\nRECENT DEVELOPMENTS:\n" + "\n".join(f"- {d}" for d in developments)

    sources_block = ""
    if sources:
        sources_block = "\nLive sources: " + " | ".join(sources[:5])

    return f"""GROK LIVE ADJUDICATION (HIGHEST PRIORITY — DO NOT OVERRIDE):
Risk Level: {risk.upper()} | Event Status: {event_status} | Attribution: {attribution_status}

{anchor_text}{confirmed_block}{contested_block}{narratives_block}{developments_block}{sources_block}

You MUST treat the above as the live factual record for this claim. Do not contradict it.
"""


def build_reality_anchor_with_grok(claim):
    adjudication = None

    if should_trigger_grok(claim):
        adjudication = get_grok_adjudication(claim)
        if adjudication:
            grok_anchor = format_grok_anchor(adjudication)
            if grok_anchor:
                print(f"GROK ADJUDICATION USED: risk={adjudication.get('risk_level')} event={adjudication.get('event_status')}", flush=True)
                return grok_anchor, adjudication
            else:
                print(f"GROK ADJUDICATION STORED (no anchor): risk={adjudication.get('risk_level')} event={adjudication.get('event_status')}", flush=True)

    manual_anchor = get_reality_anchor_for_claim(claim)
    if manual_anchor:
        return manual_anchor, adjudication

    fallback = hardcoded_reality_fallback(claim)
    return fallback, adjudication


@app.route("/increment-view/<slug>", methods=["POST"])
def increment_view(slug):
    """Increment view count for a claim. Excludes superusers."""
    if not session.get("logged_in"):
        return jsonify({"ok": False}), 200
    if session.get("true_superuser") or session.get("superuser"):
        return jsonify({"ok": False, "reason": "superuser"}), 200
    try:
        record = get_claim_by_slug(slug)
        if not record:
            return jsonify({"ok": False}), 200
        current = int(record.get("fields", {}).get("View Count", 0) or 0)
        update_airtable_record(record["id"], {"View Count": current + 1})
        return jsonify({"ok": True, "view_count": current + 1}), 200
    except Exception as e:
        print(f"VIEW COUNT ERROR: {e}", flush=True)
        return jsonify({"ok": False}), 200


@app.route("/trending", methods=["GET"])
def get_trending():
    """Return top trending claims by view count (non-superuser views only)."""
    try:
        params = {
            "filterByFormula": "AND(OR(NOT({Breakout User Excavated}), {Breakout User Excavated}!='No'), {View Count}>0)",
            "sort[0][field]": "View Count",
            "sort[0][direction]": "desc",
            "maxRecords": 10,
            "fields[]": ["Original Quote", "Stripped Claim", "URL Slug", "Overall Verdict", "View Count", "Date Added"]
        }
        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        trending = []
        for rec in records:
            f = rec.get("fields", {})
            trending.append({
                "title": clean_display_title(f.get("Original Quote") or f.get("Stripped Claim") or "Untitled"),
                "slug": f.get("URL Slug", ""),
                "verdict": f.get("Overall Verdict", "Unproven"),
                "view_count": int(f.get("View Count", 0) or 0)
            })
        return jsonify({"trending": trending}), 200
    except Exception as e:
        return jsonify({"trending": [], "error": str(e)}), 200


@app.route("/health")
def health():
    return "ok", 200


@app.route("/bootcheck")
def bootcheck():
    return jsonify({
        "status": "booted",
        "openai_key_present": bool(OPENAI_API_KEY),
        "anthropic_key_present": bool(ANTHROPIC_API_KEY),
        "xai_key_present": bool(XAI_API_KEY),
        "grok_client_ready": bool(grok_client),
        "airtable_present": bool(AIRTABLE_TOKEN and AIRTABLE_BASE_ID and AIRTABLE_TABLE_NAME),
        "users_table_present": bool(AIRTABLE_USERS_TABLE_NAME),
        "reality_table_present": bool(AIRTABLE_REALITY_TABLE_NAME)
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
        session["true_superuser"] = role == "superuser"
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
        page_mode="claim",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=recent_claims,
        current_claim=current_claim,
        archived_claims_by_topic=get_topic_archives(),
        selected_topic="",
        user_disputes=[],
        search_query="",
        search_results=[]
    )


@app.route("/archives")
def archives():
    if not session.get("logged_in"):
        return redirect("/login")
    topic = (request.args.get("topic") or "").strip()
    archives_by_topic = get_topic_archives()
    if topic:
        filtered = {topic: archives_by_topic.get(topic, [])}
    else:
        filtered = archives_by_topic
    return render_template(
        "index.html",
        page_mode="archives",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=10),
        current_claim=None,
        archived_claims_by_topic=filtered,
        selected_topic=topic,
        user_disputes=[],
        search_query="",
        search_results=[]
    )


@app.route("/claim/<slug>")
def claim_detail(slug):
    if not session.get("logged_in"):
        return redirect("/login")
    record = get_claim_by_slug(slug)
    if not record:
        latest_record = get_latest_claim()
        current_claim = build_claim_context(latest_record) if latest_record else None
    else:
        f = record.get("fields", {})
        if f.get("Breakout User Excavated") == "No":
            latest_record = get_latest_claim()
            current_claim = build_claim_context(latest_record) if latest_record else None
        else:
            current_claim = build_claim_context(record)
    return render_template(
        "index.html",
        page_mode="claim",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=10),
        current_claim=current_claim,
        archived_claims_by_topic={},
        selected_topic="",
        user_disputes=[],
        search_query="",
        search_results=[],
        claims_remaining=session.get("claims_remaining", 0)
    )
def disputes_page():
    if not session.get("logged_in"):
        return redirect("/login")
    username = session.get("username", "")
    user_disputes = get_disputes_for_user(username)
    return render_template(
        "index.html",
        page_mode="disputes",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=10),
        current_claim=None,
        archived_claims_by_topic={},
        selected_topic="",
        user_disputes=user_disputes,
        search_query="",
        search_results=[]
    )

@app.route("/editor")
def editor_page():
    if not session.get("logged_in"):
        return redirect("/login")

    if not session.get("true_superuser"):
        return "Unauthorized", 403

    try:
        params = {
            "filterByFormula": "NOT(OR({Status}='Resolved', {Status}='Closed'))",
            "sort[0][field]": "Last Updated",
            "sort[0][direction]": "desc"
        }

        records = airtable_get_all(AIRTABLE_DISPUTES_TABLE_NAME, params=params)

        editor_queue = []
        for record in records:
            f = record.get("fields", {})

            # Determine queue category — use stored value or infer from record state
            stored_category = (f.get("Editor Queue Category") or "").strip()
            is_escalated = bool(f.get("Escalated To Human", False))
            has_ai_response = bool(f.get("AI Response", ""))
            has_ai_recommended = bool(f.get("AI Recommended Changes", ""))
            pushback_count = int(f.get("Pushback Round Count", 0) or 0)

            if stored_category:
                # Airtable field is set — use it as source of truth
                queue_category = stored_category
            elif has_ai_recommended:
                # AI produced a recommendation — goes to AI Recommended Updates
                # regardless of whether it is also escalated
                queue_category = "AI Recommended Update"
            elif is_escalated:
                # Escalated with no AI recommendation — human needs to review
                queue_category = "Escalated to Human"
            elif pushback_count > 0:
                # Has gone through at least one pushback round — potential issue
                queue_category = "Potential Dispute / Pushback Issue"
            else:
                # Fresh dispute, no response yet
                queue_category = "Current Dispute"

            # Fetch claim snapshot for editor context
            claim_snap_verdict = ""
            claim_snap_qe = ""
            claim_links = f.get("Claim Record ID", [])
            if claim_links:
                try:
                    snap = get_claim_by_record_id(claim_links[0])
                    if snap:
                        snap_fields = snap.get("fields", {})
                        claim_snap_verdict = snap_fields.get("Overall Verdict", "")
                        claim_snap_qe = snap_fields.get("Quick Explanation", "")
                except Exception:
                    pass

            editor_queue.append({
                "record_id": record.get("id"),
                "title": f.get("Original Claim Title", "Untitled"),
                "claim_slug": f.get("Claim Slug", ""),
                "dispute_text": f.get("Dispute Text", ""),
                "ai_response": f.get("AI Response", ""),
                "ai_recommended_changes": f.get("AI Recommended Changes", ""),
                "ai_initial_queue_category": f.get("AI Initial Queue Category", ""),
                "editor_queue_category": queue_category,
                "status": f.get("Status", "Open"),
                "resolution_type": f.get("Resolution Type", ""),
                "sections_disputed": f.get("Sections Disputed", []),
                "date": (f.get("Date Submitted", "") or "")[:10],
                "escalated": is_escalated,
                "claim_verdict": claim_snap_verdict,
                "claim_quick_explanation": claim_snap_qe
        })

        return render_template(
            "index.html",
            page_mode="editor",
            editor_queue=editor_queue,
            superuser=session.get("superuser", False),
            true_superuser=True,
            recent_claims=get_recent_claims(limit=10),
            current_claim=None,
            archived_claims_by_topic={},
            selected_topic="",
            user_disputes=[],
            search_query="",
            search_results=[]
        )

    except Exception as e:
        return f"Editor load error: {str(e)}", 500

@app.route("/editor/resolve/<record_id>", methods=["POST"])
def resolve_editor_item(record_id):
    if not session.get("logged_in"):
        return redirect("/login")

    if not session.get("true_superuser"):
        return "Unauthorized", 403

    try:
        response = update_dispute_record(
            record_id,
            {
                "Status": "Resolved",
                "Escalated To Human": False
            }
        )

        if not response.ok:
            return f"Resolve error: {response.text}", 500

        return redirect("/editor")

    except Exception as e:
        return f"Resolve error: {str(e)}", 500




@app.route("/editor/reanalyze-dispute/<record_id>", methods=["POST"])
def editor_reanalyze_dispute(record_id):
    """Re-run AI on dispute text and write structured recommendation to Airtable."""
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    if not session.get("true_superuser"):
        return jsonify({"error": "Unauthorized"}), 403
    try:
        dispute_record = get_dispute_by_id(record_id)
        if not dispute_record:
            return jsonify({"error": "Dispute not found"}), 404

        dispute_fields = dispute_record.get("fields", {})
        claim_links = dispute_fields.get("Claim Record ID", [])
        claim_id = claim_links[0] if claim_links else None
        if not claim_id:
            return jsonify({"error": "No linked claim found"}), 400

        claim_record = get_claim_by_record_id(claim_id)
        if not claim_record:
            return jsonify({"error": "Claim record not found"}), 404

        claim_context = build_claim_context(claim_record)

        # Use AI to generate a recommendation from the dispute text
        prompt = f"""A user disputed this claim with the following text:

Dispute text: {dispute_fields.get('Dispute Text', '')}

Sections disputed: {json.dumps(dispute_fields.get('Sections Disputed', []))}

Prior AI response: {dispute_fields.get('AI Response', '')}

Current claim context:
- Title: {claim_context.get('title', '') if claim_context else ''}
- Verdict: {claim_context.get('overall_verdict', '') if claim_context else ''}
- Quick Explanation: {claim_context.get('quick_explanation', '') if claim_context else ''}

Based on the dispute, write a clear, specific recommendation for what should be changed in the claim analysis. Be precise about which sections and what the change should be. 2-4 sentences."""

        recommendation = ""
        if anthropic_client:
            resp = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system="You are an editorial assistant for a political intelligence platform. Write clear, specific recommendations for claim updates based on user disputes. No preamble. Just the recommendation.",
                messages=[{"role": "user", "content": prompt}]
            )
            recommendation = resp.content[0].text.strip()
        elif openai_client:
            resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an editorial assistant for a political intelligence platform. Write clear, specific recommendations for claim updates based on user disputes. No preamble. Just the recommendation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            recommendation = resp.choices[0].message.content.strip()

        if not recommendation:
            return jsonify({"error": "AI could not generate a recommendation"}), 500

        # Write recommendation to Airtable
        update_resp = update_dispute_record(record_id, {
            "AI Recommended Changes": recommendation,
            "Last Updated": datetime.utcnow().isoformat()
        })
        if not update_resp.ok:
            return jsonify({"error": f"Failed to save recommendation: {update_resp.text}"}), 500

        return jsonify({"ok": True, "recommendation": recommendation})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/editor/update-recommendation/<record_id>", methods=["POST"])
def editor_update_recommendation(record_id):
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    if not session.get("true_superuser"):
        return jsonify({"error": "Unauthorized"}), 403
    try:
        data = request.get_json() or {}
        recommendation = (data.get("recommendation") or "").strip()
        if not recommendation:
            return jsonify({"error": "Recommendation text is required"}), 400
        response = update_dispute_record(record_id, {
            "AI Recommended Changes": recommendation,
            "Last Updated": datetime.utcnow().isoformat()
        })
        if not response.ok:
            return jsonify({"error": response.text}), 500
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/editor/update-status/<record_id>", methods=["POST"])
def editor_update_status(record_id):
    if not session.get("logged_in"):
        return {"error": "Not logged in"}, 401
    if not session.get("true_superuser"):
        return {"error": "Unauthorized"}, 403
    try:
        data = request.get_json()
        new_status = (data or {}).get("status", "").strip()
        valid_statuses = ["Open", "AI Responded", "Escalated to Human", "In Review", "Resolved", "Closed", "Needs Review"]
        if new_status not in valid_statuses:
            return {"error": f"Invalid status: {new_status}"}, 400
        fields = {"Status": new_status}
        if new_status == "Escalated to Human":
            fields["Escalated To Human"] = True
        elif new_status in ["Resolved", "Closed"]:
            fields["Escalated To Human"] = False
        response = update_dispute_record(record_id, fields)
        if not response.ok:
            return {"error": response.text}, 500
        return {"ok": True, "status": new_status}
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/editor/update-queue-category/<record_id>", methods=["POST"])
def editor_update_queue_category(record_id):
    if not session.get("logged_in"):
        return {"error": "Not logged in"}, 401
    if not session.get("true_superuser"):
        return {"error": "Unauthorized"}, 403
    try:
        data = request.get_json()
        new_category = (data or {}).get("category", "").strip()
        valid_categories = ["Escalated to Human", "AI Recommended Update", "Potential Dispute / Pushback Issue", "Current Dispute"]
        if new_category not in valid_categories:
            return {"error": f"Invalid category: {new_category}"}, 400
        response = update_dispute_record(record_id, {"Editor Queue Category": new_category})
        if not response.ok:
            return {"error": response.text}, 500
        return {"ok": True, "category": new_category}
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/editor/update-resolution-type/<record_id>", methods=["POST"])
def editor_update_resolution_type(record_id):
    if not session.get("logged_in"):
        return {"error": "Not logged in"}, 401
    if not session.get("true_superuser"):
        return {"error": "Unauthorized"}, 403
    try:
        data = request.get_json()
        new_type = (data or {}).get("resolution_type", "").strip()
        valid_types = ["AI Recommendation", "Human Response", "Human Override", "Partial Update", "No Update", "Escalated"]
        if new_type not in valid_types:
            return {"error": f"Invalid resolution type: {new_type}"}, 400
        response = update_dispute_record(record_id, {"Resolution Type": new_type})
        if not response.ok:
            return {"error": response.text}, 500
        return {"ok": True, "resolution_type": new_type}
    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/search")
def search_page():
    if not session.get("logged_in"):
        return redirect("/login")
    query = (request.args.get("q") or "").strip()
    results = []
    if query:
        all_claims = get_all_claims()
        query_lower = query.lower()
        results = [c for c in all_claims if query_lower in (c.get("title") or "").lower() or query_lower in (c.get("stripped_claim") or "").lower() or query_lower in (c.get("speaker") or "").lower()]
    return render_template(
        "index.html",
        page_mode="search",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=10),
        current_claim=None,
        archived_claims_by_topic={},
        selected_topic="",
        search_query=query,
        search_results=results,
        user_disputes=[]
    )


@app.route("/profile")
def profile_page():
    if not session.get("logged_in"):
        return redirect("/login")
    return render_template(
        "index.html",
        page_mode="profile",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=5),
        current_claim=None,
        archived_claims_by_topic={},
        selected_topic="",
        user_disputes=[],
        search_query="",
        search_results=[]
    )


def normalize_claim_text(text):
    """Normalize claim for comparison: lowercase, strip punctuation, collapse whitespace."""
    import re as _re
    text = text.lower().strip()
    text = _re.sub(r'[^\w\s]', ' ', text)
    text = _re.sub(r'\s+', ' ', text).strip()
    return text


def keyword_overlap_score(norm_a, norm_b):
    """Jaccard overlap of meaningful keywords. Returns 0.0-1.0."""
    STOP = {'the','a','an','is','are','was','were','be','been','have','has','had',
            'do','does','did','will','would','could','should','may','might','can',
            'to','of','in','for','on','with','at','by','from','as','and','but',
            'or','not','no','it','its','this','that','they','we','you','he','she',
            'his','her','their','our','said','says','also','just','than','then',
            'when','who','what','which','how','if','so','up','out','into'}
    def kw(t):
        return {w for w in t.split() if len(w) > 3 and w not in STOP}
    a, b = kw(norm_a), kw(norm_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_duplicate_and_similar_claims(claim_text, threshold=0.30, max_similar=4):
    """Check for exact and similar claims. Returns dict with exact and similar."""
    result = {'exact': None, 'similar': []}
    if not claim_text or not AIRTABLE_TOKEN:
        return result
    norm_input = normalize_claim_text(claim_text)
    try:
        params = {
            'filterByFormula': "OR(NOT({Breakout User Excavated}), {Breakout User Excavated}!='No')",
            'fields[]': ['Original Quote', 'Stripped Claim', 'URL Slug'],
            'maxRecords': 200
        }
        response = requests.get(
            airtable_url(AIRTABLE_TABLE_NAME),
            headers={'Authorization': f'Bearer {AIRTABLE_TOKEN}'},
            params=params, timeout=20
        )
        response.raise_for_status()
        records = response.json().get('records', [])
    except Exception as e:
        print(f'DUPLICATE CHECK ERROR: {e}', flush=True)
        return result
    scored = []
    for rec in records:
        f = rec.get('fields', {})
        raw = f.get('Original Quote') or f.get('Stripped Claim') or ''
        if not raw:
            continue
        norm_existing = normalize_claim_text(raw)
        if norm_existing == norm_input:
            result['exact'] = {
                'record_id': rec.get('id'),
                'title': clean_display_title(raw),
                'slug': f.get('URL Slug', '')
            }
            return result
        score = keyword_overlap_score(norm_input, norm_existing)
        if score >= threshold:
            scored.append({
                'record_id': rec.get('id'),
                'title': clean_display_title(raw),
                'slug': f.get('URL Slug', ''),
                'score': round(score, 3)
            })
    scored.sort(key=lambda x: x['score'], reverse=True)
    result['similar'] = scored[:max_similar]
    return result


@app.route('/check-duplicate', methods=['POST'])
def check_duplicate():
    """Pre-creation duplicate/similar check — runs before /analyze."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json() or {}
    claim = (data.get('claim') or '').strip()
    if not claim:
        return jsonify({'exact': None, 'similar': []}), 200
    result = find_duplicate_and_similar_claims(claim)
    return jsonify(result), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    user_info = get_fresh_user_session_info()
    if not user_info or not user_info.get("active"):
        session.clear()
        return jsonify({"error": "Account unavailable or inactive."}), 403

    role = user_info["role"]
    claims_remaining = user_info["claims_remaining"]

    session["role"] = role
    session["superuser"] = role in ["superuser", "limited_superuser"]
    session["true_superuser"] = role == "superuser"
    session["claims_remaining"] = claims_remaining

    if role == "standard":
        return jsonify({"error": "Your account can browse existing claim files, but cannot run new excavations."}), 403

    if role == "limited_superuser" and claims_remaining <= 0:
        return jsonify({"error": "Claim limit reached. You can still browse existing claim files, but new excavations are blocked."}), 403

    data = request.get_json() or {}
    claim = (data.get("claim") or "").strip()
    mode = data.get("mode", "strip")

    if not claim:
        return jsonify({"error": "Claim is required"}), 400

    reality_anchor, grok_adjudication = build_reality_anchor_with_grok(claim)
    prompt_text = f"""
{reality_anchor}

Now analyze this claim:
"{claim}"
""".strip()

    claude_json = {}
    openai_json = {}

    try:
        if anthropic_client:
            claude_response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                temperature=0,
                system=CLAIMLAB_SYSTEM,
                messages=[{"role": "user", "content": prompt_text}]
            )
            claude_json = safe_json_parse(claude_response.content[0].text)
        else:
            claude_json = {"error": "Anthropic not configured"}
    except Exception as e:
        claude_json = {"error": str(e)}

    # ── OpenAI Challenge Layer ──
    # OpenAI reviews Claude's output + Grok context to find where Claude may be wrong
    openai_challenge = {}
    try:
        if openai_client and "error" not in claude_json:
            grok_context_block = ""
            if grok_adjudication and isinstance(grok_adjudication, dict):
                confirmed = grok_adjudication.get("confirmed_current_facts") or grok_adjudication.get("established_facts") or []
                contested = grok_adjudication.get("contested_current_facts") or grok_adjudication.get("contested_points") or []
                narratives = grok_adjudication.get("current_narratives") or []
                developments = grok_adjudication.get("recent_developments") or []
                if confirmed or contested or narratives or developments:
                    grok_context_block = "LIVE GROK CONTEXT:\n"
                    if confirmed: grok_context_block += "Confirmed: " + "; ".join(confirmed[:3]) + "\n"
                    if contested: grok_context_block += "Contested: " + "; ".join(contested[:3]) + "\n"
                    if narratives: grok_context_block += "Narratives: " + "; ".join(narratives[:2]) + "\n"
                    if developments: grok_context_block += "Recent: " + "; ".join(developments[:2]) + "\n"

            challenge_prompt = f"""You are a critical review layer for a political intelligence platform.

Claude has produced the following analysis of this claim:
CLAIM: "{claim}"

CLAUDE'S VERDICT: {claude_json.get("Overall Verdict", "Unknown")}
CLAUDE'S ONE-LINE READ: {claude_json.get("Quick Explanation", "")[:500]}
CLAUDE'S DIRECT FACTS: {claude_json.get("Direct Facts", "")[:500]}
CLAUDE'S COMMON GROUND: {claude_json.get("Common Ground", "")[:300]}

{grok_context_block}
Your job is to challenge Claude's analysis. Answer these four questions in JSON only. No markdown fences. No preamble.

Return exactly this structure:
{{
  "where_claude_is_most_likely_wrong": "One sentence identifying the single most likely error or overconfidence in Claude's analysis.",
  "what_claude_overstated": "One sentence on what Claude leaned too hard on.",
  "what_claude_missed": "One sentence on a meaningful angle, fact, or interpretation Claude did not address.",
  "strongest_alternative_interpretation": "One sentence describing the most credible alternative reading of this claim.",
  "openai_verdict": "Exactly one of: True, Mostly True, Substantially True, Plausible/Mixed, Contested, Exaggerated, Misleading, Unproven, False",
  "divergence_note": "If your verdict differs from Claude's, explain the core disagreement in one sentence. If aligned, write: Aligned with Claude's assessment."
}}"""

            challenge_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": challenge_prompt}],
                max_tokens=800,
                temperature=0
            )
            openai_challenge = safe_json_parse(challenge_response.choices[0].message.content)
            openai_json = openai_challenge  # store challenge as openai_json for compatibility
        elif not openai_client:
            openai_json = {"error": "OpenAI not configured"}
            openai_challenge = {}
        else:
            # Claude failed — run OpenAI as primary instead
            try:
                openai_response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": CLAIMLAB_SYSTEM},
                        {"role": "user", "content": prompt_text}
                    ],
                    max_tokens=4000,
                    temperature=0
                )
                openai_json = safe_json_parse(openai_response.choices[0].message.content)
                openai_challenge = {}
            except Exception as e2:
                openai_json = {"error": str(e2)}
                openai_challenge = {}
    except Exception as e:
        openai_json = {"error": str(e)}
        openai_challenge = {}

    # ── Reconciliation Layer ──
    claude_v = (claude_json.get("Overall Verdict") or "").strip()
    openai_v = (openai_challenge.get("openai_verdict") or "").strip()
    models_diverged_now = bool(claude_v and openai_v and claude_v.lower() != openai_v.lower())
    divergence_note = openai_challenge.get("divergence_note", "")
    where_wrong = openai_challenge.get("where_claude_is_most_likely_wrong", "")

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
                username=session.get("username", "Unknown"),
                existing_fields=existing_fields
            )

            fields["Claude Raw JSON"] = json.dumps(claude_json, ensure_ascii=False)[:100000]
            fields["OpenAI Raw JSON"] = json.dumps(openai_json, ensure_ascii=False)[:100000]
            if openai_challenge:
                fields["OpenAI Challenge JSON"] = json.dumps(openai_challenge, ensure_ascii=False)[:50000]
            if models_diverged_now and divergence_note:
                fields["Model Divergence Note"] = divergence_note[:2000]
                fields["Models Diverged"] = True
            else:
                fields["Models Diverged"] = False
            fields["Grok Raw JSON"] = json.dumps(grok_adjudication, ensure_ascii=False)[:100000] if grok_adjudication is not None else ""

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
                    airtable_result = {"saved": False, "status_code": response.status_code, "error": airtable_error}
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
                    airtable_result = {"saved": False, "status_code": response.status_code, "error": airtable_error}

            if airtable_result.get("saved") and role == "limited_superuser":
                new_count = max(0, claims_remaining - 1)
                update_resp = update_user_claims_remaining(user_info["record_id"], new_count)
                if update_resp.status_code == 200:
                    session["claims_remaining"] = new_count

            # Auto-run breakout detection on successful save
            if airtable_result.get("saved"):
                try:
                    saved_record_id = airtable_result.get("record_id")
                    if saved_record_id:
                        saved_record = get_claim_by_record_id(saved_record_id)
                        if saved_record:
                            bo_count = run_breakout_detection_for_claim(saved_record)
                            airtable_result["breakouts_detected"] = bo_count
                            print(f"AUTO BREAKOUT DETECTION: {bo_count} breakouts on {fields.get('URL Slug', '')}", flush=True)
                except Exception as bo_err:
                    print(f"AUTO BREAKOUT DETECTION ERROR: {bo_err}", flush=True)

    except Exception as e:
        airtable_result = {"saved": False, "error": str(e)}

    return jsonify({
        "claude": claude_json,
        "openai": openai_json,
        "openai_challenge": openai_challenge,
        "models_diverged": models_diverged_now,
        "divergence_note": divergence_note,
        "airtable": airtable_result,
        "superuser": session.get("superuser", False),
        "claims_remaining": session.get("claims_remaining"),
        "reality_anchor_used": bool(reality_anchor),
        "grok_adjudication": grok_adjudication
    })

@app.route("/submit_dispute", methods=["POST"])
def submit_dispute():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json() or {}

        claim_id = data.get("claim_id")
        claim_slug = data.get("claim_slug")
        claim_title = data.get("claim_title")
        dispute_text = (data.get("dispute_text") or "").strip()
        sections = data.get("sections_disputed")
        source_url = (data.get("source_url") or "").strip()

        if not claim_id:
            return jsonify({"error": "Claim record ID is required"}), 400

        if not claim_slug:
            return jsonify({"error": "Claim slug is required"}), 400

        if not claim_title:
            return jsonify({"error": "Claim title is required"}), 400

        if not dispute_text:
            return jsonify({"error": "Dispute text is required"}), 400

        if not sections:
            return jsonify({"error": "At least one disputed section is required"}), 400

        if not isinstance(sections, list):
            return jsonify({"error": "Sections disputed must be a list"}), 400

        username = session.get("username", "Unknown")
        role = session.get("role", "standard")

        payload = {
            "fields": {
                "Claim Record ID": [claim_id],
                "Claim Slug": claim_slug,
                "Original Claim Title": claim_title,
                "Username": username,
                "User Role at Submission": role,
                "Sections Disputed": sections,
                "Dispute Text": dispute_text,
                "User Source URL": source_url,
                "Dispute Type": "Initial Dispute",
                "Pushback Round Count": 0,
                "Status": "Open",
                "Date Submitted": datetime.utcnow().isoformat(),
                "Last Updated": datetime.utcnow().isoformat(),
                "Escalated To Human": False
            }
        }

        create_res = requests.post(
            f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_DISPUTES_TABLE_NAME}",
            json=payload,
            headers=airtable_headers(),
            timeout=30
        )

        if not create_res.ok:
            return jsonify({"error": create_res.text}), 500

        dispute_record = create_res.json()
        dispute_record_id = dispute_record.get("id")
        created_fields = dispute_record.get("fields", {})

        thread_title = (
            (created_fields.get("Stripped Claim") or "").strip()[:140]
            or (claim_title[:140] if claim_title else "Dispute Thread")
        )

        thread_init_fields = {
            "Thread Root ID": dispute_record_id,
            "Thread Sequence": 1,
            "Entry Label": "Initial Dispute",
            "Entered By": session.get("username", "Unknown"),
            "Entered By Type": "User",
            "Response Text": dispute_text,
            "Thread Title": thread_title
        }

        thread_init_res = update_dispute_record(dispute_record_id, thread_init_fields)
        if not thread_init_res.ok:
            return jsonify({"error": thread_init_res.text}), 500

        ai_review = None

        claim_record = get_claim_by_record_id(claim_id)
        if claim_record:
            claim_context = build_claim_context(claim_record)

            ai_review = review_dispute_with_ai(
                claim_context,
                {
                    "sections_disputed": sections,
                    "dispute_text": dispute_text,
                    "source_url": source_url
                }
            )

            print("AI REVIEW RESULT:", ai_review, flush=True)

            if ai_review and dispute_record_id:
                escalate = bool(ai_review.get("escalate", False))

                update_fields = {
                    "AI Response": ai_review.get("ai_response", ""),
                    "Editor Notes": ai_review.get("editor_notes", ""),
                    "Last Updated": datetime.utcnow().isoformat(),
                    "Escalated To Human": escalate,
                    "Status": "AI Responded"
                }

                if "quick_view_outcome" in ai_review:
                    update_fields["Quick View Outcome"] = ai_review.get("quick_view_outcome", "")

                if "full_excavation_outcome" in ai_review:
                    update_fields["Full Excavation Outcome"] = ai_review.get("full_excavation_outcome", "")

                update_res = update_dispute_record(dispute_record_id, update_fields)
                print("DISPUTE UPDATE STATUS:", update_res.status_code, flush=True)
                print("DISPUTE UPDATE BODY:", update_res.text, flush=True)

                if not update_res.ok:
                    print("DISPUTE AI UPDATE ERROR:", update_res.text, flush=True)
                else:
                    print("DISPUTE AI UPDATE SUCCESS", flush=True)

        # Run breakout detection on dispute text (fire-and-forget style — don't block response)
        try:
            if claim_record:
                combined = dispute_text.strip()
                if combined:
                    breakout_candidates = detect_breakout_claims(combined, source_type="Dispute")
                    claim_fields_snap = claim_record.get("fields", {})
                    parent_id = claim_record.get("id")
                    parent_identifier = claim_fields_snap.get("Claim Identifier", "")
                    root_links = claim_fields_snap.get("Root Claim", [])
                    root_id = root_links[0] if root_links else parent_id
                    dispute_had_breakouts = False
                    for b in breakout_candidates:
                        result = create_breakout_claim_record(
                            breakout_data=b,
                            parent_record_id=parent_id,
                            root_record_id=root_id,
                            origin_type="Breakout Dispute",
                            parent_identifier=parent_identifier,
                            source_section="Dispute",
                            origin_dispute_id=dispute_record_id,
                            detection_source="Dispute"
                        )
                        if result:
                            dispute_had_breakouts = True
                    if dispute_had_breakouts and dispute_record_id:
                        update_dispute_record(dispute_record_id, {"Has Breakout Claims": True})
        except Exception as bo_err:
            print(f"DISPUTE BREAKOUT DETECTION ERROR: {bo_err}", flush=True)

        return jsonify({
            "success": True,
            "dispute_id": dispute_record_id,
            "ai_review": ai_review
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/pushback_dispute", methods=["POST"])
def pushback_dispute():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json() or {}
        dispute_id = data.get("dispute_id")
        pushback_text = (data.get("pushback_text") or "").strip()

        if not dispute_id:
            return jsonify({"error": "Dispute ID is required"}), 400

        if not pushback_text:
            return jsonify({"error": "Pushback text is required"}), 400

        dispute_record = get_dispute_by_id(dispute_id)
        if not dispute_record:
            return jsonify({"error": "Dispute not found"}), 404

        fields = dispute_record.get("fields", {})

        root_dispute_id = (fields.get("Thread Root ID") or "").strip() or dispute_id
        role = (fields.get("User Role at Submission") or "standard").lower()
        max_allowed = MAX_PUSHBACKS.get(role, 1)

        claim_links = fields.get("Claim Record ID", [])
        claim_id = claim_links[0] if claim_links else None
        if not claim_id:
            return jsonify({"error": "Linked claim record missing."}), 400

        claim_record = get_claim_by_record_id(claim_id)
        if not claim_record:
            return jsonify({"error": "Claim record not found."}), 404

        claim_context = build_claim_context(claim_record)

        # Get all disputes for this claim
        all_disputes = get_disputes_for_claim_record_id(claim_id)

        thread_entries = []
        for record in all_disputes:
            record_fields = record.get("fields", {})
            record_root_id = (record_fields.get("Thread Root ID") or "").strip()

            if record_root_id == root_dispute_id or record.get("id") == root_dispute_id:
                thread_entries.append(record)

        current_round = sum(
            1
            for record in thread_entries
            if (record.get("fields", {}).get("Dispute Type") or "").strip().lower() == "pushback"
        )

        if current_round >= max_allowed:
            return jsonify({"error": "Pushback limit reached for this dispute."}), 403

        max_sequence = max(
            int(record.get("fields", {}).get("Thread Sequence", 0) or 0)
            for record in thread_entries
        ) if thread_entries else 1

        new_round = current_round + 1
        new_sequence = max_sequence + 1

        ai_review = review_pushback_with_ai(claim_context, dispute_record, pushback_text)

        if not ai_review:
            return jsonify({"error": "AI pushback review failed."}), 500

        escalate = bool(ai_review.get("escalate", False))

        root_record = None
        for record in thread_entries:
            if record.get("id") == root_dispute_id:
                root_record = record
                break

        root_fields = root_record.get("fields", {}) if root_record else fields

        payload_fields = {
            "Thread Title": root_fields.get("Thread Title") or (fields.get("Thread Title") or "Dispute Thread"),
            "Thread Root ID": root_dispute_id,
            "Thread Sequence": new_sequence,
            "Entry Label": f"Pushback {new_round}",
            "Entered By": session.get("username", "Unknown"),
            "Entered By Type": "User",
            "Response Text": pushback_text,
            "Claim Record ID": [claim_id],
            "Claim Slug": fields.get("Claim Slug", ""),
            "Original Claim Title": fields.get("Original Claim Title", ""),
            "Username": session.get("username", "Unknown"),
            "User Role at Submission": role,
            "Sections Disputed": fields.get("Sections Disputed", []),
            "Dispute Text": pushback_text,
            "User Source URL": "",
            "Parent Dispute": [root_dispute_id],
            "Dispute Type": "Pushback",
            "Pushback Round Count": new_round,
            "Status": "Escalated to Human" if escalate else "AI Responded",
            "Date Submitted": datetime.utcnow().isoformat(),
            "Last Updated": datetime.utcnow().isoformat(),
            "AI Response": ai_review.get("ai_response", ""),
            "Escalated To Human": escalate,
            "Editor Notes": ai_review.get("editor_notes", ""),
            "Quick View Outcome": ai_review.get("quick_view_outcome", ""),
            "Full Excavation Outcome": ai_review.get("full_excavation_outcome", ""),
            "AI Recommended Changes": ""
        }

        payload = {"fields": payload_fields}

        create_res = requests.post(
            f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_DISPUTES_TABLE_NAME}",
            json=payload,
            headers=airtable_headers(),
            timeout=30
        )

        if not create_res.ok:
            return jsonify({"error": create_res.text}), 500

        new_pushback_record = create_res.json()

        # Run breakout detection on pushback text
        try:
            pushback_combined = pushback_text.strip()
            if pushback_combined and claim_record:
                breakout_candidates = detect_breakout_claims(pushback_combined, source_type="Pushback")
                claim_fields_snap = claim_record.get("fields", {})
                parent_id = claim_record.get("id")
                parent_identifier = claim_fields_snap.get("Claim Identifier", "")
                root_links = claim_fields_snap.get("Root Claim", [])
                root_id = root_links[0] if root_links else parent_id
                pb_record_id = new_pushback_record.get("id")
                pushback_had_breakouts = False
                for b in breakout_candidates:
                    result = create_breakout_claim_record(
                        breakout_data=b,
                        parent_record_id=parent_id,
                        root_record_id=root_id,
                        origin_type="Breakout Pushback",
                        parent_identifier=parent_identifier,
                        source_section="Pushback",
                        origin_pushback_id=pb_record_id,
                        detection_source="Pushback"
                    )
                    if result:
                        pushback_had_breakouts = True
                if pushback_had_breakouts and pb_record_id:
                    update_dispute_record(pb_record_id, {"Has Breakout Claims": True})
        except Exception as bo_err:
            print(f"PUSHBACK BREAKOUT DETECTION ERROR: {bo_err}", flush=True)

        return jsonify({
            "success": True,
            "dispute_id": new_pushback_record.get("id"),
            "thread_root_id": root_dispute_id,
            "ai_review": ai_review,
            "pushback_round_count": new_round,
            "max_pushbacks": max_allowed,
            "escalated_to_human": escalate
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ══════════════════════════════════════════════
# EDITOR ADJUDICATION SYSTEM
# ══════════════════════════════════════════════

EDITOR_UPDATE_SYSTEM = """You are the claim update engine for Where the Truth Lies.

An editor has reviewed a user dispute and approved applying AI-recommended changes to a published claim.

Your job is to rewrite ONLY the specific sections that need updating based on the recommendation text provided.

Return ONLY valid JSON in this exact structure:

{
  "sections": [
    {
      "field": "Quick Explanation",
      "airtable_key": "Quick Explanation",
      "current": "The current text of this field",
      "proposed": "Your rewritten version of this field",
      "changed": true
    }
  ],
  "editor_note": "A 1-2 sentence summary of what was changed and why."
}

Rules:
- Only include sections that actually need to change based on the recommendation
- Keep changes minimal and precise. Do not rewrite sections that are fine
- Maintain the platform's tone: plain, direct, no political bias
- The Overall Verdict must be exactly one of: True, Mostly True, Substantially True, Plausible/Mixed, Contested, Exaggerated, Misleading, Unproven, False
- Quick Explanation must be 1-2 sentences maximum
- Do not add new information not supported by the recommendation or existing claim context
- If a section does not need to change, do not include it
"""

AIRTABLE_KEY_MAP = {
    "Quick Explanation": "Quick Explanation",
    "Overall Verdict": "Overall Verdict",
    "Direct Facts": "Direct Facts",
    "Adjacent Facts": "Adjacent Facts",
    "Root Concern": "Root Concern",
    "Values Divergence": "Values Divergence",
    "Constitutional Framework": "Constitutional Framework",
    "Common Ground": "Common Ground",
    "Bottom Line": "Strip Mode Summary",
    "Strip Mode Summary": "Strip Mode Summary",
}


def generate_update_preview(claim_record, dispute_record):
    """Call AI to generate structured before/after section rewrites."""
    if not anthropic_client and not openai_client:
        return None

    claim_fields = claim_record.get("fields", {})
    dispute_fields = dispute_record.get("fields", {})

    recommendation = (dispute_fields.get("AI Recommended Changes") or "").strip()
    sections_disputed = dispute_fields.get("Sections Disputed", [])

    if not recommendation:
        return None

    current_sections = {
        "Quick Explanation": claim_fields.get("Quick Explanation", ""),
        "Overall Verdict": claim_fields.get("Overall Verdict", ""),
        "Direct Facts": claim_fields.get("Direct Facts", ""),
        "Adjacent Facts": claim_fields.get("Adjacent Facts", ""),
        "Root Concern": claim_fields.get("Root Concern", ""),
        "Values Divergence": claim_fields.get("Values Divergence", ""),
        "Constitutional Framework": claim_fields.get("Constitutional Framework", ""),
        "Common Ground": claim_fields.get("Common Ground", ""),
        "Bottom Line": claim_fields.get("Strip Mode Summary", ""),
    }

    prompt = f"""Claim title: {claim_fields.get('Original Quote') or claim_fields.get('Stripped Claim', '')}

Sections disputed by user: {json.dumps(sections_disputed)}

Original dispute text: {dispute_fields.get('Dispute Text', '')}

AI recommendation for changes:
{recommendation}

Current claim section values:
{json.dumps(current_sections, ensure_ascii=False, indent=2)}

Based on the recommendation, return the structured update JSON."""

    try:
        if anthropic_client:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=EDITOR_UPDATE_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text
        else:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": EDITOR_UPDATE_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1500
            )
            raw = response.choices[0].message.content

        parsed = safe_json_parse(raw)
        if isinstance(parsed, dict) and parsed.get("sections"):
            return parsed
        return None

    except Exception as e:
        print("EDITOR UPDATE AI ERROR:", str(e), flush=True)
        return None


@app.route("/editor/preview-update/<record_id>", methods=["GET"])
def editor_preview_update(record_id):
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    if not session.get("true_superuser"):
        return jsonify({"error": "Unauthorized"}), 403

    try:
        dispute_record = get_dispute_by_id(record_id)
        if not dispute_record:
            return jsonify({"error": "Dispute not found"}), 404

        dispute_fields = dispute_record.get("fields", {})
        claim_links = dispute_fields.get("Claim Record ID", [])
        claim_id = claim_links[0] if claim_links else None

        if not claim_id:
            return jsonify({"error": "No linked claim found"}), 400

        claim_record = get_claim_by_record_id(claim_id)
        if not claim_record:
            return jsonify({"error": "Claim record not found"}), 404

        preview = generate_update_preview(claim_record, dispute_record)
        if not preview:
            return jsonify({"error": "Could not generate update preview. Check AI Recommended Changes field."}), 500

        return jsonify({
            "ok": True,
            "claim_title": dispute_fields.get("Original Claim Title", ""),
            "claim_slug": dispute_fields.get("Claim Slug", ""),
            "claim_record_id": claim_id,
            "sections": preview.get("sections", []),
            "editor_note": preview.get("editor_note", "")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/editor/apply-update/<record_id>", methods=["POST"])
def editor_apply_update(record_id):
    """Apply approved section rewrites to the claim and log to dispute thread."""
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    if not session.get("true_superuser"):
        return jsonify({"error": "Unauthorized"}), 403

    try:
        data = request.get_json() or {}
        approved_sections = data.get("approved_sections", [])
        editor_note = (data.get("editor_note") or "").strip()
        claim_record_id = (data.get("claim_record_id") or "").strip()

        if not approved_sections:
            return jsonify({"error": "No sections approved"}), 400
        if not claim_record_id:
            return jsonify({"error": "No claim record ID provided"}), 400

        # Build fields to write to claims table
        claim_updates = {}
        snapshot_before = {}
        snapshot_after = {}

        claim_record = get_claim_by_record_id(claim_record_id)
        claim_fields = claim_record.get("fields", {}) if claim_record else {}

        for section in approved_sections:
            field_name = section.get("field", "")
            airtable_key = AIRTABLE_KEY_MAP.get(field_name, field_name)
            proposed = section.get("proposed", "")
            current = section.get("current", "")

            if airtable_key and proposed:
                claim_updates[airtable_key] = proposed
                snapshot_before[field_name] = current
                snapshot_after[field_name] = proposed

        if not claim_updates:
            return jsonify({"error": "No valid fields to update"}), 400

        # Write to claims table
        update_url = f"{airtable_url(AIRTABLE_TABLE_NAME)}/{claim_record_id}"
        update_resp = requests.patch(
            update_url,
            headers=airtable_headers(),
            json={"fields": claim_updates},
            timeout=30
        )

        if not update_resp.ok:
            return jsonify({"error": f"Claim update failed: {update_resp.text}"}), 500

        # Write to correct Airtable field types
        applied_scope = list(snapshot_after.keys())
        editor_username = session.get("username", "Editor")

        # Editor Approved Changes — JSON snapshot (long text field)
        approved_changes_json = json.dumps({
            "editor": editor_username,
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "editor_note": editor_note or "",
            "fields": {
                field_name: {
                    "before": snapshot_before.get(field_name, ""),
                    "after": snapshot_after[field_name]
                }
                for field_name in snapshot_after
            }
        }, ensure_ascii=False)

        update_dispute_record(record_id, {
            "Status": "Resolved",
            "Escalated To Human": False,
            "Resolution Type": "AI Recommendation",
            "Editor Resolution": "Original Claim Modified",
            "Applied Update Scope": applied_scope,
            "Editor Approved Changes": approved_changes_json,
            "Last Updated": datetime.utcnow().isoformat()
        })

        return jsonify({
            "ok": True,
            "fields_updated": list(claim_updates.keys()),
            "claim_slug": claim_fields.get("URL Slug", "")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ══════════════════════════════════════════════
# CLAIM MANAGEMENT — REANALYSIS SYSTEM
# ══════════════════════════════════════════════

SELECTIVE_FIELD_MAP = {
    "Quick Explanation": "Quick Explanation",
    "Overall Verdict": "Overall Verdict",
    "Direct Facts": "Direct Facts",
    "Adjacent Facts": "Adjacent Facts",
    "Root Concern": "Root Concern",
    "Values Divergence": "Values Divergence",
    "Constitutional Framework": "Constitutional Framework",
    "Common Ground": "Common Ground",
    "Bottom Line": "Strip Mode Summary",
    "Left Perspective": "Left Perspective",
    "Right Perspective": "Right Perspective",
    "Sub Claims": "Sub-Claims",
}


def run_reanalysis_ai(claim_text, mode="full"):
    """Run the AI excavation pipeline on an existing claim text. Returns merged parsed JSON."""
    reality_anchor, grok_adjudication = build_reality_anchor_with_grok(claim_text)
    prompt_text = f"{reality_anchor}\n\nNow analyze this claim:\n\"{claim_text}\"".strip()

    claude_json = {}
    openai_json = {}

    try:
        if anthropic_client:
            resp = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                temperature=0,
                system=CLAIMLAB_SYSTEM,
                messages=[{"role": "user", "content": prompt_text}]
            )
            claude_json = safe_json_parse(resp.content[0].text)
    except Exception as e:
        claude_json = {"error": str(e)}

    try:
        if openai_client:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CLAIMLAB_SYSTEM},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=4000,
                temperature=0
            )
            openai_json = safe_json_parse(resp.choices[0].message.content)
    except Exception as e:
        openai_json = {"error": str(e)}

    primary = claude_json if "error" not in claude_json else openai_json
    return primary, claude_json, openai_json, grok_adjudication


@app.route("/editor/claims", methods=["GET"])
def editor_claims_list():
    """Return claims list for the Claim Management tab with filters."""
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    if not session.get("true_superuser"):
        return jsonify({"error": "Unauthorized"}), 403

    try:
        topic = request.args.get("topic", "").strip()
        verdict = request.args.get("verdict", "").strip()
        not_since = request.args.get("not_since", "").strip()
        limit = min(int(request.args.get("limit", 50)), 200)

        formula_parts = ["{Status}='Active'", "OR(NOT({Breakout User Excavated}), {Breakout User Excavated}!='No')"]
        if topic:
            safe_topic = escape_airtable_formula_value(topic)
            formula_parts.append(f"FIND('{safe_topic}', ARRAYJOIN({{Topic}}))")
        if verdict:
            safe_verdict = escape_airtable_formula_value(verdict)
            formula_parts.append(f"{{Overall Verdict}}='{safe_verdict}'")
        if not_since:
            # NOT({Last Reanalyzed}) catches blank/empty values safely
            # IS_BEFORE only runs when the field has a value
            formula_parts.append(
                f"OR(NOT({{Last Reanalyzed}}), IS_BEFORE({{Last Reanalyzed}}, '{not_since}'))"
            )

        formula = "AND(" + ", ".join(formula_parts) + ")" if len(formula_parts) > 1 else formula_parts[0]

        params = {
            "filterByFormula": formula,
            "sort[0][field]": "Date Added",
            "sort[0][direction]": "desc",
            "maxRecords": limit,
            "fields[]": [
                "Original Quote", "Stripped Claim", "URL Slug", "Overall Verdict",
                "Topic", "Date Added", "Last Reanalyzed", "Reanalyzed By", "Entered By"
            ]
        }

        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        claims = []
        for r in records:
            f = r.get("fields", {})
            topics = f.get("Topic", [])
            if isinstance(topics, list):
                topic_str = ", ".join(topics)
            else:
                topic_str = str(topics)
            claims.append({
                "record_id": r.get("id"),
                "title": clean_display_title(f.get("Original Quote") or f.get("Stripped Claim") or "Untitled"),
                "slug": f.get("URL Slug", ""),
                "verdict": f.get("Overall Verdict", ""),
                "topic": topic_str,
                "date_added": (f.get("Date Added") or "")[:10],
                "last_reanalyzed": (f.get("Last Reanalyzed") or "")[:10],
                "reanalyzed_by": f.get("Reanalyzed By", ""),
            })

        return jsonify({"ok": True, "claims": claims, "total": len(claims)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/editor/reanalyze-claim/<record_id>", methods=["POST"])
def editor_reanalyze_claim(record_id):
    """Reanalyze a single claim. Supports full or selective modes."""
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    if not session.get("true_superuser"):
        return jsonify({"error": "Unauthorized"}), 403

    try:
        data = request.get_json() or {}
        mode = data.get("mode", "full")  # "full" or "selective"
        selective_fields = data.get("fields", [])  # list of field names for selective

        # Fetch the claim record
        claim_record = get_claim_by_record_id(record_id)
        if not claim_record:
            return jsonify({"error": "Claim not found"}), 404

        claim_fields = claim_record.get("fields", {})
        claim_text = claim_fields.get("Original Quote") or claim_fields.get("Stripped Claim") or ""

        if not claim_text:
            return jsonify({"error": "No claim text found on this record"}), 400

        # Run AI
        primary, claude_json, openai_json, grok_adjudication = run_reanalysis_ai(claim_text, mode)

        if "error" in primary:
            return jsonify({"error": f"AI failed: {primary['error']}"}), 500

        editor_username = session.get("username", "Editor")
        now_str = datetime.utcnow().strftime("%Y-%m-%d")

        if mode == "full":
            # Full reanalysis — rebuild all fields
            update_fields = extract_primary_record_fields(
                claim=claim_text,
                parsed=primary,
                mode="full",
                username=editor_username,
                existing_fields=claim_fields
            )
            update_fields["Claude Raw JSON"] = json.dumps(claude_json, ensure_ascii=False)[:100000]
            update_fields["OpenAI Raw JSON"] = json.dumps(openai_json, ensure_ascii=False)[:100000]
            if grok_adjudication:
                update_fields["Grok Raw JSON"] = json.dumps(grok_adjudication, ensure_ascii=False)[:100000]
            update_fields["Last Reanalyzed"] = now_str
            update_fields["Reanalyzed By"] = editor_username

        else:
            # Selective — only write requested fields
            update_fields = {
                "Last Reanalyzed": now_str,
                "Reanalyzed By": editor_username,
                "Last Updated": now_str
            }
            for field_name in selective_fields:
                airtable_key = SELECTIVE_FIELD_MAP.get(field_name)
                if not airtable_key:
                    continue
                if airtable_key == "Sub-Claims":
                    sub_claims = primary.get("Sub Claims", [])
                    if isinstance(sub_claims, list):
                        update_fields["Sub-Claims"] = " | ".join(
                            [sc.get("claim", "") for sc in sub_claims if sc.get("claim")]
                        )
                        if len(sub_claims) > 0:
                            update_fields["Sub-Claim 1"] = sub_claims[0].get("claim", "")
                            if sub_claims[0].get("verdict"):
                                update_fields["Verdict: Sub-Claim1"] = sub_claims[0]["verdict"]
                elif primary.get(field_name):
                    update_fields[airtable_key] = primary[field_name]
                elif field_name == "Overall Verdict":
                    v = primary.get("Overall Verdict") or primary.get("Verdict")
                    if v:
                        update_fields["Overall Verdict"] = v

        resp = update_airtable_record(record_id, update_fields)
        if not resp.ok:
            return jsonify({"error": f"Airtable update failed: {resp.text}"}), 500

        return jsonify({
            "ok": True,
            "record_id": record_id,
            "mode": mode,
            "fields_updated": list(update_fields.keys()),
            "reanalyzed_by": editor_username,
            "last_reanalyzed": now_str,
            "slug": claim_fields.get("URL Slug", "")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/editor/reanalyze-claim-by-slug/<slug>", methods=["POST"])
def editor_reanalyze_claim_by_slug(slug):
    """Reanalyze a single claim from its claim detail page (by slug)."""
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    if not session.get("true_superuser"):
        return jsonify({"error": "Unauthorized"}), 403

    try:
        data = request.get_json() or {}
        mode = data.get("mode", "full")
        selective_fields = data.get("fields", [])

        # Find the claim by slug
        safe_slug = escape_airtable_formula_value(slug)
        params = {"filterByFormula": f"{{URL Slug}}='{safe_slug}'", "maxRecords": 1}
        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        if not records:
            return jsonify({"error": "Claim not found"}), 404

        record_id = records[0].get("id")
        # Delegate to the record_id route logic
        request._cached_json = {"mode": mode, "fields": selective_fields}

        claim_fields = records[0].get("fields", {})
        claim_text = claim_fields.get("Original Quote") or claim_fields.get("Stripped Claim") or ""
        if not claim_text:
            return jsonify({"error": "No claim text found"}), 400

        primary, claude_json, openai_json, grok_adjudication = run_reanalysis_ai(claim_text, mode)
        if "error" in primary:
            return jsonify({"error": f"AI failed: {primary['error']}"}), 500

        editor_username = session.get("username", "Editor")
        now_str = datetime.utcnow().strftime("%Y-%m-%d")

        if mode == "full":
            update_fields = extract_primary_record_fields(
                claim=claim_text, parsed=primary, mode="full",
                username=editor_username, existing_fields=claim_fields
            )
            update_fields["Claude Raw JSON"] = json.dumps(claude_json, ensure_ascii=False)[:100000]
            update_fields["OpenAI Raw JSON"] = json.dumps(openai_json, ensure_ascii=False)[:100000]
            if grok_adjudication:
                update_fields["Grok Raw JSON"] = json.dumps(grok_adjudication, ensure_ascii=False)[:100000]
        else:
            update_fields = {"Last Updated": now_str}
            for field_name in selective_fields:
                airtable_key = SELECTIVE_FIELD_MAP.get(field_name)
                if airtable_key and primary.get(field_name):
                    update_fields[airtable_key] = primary[field_name]

        update_fields["Last Reanalyzed"] = now_str
        update_fields["Reanalyzed By"] = editor_username

        resp = update_airtable_record(record_id, update_fields)
        if not resp.ok:
            return jsonify({"error": f"Airtable update failed: {resp.text}"}), 500

        return jsonify({
            "ok": True, "record_id": record_id, "mode": mode,
            "reanalyzed_by": editor_username, "last_reanalyzed": now_str,
            "fields_updated": list(update_fields.keys())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ══════════════════════════════════════════════
# BREAKOUT CLAIMS SYSTEM
# ══════════════════════════════════════════════

BREAKOUT_DETECTION_SYSTEM = """You are the Breakout Claim detection engine for Where the Truth Lies.

Your job is to read a body of text (a political claim, dispute, or pushback) and identify statements within it that are themselves distinct, standalone, falsifiable claims — separate from the primary claim being excavated.

A breakout claim must:
- Be a distinct, falsifiable statement in its own right
- Be separable from the main claim being analyzed
- Warrant its own excavation on merit

A breakout claim must NOT be:
- A restatement of the primary claim
- A general opinion or value judgment without factual content
- A vague inference
- Something already fully covered by the primary claim analysis

When multiple breakout claims share the same underlying topic or subject, group them together under a group key and group label.

Return ONLY valid JSON. No markdown fences. No preamble.

Return this exact structure:

{
  "has_breakouts": true or false,
  "breakout_claims": [
    {
      "title": "A clear, plain-language title for this breakout claim. One sentence. No hyphens or dashes.",
      "source_text": "The exact statement from the source text that triggered this breakout.",
      "group_key": "snake_case_topic_key",
      "group_label": "Plain Language Topic Label",
      "confidence": 0.85
    }
  ]
}

If no breakout claims are found, return: {"has_breakouts": false, "breakout_claims": []}

Rules:
Never use bullet points, dashes, or hyphens in any title field.
Write titles as plain declarative statements.
Group related claims under the same group_key and group_label.
Confidence should be between 0.5 and 1.0. Only include claims above 0.6 confidence.
Maximum 8 breakout claims per source text.
"""


def detect_breakout_claims(source_text, source_type="Main Claim"):
    """Run breakout detection on a text using the triple-AI pipeline."""
    if not source_text or not source_text.strip():
        return []

    prompt = f"""Analyze this text for breakout claims:

Source type: {source_type}
Text: {source_text}"""

    result = None

    # Try Grok first (live layer)
    if grok_client:
        try:
            response = grok_client.chat.completions.create(
                model="grok-3-fast",
                messages=[
                    {"role": "system", "content": BREAKOUT_DETECTION_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0
            )
            parsed = safe_json_parse(response.choices[0].message.content)
            if isinstance(parsed, dict) and "has_breakouts" in parsed:
                result = parsed
        except Exception as e:
            print(f"GROK BREAKOUT DETECTION ERROR: {e}", flush=True)

    # Fall back to Claude
    if not result and anthropic_client:
        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                temperature=0,
                system=BREAKOUT_DETECTION_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            parsed = safe_json_parse(response.content[0].text)
            if isinstance(parsed, dict) and "has_breakouts" in parsed:
                result = parsed
        except Exception as e:
            print(f"CLAUDE BREAKOUT DETECTION ERROR: {e}", flush=True)

    # Fall back to OpenAI
    if not result and openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": BREAKOUT_DETECTION_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0
            )
            parsed = safe_json_parse(response.choices[0].message.content)
            if isinstance(parsed, dict) and "has_breakouts" in parsed:
                result = parsed
        except Exception as e:
            print(f"OPENAI BREAKOUT DETECTION ERROR: {e}", flush=True)

    if not result or not result.get("has_breakouts"):
        return []

    return result.get("breakout_claims", [])


def generate_claim_identifier(parent_identifier, origin_type, child_sequence):
    """Generate a hierarchical claim identifier."""
    if not parent_identifier:
        # Root claim — no identifier yet, will be assigned on backfill
        return None

    # Determine suffix prefix based on origin type
    if origin_type == "Breakout Claim":
        prefix = "C"
    elif origin_type == "Breakout Dispute":
        prefix = "D"
    elif origin_type == "Breakout Pushback":
        prefix = "P"
    else:
        prefix = "C"

    seq = str(child_sequence).zfill(2)

    # If parent has a dot already (grandchild+), append .seq
    if "." in parent_identifier:
        return f"{parent_identifier}.{seq}"
    else:
        return f"{parent_identifier}.{prefix}{seq}"


def get_next_child_sequence(parent_record_id):
    """Count existing children of a parent to get the next child sequence number."""
    if not parent_record_id:
        return 1
    try:
        safe_id = escape_airtable_formula_value(parent_record_id)
        params = {
            "filterByFormula": f"{{Parent Claim ID}}='{safe_id}'",
            "fields[]": ["Child Sequence"]
        }
        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        if not records:
            return 1
        max_seq = max(
            int(r.get("fields", {}).get("Child Sequence", 0) or 0)
            for r in records
        )
        return max_seq + 1
    except Exception as e:
        print(f"GET NEXT CHILD SEQUENCE ERROR: {e}", flush=True)
        return 1


def create_breakout_claim_record(
    breakout_data,
    parent_record_id,
    root_record_id,
    origin_type,
    parent_identifier,
    source_section,
    origin_dispute_id=None,
    origin_pushback_id=None,
    origin_claim_record_id=None,
    detection_source="Initial Submission",
    parent_slug=None
):
    """Create a breakout claim row in Airtable with full hierarchy metadata."""
    title = breakout_data.get("title", "Untitled Breakout Claim")
    source_text = breakout_data.get("source_text", "")
    group_key = breakout_data.get("group_key", "")
    group_label = breakout_data.get("group_label", "")
    confidence = float(breakout_data.get("confidence", 0.75))

    # Deduplication — skip if breakout with same title already exists under this parent
    if parent_record_id:
        try:
            existing = get_breakout_claims_for_parent(parent_record_id)
            title_lower = title.strip().lower()
            for ex in existing:
                if ex.get("title", "").strip().lower() == title_lower:
                    print(f"BREAKOUT DEDUP: skipping '{title}' — already exists", flush=True)
                    return None
        except Exception:
            pass

    child_seq = get_next_child_sequence(parent_record_id)
    claim_identifier = generate_claim_identifier(parent_identifier, origin_type, child_seq)
    slug = slugify(title)
    dates = now_dates()

    fields = {
        "Original Quote": title,
        "Stripped Claim": title,
        "Status": "Active",
        "Published": False,
        "Human Reviewed": False,
        "Date": dates["display_date"],
        "Date Added": dates["short_date"],
        "Last Updated": dates["short_date"],
        "URL Slug": slug,
        "Origin Type": origin_type,
        "Breakout Source Section": source_section,
        "Breakout Status": "Pending Excavation",
        "Breakout Source Text": source_text,
        "Breakout Group Key": group_key,
        "Breakout Group Label": group_label,
        "Breakout Confidence": confidence,
        "Child Sequence": child_seq,
        "Claim Depth": 0,  # will calculate below
        "Lock Status": "Unlocked",
        "Claims Cost": 1,
        "Breakout Detection Source": detection_source,
        "Breakout User Excavated": "No",
    }

    if claim_identifier:
        fields["Claim Identifier"] = claim_identifier

    # Link fields
    if parent_record_id:
        fields["Parent Claim"] = [parent_record_id]
        fields["Parent Claim ID"] = parent_record_id
        # Calculate depth from parent
        try:
            parent_rec = get_claim_by_record_id(parent_record_id)
            if parent_rec:
                parent_depth = int(parent_rec.get("fields", {}).get("Claim Depth", 0) or 0)
                fields["Claim Depth"] = parent_depth + 1
        except Exception:
            fields["Claim Depth"] = 1

    if root_record_id:
        fields["Root Claim"] = [root_record_id]
    elif parent_record_id:
        fields["Root Claim"] = [parent_record_id]

    # Origin Claim — set when breakout comes from a claim body or another breakout child
    if origin_claim_record_id:
        fields["Origin Claim"] = [origin_claim_record_id]
    elif origin_type in ("Breakout Claim",) and parent_record_id:
        # Main claim origin — parent IS the origin claim
        fields["Origin Claim"] = [parent_record_id]

    if origin_dispute_id:
        fields["Origin Dispute ID"] = origin_dispute_id
    if origin_pushback_id:
        fields["Origin Pushback ID"] = origin_pushback_id

    # Create the record
    try:
        resp = create_airtable_record(fields)
        if resp.ok:
            new_record = resp.json()

            # Mark parent as having breakout children
            if parent_record_id:
                update_airtable_record(parent_record_id, {"Has Breakout Children": True})

            print(f"BREAKOUT CLAIM CREATED: {claim_identifier or slug} depth={fields['Claim Depth']} source={detection_source}", flush=True)
            return new_record
        else:
            print(f"BREAKOUT CLAIM CREATE ERROR: {resp.text}", flush=True)
            return None
    except Exception as e:
        print(f"BREAKOUT CLAIM CREATE EXCEPTION: {e}", flush=True)
        return None


def run_breakout_detection_for_claim(claim_record, detection_source_override=None):
    """
    Run full breakout detection on a claim record:
    main claim body + all disputes + all pushbacks.
    Creates breakout claim rows for any detected candidates.
    Returns count of breakouts created.
    """
    if not claim_record:
        return 0

    fields = claim_record.get("fields", {})
    record_id = claim_record.get("id")
    slug = fields.get("URL Slug", "")
    claim_text = fields.get("Original Quote") or fields.get("Stripped Claim") or ""
    parent_identifier = fields.get("Claim Identifier", "")
    root_record_id = record_id  # this claim is its own root for direct children

    # Check if root claim has a root claim link itself
    root_links = fields.get("Root Claim", [])
    if root_links:
        root_record_id = root_links[0]

    created_count = 0

    # 1. Detect from main claim body
    main_breakouts = detect_breakout_claims(claim_text, source_type="Main Claim")
    for b in main_breakouts:
        result = create_breakout_claim_record(
            breakout_data=b,
            parent_record_id=record_id,
            root_record_id=root_record_id,
            origin_type="Breakout Claim",
            parent_identifier=parent_identifier,
            source_section="Main Claim",
            detection_source=detection_source_override or "Initial Submission"
        )
        if result:
            created_count += 1

    # 2. Detect from disputes and pushbacks
    if slug:
        claim_disputes = get_disputes_for_claim(slug)
        for dispute in claim_disputes:
            dispute_text = dispute.get("dispute_text", "")
            response_text = dispute.get("response_text", "")
            dispute_type = dispute.get("dispute_type", "Initial Dispute")
            dispute_record_id = dispute.get("record_id", "")

            is_pushback = "pushback" in dispute_type.lower()
            source_type = "Pushback" if is_pushback else "Dispute"
            origin_type = "Breakout Pushback" if is_pushback else "Breakout Dispute"
            det_source = "Pushback" if is_pushback else "Dispute"

            combined_text = " ".join(filter(None, [dispute_text, response_text]))
            if not combined_text.strip():
                continue

            dispute_breakouts = detect_breakout_claims(combined_text, source_type=source_type)
            dispute_had_breakouts = False
            for b in dispute_breakouts:
                result = create_breakout_claim_record(
                    breakout_data=b,
                    parent_record_id=record_id,
                    root_record_id=root_record_id,
                    origin_type=origin_type,
                    parent_identifier=parent_identifier,
                    source_section="Dispute" if not is_pushback else "Pushback",
                    origin_dispute_id=dispute_record_id if not is_pushback else None,
                    origin_pushback_id=dispute_record_id if is_pushback else None,
                    detection_source=det_source
                )
                if result:
                    created_count += 1
                    dispute_had_breakouts = True

            # Mark the dispute record with Has Breakout Claims
            if dispute_had_breakouts and dispute_record_id:
                try:
                    update_dispute_record(dispute_record_id, {"Has Breakout Claims": True})
                except Exception as e:
                    print(f"HAS BREAKOUT CLAIMS UPDATE ERROR: {e}", flush=True)

    return created_count


def get_breakout_claims_for_parent(parent_record_id):
    """Fetch all breakout claims that have this record as parent."""
    if not parent_record_id:
        return []
    try:
        safe_id = escape_airtable_formula_value(parent_record_id)
        params = {
            "filterByFormula": f"{{Parent Claim ID}}='{safe_id}'",
            "sort[0][field]": "Breakout Group Key",
            "sort[0][direction]": "asc"
        }
        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        breakouts = []
        for r in records:
            f = r.get("fields", {})
            breakouts.append({
                "record_id": r.get("id"),
                "title": clean_display_title(f.get("Stripped Claim") or f.get("Original Quote") or "Untitled"),
                "slug": f.get("URL Slug", ""),
                "claim_identifier": f.get("Claim Identifier", ""),
                "breakout_status": f.get("Breakout Status", "Pending Excavation"),
                "breakout_source_section": f.get("Breakout Source Section", "Main Claim"),
                "breakout_group_key": f.get("Breakout Group Key", ""),
                "breakout_group_label": f.get("Breakout Group Label", ""),
                "breakout_source_text": f.get("Breakout Source Text", ""),
                "origin_type": f.get("Origin Type", "Breakout Claim"),
                "origin_dispute_id": f.get("Origin Dispute ID", ""),
                "origin_pushback_id": f.get("Origin Pushback ID", ""),
                "claim_depth": int(f.get("Claim Depth", 0) or 0),
                "has_breakout_children": bool(f.get("Has Breakout Children", False)),
                "overall_verdict": f.get("Overall Verdict", ""),
                "lock_status": f.get("Lock Status", "Unlocked"),
            })
        return breakouts
    except Exception as e:
        print(f"GET BREAKOUT CLAIMS ERROR: {e}", flush=True)
        return []


def group_breakout_claims(breakouts):
    """Group breakout claims by their group_key for display."""
    groups = {}
    ungrouped = []

    for b in breakouts:
        gk = b.get("breakout_group_key", "").strip()
        if gk:
            if gk not in groups:
                groups[gk] = {
                    "group_key": gk,
                    "group_label": b.get("breakout_group_label", gk.replace("_", " ").title()),
                    "claims": []
                }
            groups[gk]["claims"].append(b)
        else:
            ungrouped.append(b)

    result = list(groups.values())
    if ungrouped:
        result.append({
            "group_key": "__ungrouped__",
            "group_label": "Other Breakout Claims",
            "claims": ungrouped
        })

    return result


# ── BREAKOUT ROUTES ──

@app.route("/breakout/detect/<slug>", methods=["POST"])
def breakout_detect_for_claim(slug):
    """Manually trigger breakout detection for a claim. Superuser or claim owner."""
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401

    record = get_claim_by_slug(slug)
    if not record:
        return jsonify({"error": "Claim not found"}), 404

    count = run_breakout_detection_for_claim(record)
    return jsonify({"ok": True, "breakouts_created": count, "slug": slug})


@app.route("/breakout/list/<slug>", methods=["GET"])
def breakout_list_for_claim(slug):
    """Return grouped breakout claims for a claim page."""
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401

    record = get_claim_by_slug(slug)
    if not record:
        return jsonify({"error": "Claim not found"}), 404

    record_id = record.get("id")
    breakouts = get_breakout_claims_for_parent(record_id)
    grouped = group_breakout_claims(breakouts)

    return jsonify({
        "ok": True,
        "total": len(breakouts),
        "groups": grouped
    })


@app.route("/breakout/excavate", methods=["POST"])
def breakout_excavate():
    """Synchronous excavation. Requires gunicorn --timeout 120."""
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    user_info = get_fresh_user_session_info()
    if not user_info or not user_info.get("active"):
        return jsonify({"error": "Account unavailable"}), 403
    role = user_info["role"]
    claims_remaining = user_info["claims_remaining"]
    if role == "standard":
        return jsonify({"error": "Your account cannot run new excavations."}), 403
    if role == "limited_superuser" and claims_remaining <= 0:
        return jsonify({"error": "Claim limit reached."}), 403
    data = request.get_json() or {}
    breakout_record_id = (data.get("record_id") or "").strip()
    user_context = (data.get("user_context") or "").strip()
    if not breakout_record_id:
        return jsonify({"error": "Breakout record ID required"}), 400
    breakout_record = get_claim_by_record_id(breakout_record_id)
    if not breakout_record:
        return jsonify({"error": "Breakout claim not found"}), 404
    bf = breakout_record.get("fields", {})
    lock_status = (bf.get("Lock Status") or "Unlocked").strip()
    if lock_status == "Locked":
        lock_expires = (bf.get("Lock Expires At") or "")
        if lock_expires:
            try:
                exp = datetime.fromisoformat(lock_expires.replace("Z", "+00:00"))
                if datetime.utcnow().replace(tzinfo=exp.tzinfo) < exp:
                    return jsonify({"error": "This claim is currently being excavated by another user."}), 409
            except Exception:
                pass
    current_status = (bf.get("Breakout Status") or "Pending Excavation").strip()
    if current_status == "Excavated":
        existing_slug = bf.get("URL Slug", "")
        return jsonify({"ok": True, "redirect_to": f"/claim/{existing_slug}"}), 200
    lock_expiry = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    update_airtable_record(breakout_record_id, {
        "Lock Status": "Locked",
        "Locked By": session.get("username", "Unknown"),
        "Lock Expires At": lock_expiry,
        "Breakout Status": "Excavating",
        "Breakout User Excavated": "Yes"
    })
    if role == "limited_superuser":
        new_count = max(0, claims_remaining - 1)
        update_user_claims_remaining(user_info["record_id"], new_count)
        session["claims_remaining"] = new_count
    claim_text = bf.get("Original Quote") or bf.get("Stripped Claim") or ""
    if user_context:
        claim_text = f"{claim_text}\n\nAdditional context from user: {user_context}"
        update_airtable_record(breakout_record_id, {"User Added Context": user_context})
    username = session.get("username", "Unknown")
    try:
        reality_anchor, grok_adjudication = build_reality_anchor_with_grok(claim_text)
        prompt_text = f"{reality_anchor}\n\nNow analyze this claim:\n\"{claim_text}\"".strip()
        claude_json = {}
        openai_json = {}
        try:
            if anthropic_client:
                r = anthropic_client.messages.create(
                    model="claude-sonnet-4-6", max_tokens=4000, temperature=0,
                    system=CLAIMLAB_SYSTEM, messages=[{"role": "user", "content": prompt_text}]
                )
                claude_json = safe_json_parse(r.content[0].text)
        except Exception as e:
            claude_json = {"error": str(e)}
        try:
            if openai_client:
                r = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": CLAIMLAB_SYSTEM}, {"role": "user", "content": prompt_text}],
                    max_tokens=4000, temperature=0
                )
                openai_json = safe_json_parse(r.choices[0].message.content)
        except Exception as e:
            openai_json = {"error": str(e)}
        primary = claude_json if "error" not in claude_json else openai_json
        if "error" in primary:
            update_airtable_record(breakout_record_id, {
                "Lock Status": "Unlocked", "Breakout Status": "Pending Excavation", "Breakout User Excavated": "No"
            })
            return jsonify({"error": f"AI excavation failed: {primary.get('error', 'Unknown')}"}), 500
        update_fields = extract_primary_record_fields(
            claim=claim_text, parsed=primary, mode="full", username=username, existing_fields=bf
        )
        update_fields["Claude Raw JSON"] = json.dumps(claude_json, ensure_ascii=False)[:100000]
        update_fields["OpenAI Raw JSON"] = json.dumps(openai_json, ensure_ascii=False)[:100000]
        if grok_adjudication:
            update_fields["Grok Raw JSON"] = json.dumps(grok_adjudication, ensure_ascii=False)[:100000]
        update_fields["Breakout Status"] = "Excavated"
        update_fields["Lock Status"] = "Unlocked"
        update_fields["Locked By"] = ""
        update_fields["Excavation Record ID"] = breakout_record_id
        resp = update_airtable_record(breakout_record_id, update_fields)
        if not resp.ok:
            update_airtable_record(breakout_record_id, {
                "Lock Status": "Unlocked", "Breakout Status": "Pending Excavation", "Breakout User Excavated": "No"
            })
            return jsonify({"error": f"Failed to save excavation: {resp.text}"}), 500
        final_slug = update_fields.get("URL Slug", bf.get("URL Slug", ""))
        parent_links = bf.get("Parent Claim", [])
        if parent_links:
            update_airtable_record(parent_links[0], {"Has Breakout Children": True})
        refreshed = get_claim_by_record_id(breakout_record_id)
        if refreshed:
            child_count = run_breakout_detection_for_claim(refreshed, detection_source_override="Child Excavation")
            print(f"BREAKOUT COMPLETE: {final_slug} — {child_count} child breakouts", flush=True)
        return jsonify({"ok": True, "redirect_to": f"/claim/{final_slug}", "slug": final_slug, "claims_remaining": session.get("claims_remaining")})
    except Exception as e:
        print(f"BREAKOUT ERROR: {e}", flush=True)
        update_airtable_record(breakout_record_id, {
            "Lock Status": "Unlocked", "Breakout Status": "Pending Excavation", "Breakout User Excavated": "No"
        })
        return jsonify({"error": f"Excavation error: {str(e)}"}), 500



@app.route("/breakout/lock-check/<record_id>", methods=["GET"])
def breakout_lock_check(record_id):
    """Quick check if a breakout claim is locked or already excavated."""
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    record = get_claim_by_record_id(record_id)
    if not record:
        return jsonify({"error": "Not found"}), 404
    f = record.get("fields", {})
    return jsonify({
        "lock_status": f.get("Lock Status", "Unlocked"),
        "breakout_status": f.get("Breakout Status", "Pending Excavation"),
        "slug": f.get("URL Slug", "")
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
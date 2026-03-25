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
XAI_API_KEY = os.getenv("XAI_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Claims")
AIRTABLE_USERS_TABLE_NAME = os.getenv("AIRTABLE_USERS_TABLE_NAME", "Users")
AIRTABLE_REALITY_TABLE_NAME = os.getenv("AIRTABLE_REALITY_TABLE_NAME", "Reality Anchors")
AIRTABLE_DISPUTES_TABLE_NAME = os.getenv("AIRTABLE_DISPUTES_TABLE_NAME", "Disputes")

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
  "Quick Explanation": "Provide a neutral 1 to 2 sentence explanation using this structure: This claim is [verdict] because [core factual reality]. It [overstates, misrepresents, or confuses] [key distinction]. Plain language. No jargon. No sarcasm. No political tone. Not a full analytical paragraph.",
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
    claim_record_id = record.get("id")
    claim_disputes = get_disputes_for_claim(claim_record_id)
    title = fields.get("Original Quote") or fields.get("Stripped Claim") or "Untitled Claim"

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
        "disputes": claim_disputes,
        "quick_view": quick_view,
        "full_view": full_view,

        "attribution_status": attribution["status"],
        "attribution_detail": attribution["detail"],
        "claude_verdict": claude_verdict,
        "openai_verdict": openai_verdict,
        "models_diverged": models_diverged,
        "grok_adjudication": grok_adjudication,
        "grok_grounded": grok_grounded,
        "grok_status": grok_status
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
            "sort[0][field]": "Date Added",
            "sort[0][direction]": "desc"
        }
        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        recent = []
        for record in records[:limit]:
            f = record.get("fields", {})
            recent.append({
    "title": f.get("Original Quote") or f.get("Stripped Claim") or "Untitled Claim",
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
            "sort[0][field]": "Date Added",
            "sort[0][direction]": "desc"
        }
        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        claims = []
        for record in records:
            f = record.get("fields", {})
            claims.append({
                "record_id": record.get("id"),
                "title": f.get("Original Quote") or f.get("Stripped Claim") or "Untitled Claim",
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

def get_disputes_for_claim(claim_record_id):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_DISPUTES_TABLE_NAME or not claim_record_id:
        return []

    try:
        params = {
            "filterByFormula": f"FIND('{escape_airtable_formula_value(claim_record_id)}', ARRAYJOIN({{Claim Record ID}}))",
            "sort[0][field]": "Last Updated",
            "sort[0][direction]": "desc"
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
                "user_source_url": f.get("User Source URL", ""),
                "pushback_round_count": f.get("Pushback Round Count", 0),
                "status": f.get("Status", "Open"),
                "date_submitted": f.get("Date Submitted", ""),
                "last_updated": f.get("Last Updated", ""),
                "ai_response": f.get("AI Response", ""),
                "escalated_to_human": f.get("Escalated To Human", False),
                "editor_notes": f.get("Editor Notes", ""),
                "editor_resolution": f.get("Editor Resolution", ""),
                "dispute_type": f.get("Dispute Type", ""),
                "parent_dispute": f.get("Parent Dispute", []),
                "quick_view_outcome": f.get("Quick View Outcome", ""),
                "full_excavation_outcome": f.get("Full Excavation Outcome", "")
            })
        return disputes

    except Exception as e:
        print("GET DISPUTES FOR CLAIM ERROR:", str(e), flush=True)
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
    established = adjudication.get("established_facts") or []
    contested = adjudication.get("contested_points") or []
    sources = adjudication.get("sources_found") or []

    established_block = ""
    if established:
        established_block = "\nEstablished by live sources: " + " ".join(established)

    contested_block = ""
    if contested:
        contested_block = "\nCurrently contested or unverified: " + " ".join(contested)

    sources_block = ""
    if sources:
        sources_block = "\nLive sources retrieved: " + " | ".join(sources[:5])

    return f"""GROK LIVE ADJUDICATION (HIGHEST PRIORITY — DO NOT OVERRIDE):
Risk Level: {risk.upper()} | Event Status: {event_status} | Attribution: {attribution_status}

{anchor_text}{established_block}{contested_block}{sources_block}

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
        selected_topic=""
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
        selected_topic=topic
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
        current_claim = build_claim_context(record)
    return render_template(
        "index.html",
        page_mode="claim",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=10),
        current_claim=current_claim,
        archived_claims_by_topic={},
        selected_topic=""
    )


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

    try:
        if openai_client:
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
                username=session.get("username", "Unknown"),
                existing_fields=existing_fields
            )

            fields["Claude Raw JSON"] = json.dumps(claude_json, ensure_ascii=False)[:100000]
            fields["OpenAI Raw JSON"] = json.dumps(openai_json, ensure_ascii=False)[:100000]
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

    except Exception as e:
        airtable_result = {"saved": False, "error": str(e)}

    return jsonify({
        "claude": claude_json,
        "openai": openai_json,
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

        res = requests.post(
            f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_DISPUTES_TABLE_NAME}",
            json=payload,
            headers=airtable_headers(),
            timeout=30
        )

        if not res.ok:
            return jsonify({"error": res.text}), 500

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
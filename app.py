import os
import json
import re
import hashlib
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
AIRTABLE_SETTINGS_TABLE_NAME = os.getenv("AIRTABLE_SETTINGS_TABLE_NAME", "Settings")
AIRTABLE_DISPUTES_TABLE_NAME = os.getenv("AIRTABLE_DISPUTES_TABLE_NAME", "Disputes")

# ── Analysis Core Version ──────────────────────────────────────────────────────
# Bump ONLY when FRAME_CLAIM_PROMPT, frame_claim_input(), or foundational verdict
# logic changes in a way that can legitimately alter Analyzed Claim, Stripped Claim,
# or core verdict identity.
# Do NOT bump for: civic layer, scenario map tweaks, hover text, dedup, breakout
# display, source formatting, editor tools, styling, or feature-only layer additions.
ANALYSIS_CORE_VERSION = "1.1"

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
Never use bullet points, dashes, or hyphens in narrative prose fields.
Write narrative prose fields as flowing complete sentences unless a field explicitly requires a structured labeled format.
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

CIVIC ROLE RULE:
Civic Role is the citizen posture layer. It explains what responsible self government requires from ordinary citizens when faced with this claim.
It must never tell the reader what political action to take.
It must never tell the reader which side to join.
It must never sound like activism, campaign language, or moral scolding.
It must not use direct commands like vote, call, support, oppose, organize, protest, demand, or pressure.
Instead, it should explain what a serious citizen should be attentive to, what confusion or tribal temptation should be resisted, and what civic burden this issue places on a free people.
It should sound like grounded civic common sense in plain English.
It should reflect the structure of constitutional self government without quoting founders theatrically or pretending to speak as them.

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

PRE-ANALYSIS GROUNDING RULE:
Before populating any layer, establish what is actually known about this claim across the domains that are directly relevant to it. Only apply the domains the claim genuinely touches. Do not stretch to include domains that are not directly relevant — that produces hallucinated authority.
Relevant domains to consider: legal and constitutional, historical precedent, economic outcomes (observed results, not theory), social and behavioral record, philosophical assumptions underlying the claim, scientific or medical consensus where applicable, and founder intent where government power or rights are involved.
For each relevant domain: identify what is established, what is contested, and what the historical record actually shows — not what is currently argued.

PERSON AS CONTEXT RULE:
If a claim names a specific person, acknowledge that person clearly. They are the entry point and the user deserves to see their question reflected in the analysis.
However, do not center the analysis on the individual's intent, personality, character, or assumed motives. Shift immediately from who to what: what action is being taken, under what conditions, and whether those conditions and that level of action are consistent with law, precedent, and the constitutional framework.
The person is context. The action and the conditions are the subject.

CONDITIONS OVER INTENT RULE:
When analyzing claims about government action, executive authority, institutional power, or any actor exercising authority: do not attempt to infer intent. Do not characterize actions as ambitious, corrupt, genuine, or necessary based on speculation about motive.
Instead, evaluate the fit between the action and the conditions: What conditions are present? What precedent exists for comparable action under comparable conditions? Is the level of action proportionate to those conditions? Where is that fit disputed?
This applies to all actors regardless of party, office, or era.

CONSISTENCY ACROSS TIME AND ACTORS RULE:
Apply the same analytical standard regardless of who is acting, what party they belong to, or what era they operated in.
If the same behavior by a different president, party, institution, or historical figure would produce a different conclusion under the same framework, that inconsistency must be stated explicitly.
Historical precedent is not decoration. It is the test. When a claim characterizes an action as historically unprecedented, constitutionally illegitimate, or beyond legal authority, the analysis must compare that action to documented precedents — including those from administrations of both parties and from historical figures the culture broadly respects. Washington's use of federal troops during the Whiskey Rebellion, Lincoln's suspension of habeas corpus, FDR's internment orders, and comparable actions must be applied as analytical benchmarks when relevant. If the same standard would condemn those figures, the standard itself must be interrogated before it is applied to the current claim.
Charged terminology — king, dictator, fascist, traitor, authoritarian, communist, or similar — must be tested against the actual historical and legal definition of those terms, not just the underlying concern they express. If the terminology does not hold up to its own definition, that must be stated plainly.

TERMINOLOGY VALIDATION RULE:
When a claim uses charged or contested terminology — such as monarchy, authoritarian, dictator, fascist, traitor, communist, coup, insurrection, or similar — the analysis must do three things explicitly. First, briefly state what the term actually means in its legal, historical, or political context. Second, test whether the current situation meets that definition by the established standard. Third, state plainly whether the term holds up, partially applies, or does not apply — and why. In Quick View, signal that this test is happening. In Full Excavation, execute it fully.

EXPLICIT CONDITIONS RULE:
When evaluating any claim about government action, executive authority, or institutional behavior, explicitly identify the conditions present before rendering any judgment about fit or proportionality. State what is actually happening: what level of institutional breakdown exists, whether courts are functioning, whether elections are proceeding, whether legislative oversight is present or absent, and what the comparative urgency level is relative to historical crises. Do not imply conditions — state them.

INLINE PRECEDENT RULE:
When relevant, name specific historical precedents and briefly state whether current conditions are comparable. Use history as the standard, not decoration. In Quick View, signal the historical standard being applied. In Full Excavation, execute the full comparison.

STRUCTURAL EVALUATION LAYER (CRITICAL FOR CIVIC CLAIMS):

If a claim involves government power, executive authority, constitutional limits, or public concern about leadership behavior:

You MUST structure your reasoning around evaluation layers, not surface narrative details.

DO NOT prioritize:
protest size
crowd presence
slogans
movement descriptions
repetition in the claim text

These are secondary signals, not the core of the claim.

---

INSTEAD, you MUST internally evaluate the claim using these layers:

CONSTITUTIONAL REALITY:
Are institutional checks still functioning including courts, elections, and legislative oversight

HISTORICAL CONTEXT:
Has similar executive behavior occurred before under comparable conditions, or is this an outlier

THRESHOLD ANALYSIS:
What separates aggressive use of executive authority from actual authoritarian or monarch like control

RESPONSE JUSTIFICATION:
What level of conditions has historically justified large scale civic response such as protest

STRUCTURAL ACCOUNTABILITY:
Is the concern primarily about executive overreach, or about other branches failing to exercise their responsibilities

---

OUTPUT REQUIREMENT:

Your explanation must reflect these layers even if they are not explicitly labeled.

The goal is not to describe the political movement.
The goal is to evaluate whether the underlying concern meets constitutional, historical, and civic thresholds.

If surface level details conflict with structural reality, structural reality takes priority.

HOVER TERMS RULE:
Any technical term, acronym, legal citation, organization name, court case, historical document, or charged political term with a specific legal or historical definition that general readers may not recognize must be included in the Glossary.

Speaker rule:
If the claim explicitly names a speaker, use that person or entity
If the claim does not explicitly name a speaker but is strongly associated with a dominant public figure or institution, infer that speaker
If there is no clearly dominant speaker, return Unknown

Always return this exact JSON structure:

{
  "Stripped Claim": "Rewrite the claim in plain, accessible language that any ordinary person can understand immediately. Remove emotional rhetoric, dramatic framing, and inflammatory decoration. Do not substitute or sanitize specific terms — if the original claim uses a particular word or phrase, preserve it unless it is purely emotional amplification with no factual content. One sentence only.",
  "Quick Explanation": "Write exactly four labeled lines. Each line starts with its label followed by a colon. Line 1 — ONE-LINE READ: One sentence that creates friction between what the record shows and what the claim asserts. When the claim uses charged terminology like monarchy, authoritarian, dictator, king, fascist, coup, or similar, explicitly signal whether current conditions meet the legal and historical threshold for that term. When historical precedent is relevant, briefly signal the standard being applied. Line 2 — WHAT HOLDS UP: One sentence leading with the core documented government action, institutional condition, or constitutional reality that most directly bears on the claim. For civic or constitutional claims, prioritize executive actions, court constraints, congressional behavior, elections, enforcement, and historical comparisons before any public reaction, protest description, turnout, slogan, or movement detail. Line 3 — WHAT IS DISPUTED: One sentence on what remains contested, including whether the charged term meets its real definitional threshold and whether the current conditions are historically comparable to stronger or weaker precedents. Line 4 — WHERE AGREEMENT EXISTS: One sentence identifying narrow genuine common ground, or state plainly that none exists. No bullet points. No dashes. Plain language only. No preamble before the first label.",
  "Speaker": "Who made the claim, or Unknown if not specified.",
  "Topic": "Exactly one of: Iran War, Energy, Healthcare, Social Security, Medicare, Medicaid, Defense, Military, Elections, Economy, Immigration, Foreign Policy, Crime, Gender Issues, Constitutional Rights, Education, Other",
  "Sub Claims": [
    {"claim": "First distinct falsifiable claim within the statement", "verdict": "True"},
    {"claim": "Second distinct falsifiable claim", "verdict": "Contested"},
    {"claim": "Third distinct falsifiable claim if present", "verdict": "Unproven"}
  ],
  "Direct Facts": "What the documented record actually shows. 3 to 4 sentences of prose in plain English. For civic, constitutional, or executive power claims, lead with the underlying government action, institutional condition, legal constraint, court activity, congressional role, enforcement reality, and historical comparison. Do not lead with protest size, crowd behavior, slogans, or movement messaging unless the claim is specifically about those.",
  "Adjacent Facts": "What the claim omits or ignores on BOTH sides equally. 2 to 3 sentences of prose in plain English. For civic or constitutional claims, surface omitted institutional context, historical precedent, delegated authority, court involvement, congressional passivity, and definitional thresholds before discussing movement reaction or public rhetoric.",
  "Root Concern": "The legitimate underlying concern that exists even beneath a false or misleading claim. 1 to 2 sentences of prose in plain English. For civic or constitutional claims, identify the deeper concern in terms of power, institutional failure, rights, accountability, delegated authority, or civic thresholds, not the surrounding protest language or emotional framing.",
  "Values Divergence": "Where the real disagreement lives. Usually not in the facts themselves but in what people prioritize. 2 to 3 sentences of prose identifying the competing values in plain English. For civic claims, this often includes liberty versus order, restraint versus urgency, branch independence versus executive efficiency, and public distrust versus institutional continuity.",
  "Constitutional Framework": "If the claim touches government action, rights, authority, public funds, war, law enforcement, elections, or institutional power, identify the specific Article, Section, or Amendment that applies and explain relevant founder intent in plain English. If not applicable, explain briefly why not.",
  "Common Ground": "Layer 06. Identify the narrow but genuine overlap between opposing sides. 2 to 3 sentences of prose in plain English.",

  "Civic Role Quick View": "One short sentence only. This is a preview of the Civic Role section. It should briefly explain what kind of thinking or judgment this claim tests in a self governing country. Do not give instructions. Do not use commands. Do not mention political sides. Keep it simple, clear, and readable.",

  "Civic Role": "Provide three short structured lines, not bullet points and not a paragraph. Use this exact format:

  What this tests:
  One short sentence explaining what kind of thinking or judgment this claim challenges.

  What people should separate:
  One short sentence explaining what people should clearly distinguish instead of mixing together.

  Why this matters:
  One short sentence explaining why this is important in a self governing country.

  Rules:
  Do not use commands like should, must, need to, or have to.
  Do not tell the reader what action to take.
  Do not mention political sides.
  Keep each line simple, clear, and easy to read.",

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
  "Scenario Map": "MANDATORY. Populate exactly this structure in plain English. The Scenario Map must evaluate institutional and civic trajectories, not protest momentum, crowd size, slogans, or media energy unless those directly change constitutional or political outcomes. MOST SUPPORTED OUTCOME: [Short title]. [2 to 3 sentences explaining the outcome best supported by current constitutional conditions, institutional behavior, and historical precedent. Focus on whether courts, Congress, elections, agencies, and executive power continue to function within recognizable limits.] ALTERNATIVE OUTCOME 2: [Short title]. [2 to 3 sentences explaining a less likely but still plausible path driven by a meaningful shift in judicial behavior, congressional behavior, executive escalation, or public response thresholds.] ALTERNATIVE OUTCOME 3: [Short title]. [2 to 3 sentences explaining another less likely but still plausible path, especially if institutional weakness, branch passivity, or breakdown in enforcement changes the situation.] PREFERRED OUTCOME: [Short title]. [2 to 3 sentences describing the best realistic path for preserving liberty, accountability, and constitutional balance through functioning institutions rather than rhetorical escalation.] CONFIDENCE NOTE: [1 to 2 sentences explaining which constitutional, historical, and institutional facts most strongly support the ranking, and what concrete developments could reorder it.] For civic or constitutional claims, prioritize: whether institutional checks are holding, whether the behavior is historically normal or an outlier, whether thresholds for authoritarian or monarch-like rule are actually met, and whether the deeper problem is executive overreach or legislative and judicial failure to carry their own responsibilities. End with NOTE: These are plausible trajectories only. Not predictions. Only actions and time determine the actual path.",
  "Glossary": [
    {"term": "A term, acronym, agency, company, program, court case, founder reference, or concept that general readers may not recognize", "definition": "Plain language definition in 1 to 2 sentences."},
    {"term": "Another term", "definition": "Plain language definition."},
    {"term": "A third term", "definition": "Plain language definition."}
  ],
  "Sources": "Primary sources:\\nSource description one: https://url-one.com\\nSource description two: https://url-two.com\\nSource description three: https://url-three.com\\nSource description four: https://url-four.com\\nSource description five: https://url-five.com\\n\\nInclude 6 to 10 real, verifiable URLs from major news outlets, government sites, institutional bodies, or authoritative sources. Format each line exactly as: Label: URL",
  "Overall Verdict": "Exactly one of: True, Mostly True, Substantially True, Plausible/Mixed, Contested, Exaggerated, Misleading, Unproven, False",
  "Strip Mode Summary": "This is the full paid analytical paragraph, not the short Quick Explanation. Write this in the spirit of Thomas Paine's Common Sense — plain language, no jargon, no hedging, accessible to anyone regardless of political background or education level. Answer three things clearly: what is actually happening beneath the claim, why it matters in real terms, and what someone should be paying attention to next. This should reflect the full excavation without referencing the layers directly. 3 to 4 sentences. No bullet points. No dashes. Do not be condescending. Do not be sarcastic. Avoid phrases like obviously or clearly. Do not over-explain uncertainty, but do not present future outcomes as guaranteed. Let the tone reflect that situations evolve. Write with calm, grounded clarity.",
  "Analyzed Claim Candidate": "Optional. After completing the full analysis, if the excavation reveals that the foundational premise of this claim is more precisely or accurately stated than the original input suggests, provide a single clean sentence here that captures that foundational premise. This is a suggestion only — it is never automatically authoritative. Leave blank if the original framing is already accurate. Do not simply restate the Stripped Claim. This field exists for superuser review and manual override decisions only."
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

def normalize_claim_text(text):
    if not text:
        return ""

    text = text.strip()

    # Normalize quotes
    text = text.replace('“', '"').replace('”', '"')

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove repeated punctuation
    text = re.sub(r'([!?.,])\1+', r'\1', text)

    return text.strip()


def compute_framing_hash(framing_obj):
    canonical_str = json.dumps(framing_obj, sort_keys=True)
    return hashlib.sha256(canonical_str.encode()).hexdigest()

def _clean_line_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip())


def parse_quick_explanation_lines(text):
    """
    Parse the legacy four-line Quick Explanation blob into atomic fields.
    Returns normalized keys even if some lines are missing.
    """
    result = {
        "one_line_read": "",
        "what_holds_up": "",
        "what_is_disputed": "",
        "where_agreement_exists": ""
    }

    if not text:
        return result

    label_map = {
        "ONE-LINE READ:": "one_line_read",
        "WHAT HOLDS UP:": "what_holds_up",
        "WHAT IS DISPUTED:": "what_is_disputed",
        "WHERE AGREEMENT EXISTS:": "where_agreement_exists"
    }

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for label, key in label_map.items():
            if line.startswith(label):
                result[key] = _clean_line_text(line[len(label):])
                break

    return result


def parse_civic_role_lines(text):
    """
    Parse Civic Role into atomic fields so render does not depend on one blob.
    """
    result = {
        "what_this_tests": "",
        "what_people_should_separate": "",
        "why_this_matters": ""
    }

    if not text:
        return result

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    current_key = None

    key_map = {
        "What this tests:": "what_this_tests",
        "What people should separate:": "what_people_should_separate",
        "Why this matters:": "why_this_matters"
    }

    for line in lines:
        if line in key_map:
            current_key = key_map[line]
            continue

        if current_key:
            if result[current_key]:
                result[current_key] += " " + _clean_line_text(line)
            else:
                result[current_key] = _clean_line_text(line)

    return result


def build_quick_view_contract(parsed):
    """
    Authoritative Quick View contract.
    Build once at save/reanalysis time and persist atomic fields.
    """
    parsed = parsed or {}

    quick_blob = (parsed.get("Quick Explanation") or "").strip()
    quick_parts = parse_quick_explanation_lines(quick_blob)

    if not quick_parts["one_line_read"]:
        quick_parts["one_line_read"] = _clean_line_text(parsed.get("Stripped Claim") or "")

    if not quick_parts["what_holds_up"]:
        direct = _clean_line_text(parsed.get("Direct Facts") or "")
        if direct:
            quick_parts["what_holds_up"] = direct.split(". ")[0].strip()
            if quick_parts["what_holds_up"] and not quick_parts["what_holds_up"].endswith("."):
                quick_parts["what_holds_up"] += "."

    if not quick_parts["what_is_disputed"]:
        disputed = _clean_line_text(parsed.get("Adjacent Facts") or parsed.get("Root Concern") or "")
        if disputed:
            quick_parts["what_is_disputed"] = disputed.split(". ")[0].strip()
            if quick_parts["what_is_disputed"] and not quick_parts["what_is_disputed"].endswith("."):
                quick_parts["what_is_disputed"] += "."

    if not quick_parts["where_agreement_exists"]:
        agreement = _clean_line_text(parsed.get("Common Ground") or "")
        if agreement:
            quick_parts["where_agreement_exists"] = agreement.split(". ")[0].strip()
            if quick_parts["where_agreement_exists"] and not quick_parts["where_agreement_exists"].endswith("."):
                quick_parts["where_agreement_exists"] += "."

    quick_blob_rebuilt = "\n".join([
        f"ONE-LINE READ: {quick_parts['one_line_read']}".strip(),
        f"WHAT HOLDS UP: {quick_parts['what_holds_up']}".strip(),
        f"WHAT IS DISPUTED: {quick_parts['what_is_disputed']}".strip(),
        f"WHERE AGREEMENT EXISTS: {quick_parts['where_agreement_exists']}".strip()
    ])

    return {
        "quick_explanation": quick_blob_rebuilt,
        "quick_one_line_read": quick_parts["one_line_read"],
        "quick_what_holds_up": quick_parts["what_holds_up"],
        "quick_what_is_disputed": quick_parts["what_is_disputed"],
        "quick_where_agreement_exists": quick_parts["where_agreement_exists"]
    }


def build_civic_role_contract(parsed):
    """
    Authoritative Civic Role contract.
    """
    parsed = parsed or {}

    civic_role_quick_view = _clean_line_text(parsed.get("Civic Role Quick View") or "")
    civic_role_full = (parsed.get("Civic Role") or "").strip()
    civic_parts = parse_civic_role_lines(civic_role_full)

    return {
        "civic_role_quick_view": civic_role_quick_view,
        "civic_role": civic_role_full,
        "civic_what_this_tests": civic_parts["what_this_tests"],
        "civic_what_people_should_separate": civic_parts["what_people_should_separate"],
        "civic_why_this_matters": civic_parts["why_this_matters"]
    }

def detect_input_type(text):
    if len(text) > 500:
        return "long_form"
    if "?" in text:
        return "question"
    return "statement"


def detect_claim_type(claim):
    if not claim:
        return "unknown"

    c = claim.lower()

    civic_terms = [
        "constitutional", "constitution", "rights", "government", "authority",
        "executive", "president", "congress", "court", "judicial", "legislative",
        "monarch", "monarchy", "authoritarian", "dictator", "election", "due process",
        "powers", "amendment", "federal", "state", "institution", "institutional"
    ]

    economic_terms = [
        "tax", "inflation", "economy", "economic", "wages", "jobs", "price", "cost of living"
    ]

    scientific_terms = [
        "science", "medical", "disease", "virus", "study", "evidence", "research", "clinical"
    ]

    historical_terms = [
        "history", "historical", "founders", "founder", "precedent", "war", "civil war"
    ]

    if any(k in c for k in civic_terms):
        return "civic"
    if any(k in c for k in economic_terms):
        return "economic"
    if any(k in c for k in scientific_terms):
        return "scientific"
    if any(k in c for k in historical_terms):
        return "historical"

    return "general"


def extract_root_concern(claim):
    text = (claim or "").strip()
    lowered = text.lower()

    if any(k in lowered for k in ["monarch", "monarchy", "authoritarian", "dictator", "king"]):
        return "Whether executive power is being exercised beyond constitutional limits and whether institutional checks are still holding."

    if any(k in lowered for k in ["tax", "wealth", "rich", "billionaire", "economy"]):
        return "Whether public burdens and public benefits are being distributed fairly and whether institutions are serving the public rather than entrenched interests."

    if any(k in lowered for k in ["court", "rights", "amendment", "government", "authority", "executive", "congress"]):
        return "Whether constitutional structure, institutional accountability, and protected rights are being maintained in practice."

    return text

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
    # Claude remains the source of truth for structure.
    for raw_field in ["Claude Raw JSON", "OpenAI Raw JSON"]:
        raw = fields.get(raw_field)
        if raw:
            parsed = safe_json_parse(raw)
            if isinstance(parsed, dict) and parsed and "raw" not in parsed:
                return parsed
    return {}

def repair_quick_explanation(quick_explanation, parsed_json):
    """
    Ensure Quick Explanation always contains the four labeled lines
    required by the UI contract.
    """
    text = (quick_explanation or "").strip()

    labels = [
        "ONE-LINE READ:",
        "WHAT HOLDS UP:",
        "WHAT IS DISPUTED:",
        "WHERE AGREEMENT EXISTS:"
    ]

    # If the saved text already contains all required labels, keep it.
    if text and all(label in text for label in labels):
        return text

    # Try raw JSON first.
    raw_quick = (parsed_json.get("Quick Explanation") or "").strip()
    if raw_quick and all(label in raw_quick for label in labels):
        return raw_quick

    # Rebuild from available structured fields if needed.
    one_line_read = ""
    if raw_quick:
        for line in raw_quick.splitlines():
            line = line.strip()
            if line.startswith("ONE-LINE READ:"):
                one_line_read = line[len("ONE-LINE READ:"):].strip()
                break

    if not one_line_read:
        one_line_read = (parsed_json.get("Stripped Claim") or "The claim needs more complete analysis.").strip()

    what_holds_up = (parsed_json.get("Direct Facts") or "").strip()
    if what_holds_up:
        what_holds_up = what_holds_up.split(". ")[0].strip()
        if not what_holds_up.endswith("."):
            what_holds_up += "."
    else:
        what_holds_up = "The saved analysis does not currently preserve the strongest documented point."

    what_is_disputed = (parsed_json.get("Adjacent Facts") or parsed_json.get("Root Concern") or "").strip()
    if what_is_disputed:
        what_is_disputed = what_is_disputed.split(". ")[0].strip()
        if not what_is_disputed.endswith("."):
            what_is_disputed += "."
    else:
        what_is_disputed = "The main area of dispute was not preserved in the saved Quick View."

    where_agreement_exists = (parsed_json.get("Common Ground") or "").strip()
    if where_agreement_exists:
        where_agreement_exists = where_agreement_exists.split(". ")[0].strip()
        if not where_agreement_exists.endswith("."):
            where_agreement_exists += "."
    else:
        where_agreement_exists = "No clear common ground was preserved in the saved Quick View."

    return "\n".join([
        f"ONE-LINE READ: {one_line_read}",
        f"WHAT HOLDS UP: {what_holds_up}",
        f"WHAT IS DISPUTED: {what_is_disputed}",
        f"WHERE AGREEMENT EXISTS: {where_agreement_exists}"
    ])

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
    title = clean_display_title(fields.get("Analyzed Claim") or fields.get("Original Quote") or fields.get("Stripped Claim") or "Untitled Claim")

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

    quick_explanation_raw = fields.get("Quick Explanation", "") or parsed_json.get("Quick Explanation", "")
    quick_explanation = quick_explanation_raw

    if not quick_explanation.strip():
        # fallback only for legacy records
        quick_explanation = repair_quick_explanation(quick_explanation_raw, parsed_json)

    quick_one_line_read = (fields.get("Quick One-Line Read") or "").strip()
    quick_what_holds_up = (fields.get("Quick What Holds Up") or "").strip()
    quick_what_is_disputed = (fields.get("Quick What Is Disputed") or "").strip()
    quick_where_agreement_exists = (fields.get("Quick Where Agreement Exists") or "").strip()

    missing_quick_fields = not all([
        quick_one_line_read,
        quick_what_holds_up,
        quick_what_is_disputed,
        quick_where_agreement_exists
    ])

    if missing_quick_fields:
        legacy_quick_parts = parse_quick_explanation_lines(quick_explanation)

        if not quick_one_line_read:
            quick_one_line_read = legacy_quick_parts.get("one_line_read", "")

        if not quick_what_holds_up:
            quick_what_holds_up = legacy_quick_parts.get("what_holds_up", "")

        if not quick_what_is_disputed:
            quick_what_is_disputed = legacy_quick_parts.get("what_is_disputed", "")

        if not quick_where_agreement_exists:
            quick_where_agreement_exists = legacy_quick_parts.get("where_agreement_exists", "")

    stripped_claim = fields.get("Stripped Claim", "") or parsed_json.get("Stripped Claim", "")
    overall_verdict = fields.get("Overall Verdict", "Unproven") or parsed_json.get("Overall Verdict", "Unproven")
    subclaims = build_subclaims(fields, parsed_json)

    direct_facts = fields.get("Direct Facts", "") or parsed_json.get("Direct Facts", "")
    adjacent_facts = fields.get("Adjacent Facts", "") or parsed_json.get("Adjacent Facts", "")
    root_concern = fields.get("Root Concern", "") or parsed_json.get("Root Concern", "")
    values_divergence = fields.get("Values Divergence", "") or parsed_json.get("Values Divergence", "")
    constitutional_framework = fields.get("Constitutional Framework", "") or parsed_json.get("Constitutional Framework", "")
    common_ground = fields.get("Common Ground", "") or parsed_json.get("Common Ground", "")

    civic_role_quick_view = (fields.get("Civic Role Quick View") or parsed_json.get("Civic Role Quick View", "") or "").strip()
    civic_role = (fields.get("Civic Role") or parsed_json.get("Civic Role", "") or "").strip()
    civic_what_this_tests = (fields.get("Civic What This Tests") or "").strip()
    civic_what_people_should_separate = (fields.get("Civic What People Should Separate") or "").strip()
    civic_why_this_matters = (fields.get("Civic Why This Matters") or "").strip()

    missing_civic_fields = not all([
        civic_what_this_tests,
        civic_what_people_should_separate,
        civic_why_this_matters
    ])

    if missing_civic_fields:
        legacy_civic_parts = parse_civic_role_lines(civic_role)

        if not civic_what_this_tests:
            civic_what_this_tests = legacy_civic_parts.get("what_this_tests", "")

        if not civic_what_people_should_separate:
            civic_what_people_should_separate = legacy_civic_parts.get("what_people_should_separate", "")

        if not civic_why_this_matters:
            civic_why_this_matters = legacy_civic_parts.get("why_this_matters", "")

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
        "one_line_read": quick_one_line_read,
        "what_holds_up": quick_what_holds_up,
        "what_is_disputed": quick_what_is_disputed,
        "where_agreement_exists": quick_where_agreement_exists,
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
        "civic_role": civic_role,
        "civic_what_this_tests": civic_what_this_tests,
        "civic_what_people_should_separate": civic_what_people_should_separate,
        "civic_why_this_matters": civic_why_this_matters,
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
        "original_quote": fields.get("Original Quote", ""),
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
        "civic_role_quick_view": civic_role_quick_view,
        "civic_role": civic_role,
        "quick_one_line_read": quick_one_line_read,
        "quick_what_holds_up": quick_what_holds_up,
        "quick_what_is_disputed": quick_what_is_disputed,
        "quick_where_agreement_exists": quick_where_agreement_exists,
        "civic_what_this_tests": civic_what_this_tests,
        "civic_what_people_should_separate": civic_what_people_should_separate,
        "civic_why_this_matters": civic_why_this_matters,        
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


def extract_primary_record_fields(claim, parsed, mode, username, existing_fields=None, framing_data=None):
    """
    Build the Airtable field dict for a claim record.
    NOTE: This function does NOT write Analyzed Claim, URL Slug, Title Locked,
    Slug Locked, Title Source, or Analysis Core Version. Those identity/version
    fields are managed explicitly by the caller (analyze route or reanalysis routes)
    to enforce the versioned logic gate architecture.
    """
    dates = now_dates()
    existing_fields = existing_fields or {}
    parsed = parsed or {}
    is_full_reexcavate = mode == "full"

    if existing_fields and not is_full_reexcavate:
        human_reviewed_value = existing_fields.get("Human Reviewed", False)
        published_value = existing_fields.get("Published", False)
    else:
        human_reviewed_value = False
        published_value = False

    entered_by = existing_fields.get("Entered By") or username or "Unknown"

    quick_view_contract = build_quick_view_contract(parsed)
    civic_role_contract = build_civic_role_contract(parsed)

    fields = {
        "Original Quote": claim,
        "Stripped Claim": parsed.get("Stripped Claim", claim),
        "Quick Explanation": quick_view_contract["quick_explanation"],
        "Quick One-Line Read": quick_view_contract["quick_one_line_read"],
        "Quick What Holds Up": quick_view_contract["quick_what_holds_up"],
        "Quick What Is Disputed": quick_view_contract["quick_what_is_disputed"],
        "Quick Where Agreement Exists": quick_view_contract["quick_where_agreement_exists"],
        "Speaker": parsed.get("Speaker") or "Unknown",
        "Topic": [normalize_topic(parsed.get("Topic"))],
        "Human Reviewed": human_reviewed_value,
        "Published": published_value,
        "Status": "Active",
        "Mode": mode if mode in ["strip", "full"] else "strip",
        "Date": dates["display_date"],
        "Date Added": existing_fields.get("Date Added", dates["short_date"]),
        "Last Updated": dates["short_date"],
        "Last Reanalyzed": dates["short_date"],
        "Entered By": entered_by,
        "Civic Role Quick View": civic_role_contract["civic_role_quick_view"],
        "Civic Role": civic_role_contract["civic_role"],
        "Civic What This Tests": civic_role_contract["civic_what_this_tests"],
        "Civic What People Should Separate": civic_role_contract["civic_what_people_should_separate"],
        "Civic Why This Matters": civic_role_contract["civic_why_this_matters"]
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

    if framing_data and isinstance(framing_data, dict):
        if framing_data.get("input_type"):
            fields["Input Type"] = framing_data["input_type"]

        if framing_data.get("confidence_score") is not None:
            try:
                fields["Framing Confidence"] = float(framing_data["confidence_score"])
            except Exception:
                pass

    candidate = (parsed.get("Analyzed Claim Candidate") or "").strip()
    if candidate:
        fields["Analyzed Claim Candidate"] = candidate

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
                "title": clean_display_title(f.get("Analyzed Claim") or f.get("Original Quote") or f.get("Stripped Claim") or "Untitled Claim"),
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

def get_trending_claims(limit=10):
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        return []
    try:
        params = {
            "maxRecords": limit,
            "filterByFormula": "AND(OR(NOT({Breakout User Excavated}), {Breakout User Excavated}!='No'), {View Count}>0)",
            "sort[0][field]": "View Count",
            "sort[0][direction]": "desc",
            "sort[1][field]": "Date Added",
            "sort[1][direction]": "desc"
        }
        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        trending = []
        for record in records[:limit]:
            f = record.get("fields", {})
            trending.append({
                "title": clean_display_title(
                    f.get("Analyzed Claim") or f.get("Original Quote") or f.get("Stripped Claim") or "Untitled Claim"
                ),
                "slug": f.get("URL Slug", ""),
                "date": f.get("Date") or f.get("Date Added", ""),
                "verdict": f.get("Overall Verdict", "Unproven"),
                "topics": parse_topics(f.get("Topic")),
                "speaker": f.get("Speaker", "Unknown"),
                "entered_by": f.get("Entered By", ""),
                "view_count": int(f.get("View Count", 0) or 0)
            })
        return trending
    except Exception as e:
        print("TRENDING CLAIMS ERROR:", str(e), flush=True)
        return []

def get_top_trending_claim():
    if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        return None
    try:
        params = {
            "maxRecords": 1,
            "filterByFormula": "AND(OR(NOT({Breakout User Excavated}), {Breakout User Excavated}!='No'), {View Count}>0)",
            "sort[0][field]": "View Count",
            "sort[0][direction]": "desc",
            "sort[1][field]": "Date Added",
            "sort[1][direction]": "desc"
        }
        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        return records[0] if records else None
    except Exception as e:
        print("TOP TRENDING CLAIM ERROR:", str(e), flush=True)
        return None

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
                "title": clean_display_title(f.get("Analyzed Claim") or f.get("Original Quote") or f.get("Stripped Claim") or "Untitled Claim"),
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

Civic role quick view:
{claim_context.get('civic_role_quick_view', '')}

Civic role:
{claim_context.get('civic_role', '')}

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

Civic role quick view:
{claim_context.get('civic_role_quick_view', '')}

Civic role:
{claim_context.get('civic_role', '')}

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
    """Increment view count once per logged-in user session per claim. Excludes true superusers only."""
    if not session.get("logged_in"):
        return jsonify({"ok": False}), 200

    if session.get("true_superuser"):
        return jsonify({"ok": False, "reason": "true_superuser"}), 200

    try:
        if not slug:
            return jsonify({"ok": False}), 200

        viewed_claims = session.get("viewed_claim_slugs", [])
        if not isinstance(viewed_claims, list):
            viewed_claims = []

        if slug in viewed_claims:
            return jsonify({"ok": False, "reason": "already_counted_this_session"}), 200

        record = get_claim_by_slug(slug)
        if not record:
            return jsonify({"ok": False}), 200

        current = int(record.get("fields", {}).get("View Count", 0) or 0)
        new_count = current + 1

        update_airtable_record(record["id"], {"View Count": new_count})

        viewed_claims.append(slug)
        session["viewed_claim_slugs"] = viewed_claims
        session.modified = True

        return jsonify({"ok": True, "view_count": new_count}), 200

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
                "title": clean_display_title(f.get("Analyzed Claim") or f.get("Original Quote") or f.get("Stripped Claim") or "Untitled"),
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


def get_site_settings():
    """Read site mode from Settings table. Defaults to Live on any failure."""
    try:
        resp = requests.get(
            airtable_url(AIRTABLE_SETTINGS_TABLE_NAME),
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            params={"filterByFormula": "{Setting Key}='site_mode'", "maxRecords": 1},
            timeout=10
        )
        resp.raise_for_status()
        records = resp.json().get("records", [])
        if records:
            f = records[0].get("fields", {})
            return {
                "site_mode": (f.get("Site Mode") or "Live").strip(),
                "message_title": (f.get("Message Title") or "").strip(),
                "message_body": (f.get("Message Body") or "").strip()
            }
    except Exception as e:
        print(f"SETTINGS READ ERROR: {e}", flush=True)
    return {"site_mode": "Live", "message_title": "", "message_body": ""}


@app.route("/")
def home():
    if not session.get("logged_in"):
        return redirect("/login")
    trending_claims = get_trending_claims(limit=10)

    featured_record = get_top_trending_claim()

    # FALLBACK → if no views yet, use latest claim
    if not featured_record:
        featured_record = get_latest_claim()

    current_claim = build_claim_context(featured_record) if featured_record else None
    _s = get_site_settings()
    return render_template(
        "index.html",
        page_mode="claim",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=10),
        trending_claims=get_trending_claims(limit=10),
        current_claim=current_claim,
        archived_claims_by_topic={},
        selected_topic="",
        user_disputes=[],
        search_query="",
        search_results=[],
        claims_remaining=session.get("claims_remaining", 0),
        site_mode=_s["site_mode"],
        site_message_title=_s["message_title"],
        site_message_body=_s["message_body"]
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
    _s = get_site_settings()
    return render_template(
        "index.html",
        page_mode="archives",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=10),
        trending_claims=get_trending_claims(limit=10),
        current_claim=None,
        archived_claims_by_topic=filtered,
        selected_topic=topic,
        user_disputes=[],
        search_query="",
        search_results=[],
        claims_remaining=session.get("claims_remaining", 0),
        site_mode=_s["site_mode"],
        site_message_title=_s["message_title"],
        site_message_body=_s["message_body"]
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
    _s = get_site_settings()
    return render_template(
        "index.html",
        page_mode="claim",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=10),
        trending_claims=get_trending_claims(limit=10),
        current_claim=current_claim,
        archived_claims_by_topic={},
        selected_topic="",
        user_disputes=[],
        search_query="",
        search_results=[],
        claims_remaining=session.get("claims_remaining", 0),
        site_mode=_s["site_mode"],
        site_message_title=_s["message_title"],
        site_message_body=_s["message_body"]
    )
def disputes_page():
    if not session.get("logged_in"):
        return redirect("/login")
    username = session.get("username", "")
    user_disputes = get_disputes_for_user(username)
    _s = get_site_settings()
    return render_template(
        "index.html",
        page_mode="disputes",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=10),
        trending_claims=get_trending_claims(limit=10),
        current_claim=None,
        archived_claims_by_topic={},
        selected_topic="",
        user_disputes=user_disputes,
        search_query="",
        search_results=[],
        claims_remaining=session.get("claims_remaining", 0),
        site_mode=_s["site_mode"],
        site_message_title=_s["message_title"],
        site_message_body=_s["message_body"]
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
        results = [
            c for c in all_claims
            if query_lower in (c.get("title") or "").lower()
            or query_lower in (c.get("stripped_claim") or "").lower()
            or query_lower in (c.get("speaker") or "").lower()
        ]

    _s = get_site_settings()

    return render_template(
        "index.html",
        page_mode="search",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=10),
        trending_claims=get_trending_claims(limit=10),
        current_claim=None,
        archived_claims_by_topic={},
        selected_topic="",
        search_query=query,
        search_results=results,
        user_disputes=[],
        claims_remaining=session.get("claims_remaining", 0),
        site_mode=_s["site_mode"],
        site_message_title=_s["message_title"],
        site_message_body=_s["message_body"]
    )


@app.route("/profile")
def profile_page():
    if not session.get("logged_in"):
        return redirect("/login")

    _s = get_site_settings()

    return render_template(
        "index.html",
        page_mode="profile",
        superuser=session.get("superuser", False),
        true_superuser=session.get("true_superuser", False),
        recent_claims=get_recent_claims(limit=5),
        trending_claims=get_trending_claims(limit=10),
        current_claim=None,
        archived_claims_by_topic={},
        selected_topic="",
        user_disputes=[],
        search_query="",
        search_results=[],
        claims_remaining=session.get("claims_remaining", 0),
        site_mode=_s["site_mode"],
        site_message_title=_s["message_title"],
        site_message_body=_s["message_body"]
    )


def normalize_claim_text_for_dedup(text):
    """Normalize claim for duplicate comparison only."""
    import re as _re
    text = (text or "").lower().strip()
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


def find_duplicate_and_similar_claims(claim_text, threshold=0.30, max_similar=4, framing_data=None):
    """Check for exact and similar claims using canonical form + polarity guard."""
    result = {'exact': None, 'similar': []}
    if not claim_text or not AIRTABLE_TOKEN:
        return result

    # Use canonical form from framing if available, else normalize raw text
    if framing_data and framing_data.get('canonical_claim'):
        compare_text = framing_data['canonical_claim']
    else:
        compare_text = claim_text

    input_polarity = (framing_data or {}).get('polarity', '')
    input_claim_type = (framing_data or {}).get('claim_type', '')
    input_input_type = (framing_data or {}).get('input_type', '')

    norm_input = normalize_claim_text_for_dedup(compare_text)

    try:
        params = {
            'filterByFormula': "OR(NOT({Breakout User Excavated}), {Breakout User Excavated}!='No')",
            'fields[]': ['Original Quote', 'Stripped Claim', 'Analyzed Claim', 'URL Slug'],
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
        # Use Analyzed Claim for comparison if available, else Original Quote
        raw = f.get('Analyzed Claim') or f.get('Original Quote') or f.get('Stripped Claim') or ''
        if not raw:
            continue
        norm_existing = normalize_claim_text_for_dedup(raw)

        # Exact match check
        if norm_existing == norm_input:
            result['exact'] = {
                'record_id': rec.get('id'),
                'title': clean_display_title(f.get('Analyzed Claim') or f.get('Original Quote') or raw),
                'slug': f.get('URL Slug', '')
            }
            return result

        score = keyword_overlap_score(norm_input, norm_existing)
        if score >= threshold:
            scored.append({
                'record_id': rec.get('id'),
                'title': clean_display_title(f.get('Analyzed Claim') or f.get('Original Quote') or raw),
                'slug': f.get('URL Slug', ''),
                'score': round(score, 3)
            })

    # Sort by score
    scored.sort(key=lambda x: x['score'], reverse=True)
    result['similar'] = scored[:max_similar]
    return result


@app.route('/check-duplicate', methods=['POST'])
def check_duplicate():
    """Pre-creation duplicate/similar check — runs framing first, then dedup on canonical form."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json() or {}
    claim = (data.get('claim') or '').strip()
    if not claim:
        return jsonify({'exact': None, 'similar': []}), 200
    # Run framing to get canonical form and polarity for smarter dedup
    framing = frame_claim_input(claim)
    result = find_duplicate_and_similar_claims(claim, framing_data=framing)
    # Include framing data so the frontend can use it
    result['framing'] = framing
    return jsonify(result), 200


FRAME_CLAIM_PROMPT = """You are a pre-excavation claim framing engine for Where the Truth Lies.

Your job is to analyze raw user input BEFORE full excavation begins. Understand what the user actually intends to have analyzed — not just what they literally typed.

CRITICAL RULES:
1. Evaluate claims not writing ability. Slang, ebonics, dyslexic spelling, incomplete sentences, and informal grammar must all be interpreted for intent.
2. Identify the FOUNDATIONAL PREMISE first — the central animating assertion. Not downstream statements or supporting facts.
3. Rhetorical slogans and compressed narratives (like "No Kings", "Defund the Police", "Stop the Steal") must be recognized as compressed claims and unpacked to their foundational premise.
4. Sourced content, press releases, manifestos, and "About" page text must be treated as raw material — extract what the user most likely wants examined.
5. The system leads. Propose the primary framing. The user may adjust from system-generated options but cannot replace with an unrelated claim.

CIVIC MANIFESTO AND ABOUT PAGE RULES:
When the input is a movement page, manifesto, protest page, about page, open letter, grievance list, or long political narrative, do NOT frame the primary claim around turnout, slogans, emotional force, or supporter energy.
Extract the underlying assertion about government behavior, institutional conduct, constitutional limits, public harm, or civic justification.
If the text contains both a structural accusation and descriptive movement language, the structural accusation wins.
Examples:
A protest page describing a president as king like is not primarily a claim about protests. It is primarily a claim that the president is exercising power in a way comparable to monarchical or authoritarian rule.
A tax the rich style narrative is not primarily a claim about wealthy people existing. It is primarily a claim about fairness, burden sharing, institutional capture, or state allocation of resources.
A movement page can mention turnout, momentum, and reaction, but those belong in breakout_candidates or supporting context, never as the primary claim unless the actual dispute is about turnout itself.

PRIMARY CLAIM PRIORITY RULE:
When multiple layers are present, rank them in this order:
1. The structural accusation or central premise
2. The constitutional, institutional, civic, legal, or economic concern beneath it
3. The public reaction or justification argument
4. The supporting examples, numbers, slogans, and event descriptions

The primary_claim must come from level 1, and if needed level 2.
It must never come from level 3 or 4 unless the user is explicitly asking about those.

ACTOR RULES:
When input uses vague actors — "they", "them", "everyone", "the government" — do NOT convert to named entities or coordinated groups.
Use neutral grounded phrasing: "government policies", "public health authorities", "federal institutions", "elected officials".
Never invent specificity that is not in the input.

SCOPE RULES:
When input uses exaggerated scope — "controlled everything", "destroyed everything" — compress to scope-based language.
"controlled everything" becomes "extensive control over daily activities".
"destroyed the economy" becomes "caused significant economic harm".
Do not expand exaggeration into multi-domain lists.

MULTI-CLAIM RULES:
When input contains multiple claims across different domains:
Produce ONE high-level neutral primary claim capturing the overall pattern without tightly binding unrelated domains.
Push each separate domain to breakout_candidates, not the primary claim.
Example: "lockdowns destroyed businesses and forced vaccines and now they pretend it never happened" becomes primary: "Government COVID-19 policies included restrictive measures that had significant economic and social consequences." Accountability goes to breakout_candidates.

HARD PROHIBITION — primary_claim must NEVER contain:
Denial or minimization language: denied, minimized, pretended, ignored, covered up, avoided, dismissed, downplayed
Accountability or motive language: officials refuse to acknowledge, now pretend it never happened, avoiding responsibility, are denying
Narrative response framing: any clause about what actors did or said AFTER the primary policy or event occurred
When input contains both a policy or action claim AND a denial or accountability claim, stop the primary_claim at the policy and impact layer. The denial or accountability clause belongs in breakout_candidates only, never in primary_claim.

CLAIM TYPE RULES:
Preserve claim type — never convert between factual assertion, normative claim, and question.
If input is a question, the primary_claim must stay framed as an implied premise, not asserted as fact.
If input implies rather than asserts, signal this in primary_claim phrasing.
Example: "why is the US bombing Iran?" implies a premise. primary_claim: "The United States is currently conducting military operations against Iran." And set implied_premise: true.

CLAIM TYPE AND POLARITY RULES:
These are two separate classifications. Both must be returned.

claim_type — what kind of claim this is:
  factual: an assertion about what is or was true
  normative: an assertion about what should or should not happen
  inquiry: a question rather than an assertion

polarity — only applies to factual claims. What direction the factual assertion takes:
  affirming: asserts something is true or occurred ("lockdowns caused economic damage")
  rejecting: asserts something is false, unjust, or invalid ("the 2020 election was stolen")
  neutral: the claim does not clearly affirm or reject (neutral description or mixed)

For normative and inquiry claims, set polarity to "neutral" — polarity does not apply.

HARD DEDUP RULE: Claims with opposing polarity (one affirming, one rejecting the same proposition) must NEVER be treated as duplicates or similar matches, even if their canonical forms overlap.
Example: "COVID lockdowns were justified" (affirming) vs "COVID lockdowns were unconstitutional" (rejecting) — same topic, opposite polarity, never a dedup match.

Claims of different claim_type must also never be treated as duplicates.
Example: "The US is bombing Iran" (factual, affirming) vs "Why is the US bombing Iran?" (inquiry) — never a dedup match.

CANONICAL NORMALIZATION:
After framing, generate a canonical form by:
1. Stripping intensity modifiers: remove severe, catastrophic, primary, sustained, extreme, massive
2. Normalizing causality: destroyed/wrecked/caused major damage all become "caused economic harm"
3. Collapsing time scope unless time is core to claim: "during COVID", "long-term", "beyond restrictions" become neutral baseline
4. Stripping named actors when the claim works without them for comparison purposes
The canonical form is used for deduplication only. Never display it.

INPUT TYPES: single_claim, multi_claim, sourced_content, rhetorical_slogan, question, unclear
CONFIDENCE: 0.85+ auto proceed. 0.60-0.84 inline banner. Under 0.60 full modal.

Return ONLY valid JSON. No markdown. No preamble.

{
  "input_type": "single_claim",
  "primary_claim": "The foundational premise. One sentence.",
  "clarified_text": "Clean readable version preserving meaning.",
  "alternate_claims": ["Second interpretation", "Third if genuinely distinct"],
  "breakout_candidates": ["Distinct sub-claim that can stand alone"],
  "confidence_score": 0.0,
  "needs_clarification": false,
  "framing_note": "One sentence explaining the framing choice.",
  "canonical_claim": "Stripped normalized version for dedup comparison only. No intensity modifiers. Normalized causality. No named actors unless essential.",
  "claim_type": "factual | normative | inquiry",
  "polarity": "affirming | rejecting | neutral",
  "implied_premise": false
}
"""


def frame_claim_input(raw_input):
    fallback = {
        "input_type": "single_claim",
        "primary_claim": raw_input,
        "clarified_text": raw_input,
        "alternate_claims": [],
        "breakout_candidates": [],
        "confidence_score": 0.9,
        "needs_clarification": False,
        "framing_note": "",
        "canonical_claim": raw_input,
        "claim_type": "general",
        "polarity": "neutral",
        "implied_premise": False
    }
    if not raw_input or not anthropic_client:
        return fallback
    try:
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=800, temperature=0,
            system=FRAME_CLAIM_PROMPT,
            messages=[{"role": "user", "content": f"Frame this input:\n\n{raw_input}"}]
        )
        result = safe_json_parse(response.content[0].text)
        if not isinstance(result, dict) or "primary_claim" not in result:
            return fallback
        score = float(result.get("confidence_score", 0.9))
        if "needs_clarification" not in result:
            result["needs_clarification"] = score < 0.60

        # Post-processing guard: strip accountability or denial clauses from primary_claim
        # If the model leaks them into primary_claim, move them into breakout_candidates
        primary = str(result.get("primary_claim", "") or "").strip()
        accountability_signals = [
            "denied", "minimized", "pretended", "pretend", "ignored",
            "covered up", "avoided", "dismissed", "downplayed",
            "now pretend", "refuse to acknowledge", "never happened",
            "avoiding responsibility", "officials have minimized",
            "have downplayed", "are denying", "won't acknowledge"
        ]

        has_accountability = any(sig in primary.lower() for sig in accountability_signals)
        has_policy = any(w in primary.lower() for w in [
            "polic", "lockdown", "mandates", "restrictions", "government",
            "administration", "officials", "law", "order", "executive"
        ])

        if has_accountability and has_policy:
            import re as _re

            cleaned = _re.split(
                r'\s+and\s+(now|they|officials|those)\s+',
                primary,
                flags=_re.IGNORECASE
            )[0]
            cleaned = _re.split(
                r',\s+(while|but|yet)\s+\w+\s+(denied|minimized|pretended|ignored)',
                cleaned,
                flags=_re.IGNORECASE
            )[0]
            cleaned = cleaned.strip().rstrip(',').strip()

            if cleaned and len(cleaned) > 20 and cleaned != primary:
                accountability_clause = primary[len(cleaned):].strip()
                accountability_clause = _re.sub(
                    r'^(and|but|while|yet)\b',
                    '',
                    accountability_clause,
                    flags=_re.IGNORECASE
                ).strip()
                accountability_clause = accountability_clause.lstrip(',').strip()
                result["primary_claim"] = cleaned

                existing_breakouts = result.get("breakout_candidates", [])
                if not isinstance(existing_breakouts, list):
                    existing_breakouts = []

                if accountability_clause:
                    result["breakout_candidates"] = existing_breakouts + [accountability_clause]
        primary_claim = str(result.get("primary_claim", "") or "").strip()
        breakout_candidates = result.get("breakout_candidates", [])
        clarified_text = str(result.get("clarified_text", "") or "").strip()
        canonical_claim = str(result.get("canonical_claim", "") or "").strip()
        input_type = str(result.get("input_type", "") or "").strip() or detect_input_type(raw_input)
        claim_type = str(result.get("claim_type", "") or "").strip() or detect_claim_type(primary_claim)
        polarity = str(result.get("polarity", "") or "").strip() or "neutral"
        implied_premise = bool(result.get("implied_premise", False))
        confidence_score = float(result.get("confidence_score", 0.9) or 0.9)
        framing_note = str(result.get("framing_note", "") or "").strip()

        framing_obj = {
            "input_type": input_type,
            "claim_type": claim_type,
            "polarity": polarity,
            "primary_claim": primary_claim,
            "clarified_text": clarified_text,
            "canonical_claim": canonical_claim or primary_claim,
            "supporting_claims": breakout_candidates if isinstance(breakout_candidates, list) else [],
            "foundational_concern": extract_root_concern(primary_claim),
            "implied_premise": implied_premise,
            "confidence_score": confidence_score,
            "framing_note": framing_note,
            "framing_version": "1.1"
        }

        result["framing_obj"] = framing_obj
        return result
    except Exception as e:
        print(f"FRAME CLAIM ERROR: {e}", flush=True)
        return fallback


@app.route("/frame-claim", methods=["POST"])
def frame_claim():
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json() or {}
    raw_input = (data.get("claim") or "").strip()
    if not raw_input:
        return jsonify({"error": "Claim required"}), 400
    return jsonify(frame_claim_input(raw_input)), 200


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

            challenge_prompt = f"""You are the OpenAI challenge layer for Where the Truth Lies.

Your role is not to be more emotional, more activist, more punitive, or more dramatic than Claude.
Your role is to test Claude's analysis for overreach, missing context, weak framing, definitional drift, historical inconsistency, and unjustified certainty.

This system may receive claims from politics, law, science, medicine, technology, business, culture, history, media, art, philosophy, religion, race, gender, satire, dark humor, offensive rhetoric, nihilistic statements, disturbing content, or emotionally loaded personal and public claims.
Your job is not to moralize about the wording. Your job is to identify what is actually being asserted, what Claude handled well, what Claude may have overstated, and what meaningful context may be missing.
If a layer is not applicable to the claim type, do not invent relevance. Focus on the actual assertion and the strongest grounded alternative reading.

CRITICAL RULES:
Use plain English.
Do not use first person. Never write I, me, my, we, our, or us.
Do not sound like an activist, pundit, therapist, prosecutor, or partisan.
Do not intensify rhetoric.
Do not use emotionally loaded adjectives unless they are necessary and supported by the record.
Do not assume the most sinister interpretation when a narrower one is better supported.
Do not flatten structural context. When relevant, acknowledge historical precedent, institutional incentives, delegated authority, legal ambiguity, definitional thresholds, and the possibility that modern power expansions may be systemic rather than unique to one actor.
When the claim concerns executive power, democratic norms, constitutional conflict, or institutional overreach, explicitly consider whether Congress, the courts, administrative agencies, or long running bipartisan precedent contributed to the present condition.
If Claude uses a charged concept like authoritarian, monarchy, fascist, coup, insurrection, racist, sexist, genocidal, suicidal, or similar, test whether the definition is actually met instead of merely responding to the emotional force of the term.
If the claim is satirical, rhetorical, grief driven, offensive, racist, sexist, morbid, or psychologically loaded, identify the factual core if one exists. If no meaningful factual core exists, say so plainly rather than pretending the claim supports a full factual disagreement.
If Claude is directionally reasonable but incomplete, say so. Challenge overconfidence without manufacturing disagreement.

Claude has produced the following analysis of this claim:
CLAIM: "{claim}"

CLAUDE'S VERDICT: {claude_json.get("Overall Verdict", "Unknown")}
CLAUDE'S ONE-LINE READ: {claude_json.get("Quick Explanation", "")[:500]}
CLAUDE'S DIRECT FACTS: {claude_json.get("Direct Facts", "")[:500]}
CLAUDE'S COMMON GROUND: {claude_json.get("Common Ground", "")[:300]}

{grok_context_block}
Your job is to challenge Claude's analysis in a disciplined way. Return JSON only. No markdown fences. No preamble.

Return exactly this structure:
{{
  "where_claude_is_most_likely_wrong": "One sentence identifying the single most likely error, overreach, or unsupported assumption in Claude's analysis.",
  "what_claude_overstated": "One sentence on what Claude leaned too hard on, if anything. If nothing material was overstated, say that plainly.",
  "what_claude_missed": "One sentence on a meaningful context gap, definitional issue, structural factor, historical comparison, or alternative interpretation Claude did not address.",
  "strongest_alternative_interpretation": "One sentence describing the strongest credible alternative reading of the claim that remains grounded in the record.",
  "openai_verdict": "Exactly one of: True, Mostly True, Substantially True, Plausible/Mixed, Contested, Exaggerated, Misleading, Unproven, False",
  "divergence_note": "If your verdict differs from Claude's, write one calm, neutral sentence beginning with 'OpenAI and Claude differ on'. Attribute the disagreement explicitly to OpenAI and Claude. Describe the disagreement in terms of evidence weighting, scope, definition, historical comparison, institutional context, or degree of certainty. Do not use first person. Do not use vague phrases like more severe unless you specify what is being weighed more heavily. If aligned, write exactly: OpenAI and Claude are aligned on the core assessment."
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

            # ── Run framing to get Analyzed Claim and URL Slug ──────────────────
            # framing_data is the authoritative source for Analyzed Claim identity.
            # On re-submission of an existing record, preserve the stored identity
            # fields unless they are empty (first time or were never set).
            submission_framing = frame_claim_input(claim)
            primary_claim_title = (submission_framing or {}).get("primary_claim", "").strip()

            existing_analyzed = (existing_fields.get("Analyzed Claim") or "").strip()
            existing_slug = (existing_fields.get("URL Slug") or "").strip()
            is_new_record = not existing_record

            fields = extract_primary_record_fields(
                claim=claim,
                parsed=primary,
                mode=mode,
                username=session.get("username", "Unknown"),
                existing_fields=existing_fields,
                framing_data=submission_framing
            )

            # ── Identity fields: set on new records, preserve on existing ───────
            if is_new_record:
                # New submission — write all identity and version fields fresh
                analyzed_claim = primary_claim_title or claim
                url_slug = slugify(analyzed_claim)
                fields["Analyzed Claim"] = analyzed_claim
                fields["URL Slug"] = url_slug
                fields["Title Locked"] = False
                fields["Slug Locked"] = True
                fields["Title Source"] = "AI Initial"
                fields["Analysis Core Version"] = ANALYSIS_CORE_VERSION
            else:
                # Re-submission of existing record — preserve identity fields
                fields["Analyzed Claim"] = existing_analyzed or primary_claim_title or claim
                fields["URL Slug"] = existing_slug or slugify(existing_analyzed or primary_claim_title or claim)
                # Preserve lock fields if already set; do not overwrite
                if "Title Locked" not in existing_fields:
                    fields["Title Locked"] = False
                if "Slug Locked" not in existing_fields:
                    fields["Slug Locked"] = True
                if not existing_fields.get("Title Source"):
                    fields["Title Source"] = "AI Initial"
                if not existing_fields.get("Analysis Core Version"):
                    fields["Analysis Core Version"] = ANALYSIS_CORE_VERSION
                # Protect Quick View structure before save
                    parsed_for_repair = claude_json if isinstance(claude_json, dict) and claude_json else {}
                    fields["Quick Explanation"] = repair_quick_explanation(
                        fields.get("Quick Explanation", "") or parsed_for_repair.get("Quick Explanation", ""),
                        parsed_for_repair
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
    "Civic Role Quick View": "Civic Role Quick View",
    "Civic Role": "Civic Role",
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
    "Civic Role Quick View": claim_fields.get("Civic Role Quick View", ""),
    "Civic Role": claim_fields.get("Civic Role", ""),
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
    "Civic Role Quick View": "Civic Role Quick View",
    "Civic Role": "Civic Role",
    "Bottom Line": "Strip Mode Summary",
    "Left Perspective": "Left Perspective",
    "Right Perspective": "Right Perspective",
    "Scenario Map": "Scenario Map",
    "Sub Claims": "Sub-Claims",
}


def run_reanalysis_ai(claim_text, mode="full", canonical_anchor=None):
    """Run the AI excavation pipeline on an existing claim text. Returns merged parsed JSON."""
    reality_anchor, grok_adjudication = build_reality_anchor_with_grok(claim_text)

    canonical_block = ""
    if canonical_anchor and isinstance(canonical_anchor, dict):
        anchored_title = (canonical_anchor.get("analyzed_claim") or "").strip()
        anchored_stripped = (canonical_anchor.get("stripped_claim") or "").strip()
        anchored_quick = (canonical_anchor.get("quick_explanation") or "").strip()

        canonical_parts = []
        if anchored_title:
            canonical_parts.append(f"Accepted analyzed claim: {anchored_title}")
        if anchored_stripped:
            canonical_parts.append(f"Accepted stripped claim: {anchored_stripped}")
        if anchored_quick:
            canonical_parts.append(f"Accepted quick explanation: {anchored_quick}")

        if canonical_parts:
            canonical_block = (
                "CANONICAL CLAIM IDENTITY FOR FEATURE REFRESH:\n"
                + "\n".join(canonical_parts)
                + "\n\n"
                + "Treat the accepted claim identity above as fixed for this refresh. "
                + "Do not materially reframe, narrow, broaden, or replace it. "
                + "Refresh supporting layers around that accepted claim identity.\n\n"
            )

    prompt_text = (
        f"{reality_anchor}\n\n"
        f"{canonical_block}"
        f"Now analyze this claim:\n\"{claim_text}\""
    ).strip()

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
        if openai_client and "error" not in claude_json:
            challenge_prompt = f"""
You are reviewing an AI-generated claim analysis.

Your job is NOT to rewrite the analysis.

Your job is to:
- identify if the analysis overstates conclusions
- identify if it misses key counterpoints
- identify if it misinterprets the claim

Return ONLY JSON:

{{
  "agreement": "agree" or "partial" or "disagree",
  "issues": ["short list of issues if any"],
  "notes": "brief explanation"
}}

CLAIM:
{claim_text}

ANALYSIS:
{json.dumps(claude_json, ensure_ascii=False)}
"""
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a strict analytical reviewer."},
                    {"role": "user", "content": challenge_prompt}
                ],
                max_tokens=800,
                temperature=0
            )
            openai_json = safe_json_parse(resp.choices[0].message.content)
    except Exception as e:
        openai_json = {"error": str(e)}

    primary = claude_json if "error" not in claude_json else openai_json
    return primary, claude_json, openai_json, grok_adjudication


@app.route("/editor/update-site-mode", methods=["POST"])
def editor_update_site_mode():
    """Update Site Mode in existing Settings record. True superuser only."""
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    if not session.get("true_superuser"):
        return jsonify({"error": "Unauthorized"}), 403
    data = request.get_json() or {}
    new_mode = (data.get("site_mode") or "").strip()
    valid = ["Live", "Construction/ Site Maintenance", "Testing"]
    if new_mode not in valid:
        return jsonify({"error": f"Invalid mode. Must be one of: {valid}"}), 400
    try:
        params = {"filterByFormula": "{Setting Key}='site_mode'", "maxRecords": 1}
        resp = requests.get(
            airtable_url(AIRTABLE_SETTINGS_TABLE_NAME),
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            params=params, timeout=10
        )
        resp.raise_for_status()
        records = resp.json().get("records", [])
        if not records:
            return jsonify({"error": "Settings record not found"}), 404
        record_id = records[0]["id"]
        patch = requests.patch(
            f"{airtable_url(AIRTABLE_SETTINGS_TABLE_NAME)}/{record_id}",
            headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}", "Content-Type": "application/json"},
            json={"fields": {
                "Site Mode": new_mode,
                "Updated By": session.get("username", "Editor")
            }},
            timeout=10
        )
        if not patch.ok:
            return jsonify({"error": f"Airtable error: {patch.text}"}), 500
        return jsonify({"ok": True, "site_mode": new_mode}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
                "title": clean_display_title(f.get("Analyzed Claim") or f.get("Original Quote") or f.get("Stripped Claim") or "Untitled"),
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
    """
    Reanalyze a single claim by record ID.

    Reanalysis modes (passed in JSON body):
      reanalysis_mode: "feature_refresh" (default) | "core_logic_refresh"
      force: true  — override Title Locked on core_logic_refresh
      regenerate_slug: true — also regenerate URL Slug (only on core_logic_refresh,
                               blocked if Slug Locked unless force=true)

    feature_refresh:  Updates all analysis layers. Never touches Analyzed Claim,
                      URL Slug, Title Locked, Slug Locked, Title Source, or
                      Analysis Core Version.

    core_logic_refresh: Runs frame_claim_input() fresh from Original Quote, then
                        runs full analysis. May update Analyzed Claim and version
                        fields subject to lock rules.
    """
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    if not session.get("true_superuser"):
        return jsonify({"error": "Unauthorized"}), 403

    try:
        data = request.get_json() or {}
        reanalysis_mode = data.get("reanalysis_mode", "feature_refresh")
        selective_fields = data.get("fields", [])
        force = bool(data.get("force", False))
        regenerate_slug = bool(data.get("regenerate_slug", False))

        claim_record = get_claim_by_record_id(record_id)
        if not claim_record:
            return jsonify({"error": "Claim not found"}), 404

        claim_fields = claim_record.get("fields", {})
        raw_claim_text = claim_fields.get("Original Quote") or claim_fields.get("Stripped Claim") or ""
        if not raw_claim_text:
            return jsonify({"error": "No claim text found on this record"}), 400

        editor_username = session.get("username", "Editor")
        now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
        now_date = datetime.utcnow().strftime("%Y-%m-%d")

        canonical_anchor = None
        if reanalysis_mode == "feature_refresh":
            canonical_anchor = {
                "analyzed_claim": claim_fields.get("Analyzed Claim", ""),
                "stripped_claim": claim_fields.get("Stripped Claim", ""),
                "quick_explanation": claim_fields.get("Quick Explanation", "")
            }

        # Feature refresh uses canonical accepted framing.
        # Core logic refresh reruns without the canonical anchor.
        primary, claude_json, openai_json, grok_adjudication = run_reanalysis_ai(
            raw_claim_text,
            "full",
            canonical_anchor=canonical_anchor
        )
        if "error" in primary:
            return jsonify({"error": f"AI failed: {primary['error']}"}), 500

        # Feature refresh must NOT silently reframe the claim.
        # Preserve locked top-level framing fields in both visible outputs and raw JSON.
        preserve_stripped = (reanalysis_mode == "feature_refresh")
        preserve_quick = False

        existing_stripped = (claim_fields.get("Stripped Claim") or "").strip()
        existing_quick = (claim_fields.get("Quick Explanation") or "").strip()

        if preserve_stripped and existing_stripped:
            primary["Stripped Claim"] = existing_stripped
            if isinstance(claude_json, dict):
                claude_json["Stripped Claim"] = existing_stripped
            if isinstance(openai_json, dict):
                openai_json["Stripped Claim"] = existing_stripped


        # Build update payload
        if selective_fields:
            update_fields = {
                "Last Reanalyzed": now_date,
                "Reanalyzed By": editor_username,
                "Last Feature Refresh": now_iso
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
                        if len(sub_claims) > 1:
                            update_fields["Sub-Claim 2"] = sub_claims[1].get("claim", "")
                            if sub_claims[1].get("verdict"):
                                update_fields["Verdict: Sub-Claim 2"] = sub_claims[1]["verdict"]
                        if len(sub_claims) > 2:
                            update_fields["Sub-Claim 3"] = sub_claims[2].get("claim", "")
                            if sub_claims[2].get("verdict"):
                                update_fields["Verdict: Sub-Claim 3"] = sub_claims[2]["verdict"]

                elif field_name == "Overall Verdict":
                    v = primary.get("Overall Verdict") or primary.get("Verdict")
                    if v:
                        update_fields["Overall Verdict"] = v

                elif primary.get(field_name):
                    update_fields[airtable_key] = primary[field_name]

        else:
            update_fields = extract_primary_record_fields(
                claim=raw_claim_text,
                parsed=primary,
                mode="full",
                username=editor_username,
                existing_fields=claim_fields
            )
            update_fields["Claude Raw JSON"] = json.dumps(claude_json, ensure_ascii=False)[:100000]
            update_fields["OpenAI Raw JSON"] = json.dumps(openai_json, ensure_ascii=False)[:100000]
            if grok_adjudication:
                update_fields["Grok Raw JSON"] = json.dumps(grok_adjudication, ensure_ascii=False)[:100000]
            update_fields["Last Reanalyzed"] = now_date
            update_fields["Reanalyzed By"] = editor_username

            for protected in ["Analyzed Claim", "URL Slug", "Title Locked", "Slug Locked",
                              "Title Source", "Analysis Core Version"]:
                update_fields.pop(protected, None)

        title_updated = False
        slug_updated = False
        warning = None

        if reanalysis_mode == "feature_refresh":
            update_fields["Last Feature Refresh"] = now_iso

            # Preserve top-level framing stability fields during normal refresh
            # unless they were explicitly targeted via selective reanalysis.
            if not selective_fields:
                if claim_fields.get("Stripped Claim"):
                    update_fields["Stripped Claim"] = claim_fields.get("Stripped Claim", "")    

        elif reanalysis_mode == "core_logic_refresh":

            normalized_input = normalize_claim_text(raw_claim_text)

            new_framing = frame_claim_input(normalized_input)
            new_primary_claim = (new_framing or {}).get("primary_claim", "").strip()
            framing_obj = (new_framing or {}).get("framing_obj", {})

            new_hash = compute_framing_hash(framing_obj)
            old_hash = claim_fields.get("Canonical Framing Hash")

            # Always store framing object
            update_fields["Canonical Framing JSON"] = json.dumps(framing_obj, ensure_ascii=False)[:100000]
            update_fields["Framing Version"] = "1.0"

            title_locked = bool(claim_fields.get("Title Locked", False))
            slug_locked = bool(claim_fields.get("Slug Locked", True))

            if not old_hash:
                # First run → establish canonical baseline
                update_fields["Canonical Framing Hash"] = new_hash

                if new_primary_claim:
                    if not title_locked or force:
                        update_fields["Analyzed Claim"] = new_primary_claim
                        update_fields["Analysis Core Version"] = ANALYSIS_CORE_VERSION
                        update_fields["Last Core Logic Refresh"] = now_iso
                        update_fields["Title Source"] = "AI Core Refresh"
                        title_updated = True
                    else:
                        warning = "Title locked. Cannot set initial canonical claim."

            else:
                if new_hash == old_hash:
                    # Stable → no drift
                    update_fields["Last Core Logic Refresh"] = now_iso

                else:
                    # Drift detected → DO NOT overwrite
                    update_fields["Pending Framing Candidate"] = new_primary_claim
                    warning = (
                        "Framing drift detected. New candidate stored instead of overwriting canonical claim."
                    )

            # Slug regeneration (only if allowed)
            if regenerate_slug:
                if slug_locked and not force:
                    warning = (warning or "") + " Slug locked."
                else:
                    slug_source = update_fields.get("Analyzed Claim") or new_primary_claim or raw_claim_text
                    update_fields["URL Slug"] = slugify(slug_source)
                    slug_updated = True

        resp = update_airtable_record(record_id, update_fields)
        if not resp.ok:
            return jsonify({"error": f"Airtable update failed: {resp.text}"}), 500

        result = {
            "ok": True,
            "record_id": record_id,
            "reanalysis_mode": reanalysis_mode,
            "fields_updated": list(update_fields.keys()),
            "reanalyzed_by": editor_username,
            "last_reanalyzed": now_date,
            "title_updated": title_updated,
            "slug_updated": slug_updated,
            "slug": update_fields.get("URL Slug", claim_fields.get("URL Slug", ""))
        }
        if warning:
            result["warning"] = warning.strip()
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/editor/reanalyze-claim-by-slug/<slug>", methods=["POST"])
def editor_reanalyze_claim_by_slug(slug):
    """
    Reanalyze a single claim from its claim detail page (by slug).

    Identical logic to /editor/reanalyze-claim/<record_id> — resolves the record
    by slug then applies the same versioned logic gate. Always runs AI against
    Original Quote, never against Analyzed Claim.

    Reanalysis modes (passed in JSON body):
      reanalysis_mode: "feature_refresh" (default) | "core_logic_refresh"
      force: true  — override Title Locked / Slug Locked on core_logic_refresh
      regenerate_slug: true — also regenerate URL Slug (core_logic_refresh only)
    """
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401
    if not session.get("true_superuser"):
        return jsonify({"error": "Unauthorized"}), 403

    try:
        data = request.get_json() or {}
        reanalysis_mode = data.get("reanalysis_mode", "feature_refresh")
        selective_fields = data.get("fields", [])
        force = bool(data.get("force", False))
        regenerate_slug = bool(data.get("regenerate_slug", False))

        safe_slug = escape_airtable_formula_value(slug)
        params = {"filterByFormula": f"{{URL Slug}}='{safe_slug}'", "maxRecords": 1}
        records = airtable_get_all(AIRTABLE_TABLE_NAME, params=params)
        if not records:
            return jsonify({"error": "Claim not found"}), 404

        record_id = records[0].get("id")
        claim_fields = records[0].get("fields", {})
        raw_claim_text = claim_fields.get("Original Quote") or claim_fields.get("Stripped Claim") or ""
        if not raw_claim_text:
            return jsonify({"error": "No claim text found"}), 400

        editor_username = session.get("username", "Editor")
        now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
        now_date = datetime.utcnow().strftime("%Y-%m-%d")

        canonical_anchor = None
        if reanalysis_mode == "feature_refresh":
            canonical_anchor = {
                "analyzed_claim": claim_fields.get("Analyzed Claim", ""),
                "stripped_claim": claim_fields.get("Stripped Claim", ""),
                "quick_explanation": claim_fields.get("Quick Explanation", "")
            }

        # Feature refresh uses canonical accepted framing.
        # Core logic refresh reruns without the canonical anchor.
        primary, claude_json, openai_json, grok_adjudication = run_reanalysis_ai(
            raw_claim_text,
            "full",
            canonical_anchor=canonical_anchor
        )
        if "error" in primary:
            return jsonify({"error": f"AI failed: {primary['error']}"}), 500

        # Feature refresh must NOT silently reframe the claim.
        # Preserve locked top-level framing fields in both visible outputs and raw JSON.
        preserve_stripped = (reanalysis_mode == "feature_refresh")
        preserve_quick = False

        existing_stripped = (claim_fields.get("Stripped Claim") or "").strip()
        existing_quick = (claim_fields.get("Quick Explanation") or "").strip()

        if preserve_stripped and existing_stripped:
            primary["Stripped Claim"] = existing_stripped
            if isinstance(claude_json, dict):
                claude_json["Stripped Claim"] = existing_stripped
            if isinstance(openai_json, dict):
                openai_json["Stripped Claim"] = existing_stripped

        # Build update payload
        if selective_fields:
            update_fields = {
                "Last Reanalyzed": now_date,
                "Reanalyzed By": editor_username,
                "Last Feature Refresh": now_iso
            }

            for field_name in selective_fields:
                airtable_key = SELECTIVE_FIELD_MAP.get(field_name)
                if not airtable_key:
                    continue

                if field_name == "Sub Claims":
                    sub_claims = primary.get("Sub Claims", [])
                    if isinstance(sub_claims, list):
                        update_fields["Sub-Claims"] = " | ".join(
                            [sc.get("claim", "") for sc in sub_claims if sc.get("claim")]
                        )
                        if len(sub_claims) > 0:
                            update_fields["Sub-Claim 1"] = sub_claims[0].get("claim", "")
                            if sub_claims[0].get("verdict"):
                                update_fields["Verdict: Sub-Claim1"] = sub_claims[0]["verdict"]
                        if len(sub_claims) > 1:
                            update_fields["Sub-Claim 2"] = sub_claims[1].get("claim", "")
                            if sub_claims[1].get("verdict"):
                                update_fields["Verdict: Sub-Claim 2"] = sub_claims[1]["verdict"]
                        if len(sub_claims) > 2:
                            update_fields["Sub-Claim 3"] = sub_claims[2].get("claim", "")
                            if sub_claims[2].get("verdict"):
                                update_fields["Verdict: Sub-Claim 3"] = sub_claims[2]["verdict"]

                elif field_name == "Overall Verdict":
                    v = primary.get("Overall Verdict") or primary.get("Verdict")
                    if v:
                        update_fields["Overall Verdict"] = v

                elif primary.get(field_name):
                    update_fields[airtable_key] = primary[field_name]

        else:
            update_fields = extract_primary_record_fields(
                claim=raw_claim_text,
                parsed=primary,
                mode="full",
                username=editor_username,
                existing_fields=claim_fields
            )
        if not selective_fields:
            update_fields["Claude Raw JSON"] = json.dumps(claude_json, ensure_ascii=False)[:100000]
            update_fields["OpenAI Raw JSON"] = json.dumps(openai_json, ensure_ascii=False)[:100000]
            if grok_adjudication:
                update_fields["Grok Raw JSON"] = json.dumps(grok_adjudication, ensure_ascii=False)[:100000]
            update_fields["Last Reanalyzed"] = now_date
            update_fields["Reanalyzed By"] = editor_username

            # Remove any identity fields that may have leaked in
            for protected in ["Analyzed Claim", "URL Slug", "Title Locked", "Slug Locked",
                              "Title Source", "Analysis Core Version"]:
                update_fields.pop(protected, None)

        title_updated = False
        slug_updated = False
        old_slug = slug
        new_slug = old_slug
        warning = None

        if reanalysis_mode == "feature_refresh":
            update_fields["Last Feature Refresh"] = now_iso

            # Preserve top-level framing stability fields during normal refresh
            # unless they were explicitly targeted via selective reanalysis.
            if not selective_fields:
                if claim_fields.get("Stripped Claim"):
                    update_fields["Stripped Claim"] = claim_fields.get("Stripped Claim", "")
        elif reanalysis_mode == "core_logic_refresh":

            normalized_input = normalize_claim_text(raw_claim_text)

            new_framing = frame_claim_input(normalized_input)
            new_primary_claim = (new_framing or {}).get("primary_claim", "").strip()
            framing_obj = (new_framing or {}).get("framing_obj", {})

            new_hash = compute_framing_hash(framing_obj)
            old_hash = claim_fields.get("Canonical Framing Hash")

            # Always store framing object
            update_fields["Canonical Framing JSON"] = json.dumps(framing_obj, ensure_ascii=False)[:100000]
            update_fields["Framing Version"] = "1.0"

            title_locked = bool(claim_fields.get("Title Locked", False))
            slug_locked = bool(claim_fields.get("Slug Locked", True))

            if not old_hash:
                # First run → establish canonical baseline
                update_fields["Canonical Framing Hash"] = new_hash

                if new_primary_claim:
                    if not title_locked or force:
                        update_fields["Analyzed Claim"] = new_primary_claim
                        update_fields["Analysis Core Version"] = ANALYSIS_CORE_VERSION
                        update_fields["Last Core Logic Refresh"] = now_iso
                        update_fields["Title Source"] = "AI Core Refresh"
                        title_updated = True
                    else:
                        warning = "Title locked. Cannot set initial canonical claim."

            else:
                if new_hash == old_hash:
                    # Stable → no drift
                    update_fields["Last Core Logic Refresh"] = now_iso

                else:
                    # Drift detected → DO NOT overwrite
                    update_fields["Pending Framing Candidate"] = new_primary_claim
                    warning = (
                        "Framing drift detected. New candidate stored instead of overwriting canonical claim."
                    )

            if regenerate_slug:
                if slug_locked and not force:
                    warning = (warning or "") + " Slug locked."
                else:
                    slug_source = update_fields.get("Analyzed Claim") or new_primary_claim or raw_claim_text
                    new_slug = slugify(slug_source)
                    update_fields["URL Slug"] = new_slug
                    slug_updated = True

        resp = update_airtable_record(record_id, update_fields)
        if not resp.ok:
            return jsonify({"error": f"Airtable update failed: {resp.text}"}), 500

        slug_changed = new_slug != old_slug
        result = {
            "ok": True,
            "record_id": record_id,
            "reanalysis_mode": reanalysis_mode,
            "fields_updated": list(update_fields.keys()),
            "reanalyzed_by": editor_username,
            "last_reanalyzed": now_date,
            "title_updated": title_updated,
            "slug_updated": slug_updated,
            "new_slug": new_slug,
            "slug_changed": slug_changed,
            "redirect_to": f"/claim/{new_slug}" if slug_changed else None
        }
        if warning:
            result["warning"] = warning.strip()
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ══════════════════════════════════════════════
# BREAKOUT CLAIMS SYSTEM
# ══════════════════════════════════════════════

BREAKOUT_DETECTION_SYSTEM = """You are the Breakout Claim detection engine for Where the Truth Lies.

Your job is to identify EXCAVATION-WORTHY breakout claims — branches that deserve their own full analysis, separate from the primary claim being examined.

This is a two-step process. Do not skip step one.

STEP ONE — EXTRACTION:
Identify all distinct, falsifiable assertions in the text. Note each one's actor, domain, and the type of evidence and adjudication it would require.

STEP TWO — COLLAPSING:
Before creating any breakout claim, apply this rule: when multiple extracted assertions share the same actor, same domain, same underlying accusation, and would require substantially the same evidence and adjudication logic — collapse them into ONE representative breakout claim that captures the pattern.

A representative breakout claim is a single sentence that synthesizes the pattern AND carries the tension of what is being tested. It is NOT a list. It is NOT a header. It stands alone as a testable assertion that signals what the excavation will examine.

Examples of correct collapsing:
Multiple immigration enforcement assertions (targeting families, detentions without warrants, profiling) → ONE breakout: "The Trump administration is conducting immigration enforcement operations that critics argue violate due process protections — and whether those operations fall within existing legal authority is disputed."
Multiple election integrity assertions (rigging maps, threatening elections, suppressing voters) → ONE breakout: "The Trump administration is taking actions that critics argue threaten electoral integrity and voter access — and whether those actions cross a constitutional or statutory line remains contested."
Multiple spending assertions (missile strikes, billionaire giveaways) → ONE breakout: "The administration is pursuing spending and economic decisions that critics argue systematically favor wealthy interests — and whether those decisions represent legitimate policy or improper favoritism is disputed."

Keep separate ONLY when claims would require meaningfully different evidence, different legal standards, or different historical comparisons to adjudicate.

A breakout claim must:
- Be independently analyzable against a factual record without relying on the primary claim
- Represent a genuinely distinct branch of investigation
- Carry the tension of what is being tested, not just describe a category

A breakout claim must NOT be:
- A restatement of the primary claim
- A general opinion or value judgment without factual content
- An event description or narrative moment (e.g. "the parade was drowned out") — observations are not claims
- A description of what happened rather than an assertion about what is true
- Rhetorical framing without a falsifiable factual core
- One of several nearly identical assertions that should be collapsed into one

Before finalizing each breakout claim, ask: would analyzing this separately produce meaningfully different evidence, reasoning, or conclusions than analyzing it as part of the main claim? If no, merge it or exclude it.

Target 3 to 5 strong grouped breakout claims. Only exceed 5 if the text contains genuinely unrelated topics requiring different adjudication paths. Never exceed 7.

Return ONLY valid JSON. No markdown fences. No preamble.

{
  "has_breakouts": true or false,
  "breakout_claims": [
    {
      "title": "One sentence representative claim capturing the pattern and its tension. Plain language. No hyphens or dashes.",
      "source_text": "The key assertion from the source text that triggered this breakout.",
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
Confidence should be between 0.5 and 1.0. Only include claims above 0.6 confidence.
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
    # For breakout detection, use the richest available source text
    # Original Quote has the full assertions; supplement with analyzed content
    original_quote = fields.get("Original Quote") or ""
    stripped_claim = fields.get("Stripped Claim") or ""
    direct_facts = fields.get("Direct Facts") or ""
    sub_claims_raw = fields.get("Sub-Claims") or ""

    # Build detection text: original quote is primary (has all assertions)
    # Fall back to stripped claim + direct facts if no original quote
    if original_quote and len(original_quote) > 50:
        claim_text = original_quote
    elif stripped_claim:
        # Combine stripped claim with direct facts for richer detection
        parts = [stripped_claim]
        if direct_facts:
            parts.append(direct_facts)
        if sub_claims_raw:
            parts.append(sub_claims_raw)
        claim_text = " ".join(parts)
    else:
        claim_text = ""
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
                "title": clean_display_title(f.get("Analyzed Claim") or f.get("Stripped Claim") or f.get("Original Quote") or "Untitled"),
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
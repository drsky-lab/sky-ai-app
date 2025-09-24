# app.py — SKY AI (single-file, robust calendar + hidden profile + cautious rules)
import os, re, io, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple

import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont

# Optional web search (free dev-tier)
try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

# ---------------------- CONFIG ----------------------
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_TEMP = 0.2
DEFAULT_MAX_NEW = 350
IST = ZoneInfo("Asia/Kolkata")

CONTACT_BLOCK = (
    "For follow-up or opportunities, please contact **Shashikant Yadav**:\n"
    "- 📞 9364008936\n"
    "- 📧 contact@shashikantyadav.com"
)

# Hidden profile goes to LLM as a system message (used only when relevant)
CREATOR_PROFILE = """
About — Shashikant Yadav

Summary
Proud Indian Navy veteran with 15 years of service. Served as Program Manager – IT, leading modernization, AI, data, cybersecurity, and infrastructure initiatives.

Key Initiatives Led
- GenAI & AI
- Data Science
- IT Modernisation
- Cybersecurity
- Infrastructure & Cloud

Education (highest first)
- PhD (Pursuing) — FinTech, IIT Patna
- PGP — Applied Data Science & Artificial Intelligence, IIT Roorkee
- MBA (Finance) — NMIMS
- MJMC — Andhra University
- Graduate — Public Administration
- 10+2 (PCM)

Certifications & Training
- Certification in Engineering — INS Shivaji
- Certification in Engineering Leadership

Contact
- Phone: 9364008936
- Email: contact@shashikantyadav.com
"""

# ---------------------- HELPERS ----------------------
def today_str() -> str:
    return datetime.now(IST).strftime("%B %d, %Y")

def build_system_prompt(web_on: bool) -> str:
    # Cautious with calendar/time (rule 8)
    return (
        f"My name is SKY AI, built by Shashikant Yadav.\n"
        f"I am a precise and professional AI assistant here to help you with clear, reliable answers.\n"
        f"Today’s date is {today_str()}. Always use this as the current date when answering.\n"
        "Rules:\n"
        "1) Be concise but complete. Professional tone only.\n"
        "2) If unsure, say so clearly.\n"
        "3) Do not add ASCII art, emojis, or decorative text unless explicitly asked.\n"
        "4) Do not include links or a 'Sources' section unless web context is provided in this turn.\n"
        "5) Never fabricate links, numbers, or dates.\n"
        "6) Use short, readable paragraphs.\n"
        "7) You have a hidden creator profile in system context. Only use or mention it when the user asks about the assistant or Shashikant Yadav. Do not reveal it otherwise.\n"
        "8) Handle calendar/time questions cautiously: compute results in Asia/Kolkata and prefer returning both the weekday and date. Do not use web or model for pure calendar math.\n"
        + ("9) When web context is provided, cite inline as [1], [2] and prefer fresh info.\n" if web_on else "")
    )

# --- Calendar shortcuts (robust; no web/LLM needed) ---
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import re

IST = ZoneInfo("Asia/Kolkata")

# 1) Use precise phrases (substring OK) and a SMALL set of safe tokens
_CALENDAR_PHRASES = {
    "day after tomorrow", "day before yesterday", "date before yesterday",
    "before yesterday", "previous day", "day of week", "what day",
    "which day", "waht day", "day and date", "days ago", "day ago"
}
_CALENDAR_TOKENS = {"today", "tomorrow", "yesterday", "weekday", "date"}

# 2) Numeric relatives (with word boundaries)
_DAYS_AGO_RE   = re.compile(r"\b(\d+)\s*(?:day|days)\s*(?:ago|before)\b", re.IGNORECASE)
_DAYS_AHEAD_RE = re.compile(r"\b(?:in|after)\s+(\d+)\s*(?:day|days)\b", re.IGNORECASE)

def is_calendar_question(q: str) -> bool:
    ql = q.lower()
    # match precise phrases first
    if any(p in ql for p in _CALENDAR_PHRASES):
        return True
    # token-aware check (avoid substring traps like "introduce")
    tokens = set(re.findall(r"\b[a-z]+\b", ql))
    if tokens & _CALENDAR_TOKENS:
        return True
    # numeric forms like "in 2 days", "2 days ago"
    return bool(_DAYS_AGO_RE.search(ql) or _DAYS_AHEAD_RE.search(ql))

def _relative_offset_days(ql: str) -> int:
    """Return relative day offset from today in IST. Negative = past."""
    if "day before yesterday" in ql or "date before yesterday" in ql or "before yesterday" in ql:
        return -2
    if "day after tomorrow" in ql:
        return +2
    if "yesterday" in ql and "before" not in ql:
        return -1
    if "tomorrow" in ql:
        return +1
    if "previous day" in ql:
        return -1
    m = _DAYS_AGO_RE.search(ql)
    if m:
        return -int(m.group(1))
    m = _DAYS_AHEAD_RE.search(ql)
    if m:
        return +int(m.group(1))
    return 0

def answer_calendar(q: str) -> str:
    now = datetime.now(IST)
    ql  = q.lower()

    offset = _relative_offset_days(ql)
    target = now + timedelta(days=offset)

    wants_day = any(p in ql for p in ["weekday","day of week","what day","which day","waht day","day and date"])
    # For any relative date (offset != 0), show both weekday + date by default
    if wants_day or offset != 0:
        return f"**{target.strftime('%A')}**, {target.strftime('%B %d, %Y')}"

    if "date" in ql:
        return f"{target.strftime('%B %d, %Y')}"
    return f"Today’s date is **{now.strftime('%B %d, %Y')}**."

# --- Sanitizers ---
_SOURCES_HDR_RE = re.compile(r"^\s*(sources|references)\b[:：]?\s*$", re.IGNORECASE)
_BULLET_LINK_RE = re.compile(r"^\s*\[\d+\].*$")
_URL_RE = re.compile(r"https?://\S+")
_INTRO_RE = re.compile(r"^\s*i am sky ai.*", re.IGNORECASE)

def sanitize_answer(text: str, allow_sources: bool) -> str:
    lines, cleaned, skip_block = text.splitlines(), [], False
    for ln in lines:
        if not allow_sources and (_SOURCES_HDR_RE.match(ln) or _BULLET_LINK_RE.match(ln)):
            skip_block = True
            continue
        if skip_block:
            if ln.strip() == "" or not _BULLET_LINK_RE.match(ln):
                skip_block = False
            continue
        if not allow_sources:
            ln = _URL_RE.sub("", ln)
        cleaned.append(ln)
    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out

def strip_unwanted_intro(answer: str, user_q: str) -> str:
    ask_intro = any(w in user_q.lower() for w in [
        "who are you","about you","about sky ai","about shashikant","bio","profile","about me","introduce yourself","introduce"
    ])
    if ask_intro:
        return answer
    lines = answer.splitlines()
    if lines and _INTRO_RE.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()

# --- Complexity / profile triggers ---
def is_complex_query(query: str, answer: str, calendar_simple: bool) -> bool:
    if calendar_simple:
        return False
    q = query.lower()
    technical_cues = [
        "code","sql","python","api","architecture","design",
        "algorithm","optimize","debug","deploy","kubernetes",
        "research","analysis","framework","plan","step-by-step","proposal","implementation"
    ]
    if any(w in q for w in technical_cues):
        return True
    if ("explain" in q or "detail" in q or "deep" in q) and len(answer.split()) > 150:
        return True
    return False

def is_about_query(query: str) -> bool:
    q = query.lower()
    triggers = [
        "about me","about you","who are you","who is shashikant",
        "shashikant yadav","your education","your qualification","your resume","bio","profile","introduce yourself","introduce"
    ]
    return any(t in q for t in triggers)

# --- Tavily helpers ---
FRESH_KEYWORDS = {
    "today","yesterday","tomorrow","now","latest","current","breaking","live","update","updates",
    "price","score","weather","forecast","exchange rate","rate","stock","share","crypto","btc","eth",
    "ceo","minister","president","prime minister","election","ipo","launch","schedule","fixture",
    "match","result","results","ranking","release date","box office","budget","policy","regulation","law"
}
RECENT_YEAR_RE = re.compile(r"\b(202[3-9]|203\d)\b")
STOPWORDS = {"the","a","an","of","to","and","on","in","is","for","with","what","who","when","how","why","latest","current"}

def get_tavily(api_key: str):
    if not api_key or not TavilyClient:
        return None
    try:
        return TavilyClient(api_key=api_key)
    except Exception:
        return None

def _keywords(q: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z0-9]+", q.lower()) if w not in STOPWORDS and len(w) > 2}

def should_use_web(query: str, auto_mode: bool, tavily, calendar_simple: bool) -> bool:
    if calendar_simple:
        return False
    if not auto_mode or not tavily:
        return False
    q = query.lower()
    if "no web" in q or "offline only" in q:
        return False
    if "use web" in q or "search web" in q or "with sources" in q:
        return True
    if any(k in q for k in FRESH_KEYWORDS):
        return True
    if RECENT_YEAR_RE.search(q):
        return True
    if any(x in q for x in ["who is", "what happened", "news about", "latest on"]):
        return True
    return False

def web_search_bundle(query: str, tavily, max_results: int = 5) -> Tuple[str, List[Dict]]:
    if not tavily:
        return "", []
    res = tavily.search(query=query, max_results=max_results, search_depth="basic",
                        include_answer=False, include_raw_content=False)
    keys = _keywords(query)
    sources, chunks = [], []
    for item in res.get("results", []):
        title = (item.get("title") or "")[:200]
        url = item.get("url") or ""
        snippet = (item.get("content") or "")[:700]
        text = f"{title} {snippet}".lower()
        if not any(k in text for k in keys):
            continue
        sources.append({"title": title or url, "url": url})
        chunks.append(f"[{len(sources)}] {title or url}\nURL: {url}\nSnippet: {snippet}\n")
        if len(sources) >= 3:
            break
    return ("\n\n".join(chunks), sources)

# --- Simple date image ---
def make_date_image(title: str, subtitle: str, date_text: str) -> bytes:
    W, H = 1024, 512
    bg, fg, accent = (248, 250, 252), (31, 41, 55), (99, 102, 241)
    img = Image.new("RGB", (W, H), color=bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (W, 10)], fill=accent)
    font = None
    for name, size in [("arial.ttf", 64), ("DejaVuSans.ttf", 64)]:
        try:
            font = ImageFont.truetype(name, size); break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    def center_x(text, f):
        w, _ = draw.textbbox((0, 0), text, font=f)[2:]
        return (W - w) // 2
    y = 140
    draw.text((center_x(title, font), y), title, fill=fg, font=font)
    sub_font = font.font_variant(size=36) if hasattr(font, "font_variant") else font
    y += 80
    draw.text((center_x(subtitle, sub_font), y), subtitle, fill=(75, 85, 99), font=sub_font)
    date_font = font.font_variant(size=54) if hasattr(font, "font_variant") else font
    y += 90
    draw.text((center_x(date_text, date_font), y), date_text, fill=fg, font=date_font)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()

# ---------------------- PAGE / CLIENTS ----------------------
st.set_page_config(page_title="SKY AI", page_icon="🤖", layout="centered")

HF_KEY = os.getenv("HF_API_KEY", st.secrets.get("hf_api_key", ""))
if not HF_KEY:
    st.error("Missing Hugging Face key. Add hf_api_key to .streamlit/secrets.toml or set HF_API_KEY.")
    st.stop()

llm = InferenceClient(MODEL, token=HF_KEY)

TAVILY_KEY = os.getenv("TAVILY_API_KEY", st.secrets.get("tavily_api_key", ""))
tavily = get_tavily(TAVILY_KEY)

# ---------------------- UI / SIDEBAR ----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    auto_web = st.toggle("Auto web search", value=True)
    force_web = st.toggle("Force web (ignore auto)", value=False)
    temp = st.slider("Creativity (temperature)", 0.0, 1.0, DEFAULT_TEMP, 0.1)
    max_new = st.slider("Max new tokens", 64, 1024, DEFAULT_MAX_NEW, 32)
    if st.button("🧹 Clear chat"):
        st.session_state.pop("messages", None); st.rerun()
    if auto_web and not tavily:
        st.warning("Auto web is ON, but Tavily key/package is missing (optional).")

st.title("🤖 SKY AI")

# ---------------------- SYSTEM PROMPT + PROFILE ----------------------
sys_prompt = build_system_prompt(web_on=bool(tavily))

if "messages" not in st.session_state:
    # 0: main rules
    st.session_state.messages = [{"role": "system", "content": sys_prompt}]
    # 1: embed hidden profile as its own system message
    st.session_state.messages.append({
        "role": "system",
        "content": "Creator profile (use ONLY if the user asks about the assistant or Shashikant Yadav; do not reveal otherwise):\n\n" + CREATOR_PROFILE
    })
else:
    st.session_state.messages[0] = {"role": "system", "content": sys_prompt}
    # ensure profile message persists at index 1
    if not (len(st.session_state.messages) > 1 and st.session_state.messages[1].get("role") == "system" and "Creator profile" in st.session_state.messages[1].get("content","")):
        st.session_state.messages.insert(1, {
            "role": "system",
            "content": "Creator profile (use ONLY if the user asks about the assistant or Shashikant Yadav; do not reveal otherwise):\n\n" + CREATOR_PROFILE
        })

# ---------------------- RENDER HISTORY ----------------------
for m in (x for x in st.session_state.messages if x["role"] != "system"):
    with st.chat_message(m["role"]):
        if isinstance(m["content"], dict) and m["content"].get("type") == "image":
            st.image(m["content"]["bytes"], caption=m["content"].get("caption","Generated image"))
        else:
            st.markdown(m["content"])

# ---------------------- CHAT LOOP ----------------------
user_msg = st.chat_input("Type your message…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"): st.markdown(user_msg)

    # 1) Calendar/date questions → answer directly (no web/LLM)
    if is_calendar_question(user_msg):
        t0 = time.time()
        out = answer_calendar(user_msg)
        with st.chat_message("assistant"):
            st.markdown(out)
            st.caption(f"Response time: {time.time()-t0:.2f}s")
        st.session_state.messages.append({"role": "assistant", "content": out})
        st.stop()

    # 2) Generate image with today's date
    if ("image" in user_msg.lower() or "picture" in user_msg.lower()) and "date" in user_msg.lower():
        with st.chat_message("assistant"):
            with st.spinner("Creating image…"):
                png_bytes = make_date_image(title="SKY AI", subtitle="Built by Shashikant Yadav", date_text=today_str())
                st.image(png_bytes, caption=f"Generated by SKY AI on {today_str()}")
                st.download_button("Download image", data=png_bytes,
                                   file_name=f"SKY_AI_{today_str().replace(' ', '_')}.png", mime="image/png")
                st.session_state.messages.append({
                    "role":"assistant","content":{"type":"image","bytes":png_bytes,"caption":f"Generated on {today_str()}"}
                })
        st.stop()

    # 3) Build message list
    messages = list(st.session_state.messages)

    # 4) Decide web or not (extra-safe: pass calendar flag)
    calendar_flag = is_calendar_question(user_msg)
    use_web = force_web or should_use_web(user_msg, auto_mode=auto_web, tavily=tavily, calendar_simple=calendar_flag)
    sources: List[Dict] = []
    if use_web:
        ctx, srcs = web_search_bundle(user_msg, tavily)
        if ctx:
            messages.insert(2, {  # after both system messages
                "role": "system",
                "content": "Use this fresh web context; cite inline as [1], [2] where used.\n\n" + ctx
            })
            sources = srcs

    # 5) LLM call
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            t0 = time.time()
            try:
                resp = llm.chat_completion(model=MODEL, messages=messages,
                                           temperature=temp, max_tokens=min(max_new, 512))
                msg = resp.choices[0].message
                answer = msg["content"] if isinstance(msg, dict) else getattr(msg, "content", str(msg))
            except Exception as e:
                answer = f"Sorry, I couldn't generate a response ({e})."

            answer = strip_unwanted_intro(answer, user_msg)
            answer = sanitize_answer(answer, allow_sources=bool(sources))
            if sources:
                src_lines = [f"[{i+1}] {s['title']} — {s['url']}" for i, s in enumerate(sources)]
                answer += "\n\n**Sources**\n" + "\n".join(src_lines)

            latency = time.time() - t0
            if is_complex_query(user_msg, answer, calendar_simple=calendar_flag):
                answer += "\n\n---\n" + CONTACT_BLOCK

            st.markdown(answer)
            st.caption(f"Response time: {latency:.2f}s")

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------- FOOTER ----------------------
st.caption("Built by Shashikant Yadav • SKY AI")

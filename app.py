"""
Streamlit UI for the AI Video Assistant pipeline — "Cinematic Editorial" theme.

Run with:
    streamlit run streamlit_app.py
"""

import re
import os
import json
import urllib.request
import urllib.parse
import urllib.error

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="Reel — AI Video Assistant",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def extract_youtube_video_id(url: str) -> str | None:
    patterns = [
        r"(?:youtube\.com/watch\?v=)([^&]+)",
        r"(?:youtu\.be/)([^?&]+)",
        r"(?:youtube\.com/shorts/)([^?&]+)",
        r"(?:youtube\.com/live/)([^?&]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


def get_youtube_transcript(url: str, language: str) -> str:
    video_id = extract_youtube_video_id(url)

    if not video_id:
        raise ValueError("Invalid YouTube URL.")

    api_key = st.secrets.get("SUPADATA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "SUPADATA_API_KEY is not configured.\n\n"
            "Please add your Supadata API key to Streamlit "
            "Cloud Secrets."
        )

    encoded_url = urllib.parse.quote(url, safe="")

    api_url = (
        "https://api.supadata.ai/v1/transcript"
        f"?url={encoded_url}"
    )

    request = urllib.request.Request(
        api_url,
        headers={
            "x-api-key": api_key,
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as e:

        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            error_body = ""

        raise RuntimeError(
            f"Supadata transcript request failed "
            f"(HTTP {e.code}).\n\n"
            f"{error_body}"
        )

    except urllib.error.URLError as e:

        raise RuntimeError(
            "Could not connect to the transcript service.\n\n"
            f"Details: {e.reason}"
        )

    except Exception as e:

        raise RuntimeError(
            "Could not retrieve the YouTube transcript.\n\n"
            f"Details: {e}"
        )

    # Supadata can return transcript content
    # either as a string or as timestamped segments.

    content = data.get("content")

    if isinstance(content, str):
        transcript = content.strip()

    elif isinstance(content, list):

        transcript = " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        ).strip()

    else:
        transcript = ""

    if not transcript:
        raise RuntimeError(
            "Supadata returned an empty transcript.\n\n"
            "This video may not be accessible or "
            "could not be transcribed."
        )

    return transcript
# ─────────────────────────────────────────────────────────────────────────────
# THEME — "Cinematic Editorial": near-black + film-amber + a teal accent,
# a serif display face paired with a clean grotesk body and mono for data.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0b0b0d;
    --bg-2: #131316;
    --panel: #17171b;
    --panel-2: #1e1e23;
    --line: #2b2b31;
    --amber: #e8a33d;
    --amber-soft: rgba(232,163,61,0.14);
    --teal: #37c9b0;
    --teal-soft: rgba(55,201,176,0.12);
    --ink: #ece9e2;
    --ink-dim: #9c9a97;
    --danger: #e2634f;
}

html, body, [class*="css"] { background-color: var(--bg) !important; color: var(--ink) !important; }
.stApp { background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(232,163,61,0.06), transparent), var(--bg) !important; }

* { font-family: 'Space Grotesk', sans-serif; }
h1, h2, h3, .display { font-family: 'Fraunces', serif !important; }
code, .mono, .stTextArea textarea, .stCodeBlock, pre { font-family: 'JetBrains Mono', monospace !important; }

/* film-strip top rule */
.filmrule {
    height: 6px;
    width: 100%;
    background: repeating-linear-gradient(90deg, var(--amber) 0 14px, transparent 14px 26px);
    opacity: 0.55;
    border-radius: 3px;
    margin-bottom: 1.6rem;
}

/* ── Hero ── */
.hero-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.4rem;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: clamp(2.1rem, 4.4vw, 3.6rem);
    line-height: 1.05;
    color: var(--ink);
    margin: 0;
}
.hero-title em { color: var(--amber); font-style: italic; }
.hero-sub {
    color: var(--ink-dim);
    font-size: 0.95rem;
    margin-top: 0.6rem;
    max-width: 560px;
    line-height: 1.6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-2) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }
.side-brand {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.side-brand span { color: var(--amber); }
.side-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ink-dim);
    margin-bottom: 1.4rem;
}

/* stepper */
.step-row { display:flex; align-items:center; gap:0.65rem; padding: 0.4rem 0; font-size: 0.82rem; }
.step-icon {
    width: 20px; height: 20px; border-radius: 50%;
    display:flex; align-items:center; justify-content:center;
    font-size: 0.65rem; flex-shrink:0; font-family:'JetBrains Mono',monospace;
    border: 1px solid var(--line);
}
.step-done  { background: var(--teal-soft); border-color: var(--teal); color: var(--teal); }
.step-active{ background: var(--amber-soft); border-color: var(--amber); color: var(--amber); animation: blink 1.3s infinite; }
.step-idle  { color: var(--ink-dim); }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.35;} }
.step-label-done   { color: var(--ink); }
.step-label-active { color: var(--amber); }
.step-label-idle   { color: var(--ink-dim); }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea textarea {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
    color: var(--ink) !important;
}
.stTextInput > div > div > input:focus { border-color: var(--amber) !important; box-shadow: 0 0 0 1px var(--amber) !important; }
label, .stRadio label { color: var(--ink-dim) !important; font-size: 0.8rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--amber) !important;
    color: #16130a !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover { filter: brightness(1.08); transform: translateY(-1px); box-shadow: 0 6px 18px rgba(232,163,61,0.25); }
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--ink-dim) !important;
    border: 1px solid var(--line) !important;
}
.stDownloadButton > button {
    background: var(--panel-2) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--ink-dim);
    font-size: 0.82rem; font-weight: 500; padding: 0.6rem 1rem;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] { color: var(--amber) !important; border-bottom: 2px solid var(--amber) !important; }

/* ── Stat pills ── */
.stat-row { display:flex; gap: 0.75rem; flex-wrap: wrap; margin: 1.1rem 0 1.6rem; }
.stat-pill {
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
    padding: 0.6rem 1rem; min-width: 108px;
}
.stat-num { font-family:'Fraunces', serif; font-size: 1.35rem; color: var(--amber); font-weight:600; line-height:1; }
.stat-label { font-family:'JetBrains Mono', monospace; font-size: 0.62rem; letter-spacing:0.12em; text-transform:uppercase; color: var(--ink-dim); margin-top:0.3rem; }

/* ── Content panel ── */
.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.5rem 1.6rem;
    position: relative;
}
.panel-title {
    font-family:'JetBrains Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--teal); margin-bottom: 0.85rem;
}
.panel-body { font-size: 0.92rem; line-height: 1.75; color: var(--ink); }

.transcript-panel {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.2rem;
    max-height: 460px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.85;
    color: var(--ink-dim);
    white-space: pre-wrap;
}

/* ── Chat bubbles (native st.chat_message override) ── */
[data-testid="stChatMessage"] {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
}
.stChatInputContainer, [data-testid="stChatInput"] {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
}

hr { border: none !important; border-top: 1px solid var(--line) !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_steps": {},
    "pipeline_done": False,
    "processed_source": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

STEPS = [
    ("audio", "🔊", "Splitting audio"),
    ("transcript", "📝", "Transcribing"),
    ("title", "🏷️", "Naming the reel"),
    ("summary", "📋", "Summarising"),
    ("extract", "🔍", "Extracting signal"),
    ("rag", "🧠", "Indexing for chat"),
]


def set_step(key, state):
    st.session_state.pipeline_steps[key] = state


def render_stepper():
    for key, icon, label in STEPS:
        state = st.session_state.pipeline_steps.get(key, "idle")
        icon_cls = {"done": "step-done", "active": "step-active", "idle": "step-idle"}[state]
        label_cls = {"done": "step-label-done", "active": "step-label-active", "idle": "step-label-idle"}[state]
        glyph = "✓" if state == "done" else ("●" if state == "active" else "○")
        st.markdown(
            f"""<div class="step-row">
                    <div class="step-icon {icon_cls}">{glyph}</div>
                    <div class="{label_cls}">{icon} {label}</div>
                </div>""",
            unsafe_allow_html=True,
        )


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def count_items(text: str) -> int:
    if not text:
        return 0
    lines = [l for l in text.splitlines() if l.strip()]
    return max(len(lines), 1) if lines else 0


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="side-brand">🎞️ <span>Reel</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-tag">AI Video Assistant</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Source",
        ["YouTube URL", "Local file / path"],
        label_visibility="collapsed"
    )

    source = st.text_input(
        "Source",
        placeholder="https://youtube.com/watch?v=… or /path/to/file.mp4",
        label_visibility="collapsed",
    )

    language = st.selectbox(
        "Language",
        ["english", "hinglish"],
        index=0
    )

    run_clicked = st.button(
        "▶  Analyse",
        use_container_width=True
    )
    
    if st.session_state.pipeline_steps:
        st.markdown("---")
        st.markdown('<div class="side-tag">Pipeline</div>', unsafe_allow_html=True)
        render_stepper()

    if st.session_state.result:
        st.markdown("---")
        st.download_button(
            "⬇ Transcript (.txt)",
            data=st.session_state.result["transcript"].encode("utf-8"),
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )
        if st.button("↺ New session", use_container_width=True, type="secondary"):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.pipeline_steps = {}
            st.session_state.pipeline_done = False
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUN
# ─────────────────────────────────────────────────────────────────────────────
if run_clicked and source.strip():
    # Prevent accidental duplicate processing
    if st.session_state.processed_source == source.strip():
        st.info("This source has already been analysed.")
    else:
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        st.session_state.pipeline_done = False
        st.session_state.processed_source = None

        status_box = st.empty()
        try:
                    # ---------------------------------------------------------
                    # STEP 1 & 2: GET TRANSCRIPT
                    # ---------------------------------------------------------
                    if input_mode == "YouTube URL":
                        set_step("audio", "active")
                        status_box.info("🔊 Getting YouTube transcript…")

                        transcript = get_youtube_transcript(source, language)

                        set_step("audio", "done")
                        set_step("transcript", "done")

                    else:
                        set_step("audio", "active")
                        status_box.info("🔊 Splitting audio into chunks…")

                        chunks = process_input(source)

                        set_step("audio", "done")

                        set_step("transcript", "active")
                        status_box.info("📝 Transcribing…")

                        transcript = transcribe_all(chunks, language)

                        set_step("transcript", "done")

                    # ---------------------------------------------------------
                    # STEP 3: TITLE
                    # ---------------------------------------------------------
                    set_step("title", "active")
                    status_box.info("🏷️ Naming the reel…")

                    title = generate_title(transcript)

                    set_step("title", "done")

                    # ---------------------------------------------------------
                    # STEP 4: SUMMARY
                    # ---------------------------------------------------------
                    set_step("summary", "active")
                    status_box.info("📋 Summarising…")

                    summary = summarize(transcript)

                    set_step("summary", "done")

                    # ---------------------------------------------------------
                    # STEP 5: EXTRACT INFORMATION
                    # ---------------------------------------------------------
                    set_step("extract", "active")
                    status_box.info(
                        "🔍 Extracting action items, decisions, questions…"
                    )

                    action_items = extract_action_items(transcript)
                    decisions = extract_key_decisions(transcript)
                    questions = extract_questions(transcript)

                    set_step("extract", "done")

                    # ---------------------------------------------------------
                    # STEP 6: RAG
                    # ---------------------------------------------------------
                    set_step("rag", "active")
                    status_box.info("🧠 Indexing transcript for chat…")

                    rag_chain = build_rag_chain(transcript)

                    set_step("rag", "done")

                    # ---------------------------------------------------------
                    # SAVE RESULT
                    # ---------------------------------------------------------
                    st.session_state.result = {
                        "title": title,
                        "transcript": transcript,
                        "summary": summary,
                        "action_items": action_items,
                        "key_decisions": decisions,
                        "open_questions": questions,
                        "rag_chain": rag_chain,
                    }

                    st.session_state.pipeline_done = True
                    st.session_state.processed_source = source.strip()

                    status_box.empty()
                    st.rerun()

        except Exception as e:
                    for key, _, _ in STEPS:
                        if st.session_state.pipeline_steps.get(key) == "active":
                            st.session_state.pipeline_steps[key] = "idle"

                    status_box.error(f"⚠ Pipeline failed: {e}")
# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="filmrule"></div>', unsafe_allow_html=True)

result = st.session_state.result

if not result:
    st.markdown('<div class="hero-kicker">Meeting &amp; video intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Turn any recording<br>into an <em>answerable</em> document.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Drop a YouTube link or a local file in the sidebar. '
        "Reel transcribes it, pulls out the summary, action items, decisions and "
        "open questions — then lets you chat with the transcript directly.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    for col, icon, label, desc in [
        (c1, "📝", "Transcribe", "Accurate, language-aware transcription."),
        (c2, "🔍", "Extract", "Action items, decisions, open questions — auto-pulled."),
        (c3, "💬", "Chat", "Ask follow-ups; answers are grounded in the transcript."),
    ]:
        with col:
            st.markdown(
                f'<div class="panel"><div class="panel-title">{icon} {label}</div>'
                f'<div class="panel-body" style="color:var(--ink-dim)">{desc}</div></div>',
                unsafe_allow_html=True,
            )

else:
    st.markdown('<div class="hero-kicker">Now viewing</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-title">{result["title"]}</div>', unsafe_allow_html=True)

    wc = word_count(result["transcript"])
    read_min = max(1, round(wc / 160))
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-pill"><div class="stat-num">{wc:,}</div><div class="stat-label">Words</div></div>
            <div class="stat-pill"><div class="stat-num">{read_min}</div><div class="stat-label">Min read</div></div>
            <div class="stat-pill"><div class="stat-num">{count_items(result['action_items'])}</div><div class="stat-label">Action items</div></div>
            <div class="stat-pill"><div class="stat-num">{count_items(result['key_decisions'])}</div><div class="stat-label">Decisions</div></div>
            <div class="stat-pill"><div class="stat-num">{count_items(result['open_questions'])}</div><div class="stat-label">Open Q's</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["Overview", "Action Items", "Decisions", "Open Questions", "Transcript", "Chat"]
    )

    with tab_overview:
        st.markdown(
            f'<div class="panel"><div class="panel-title">📋 Summary</div>'
            f'<div class="panel-body">{result["summary"]}</div></div>',
            unsafe_allow_html=True,
        )

    with tab_actions:
        st.markdown(
            f'<div class="panel"><div class="panel-title">✅ Action Items</div>'
            f'<div class="panel-body">{result["action_items"]}</div></div>',
            unsafe_allow_html=True,
        )

    with tab_decisions:
        st.markdown(
            f'<div class="panel"><div class="panel-title">🔑 Key Decisions</div>'
            f'<div class="panel-body">{result["key_decisions"]}</div></div>',
            unsafe_allow_html=True,
        )

    with tab_questions:
        st.markdown(
            f'<div class="panel"><div class="panel-title">❓ Open Questions</div>'
            f'<div class="panel-body">{result["open_questions"]}</div></div>',
            unsafe_allow_html=True,
        )

    with tab_transcript:
        search = st.text_input("Search transcript", placeholder="Find a word or phrase…")
        display_text = result["transcript"]
        if search.strip():
            hits = display_text.lower().count(search.lower())
            st.caption(f"{hits} match{'es' if hits != 1 else ''} for “{search}”")
        st.markdown(f'<div class="transcript-panel">{display_text}</div>', unsafe_allow_html=True)

    with tab_chat:
        st.caption("Answers are grounded in this transcript via retrieval-augmented generation.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎞️"):
                st.markdown(msg["content"])

        question = st.chat_input("Ask something about this recording…")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(question)

            with st.chat_message("assistant", avatar="🎞️"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as e:
                        answer = f"⚠ Error answering question: {e}"
                st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.session_state.chat_history:
            if st.button("🗑 Clear chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
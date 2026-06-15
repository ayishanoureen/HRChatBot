"""
app.py —  HR Policy Chatbot  |  Streamlit UI
Run with:  streamlit run app.py
Requires:  python main.py --phase all  (to build the vector store first)
"""

import os
import sys
import logging
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title            = " HR Assistant",
    page_icon             = "🏢",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

for _log in ("sentence_transformers", "chromadb", "httpx", "urllib3"):
    logging.getLogger(_log).setLevel(logging.ERROR)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* Main App */
.stApp {
    background: #f3f5f9;
}

footer {
    visibility: hidden !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* HEADER & NAVIGATION */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Hide top-right buttons (Deploy, Menu) specifically */
[data-testid="stAppDeploy"], 
[data-testid="stMainMenu"], 
.stDeployButton,
[data-testid="stHeaderAction"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    padding-top: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SIDEBAR */
/* ═══════════════════════════════════════════════════════════════════════════ */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16233b 0%, #0e1729 100%) !important;
    border-right: 1px solid #1f2d44;
    width: 378px !important;
}

section[data-testid="stSidebar"] > div {
    width: 378px !important;
}

[data-testid="stSidebarContent"] {
    padding: 2rem 1.6rem 1.5rem !important;
}

/* Sidebar Text */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {
    color: #d6deea !important;
}

/* Sidebar Divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 1.6rem 0 !important;
}

/* Sidebar Metrics */
[data-testid="stSidebar"] .stMetric {
    background: transparent !important;
}

[data-testid="stSidebar"] .stMetric label {
    color: #aab7c9 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

/* Sidebar Buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    border-radius: 14px !important;
    border: 1px solid rgba(99,102,241,0.08) !important;
    background: rgba(59,130,246,0.10) !important;
    color: #dbeafe !important;
    padding: 0.95rem 1rem !important;
    text-align: left !important;
    font-size: 0.90rem !important;
    font-weight: 500 !important;
    transition: all .18s ease;
    box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(59,130,246,0.18) !important;
    border-color: rgba(99,102,241,0.20) !important;
    transform: translateX(2px);
}

/* Sidebar Collapse & Toggle Fix */
section[data-testid="stSidebar"][aria-expanded="false"] {
    background: transparent !important;
    border-right: none !important;
    box-shadow: none !important;
}

/* Force the open button (hamburger) to a fixed, visible position */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    position: fixed !important;
    top: 1rem !important;
    left: 1rem !important;
    z-index: 1000001 !important;
    display: flex !important;
    visibility: visible !important;
}

button[kind="header"] {
    background: #ffffff !important;
    color: #16233b !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    width: 42px !important;
    height: 42px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    pointer-events: auto !important;
}

button[kind="header"]:hover {
    background: #ffffff !important;
    color: #4f46e5 !important;
    transform: scale(1.05);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* PAGE HEADER */
/* ═══════════════════════════════════════════════════════════════════════════ */

.page-wrapper {
    width: 100%;
}

/* Reduced top gap */
.page-header {
    text-align: center;
    padding-top: 2.2rem;
    padding-bottom: 0.2rem;
    margin-bottom: 0.5rem;
}

.page-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: #1e293b;
    margin: 0;
    letter-spacing: -0.04em;
    line-height: 1.2;
}

.page-header p {
    margin-top: 0.55rem;
    color: #64748b;
    font-size: 1rem;
    font-weight: 500;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* WELCOME CARD */
/* ═══════════════════════════════════════════════════════════════════════════ */

.welcome-card {
    max-width: 860px;
    margin: 1rem auto 1.5rem auto;   /* reduced huge gap */
    background: #ffffff;
    border-radius: 26px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 24px rgba(15,23,42,0.05);
    padding: 2.2rem 2.5rem;
    text-align: center;
    transition: all .25s ease;
    overflow: hidden;
}

.welcome-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 34px rgba(15,23,42,.08)
}

.welcome-card h2 {
    font-size: 2.15rem;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 1.2rem;
    letter-spacing: -0.03em;
    line-height: 1.25;
}

.welcome-text {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    margin-top: 1.4rem;
}

.welcome-text div {
    font-size: 1.18rem;
    line-height: 2rem;
    color: #5b6472;
    font-weight: 400;
}

.welcome-card strong {
    color: #334155;
    font-weight: 700;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* CHAT */
/* ═══════════════════════════════════════════════════════════════════════════ */

[data-testid="stChatMessage"] {
    margin-bottom: 1.2rem !important;
}

/* User */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageContent {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important;
    border-radius: 20px 20px 6px 20px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 6px 18px rgba(79,70,229,.18) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
    color: white !important;
}

/* Assistant */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stChatMessageContent {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px 20px 20px 8px !important;
    padding: 1.25rem 1.4rem !important;
    box-shadow: 0 4px 18px rgba(15,23,42,.05) !important;
    color: #1e293b !important;
    line-height: 1.8 !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) li {
    color: #334155 !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) strong {
    color: #4338ca !important;
}

/* Avatars */
[data-testid="chatAvatarIcon-user"] {
    background: #4f46e5 !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: white !important;
    border: 2px solid #e2e8f0 !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SOURCES */
/* ═══════════════════════════════════════════════════════════════════════════ */

.sources-section {
    margin-top: 1rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: .9rem;
}

.sources-label {
    font-size: .75rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: .6rem;
}

.src-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.src-chip {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 5px 12px;
    font-size: .76rem;
    font-weight: 600;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* THINKING */
/* ═══════════════════════════════════════════════════════════════════════════ */

.thinking-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: .92rem;
    color: #64748b;
    font-style: italic;
}

.dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #6366f1;
    animation: pulse 1.2s infinite ease-in-out;
}

.dot:nth-child(2){animation-delay:.22s;}
.dot:nth-child(3){animation-delay:.44s;}

@keyframes pulse {
    0%,80%,100% {
        transform: scale(.6);
        opacity: .35;
    }
    40% {
        transform: scale(1.15);
        opacity: 1;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* CHAT INPUT */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Target the outer Streamlit container wrap to fully override the native theme colors */
div[data-testid="stChatInput"], 
.stChatInputContainer,
div[data-testid="stChatInput"] > div {
    border: 1.5px solid #f1f5f9 !important; 
    border-radius: 22px !important;
    background: #f1f5f9 !important;
    transition: all 0.2s ease !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Apply only the clean blue color to the entire wrapper layout when active */
div[data-testid="stChatInput"]:focus-within,
.stChatInputContainer:focus-within,
div[data-testid="stChatInput"] > div:focus-within {
    border: 1.5px solid #2563eb !important;
    border-radius: 22px !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.10) !important;
    outline: none !important;
}



/* ═══════════════════════════════════════════════════════════════════════════ */
/* ERROR CARD */
/* ═══════════════════════════════════════════════════════════════════════════ */

.err-card {
    max-width: 700px;
    margin: 2rem auto;
    background: #fff7f7;
    border: 1px solid #fecaca;
    border-left: 4px solid #ef4444;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    color: #7f1d1d;
}

/* Summary */
.summary-block {
    margin-top: 1rem;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-left: 4px solid #4f46e5;
    border-radius: 12px;
    padding: .9rem 1rem;
    color: #312e81;
    font-size: .93rem;
}

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Backend
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_embedding_model():
    from src.embeddings import get_embedding_model
    return get_embedding_model()


def _get_collection():
    from src.vector_store import get_or_create_collection
    return get_or_create_collection()



def _ask(query: str):
    from src.rag_chain import ask
    return ask(query, collection=_get_collection())


# ══════════════════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════════════════


def _init_state():
    for k, v in {
        "messages": [],
        "total_queries": 0,
        "backend_ready": False,
        "backend_error": None
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _ts() -> str:
    return datetime.now().strftime("%H:%M")


def _source_chip_html(src: dict) -> str:
    name = os.path.splitext(src.get("source", "?"))[0].replace("_", " ")
    name = name[:24] + "…" if len(name) > 26 else name

    return (
        f'<span class="src-chip">📄 {name} · '
        f'p.{src.get("page", "?")} · '
        f'{src.get("similarity", 0):.0%}</span>'
    )


def _friendly_error(exc: Exception) -> str:
    m = str(exc).lower()

    if any(x in m for x in ("quota", "429", "rate")):
        return "⚠️ Gemini API rate limit reached. Please wait a moment and try again."

    if any(x in m for x in ("api key", "401", "403")):
        return "🔑 Gemini API key issue. Check GEMINI_API_KEY in your .env file."

    if any(x in m for x in ("network", "connection", "timeout")):
        return "🌐 Network error. Check your internet connection and try again."

    return f"❌ Unexpected error: {exc}"


def _post_process(text: str) -> str:
    lines = text.split("\n")
    out = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("**Summary:**") or stripped.startswith("**Summary**:"):
            content = stripped.replace("**Summary:**", "").replace("**Summary**:", "").strip()
            out.append(
                f'<div class="summary-block">📌 <strong>Summary:</strong> {content}</div>'
            )
        else:
            out.append(line)

    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# Rendering
# ══════════════════════════════════════════════════════════════════════════════


def _render_user_msg(text: str, ts: str):
    with st.chat_message("user"):
        st.markdown(text)
        st.caption(ts)



def _render_bot_msg(text: str, sources: list, ts: str):
    with st.chat_message("assistant"):
        st.markdown(_post_process(text), unsafe_allow_html=True)

        if sources:
            chips = "".join(_source_chip_html(s) for s in sources)

            st.markdown(f"""
            <div class="sources-section">
                <div class="sources-label">📚 Sources</div>
                <div class="src-chips">{chips}</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(ts)



def _render_thinking():
    with st.chat_message("assistant"):
        st.markdown("""
        <div class="thinking-wrap">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
            &nbsp;Searching HR policies…
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

_QUICK_QUESTIONS = [
    "What is the leave policy?",
    "How does the appraisal process work?",
    "What are the rules for late attendance?",
    "Explain the travel reimbursement policy.",
    "What is the notice period for resignation?",
    "What are the employee benefits?",
]



def _render_sidebar():
    with st.sidebar:

        st.markdown("## 🏢  HR Assistant")
        st.caption("Powered by Gemini · ChromaDB · SentenceTransformers")

        st.divider()

        st.markdown("### ⚙️ System Status")

        if st.session_state.backend_error:
            st.error("❌ Backend error — see main panel")
        elif st.session_state.backend_ready:
            st.success("✅ Ready")
        else:
            st.info("⏳ Initialising…")

        st.divider()

        st.markdown("### 📊 Session")

        c1, c2 = st.columns(2)
        c1.metric("Questions", st.session_state.total_queries)
        c2.metric("Messages", len(st.session_state.messages))

        st.divider()

        st.markdown("### 💡 Quick Questions")

        for q in _QUICK_QUESTIONS:
            if st.button(q, key=f"qq_{q}"):
                st.session_state["_pending_quick"] = q

        st.divider()

        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.rerun()

        st.divider()

        st.caption("HR Policy RAG Chatbot · v1.3")


# ══════════════════════════════════════════════════════════════════════════════
# Page Sections
# ══════════════════════════════════════════════════════════════════════════════


def _render_header():
    st.markdown(
        '<div class="page-wrapper">'
        '<div class="page-header">'
        '<h1>🏢  HR Policy Assistant</h1>'
        '<p>Ask anything about \'s HR policies — leave, travel, appraisals, benefits, and more.</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )




def _render_welcome():
    st.markdown(
        '<div class="welcome-card">'
        '<h2>👋 Hello! How can I help?</h2>'
        '<div class="welcome-text">'
        '<div>I\'m your <strong> HR Policy Assistant</strong>.</div>'
        '<div>'
        'I can answer questions about '
        '<strong>leave</strong>, '
        '<strong>attendance</strong>, '
        '<strong>travel reimbursement</strong>, '
        '<strong>performance appraisals</strong>, '
        '<strong>benefits</strong>, '
        '<strong>conduct policies</strong>, and more.'
        '</div>'
        '<div>'
        'Use the <strong>Quick Questions</strong> in the sidebar, '
        'or type your own question below.'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )




def _render_error_panel(msg: str):
    st.markdown(
        f'<div class="err-card"><strong>❌ Backend Error</strong><br>{msg}</div>',
        unsafe_allow_html=True
    )

    st.code("python main.py --phase all", language="bash")



def _process_query(query: str):
    ts = _ts()

    st.session_state.messages.append({
        "role": "user",
        "text": query,
        "sources": [],
        "ts": ts
    })

    st.session_state.total_queries += 1

    placeholder = st.empty()

    with placeholder.container():
        _render_thinking()

    status = st.empty()
    status.caption("🔍 Searching policies…")

    try:
        status.caption("⚡ Generating answer…")

        resp = _ask(query)
        answer = resp.answer
        sources = resp.sources or []

    except Exception as exc:
        answer = _friendly_error(exc)
        sources = []

    finally:
        placeholder.empty()
        status.empty()

    st.session_state.messages.append({
        "role": "bot",
        "text": answer,
        "sources": sources,
        "ts": _ts()
    })


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════


def main():

    _render_sidebar()
    _render_header()

    # One-time backend init
    if not st.session_state.backend_ready and not st.session_state.backend_error:

        with st.spinner("⏳ Loading embedding model and vector store…"):
            try:
                _load_embedding_model()
                col = _get_collection()

                if col.count() == 0:
                    st.session_state.backend_error = (
                        "Vector store is empty. Run: python main.py --phase all"
                    )
                else:
                    st.session_state.backend_ready = True

            except Exception as e:
                st.session_state.backend_error = str(e)

    if st.session_state.backend_error:
        _render_error_panel(st.session_state.backend_error)
        return

    # Chat history
    if not st.session_state.messages:
        _render_welcome()
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                _render_user_msg(msg["text"], msg.get("ts", ""))
            else:
                _render_bot_msg(
                    msg["text"],
                    msg.get("sources", []),
                    msg.get("ts", "")
                )

    # Quick Question
    pending = st.session_state.pop("_pending_quick", None)

    if pending and st.session_state.backend_ready:
        _process_query(pending)
        st.rerun()

    # Chat Input
    if st.session_state.backend_ready:

        user_input = st.chat_input(
            "Ask about HR policies — leave, appraisals, travel, benefits…"
        )

        if user_input and user_input.strip():
            _process_query(user_input.strip())
            st.rerun()


if __name__ == "__main__":
    main()
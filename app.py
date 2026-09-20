"""Streamlit front-end for the AI Research Agent.  Run with:  streamlit run app.py"""

import os
import re
import traceback

# These must be set BEFORE crewai is imported (it happens inside research_agent).
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

import streamlit as st  # noqa: E402

from research_agent import DEPTH_SETTINGS, MODEL_CHOICES, PATCH_STATUS, run_research  # noqa: E402

st.set_page_config(page_title="AI Research Agent", page_icon="🔎")


def get_api_key(user_key: str) -> str | None:
    """Priority: key typed in the sidebar > Streamlit secrets > environment variable."""
    if user_key.strip():
        return user_key.strip()
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:  # no secrets file / key not set
        return os.getenv("GROQ_API_KEY")


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    user_key = st.text_input(
        "Groq API key (optional)",
        type="password",
        help="Leave empty to use the key saved in Streamlit secrets.",
    )
    depth = st.radio("Report depth", list(DEPTH_SETTINGS), index=1)
    model_label = st.selectbox("Model", list(MODEL_CHOICES))
    model = MODEL_CHOICES[model_label]
    st.caption("Search: DuckDuckGo (free)")
    active = ", ".join(name for name, ok in PATCH_STATUS.items() if ok) or "NOT applied"
    st.caption(f"Groq compatibility patch: {active}")

# ---------- Main page ----------
st.title("🔎 AI Research Agent")
st.write("Enter a topic. The agent searches the web and writes a report with sources.")

topic = st.text_input(
    "Research topic",
    placeholder="e.g. Solid-state batteries for electric vehicles",
)

if st.button("Generate report", type="primary"):
    api_key = get_api_key(user_key)
    st.session_state.pop("report", None)  # clear the previous report

    if not topic.strip():
        st.warning("Please enter a research topic first.")
    elif not api_key:
        st.error("No Groq API key found. Add it in the sidebar or in Streamlit secrets.")
    else:
        try:
            with st.spinner("Researching and writing... this can take 30-90 seconds."):
                report = run_research(topic.strip(), depth, api_key, model)
            # Save in session_state so the report survives Streamlit's reruns
            # (for example when you click the download button).
            st.session_state["report"] = report
            st.session_state["topic"] = topic.strip()
        except Exception as exc:
            details = traceback.format_exc()
            print(details)  # also goes to the logs ("Manage app" on Streamlit Cloud)
            st.error(f"Something went wrong: {exc}")
            with st.expander("Technical details (copy this if you need help)"):
                st.code(details)
            st.info(
                "Common causes: wrong API key, or the Groq rate limit was hit "
                "(wait a minute and try again, or choose 'Quick' depth)."
            )

if "report" in st.session_state:
    st.divider()
    st.markdown(st.session_state["report"])

    filename = re.sub(r"[^a-z0-9]+", "-", st.session_state["topic"].lower()).strip("-")[:50]
    st.download_button(
        "⬇️ Download report (.md)",
        data=st.session_state["report"],
        file_name=f"{filename or 'report'}.md",
        mime="text/markdown",
    )

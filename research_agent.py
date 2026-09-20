"""The CrewAI part: one agent, one task, one crew."""

import functools
import inspect
from datetime import date

from crewai import LLM, Agent, Crew, Process, Task

from tools import duckduckgo_search

# ---------------------------------------------------------------------------
# Compatibility patch for a known CrewAI bug (GitHub issue #5886).
# CrewAI adds an Anthropic-only field called "cache_breakpoint" to every message.
# Groq rejects unknown fields ("property 'cache_breakpoint' is unsupported"),
# so we remove the field right before the request is sent.
# It only affects Groq requests, and it is harmless once CrewAI fixes the bug.
#
# Two layers, so that it keeps working even if CrewAI renames something:
#   1) "litellm": wraps the final litellm.completion() call (most reliable)
#   2) "class":   wraps two internal methods of CrewAI's LLM class
# PATCH_STATUS shows which layers are active (the app displays it in the sidebar).
# ---------------------------------------------------------------------------
_CACHE_KEY = "cache_breakpoint"
PATCH_STATUS = {"litellm": False, "class": False}


def _strip_marker(messages):
    if not isinstance(messages, list):
        return messages
    return [
        {k: v for k, v in m.items() if k != _CACHE_KEY} if isinstance(m, dict) else m
        for m in messages
    ]


def _is_groq_model(model) -> bool:
    return str(model or "").startswith("groq/")


def _clean_call_args(args, kwargs):
    """Strip the marker from a litellm call, but only for Groq models."""
    model = kwargs.get("model", args[0] if args else "")
    is_groq = (
        _is_groq_model(model)
        or kwargs.get("custom_llm_provider") == "groq"
        or "api.groq.com" in str(kwargs.get("api_base") or "")
    )
    if is_groq:
        if "messages" in kwargs:
            kwargs = {**kwargs, "messages": _strip_marker(kwargs["messages"])}
        elif len(args) > 1:
            args = (args[0], _strip_marker(args[1]), *args[2:])
    return args, kwargs


def _patch_litellm() -> None:
    try:
        import litellm
    except ImportError:
        return
    if getattr(litellm, "_groq_cache_patch", False):
        PATCH_STATUS["litellm"] = True
        return  # Streamlit re-runs this file often; patch only once

    orig_sync = getattr(litellm, "completion", None)
    if callable(orig_sync):
        @functools.wraps(orig_sync)
        def sync_wrapper(*args, **kwargs):
            args, kwargs = _clean_call_args(args, kwargs)
            return orig_sync(*args, **kwargs)

        litellm.completion = sync_wrapper

    orig_async = getattr(litellm, "acompletion", None)
    if callable(orig_async):
        @functools.wraps(orig_async)
        async def async_wrapper(*args, **kwargs):
            args, kwargs = _clean_call_args(args, kwargs)
            return await orig_async(*args, **kwargs)

        litellm.acompletion = async_wrapper

    litellm._groq_cache_patch = True
    PATCH_STATUS["litellm"] = True


def _patch_llm_class() -> None:
    if getattr(LLM, "_groq_cache_patch", False):
        PATCH_STATUS["class"] = True
        return

    def is_groq(llm) -> bool:
        return _is_groq_model(getattr(llm, "model", ""))

    fmt = inspect.getattr_static(LLM, "_format_messages_for_provider", None)
    if inspect.isfunction(fmt):
        def patched_format(self, messages, *args, **kwargs):
            if is_groq(self):
                messages = _strip_marker(messages)
            return fmt(self, messages, *args, **kwargs)

        LLM._format_messages_for_provider = patched_format
        PATCH_STATUS["class"] = True

    prep = inspect.getattr_static(LLM, "_prepare_completion_params", None)
    if inspect.isfunction(prep):
        def patched_prepare(self, *args, **kwargs):
            params = prep(self, *args, **kwargs)
            if is_groq(self) and isinstance(params, dict) and "messages" in params:
                params["messages"] = _strip_marker(params["messages"])
            return params

        LLM._prepare_completion_params = patched_prepare
        PATCH_STATUS["class"] = True

    if PATCH_STATUS["class"]:
        LLM._groq_cache_patch = True


for _patch in (_patch_litellm, _patch_llm_class):
    try:
        _patch()
    except Exception as exc:  # a failed patch must never crash the app
        print(f"[groq patch] {_patch.__name__} failed: {exc}")

# "groq/" tells CrewAI (via LiteLLM) to use Groq. The rest is Groq's own model ID.
MODEL_NAME = "groq/openai/gpt-oss-120b"

# How much work the agent should do. Smaller = faster and uses fewer tokens.
DEPTH_SETTINGS = {
    "Quick": {"searches": 2, "words": "500-700"},
    "Standard": {"searches": 4, "words": "800-1200"},
    "Deep": {"searches": 5, "words": "1200-1600"},
}


def run_research(topic: str, depth: str, api_key: str) -> str:
    """Run the research agent and return the finished report as Markdown text."""
    settings = DEPTH_SETTINGS[depth]

    # 1) The brain: Groq-hosted gpt-oss-120b
    llm = LLM(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.3,           # low = more factual, less "creative"
        max_tokens=4000,           # upper limit for the report length (in tokens)
        reasoning_effort="medium",  # gpt-oss option: "low" | "medium" | "high"
    )

    # 2) The agent: who it is + which tools it may use
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Research a topic on the web and write an accurate, well-structured report with sources.",
        backstory=(
            "You are a careful analyst. You search the web, compare sources, "
            "never invent facts, and clearly say when information is uncertain."
        ),
        tools=[duckduckgo_search],
        llm=llm,
        allow_delegation=False,  # single agent: nobody to delegate to
        max_iter=10,             # safety limit on think/search loops
        verbose=True,            # prints the agent's steps to the terminal / app logs
    )

    # 3) The task: what to do and what the result should look like
    #    {topic}, {max_searches}, {word_range}, {today} are filled in by kickoff(inputs=...)
    task = Task(
        description=(
            "Research the topic: {topic}\n\n"
            "Steps:\n"
            "1. Use the DuckDuckGo Search tool up to {max_searches} times, "
            "with different, specific queries.\n"
            "2. Note key facts, numbers, dates and differing viewpoints.\n"
            "3. Write a report of roughly {word_range} words.\n\n"
            "Rules:\n"
            "- Only state facts supported by your search results. Do not invent "
            "facts, numbers or URLs.\n"
            "- If sources disagree or information is thin, say so.\n"
            "- Today's date is {today}; prefer recent information."
        ),
        expected_output=(
            "A Markdown report with: a title (# heading), a short Executive Summary, "
            "3-5 sections with ## headings, a 'Key Takeaways' bullet list, and a "
            "'Sources' section listing the URLs you actually used. "
            "Do not wrap the report in a code block."
        ),
        agent=researcher,
    )

    # 4) The crew: runs the task(s) with the agent(s)
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff(
        inputs={
            "topic": topic,
            "max_searches": settings["searches"],
            "word_range": settings["words"],
            "today": date.today().strftime("%B %d, %Y"),
        }
    )
    return result.raw  # the final text answer

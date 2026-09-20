"""Custom tool: lets the agent search the web with DuckDuckGo (free, no API key)."""

from crewai.tools import tool
from ddgs import DDGS  # the package was renamed from `duckduckgo-search` to `ddgs`

MAX_RESULTS = 5      # how many results the agent sees per search
SNIPPET_CHARS = 300  # keep results short -> fewer tokens -> fewer rate-limit errors


@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web with DuckDuckGo. Input: a short, specific search query.
    Returns the top results as: title, URL and a short text snippet."""
    try:
        results = DDGS().text(query, max_results=MAX_RESULTS)
    except Exception as exc:  # network problems, rate limits, no results...
        return f"Search failed ({exc}). Try a different or simpler query."

    if not results:
        return "No results found. Try rephrasing the query."

    lines = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "").strip()
        url = item.get("href", "").strip()
        snippet = item.get("body", "").strip()[:SNIPPET_CHARS]
        lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n".join(lines)

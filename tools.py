"""Custom tools for the agent: web search and reading a web page (both free, no API key)."""

import ipaddress
import re
import socket
from html import unescape
from urllib.parse import urlparse

import requests
from crewai.tools import tool
from ddgs import DDGS  # the package was renamed from `duckduckgo-search` to `ddgs`

MAX_RESULTS = 5       # how many results the agent sees per search
SNIPPET_CHARS = 300   # keep results short -> fewer tokens -> fewer rate-limit errors
MAX_PAGE_CHARS = 2500  # how much text of a web page the agent gets to read


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


def _is_public_http_url(url: str) -> bool:
    """Only allow normal public web addresses (blocks localhost / private networks)."""
    try:
        parts = urlparse(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return False
        for info in socket.getaddrinfo(parts.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except Exception:
        return False


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", html)
    text = unescape(re.sub(r"(?s)<[^>]+>", " ", html))
    return re.sub(r"\s+", " ", text).strip()


@tool("Read Webpage")
def read_webpage(url: str) -> str:
    """Open a web page and return its text (shortened). Input: one full URL that starts
    with http:// or https:// and came from a search result."""
    if not _is_public_http_url(url):
        return "Cannot open this URL. Use a public http(s) link taken from the search results."

    text = ""
    try:  # 1st try: ddgs has a built-in page extractor
        data = DDGS().extract(url, fmt="text_plain")
        content = data.get("content", "") if isinstance(data, dict) else ""
        text = content if isinstance(content, str) else ""
    except Exception:
        text = ""

    if not text.strip():  # 2nd try: plain download + strip the HTML tags
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (research-agent)"})
            resp.raise_for_status()
            if not resp.headers.get("content-type", "").lower().startswith(("text/", "application/xhtml")):
                return "This link is not a normal web page (maybe a PDF). Try another source."
            text = _html_to_text(resp.text)
        except Exception as exc:
            return f"Could not open the page ({exc}). Try another source."

    text = text.strip()
    if not text:
        return "The page had no readable text. Try another source."
    return text[:MAX_PAGE_CHARS]

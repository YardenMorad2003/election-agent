"""
Operational web search tool.

Keeps the existing DuckDuckGo/Wikipedia search flow for evergreen or older
questions, but switches to an RSS-backed recent-news search when the user is
explicitly asking for very recent developments such as the last 24 hours or
last week.
"""
from __future__ import annotations

import email.utils
import re
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Literal
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from langchain_core.tools import tool

from tools.web_search import web_search as legacy_web_search


USER_AGENT = "Mozilla/5.0 (compatible; ElectionAgent/1.0; RSSNewsSearch)"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

RECENT_24H_HINTS = (
    "today", "past 24 hours", "last 24 hours", "last day", "past day",
    "since yesterday", "overnight", "breaking", "just now",
)
RECENT_WEEK_HINTS = (
    "this week", "past week", "last week", "last 7 days", "past 7 days",
)
NEWS_EVENT_HINTS = (
    "news", "headline", "headlines", "breaking", "developments", "update",
    "updates", "press conference", "announced", "announcement",
)
SOFT_RECENCY_HINTS = ("latest", "recent", "current", "today", "newest")
PRESENT_TIME_HINTS = ("current", "currently", "today", "latest", "now")
FACT_QUERY_PREFIXES = ("who is", "who's", "what is", "what's", "which is", "name")
EXPLICIT_PRESENT_TIME_PHRASES = ("as of today", "right now", "at the moment")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "on", "in", "to", "for", "about",
    "give", "me", "latest", "recent", "current", "news", "what", "who", "when",
    "where", "why", "how", "is", "are", "was", "were", "us",
}
LOW_QUALITY_TITLE_HINTS = (
    "opinion", "analysis", "explainer", "live updates", "newsletter",
    "podcast", "watch live", "photo", "photos", "video",
)
LOW_QUALITY_SNIPPET_HINTS = (
    "opinion", "analysis", "column", "newsletter", "listen to", "watch",
)


def _http_get(url: str, timeout: int = 10) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _infer_window(query: str) -> Literal["24h", "7d"] | None:
    q = query.lower()
    if any(hint in q for hint in RECENT_24H_HINTS):
        return "24h"
    if any(hint in q for hint in RECENT_WEEK_HINTS):
        return "7d"
    return None


def _is_simple_current_fact_query(query: str) -> bool:
    q = query.lower()
    return (
        any(q.startswith(prefix) for prefix in FACT_QUERY_PREFIXES)
        and _has_present_time_intent(q)
    )


def _has_present_time_intent(query: str) -> bool:
    q = query.lower()
    return any(hint in q for hint in PRESENT_TIME_HINTS) or any(
        phrase in q for phrase in EXPLICIT_PRESENT_TIME_PHRASES
    )


def should_use_rss_news_search(query: str) -> bool:
    """Return True when the query is explicitly asking for fresh news."""
    if _is_simple_current_fact_query(query):
        return False
    q = query.lower()
    has_explicit_time_window = _infer_window(query) is not None
    has_news_intent = any(hint in q for hint in NEWS_EVENT_HINTS)
    has_soft_recency = any(hint in q for hint in SOFT_RECENCY_HINTS)
    return has_explicit_time_window or has_news_intent or has_soft_recency


def _default_news_window(query: str) -> Literal["24h", "7d"] | None:
    q = query.lower()
    if any(hint in q for hint in RECENT_24H_HINTS):
        return "24h"
    if should_use_rss_news_search(query):
        return "7d"
    return None


def _normalize_search_query(query: str) -> str:
    """Strip guessed stale years from present-tense factual lookups."""
    current_year = datetime.now(timezone.utc).year

    def _replace_year(match: re.Match[str]) -> str:
        year = int(match.group(0))
        return "" if year < current_year else match.group(0)

    if _has_present_time_intent(query):
        normalized = re.sub(r"\b20\d{2}\b", _replace_year, query)
    else:
        normalized = query
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,")
    return normalized or query


def _google_news_query(query: str, window: Literal["24h", "7d"]) -> str:
    time_filter = "when:1d" if window == "24h" else "when:7d"
    return GOOGLE_NEWS_RSS.format(query=quote_plus(f"{query} {time_filter}"))


def _parse_pubdate(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _clean_google_news_link(link: str) -> str:
    # Google News RSS often provides a redirectable canonical URL already.
    return link.strip()


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_query_terms(query: str) -> set[str]:
    terms = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
    return {term for term in terms if term not in STOPWORDS}


def _score_news_item(item: dict, query_terms: set[str]) -> float:
    title = item.get("title", "").lower()
    snippet = item.get("snippet", "").lower()
    haystack = f"{title} {snippet}"

    score = 0.0

    # Prefer direct topical overlap with the user's query.
    score += sum(1.5 for term in query_terms if term in title)
    score += sum(0.75 for term in query_terms if term in snippet)

    # Prefer items with usable summary text.
    if item.get("snippet"):
        score += 1.5

    # Prefer very recent items.
    published = item.get("published")
    if published is not None:
        age_hours = max((datetime.now(timezone.utc) - published).total_seconds() / 3600, 0)
        score += max(0, 2 - age_hours / 24)

    # Penalize weak result types that tend to read like commentary or hubs.
    if any(hint in title for hint in LOW_QUALITY_TITLE_HINTS):
        score -= 3
    if any(hint in snippet for hint in LOW_QUALITY_SNIPPET_HINTS):
        score -= 2
    if title.startswith(("how ", "why ")):
        score -= 1.5
    if "opinion" in haystack or "analysis" in haystack:
        score -= 2

    # Penalize obvious section/topic pages.
    if len(title.split()) < 4:
        score -= 1

    return score


def _parse_rss_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = _clean_google_news_link(item.findtext("link") or "")
        pub_date = _parse_pubdate((item.findtext("pubDate") or "").strip())
        description = _strip_html(item.findtext("description") or "")
        source = ""
        source_node = item.find("source")
        if source_node is not None and source_node.text:
            source = source_node.text.strip()
        if not source and " - " in title:
            title, source = title.rsplit(" - ", 1)
        if not title or not link:
            continue
        items.append({
            "title": title.strip(),
            "url": link,
            "source": source or "Google News RSS",
            "published": pub_date,
            "snippet": description,
        })
    return items


def _filter_recent(items: list[dict], window: Literal["24h", "7d"]) -> list[dict]:
    now = datetime.now(timezone.utc)
    cutoff = now - (timedelta(days=1) if window == "24h" else timedelta(days=7))
    recent = []
    seen: set[str] = set()
    for item in items:
        published = item.get("published")
        dedupe_key = item["url"]
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        if published is not None and published < cutoff:
            continue
        recent.append(item)
    recent.sort(
        key=lambda item: item.get("published") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return recent


def _rank_news_items(query: str, items: list[dict], limit: int = 5) -> list[dict]:
    query_terms = _extract_query_terms(query)
    ranked = sorted(
        items,
        key=lambda item: (
            _score_news_item(item, query_terms),
            item.get("published") or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _format_rss_results(query: str, window: Literal["24h", "7d"], results: list[dict]) -> str:
    lines = [
        "STATUS: OK",
        "Method: RSS news search",
        f"Window: {'last 24 hours' if window == '24h' else 'last 7 days'}",
        f"Query: {query}",
    ]
    for i, item in enumerate(results, 1):
        lines.append(f"{i}. {item['title']}")
        lines.append(f"   Source: {item['source']}")
        if item.get("published") is not None:
            lines.append(
                f"   Published: {item['published'].astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
        if item.get("snippet"):
            lines.append(f"   Snippet: {item['snippet'][:300]}")
        lines.append(f"   URL: {item['url']}")
    return "\n".join(lines)


def _search_recent_news_rss(query: str, window: Literal["24h", "7d"]) -> str | None:
    rss_url = _google_news_query(query, window)
    xml_text = _http_get(rss_url)
    parsed = _parse_rss_feed(xml_text)
    recent = _filter_recent(parsed, window)
    ranked = _rank_news_items(query, recent, limit=5)
    if not ranked:
        return None
    return _format_rss_results(query, window, ranked)


def _with_query_used(result: str, query_used: str) -> str:
    if not result:
        return f"STATUS: ERROR\nQuery Used: {query_used}\nWeb search returned no content."
    lines = result.splitlines()
    if lines and lines[0].startswith("STATUS:"):
        return "\n".join([lines[0], f"Query Used: {query_used}", *lines[1:]])
    return f"Query Used: {query_used}\n{result}"


@tool
def web_search(query: str) -> str:
    """Search the web operationally.

    Pass the user's search intent with minimal rewriting. Do not append a year
    or date constraint unless the user explicitly asked for one.

    For explicit last-24-hours or last-week news questions, use RSS-backed
    news retrieval. For evergreen,
    background, or older information, fall back to the existing search tool.
    """
    try:
        normalized_query = _normalize_search_query(query)

        if _is_simple_current_fact_query(normalized_query):
            return _with_query_used(legacy_web_search.invoke(normalized_query), normalized_query)

        window = _default_news_window(normalized_query)
        if should_use_rss_news_search(normalized_query) and window is not None:
            rss_result = _search_recent_news_rss(normalized_query, window)
            if rss_result:
                return _with_query_used(rss_result, normalized_query)
        return _with_query_used(legacy_web_search.invoke(normalized_query), normalized_query)
    except Exception as exc:
        try:
            normalized_query = _normalize_search_query(query)
            return _with_query_used(legacy_web_search.invoke(normalized_query), normalized_query)
        except Exception:
            return f"STATUS: ERROR\nOperational web search failed. Error: {exc}"

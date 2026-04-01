"""
RSS tool for recent Israeli political news.
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from langchain_core.tools import tool


USER_AGENT = "Mozilla/5.0 (compatible; ElectionAgent/1.0)"
MAX_ITEMS = 6
RSS_TIMEOUT_SECONDS = 10

DEFAULT_FEED_QUERIES = [
    "Israeli politics OR Knesset OR Netanyahu OR coalition Israel",
    "Israel election OR Knesset election OR Israeli parties",
]

POLITICS_KEYWORDS = {
    "israel",
    "israeli",
    "knesset",
    "coalition",
    "government",
    "opposition",
    "election",
    "elections",
    "netanyahu",
    "lapid",
    "gantz",
    "ben-gvir",
    "smotrich",
    "shas",
    "likud",
    "party",
    "parties",
    "cabinet",
    "minister",
    "ministers",
    "judicial",
    "hostage",
    "ceasefire",
}

GENERIC_QUERY_TOKENS = {
    "about",
    "breaking",
    "current",
    "currently",
    "developments",
    "government",
    "headline",
    "headlines",
    "israel",
    "israeli",
    "latest",
    "live",
    "minister",
    "ministers",
    "most",
    "news",
    "political",
    "politics",
    "recent",
    "regarding",
    "realtime",
    "real",
    "time",
    "today",
    "update",
    "updates",
    "what",
    "who",
}


def _http_get(url: str, timeout: int = RSS_TIMEOUT_SECONDS) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _clean_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_direct_url_from_description(description: str) -> str:
    if not description:
        return ""
    urls = re.findall(r'https?://[^\s"<>()]+', description)
    for url in urls:
        if "news.google.com" not in urlparse(url).netloc:
            return url
    return ""


def _select_item_url(link: str, description: str, publisher_url: str) -> str:
    direct_url = _extract_direct_url_from_description(description)
    if direct_url:
        return direct_url

    if publisher_url:
        return publisher_url

    if link and "news.google.com" not in urlparse(link).netloc:
        return link

    return ""


def _google_news_rss_url(query: str) -> str:
    return (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )


def _build_feed_queries(query: str) -> list[str]:
    cleaned_query = _clean_text(query)
    freshness_suffix = ""
    lowered = cleaned_query.lower()
    if "today" in lowered:
        freshness_suffix = " when:1d"
    elif any(token in lowered for token in ["latest", "recent", "most recent", "breaking"]):
        freshness_suffix = " when:7d"

    tokens = [
        token for token in re.findall(r"[a-zA-Z]{3,}", cleaned_query.lower())
        if token not in GENERIC_QUERY_TOKENS
    ]
    focused_query = " ".join(tokens)

    queries: list[str] = []
    for candidate in [cleaned_query + freshness_suffix, focused_query + freshness_suffix]:
        if candidate and candidate not in queries:
            queries.append(candidate)

    for candidate in DEFAULT_FEED_QUERIES:
        if candidate not in queries:
            queries.append(candidate)
    return queries


def _recency_window_hours(query: str) -> int | None:
    lowered = query.lower()
    if "today" in lowered:
        return 24
    if any(token in lowered for token in ["latest", "recent", "most recent", "breaking"]):
        return 168
    return None


def _parse_date(text: str | None) -> datetime | None:
    if not text:
        return None

    raw = text.strip()
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _text_or_empty(node: ET.Element | None, path: str) -> str:
    if node is None:
        return ""
    found = node.find(path)
    return _clean_text(found.text if found is not None else "")


def _extract_atom_link(entry: ET.Element) -> str:
    for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
        href = link.attrib.get("href", "").strip()
        if href:
            return href
    return ""


def _extract_item_source(item: ET.Element, fallback: str) -> tuple[str, str]:
    source_node = item.find("source")
    if source_node is None:
        return fallback, ""
    return _clean_text(source_node.text) or fallback, source_node.attrib.get("url", "").strip()


def _clean_google_news_title(title: str, publisher: str) -> str:
    cleaned = title.strip()
    if publisher and cleaned.endswith(f" - {publisher}"):
        return cleaned[: -(len(publisher) + 3)].strip()
    return cleaned


def _format_published_timestamp(published: datetime | None) -> str:
    if not published:
        return "Unknown"
    return published.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _parse_rss_items(xml_bytes: bytes, source_name: str) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    items: list[dict] = []

    channel = root.find("channel")
    if channel is not None:
        for item in channel.findall("item"):
            publisher, publisher_url = _extract_item_source(item, source_name)
            title = _clean_google_news_title(_text_or_empty(item, "title"), publisher)
            description = _text_or_empty(item, "description")
            link = _text_or_empty(item, "link")
            resolved_link = _select_item_url(link, description, publisher_url)
            published = _parse_date(_text_or_empty(item, "pubDate"))
            items.append(
                {
                    "title": title,
                    "summary": description,
                    "url": resolved_link,
                    "published": published,
                    "source": publisher,
                    "source_url": publisher_url,
                }
            )
        return items

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", ns):
        title = _text_or_empty(entry, "atom:title")
        summary = _text_or_empty(entry, "atom:summary") or _text_or_empty(entry, "atom:content")
        published = _parse_date(
            _text_or_empty(entry, "atom:updated") or _text_or_empty(entry, "atom:published")
        )
        items.append(
            {
                "title": title,
                "summary": summary,
                "url": _extract_atom_link(entry),
                "published": published,
                "source": source_name,
                "source_url": "",
            }
        )
    return items


def _is_politics_item(item: dict, query: str) -> bool:
    haystack = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    title = item.get("title", "").lower()
    query_lower = query.lower()
    query_tokens = {
        token for token in re.findall(r"[a-zA-Z]{3,}", query_lower)
        if token not in GENERIC_QUERY_TOKENS
    }

    phrase_patterns = [
        "defense minister",
        "prime minister",
        "foreign minister",
        "finance minister",
        "knesset speaker",
    ]
    matched_phrases = [phrase for phrase in phrase_patterns if phrase in query_lower]

    if matched_phrases:
        # For office-holder queries, require the exact role phrase to appear.
        # This avoids broad Israel-politics stories being misclassified as being
        # "about" the role when they only mention adjacent topics.
        if not any(phrase in haystack for phrase in matched_phrases):
            return False

    score = 0
    for phrase in matched_phrases:
        if phrase in haystack:
            score += 4
        elif all(word in haystack for word in phrase.split()):
            score += 2

    for token in query_tokens:
        if token in title:
            score += 3
        elif token in haystack:
            score += 1

    if matched_phrases or query_tokens:
        return score >= 3
    return any(keyword in haystack for keyword in POLITICS_KEYWORDS)


def _dedupe_and_sort(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []

    for item in items:
        key = item.get("url") or item.get("title") or ""
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    deduped.sort(
        key=lambda item: item.get("published") or datetime(1970, 1, 1, tzinfo=timezone.utc),
        reverse=True,
    )
    return deduped


def _filter_by_recency(items: list[dict], query: str) -> list[dict]:
    window_hours = _recency_window_hours(query)
    if window_hours is None:
        return items

    now = datetime.now(timezone.utc)
    fresh_items = []
    for item in items:
        published = item.get("published")
        if published is None:
            continue
        age_hours = (now - published).total_seconds() / 3600
        if age_hours <= window_hours:
            fresh_items.append(item)

    return fresh_items if fresh_items else items


def _format_age(published: datetime | None) -> str:
    if not published:
        return "Unknown time"

    now = datetime.now(timezone.utc)
    delta = now - published
    total_hours = int(delta.total_seconds() // 3600)
    if total_hours < 1:
        minutes = max(1, int(delta.total_seconds() // 60))
        return f"{minutes}m ago"
    if total_hours < 48:
        return f"{total_hours}h ago"
    return published.strftime("%Y-%m-%d")


def _format_results(query: str, items: list[dict]) -> str:
    if not items:
        return f"STATUS: NO_RESULTS\nQuery: {query}\nNo recent Israeli political RSS items matched."

    lines = [f"STATUS: OK", f"Query: {query}", f"Items: {min(len(items), MAX_ITEMS)}"]
    for idx, item in enumerate(items[:MAX_ITEMS], 1):
        lines.append(f"{idx}. {item['title']}")
        lines.append(f"   Source: {item['source']}")
        lines.append(f"   Published: {_format_published_timestamp(item.get('published'))} ({_format_age(item.get('published'))})")
        if item.get("summary"):
            lines.append(f"   Summary: {item['summary'][:280]}")
        if item.get("url"):
            lines.append(f"   URL: {item['url']}")
        if item.get("source_url"):
            lines.append(f"   Publisher URL: {item['source_url']}")
    return "\n".join(lines)


@tool
def israel_politics_rss(query: str = "latest Israeli political developments") -> str:
    """Fetch recent RSS headlines about Israeli politics and current events.
    Use for live or recent political developments, coalition negotiations,
    leadership news, Knesset activity, and breaking updates."""
    try:
        items: list[dict] = []
        for feed_query in _build_feed_queries(query):
            source_name = f"Google News RSS: {feed_query}"
            parsed_items = _parse_rss_items(_http_get(_google_news_rss_url(feed_query)), source_name)
            matched_items = [item for item in parsed_items if _is_politics_item(item, query)]
            items.extend(matched_items)

        return _format_results(query, _filter_by_recency(_dedupe_and_sort(items), query))
    except Exception as e:
        return f"STATUS: ERROR\nRSS news is currently unavailable. Error: {e}"

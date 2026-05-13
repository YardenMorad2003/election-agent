"""
Web Search Tool — fetches concise external context from public web endpoints.
"""
import json
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import tool


USER_AGENT = "Mozilla/5.0 (compatible; ElectionAgent/1.0)"


def _normalize_url(raw_url: str) -> str:
    if raw_url.startswith("http://") or raw_url.startswith("https://"):
        return raw_url

    parsed = urlparse(raw_url)
    params = parse_qs(parsed.query)
    if "uddg" in params and params["uddg"]:
        return unquote(params["uddg"][0])

    if raw_url.startswith("//"):
        return f"https:{raw_url}"

    if raw_url.startswith("/"):
        return f"https://lite.duckduckgo.com{raw_url}"

    return raw_url


def _http_get(url: str, timeout: int = 10) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _search_duckduckgo_instant(query: str) -> list[dict]:
    params = urlencode({
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    })
    payload = json.loads(_http_get(f"https://api.duckduckgo.com/?{params}"))

    results = []
    if payload.get("AbstractText"):
        results.append({
            "title": payload.get("Heading") or "DuckDuckGo Instant Answer",
            "snippet": payload["AbstractText"],
            "url": payload.get("AbstractURL", ""),
            "source": "DuckDuckGo",
        })

    for topic in payload.get("RelatedTopics", []):
        entries = topic.get("Topics", []) if isinstance(topic, dict) and "Topics" in topic else [topic]
        for entry in entries:
            if not entry.get("Text"):
                continue
            results.append({
                "title": entry.get("FirstURL", "").rsplit("/", 1)[-1].replace("_", " ") or "Related topic",
                "snippet": entry["Text"],
                "url": entry.get("FirstURL", ""),
                "source": "DuckDuckGo",
            })
            if len(results) >= 5:
                return results
    return results[:5]


class _DuckDuckGoLiteParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_link = False
        self.current_href = ""
        self.current_text: list[str] = []
        self.results: list[dict] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        normalized_href = _normalize_url(href)
        if not normalized_href.startswith("http"):
            return
        # Skip DDG-internal links (homepage, lite, help, about). When DDG
        # rate-limits or bot-deflects, the response contains only links back
        # to itself — accepting those as results lets the synthesizer LLM
        # treat empty output as STATUS: OK and hallucinate from training.
        host = urlparse(normalized_href).netloc.lower()
        if host.endswith("duckduckgo.com"):
            return
        self.in_link = True
        self.current_href = normalized_href
        self.current_text = []

    def handle_endtag(self, tag):
        if tag != "a" or not self.in_link:
            return
        title = unescape("".join(self.current_text)).strip()
        if title and len(self.results) < 5:
            self.results.append({
                "title": title,
                "snippet": "",
                "url": self.current_href,
                "source": "DuckDuckGo Lite",
            })
        self.in_link = False
        self.current_href = ""
        self.current_text = []

    def handle_data(self, data):
        if self.in_link:
            self.current_text.append(data)


def _search_duckduckgo_lite(query: str) -> list[dict]:
    # POST: the lite endpoint frequently returns the DDG homepage (no results)
    # when queried with GET, even with a valid User-Agent. POST is reliable.
    data = urlencode({"q": query}).encode("utf-8")
    req = Request(
        "https://lite.duckduckgo.com/lite/",
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    parser = _DuckDuckGoLiteParser()
    parser.feed(html)
    return parser.results[:5]


def _search_wikipedia(query: str) -> list[dict]:
    params = urlencode({
        "action": "opensearch",
        "search": query,
        "limit": 3,
        "namespace": 0,
        "format": "json",
    })
    payload = json.loads(_http_get(f"https://en.wikipedia.org/w/api.php?{params}"))
    titles = payload[1] if len(payload) > 1 else []
    descriptions = payload[2] if len(payload) > 2 else []
    urls = payload[3] if len(payload) > 3 else []

    return [
        {
            "title": title,
            "snippet": description or "Wikipedia result",
            "url": url,
            "source": "Wikipedia",
        }
        for title, description, url in zip(titles, descriptions, urls)
    ]


def _search_wikipedia_summary(query: str) -> list[dict]:
    seed_results = _search_wikipedia(query)
    if not seed_results:
        return []

    top = seed_results[0]
    title = top["title"].replace(" ", "_")
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    payload = json.loads(_http_get(summary_url))

    extract = payload.get("extract")
    page_url = payload.get("content_urls", {}).get("desktop", {}).get("page", top["url"])
    if not extract:
        return seed_results

    return [{
        "title": payload.get("title", top["title"]),
        "snippet": extract,
        "url": page_url,
        "source": "Wikipedia Summary",
    }]


def _format_results(results: list[dict]) -> str:
    if not results:
        return "STATUS: NO_RESULTS\nNo web results found."

    lines = []
    lines.append("STATUS: OK")
    for i, item in enumerate(results[:5], 1):
        lines.append(f"{i}. {item['title']}")
        lines.append(f"   Source: {item['source']}")
        if item.get("snippet"):
            lines.append(f"   Snippet: {item['snippet']}")
        if item.get("url"):
            lines.append(f"   URL: {item['url']}")
    return "\n".join(lines)


def _is_fact_lookup(query: str) -> bool:
    q = query.lower().strip()
    prefixes = (
        "who is", "who's", "what is", "what's", "which is",
        "when was", "when did", "when is", "name",
    )
    fact_terms = (
        "current", "appointed", "appointment", "minister", "secretary",
        "president", "prime minister", "governor", "leader", "mayor",
    )
    return q.startswith(prefixes) or any(term in q for term in fact_terms)


@tool
def web_search(query: str) -> str:
    """Search the public web for context not covered by the local election database.
    Use for current events, external background, party history, leaders, or
    questions that explicitly ask for web/search/news results."""
    try:
        results = []
        fact_lookup = _is_fact_lookup(query)

        if fact_lookup:
            # Live SERP first. DDG Instant Answer returns the institutional
            # concept abstract for officeholder queries (e.g. the Wikipedia
            # article on "President of the United States"), which never names
            # the current officeholder. Letting that block the cascade left
            # the synthesizer LLM to hallucinate the name from its training
            # cutoff (gave "Joe Biden" for "who is the us president?").
            results = _search_duckduckgo_lite(query)
            if not results:
                results = _search_wikipedia_summary(query)
            if not results:
                results = _search_duckduckgo_instant(query)
        else:
            results = _search_duckduckgo_instant(query)
            if not results:
                results = _search_wikipedia_summary(query)
            if not results:
                results = _search_duckduckgo_lite(query)

        if not results:
            results = _search_wikipedia(query)
        if not results and any(token in query.lower() for token in ["current", "currently", "latest", "today"]):
            stripped_query = query
            for token in [" current", " currently", " latest", " today", " 2023", " 2024", " 2025", " 2026"]:
                stripped_query = stripped_query.replace(token, "")
            results = _search_wikipedia_summary(stripped_query.strip())
        return _format_results(results)
    except Exception as e:
        return f"STATUS: ERROR\nWeb search is currently unavailable. Error: {e}"

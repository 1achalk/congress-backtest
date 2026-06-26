"""
Shared scrape mechanics for the official Senate eFD portal (efdsearch.senate.gov).
Used by both efd_probe.py (recon) and ingest.py (the real pull). See CLAUDE.md D0.

NOTE: confirm against the LIVE site before reuse -- gov sites drift.
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = "https://efdsearch.senate.gov"
SEARCH_URL = f"{ROOT}/search/"
LANDING_URL = f"{ROOT}/search/home/"
REPORTS_URL = f"{ROOT}/search/report/data/"

PAGE_LENGTH = 100
REPORT_TYPES = "[11]"        # Periodic Transaction Report
FILER_TYPES = "[1,5]"        # current + former Senators (4=Candidate -> 0 rows in 2021+ window)
THROTTLE_SECS = 1.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 (congress-backtest scraper)"
)

PAPER_PREFIX = "/search/view/paper/"
PTR_PREFIX = "/search/view/ptr/"

TX_COLUMNS = [
    "transaction_date", "owner", "ticker", "asset_description",
    "asset_type", "type", "amount", "comment",
]


def new_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def authenticate(session):
    """Clear the click-through agreement gate; return the live CSRF token."""
    r = session.get(SEARCH_URL, timeout=15)
    if not r.url.rstrip("/").endswith("/search/home"):
        raise RuntimeError(f"Unexpected redirect from /search/: {r.url}")
    m = re.search(r'csrfmiddlewaretoken"\s+value="([^"]+)"', r.text)
    if not m:
        raise RuntimeError("Could not find csrfmiddlewaretoken on landing page")
    token = m.group(1)
    r2 = session.post(
        LANDING_URL,
        data={"csrfmiddlewaretoken": token, "prohibition_agreement": "1"},
        headers={"Referer": LANDING_URL},
        timeout=15,
    )
    if "filedReports" not in r2.text:
        raise RuntimeError("Agreement POST did not reach the search results page")
    return token


def cache_path(cache_dir, *parts):
    p = Path(cache_dir).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fetch_search_page(session, token, start, submitted_start_date, cache_dir):
    cache_file = cache_path(cache_dir, "search_pages", f"start_{start:05d}.json")
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    payload = {
        "start": str(start),
        "length": str(PAGE_LENGTH),
        "report_types": REPORT_TYPES,
        "filer_types": FILER_TYPES,
        "submitted_start_date": submitted_start_date,
        "submitted_end_date": "",
        "candidate_state": "",
        "senator_state": "",
        "office_id": "",
        "first_name": "",
        "last_name": "",
        "csrfmiddlewaretoken": token,
    }
    r = session.post(REPORTS_URL, data=payload, headers={"Referer": SEARCH_URL}, timeout=20)
    r.raise_for_status()
    data = r.json()
    cache_file.write_text(json.dumps(data))
    time.sleep(THROTTLE_SECS)
    return data


def all_search_rows(session, token, submitted_start_date, cache_dir):
    rows, start = [], 0
    while True:
        page = fetch_search_page(session, token, start, submitted_start_date, cache_dir)
        batch = page["data"]
        if not batch:
            break
        rows.extend(batch)
        start += PAGE_LENGTH
        if start >= page["recordsTotal"]:
            break
    return rows


def parse_search_row(row):
    first, last, office, link_html, date_received = row
    a = BeautifulSoup(link_html, "lxml").find("a")
    href = a["href"]
    title = a.get_text(strip=True)
    return {
        "senator": f"{first.strip()} {last.strip()}",
        "office": office.strip(),
        "ptr_link": href,
        "date_received": date_received.strip(),
        "is_paper": href.startswith(PAPER_PREFIX),
        "is_amendment": "Amendment" in title,
        "title": title,
    }


def fetch_electronic_report_html(session, token, href, cache_dir):
    uid = href.strip("/").split("/")[-1]
    cache_file = cache_path(cache_dir, "reports", f"{uid}.html")
    if cache_file.exists():
        return cache_file.read_text(), token

    url = ROOT + href
    r = session.get(url, timeout=20)
    if r.url.rstrip("/").endswith("/search/home"):
        # session/agreement expired mid-scrape -- re-auth once and retry
        token = authenticate(session)
        r = session.get(url, timeout=20)
    r.raise_for_status()
    cache_file.write_text(r.text)
    time.sleep(THROTTLE_SECS)
    return r.text, token


def parse_transactions(html):
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None or table.find("tbody") is None:
        return []
    txs = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 9:
            continue
        # cells[0] is the "#" row index; 1..8 map to TX_COLUMNS. Use a space
        # separator everywhere -- some cells (ticker, asset description) contain
        # multiple text nodes joined by <br/> with no whitespace between them, and
        # a plain get_text(strip=True) silently glues them into one bogus token
        # (e.g. a spin-off distribution row listing both "DELL" and "VMW" tickers
        # becomes "DELLVMW"). The ticker cell specifically holds 0-2 <a> links
        # (0 for "--", 2 for a spin-off/exchange row crediting two securities) --
        # extract those directly rather than the cell's raw text.
        cells = [td.get_text(" ", strip=True) for td in tds]
        ticker_links = [a.get_text(strip=True) for a in tds[3].find_all("a")]
        if ticker_links:
            cells[3] = " / ".join(ticker_links)
        tx = dict(zip(TX_COLUMNS, cells[1:9]))
        txs.append(tx)
    return txs


def scrape_flattened_transactions(session, token, submitted_start_date, cache_dir):
    """
    Full pull: search rows -> dedupe exact-duplicate ptr_links -> fetch electronic
    reports -> flatten to one row per transaction. Paper reports are skipped (not
    fetched) and counted. Carries is_amendment down to each transaction row so D12
    can be applied downstream -- this function does NOT apply D12 itself.
    """
    raw_rows = all_search_rows(session, token, submitted_start_date, cache_dir)
    reports = [parse_search_row(r) for r in raw_rows]

    seen, deduped = set(), []
    for rep in reports:
        if rep["ptr_link"] in seen:
            continue
        seen.add(rep["ptr_link"])
        deduped.append(rep)

    flattened = []
    for rep in deduped:
        if rep["is_paper"]:
            continue
        html, token = fetch_electronic_report_html(session, token, rep["ptr_link"], cache_dir)
        for tx in parse_transactions(html):
            flattened.append({
                "senator": rep["senator"],
                "transaction_date": tx["transaction_date"],
                "date_received": rep["date_received"],
                "ticker": tx["ticker"],
                "asset_description": tx["asset_description"],
                "asset_type": tx["asset_type"],
                "type": tx["type"],
                "amount": tx["amount"],
                "owner": tx["owner"],
                "comment": tx["comment"],
                "ptr_link": rep["ptr_link"],
                "is_amendment": rep["is_amendment"],
            })

    return flattened, deduped
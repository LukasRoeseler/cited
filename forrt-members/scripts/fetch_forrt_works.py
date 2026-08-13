"""Fetch FORRT contributors and their OpenAlex works.

Network-only. Run this once, then the report reads the CSVs it writes.
Never call this from the Quarto render.

Usage:
    py -3.12 forrt-members/scripts/fetch_forrt_works.py

Writes forrt-members/data/*.csv and caches raw per-author payloads in
forrt-members/cache/ (gitignored) so re-runs and interrupted runs are cheap.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

import yaml

MAILTO = "lukas.roeseler@uni-muenster.de"
UA = f"cited-forrt-analysis/1.0 (mailto:{MAILTO})"

CONTRIB_CSV_URL = (
    "https://raw.githubusercontent.com/forrtproject/forrtproject.github.io/"
    "build-resources/scripts/forrt_contribs/contributors_cache.csv"
)
PUBS_YAML_URL = (
    "https://raw.githubusercontent.com/forrtproject/forrtproject.github.io/"
    "main/data/publications.yaml"
)

WORK_SELECT = "id,doi,publication_year,cited_by_count,type,primary_location,authorships"
SOURCE_SELECT = "id,display_name,type,summary_stats"

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
DATA = os.path.join(BASE, "data")
CACHE = os.path.join(BASE, "cache", "works")

ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")

n_api_calls = 0


def http_json(url: str, tries: int = 4) -> dict:
    """GET JSON with retries and a polite delay."""
    global n_api_calls
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as resp:
                n_api_calls += 1
                time.sleep(0.11)
                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = exc
            code = getattr(exc, "code", None)
            if code in (400, 404):
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after {tries} tries: {url}") from last


def http_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read().decode("utf-8")


def normalize_orcid(raw: str | None) -> str:
    """Strip scheme/host/slashes and uppercase, mirroring the game's normalizeOrcid."""
    s = (raw or "").strip()
    s = re.sub(r"^https?://", "", s, flags=re.I)
    s = re.sub(r"^(www\.)?orcid\.org/", "", s, flags=re.I)
    return s.strip("/").strip().upper()


def normalize_doi(raw: str | None) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.I)
    return s.strip().lower()


def person_id(orcid: str) -> str:
    return hashlib.sha1(orcid.encode()).hexdigest()[:12]


def short_id(url: str | None) -> str | None:
    if not url:
        return None
    return url.rstrip("/").rsplit("/", 1)[-1]


def trim_work(w: dict) -> dict:
    """Keep only what the analysis needs. The authorships array is dropped after
    counting, because storing it would balloon the cache."""
    src = (w.get("primary_location") or {}).get("source") or {}
    auths = w.get("authorships") or []
    orcids = []
    for a in auths:
        o = (a.get("author") or {}).get("orcid")
        if o:
            orcids.append(normalize_orcid(o))
    return {
        "work_id": short_id(w.get("id")),
        "doi": normalize_doi(w.get("doi")),
        "publication_year": w.get("publication_year"),
        "type": w.get("type"),
        "cited_by_count": w.get("cited_by_count"),
        "source_id": short_id(src.get("id")),
        "source_type": src.get("type"),
        "n_authors": len(auths),
        "author_orcids": orcids,
    }


def fetch_author_works(orcid: str) -> list[dict]:
    """All works for one ORCID, cursor-paginated. Cached per ORCID."""
    path = os.path.join(CACHE, f"{orcid}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    out: list[dict] = []
    cursor = "*"
    while cursor:
        url = (
            "https://api.openalex.org/works?"
            + urllib.parse.urlencode(
                {
                    "filter": f"author.orcid:{orcid}",
                    "per_page": 200,
                    "cursor": cursor,
                    "select": WORK_SELECT,
                    "mailto": MAILTO,
                }
            )
        )
        data = http_json(url)
        results = data.get("results") or []
        out.extend(trim_work(w) for w in results)
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not results:
            break

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return out


def fetch_works_by_doi(dois: list[str]) -> dict[str, dict]:
    """Resolve FORRT publication DOIs directly, so listed works are present even if
    a contributor's ORCID is missing from OpenAlex."""
    found: dict[str, dict] = {}
    for i in range(0, len(dois), 25):
        chunk = dois[i : i + 25]
        flt = "doi:" + "|".join(chunk)
        url = (
            "https://api.openalex.org/works?"
            + urllib.parse.urlencode(
                {"filter": flt, "per_page": 50, "select": WORK_SELECT, "mailto": MAILTO}
            )
        )
        data = http_json(url)
        for w in data.get("results") or []:
            tw = trim_work(w)
            if tw["doi"]:
                found[tw["doi"]] = tw
    return found


def fetch_sources(source_ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(source_ids), 50):
        chunk = source_ids[i : i + 50]
        url = (
            "https://api.openalex.org/sources?"
            + urllib.parse.urlencode(
                {
                    "filter": "ids.openalex:" + "|".join(chunk),
                    "per_page": len(chunk),
                    "select": SOURCE_SELECT,
                    "mailto": MAILTO,
                }
            )
        )
        data = http_json(url)
        for s in data.get("results") or []:
            stats = s.get("summary_stats") or {}
            mc = stats.get("2yr_mean_citedness")
            out[short_id(s.get("id"))] = {
                "source_id": short_id(s.get("id")),
                "display_name": s.get("display_name"),
                "type": s.get("type"),
                "mean_citedness_2yr": mc if isinstance(mc, (int, float)) else "",
                "works_count": stats.get("works_count", ""),
                "h_index": stats.get("h_index", ""),
            }
        print(f"    sources {min(i + 50, len(source_ids))}/{len(source_ids)}", flush=True)
    return out


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"  wrote {name} ({len(rows)} rows)")


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)

    print("1. FORRT contributor CSV")
    text = http_text(CONTRIB_CSV_URL)
    rows = list(csv.DictReader(text.splitlines()))
    print(f"  {len(rows)} person-project rows")

    # ORCID normalization audit trail
    norm_rows = []
    seen_raw = set()
    for r in rows:
        raw = (r.get("ORCID iD") or "").strip()
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        clean = normalize_orcid(raw)
        if not raw:
            status = "missing"
        elif ORCID_RE.match(clean):
            status = "valid"
        else:
            status = "malformed"
        norm_rows.append({"orcid_raw": raw, "orcid_clean": clean, "status": status})

    valid_orcids = sorted({r["orcid_clean"] for r in norm_rows if r["status"] == "valid"})
    n_malformed = sum(1 for r in norm_rows if r["status"] == "malformed")
    n_no_orcid_rows = sum(1 for r in rows if not (r.get("ORCID iD") or "").strip())
    print(f"  {len(valid_orcids)} valid unique ORCIDs, {n_malformed} malformed")

    # people and projects
    people: dict[str, dict] = {}
    projects: list[dict] = []
    names_without_orcid = set()
    for r in rows:
        clean = normalize_orcid(r.get("ORCID iD"))
        name = (
            (r.get("First name") or "").strip(),
            (r.get("Middle name") or "").strip(),
            (r.get("Surname") or "").strip(),
        )
        if not ORCID_RE.match(clean):
            names_without_orcid.add(name)
            continue
        pid = person_id(clean)
        if pid not in people:
            people[pid] = {
                "person_id": pid,
                "orcid": clean,
                "first_name": name[0],
                "middle_name": name[1],
                "surname": name[2],
                "n_projects": 0,
            }
        people[pid]["n_projects"] += 1
        projects.append(
            {
                "person_id": pid,
                "project_name": (r.get("Project Name") or "").strip(),
                "project_url": (r.get("Project URL") or "").strip(),
            }
        )

    print("2. FORRT publications.yaml")
    pubs = yaml.safe_load(http_text(PUBS_YAML_URL))
    pub_entries = []
    for e in pubs or []:
        # Take the union of both DOI fields, not just the first one found. For two
        # entries these disagree: altmetric_doi points at the preprint while
        # links.doi points at the published journal article. Keeping only the
        # first would have discarded the journal version, which is the version
        # that carries a journal mean citedness.
        dois = []
        for c in (e.get("altmetric_doi"), (e.get("links") or {}).get("doi")):
            d = normalize_doi(str(c) if c else "")
            if d.startswith("10.") and d not in dois:
                dois.append(d)
        for d in dois or [""]:
            pub_entries.append(
                {
                    "doi": d,
                    "title": (e.get("title") or "").strip(),
                    "year": e.get("year", ""),
                    "type": e.get("type", ""),
                    "status": e.get("status", ""),
                }
            )
    forrt_dois = sorted({p["doi"] for p in pub_entries if p["doi"]})
    print(f"  {len(pub_entries)} entries, {len(forrt_dois)} unique DOIs")

    print("3. Resolving FORRT DOIs in OpenAlex")
    forrt_works = fetch_works_by_doi(forrt_dois)
    print(f"  matched {len(forrt_works)}/{len(forrt_dois)}")

    print(f"4. Fetching works for {len(valid_orcids)} ORCIDs")
    works: dict[str, dict] = {}
    author_works: set[tuple[str, str]] = set()
    errored: list[str] = []
    zero_works: list[str] = []

    def absorb(orcid: str, recs: list[dict]) -> None:
        pid = person_id(orcid)
        for w in recs:
            wid = w["work_id"]
            if not wid:
                continue
            if wid not in works:
                works[wid] = {k: v for k, v in w.items() if k != "author_orcids"}
                works[wid]["_orcids"] = set(w.get("author_orcids") or [])
            else:
                works[wid]["_orcids"].update(w.get("author_orcids") or [])
            author_works.add((pid, wid))

    for i, orcid in enumerate(valid_orcids, 1):
        try:
            recs = fetch_author_works(orcid)
        except Exception as exc:  # noqa: BLE001
            errored.append(f"{orcid}: {type(exc).__name__}")
            print(f"  [{i}/{len(valid_orcids)}] {orcid} ERROR {exc}", flush=True)
            continue
        if not recs:
            zero_works.append(orcid)
        absorb(orcid, recs)
        if i % 25 == 0 or i == len(valid_orcids):
            print(
                f"  [{i}/{len(valid_orcids)}] works={len(works)} calls={n_api_calls}",
                flush=True,
            )

    # Fold in the directly resolved FORRT works, and record which contributors
    # authored them even if that work did not surface in their own crawl.
    valid_set = set(valid_orcids)
    for doi, w in forrt_works.items():
        wid = w["work_id"]
        if wid and wid not in works:
            works[wid] = {k: v for k, v in w.items() if k != "author_orcids"}
            works[wid]["_orcids"] = set(w.get("author_orcids") or [])
        elif wid:
            works[wid]["_orcids"].update(w.get("author_orcids") or [])
        for o in w.get("author_orcids") or []:
            if o in valid_set and wid:
                author_works.add((person_id(o), wid))

    forrt_work_ids = {w["work_id"] for w in forrt_works.values() if w["work_id"]}

    print("5. Fetching journal metadata")
    source_ids = sorted({w["source_id"] for w in works.values() if w.get("source_id")})
    print(f"  {len(source_ids)} distinct sources")
    sources = fetch_sources(source_ids)

    # finalize works rows
    n_forrt_authors_by_work: dict[str, int] = defaultdict(int)
    for pid, wid in author_works:
        n_forrt_authors_by_work[wid] += 1

    work_rows = []
    for wid, w in works.items():
        n_auth = w.get("n_authors") or 0
        work_rows.append(
            {
                "work_id": wid,
                "doi": w.get("doi") or "",
                "publication_year": w.get("publication_year") or "",
                "type": w.get("type") or "",
                "cited_by_count": w.get("cited_by_count")
                if w.get("cited_by_count") is not None
                else "",
                "source_id": w.get("source_id") or "",
                "source_type": w.get("source_type") or "",
                "n_forrt_authors": n_forrt_authors_by_work.get(wid, 0),
                "n_authors": n_auth,
                "n_authors_censored": int(n_auth >= 100),
                "forrt_listed": int(wid in forrt_work_ids),
            }
        )

    doi_to_wid = {w["doi"]: w["work_id"] for w in forrt_works.values() if w.get("doi")}
    pub_rows = [
        {**p, "matched_work_id": doi_to_wid.get(p["doi"], "")} for p in pub_entries
    ]

    print("6. Writing CSVs")
    write_csv(
        "contributors.csv",
        ["person_id", "orcid", "first_name", "middle_name", "surname", "n_projects"],
        sorted(people.values(), key=lambda r: r["person_id"]),
    )
    write_csv(
        "contributor_projects.csv",
        ["person_id", "project_name", "project_url"],
        projects,
    )
    write_csv(
        "orcid_normalization.csv", ["orcid_raw", "orcid_clean", "status"], norm_rows
    )
    write_csv(
        "works.csv",
        [
            "work_id",
            "doi",
            "publication_year",
            "type",
            "cited_by_count",
            "source_id",
            "source_type",
            "n_forrt_authors",
            "n_authors",
            "n_authors_censored",
            "forrt_listed",
        ],
        sorted(work_rows, key=lambda r: r["work_id"]),
    )
    write_csv(
        "author_works.csv",
        ["person_id", "work_id"],
        [{"person_id": p, "work_id": w} for p, w in sorted(author_works)],
    )
    write_csv(
        "sources.csv",
        [
            "source_id",
            "display_name",
            "type",
            "mean_citedness_2yr",
            "works_count",
            "h_index",
        ],
        sorted(sources.values(), key=lambda r: r["source_id"]),
    )
    write_csv(
        "forrt_publications.csv",
        ["doi", "title", "year", "type", "status", "matched_work_id"],
        pub_rows,
    )

    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "openalex_snapshot_year": datetime.now(timezone.utc).year,
        "script_version": "1.0",
        "n_api_calls": n_api_calls,
        "n_contributor_project_rows": len(rows),
        "n_valid_orcids": len(valid_orcids),
        "n_malformed_orcids": n_malformed,
        "n_rows_without_orcid": n_no_orcid_rows,
        "n_names_without_orcid": len(names_without_orcid),
        "n_unique_works": len(work_rows),
        "n_author_work_links": len(author_works),
        "n_sources": len(sources),
        "n_forrt_dois_listed": len(forrt_dois),
        "n_forrt_dois_resolved": len(forrt_works),
        "forrt_dois_unresolved": sorted(set(forrt_dois) - set(forrt_works)),
        "orcids_errored": errored,
        "orcids_zero_works": zero_works,
    }
    with open(os.path.join(DATA, "fetch_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("  wrote fetch_meta.json")
    print(
        f"\nDone. {len(work_rows)} unique works, {len(author_works)} author-work links, "
        f"{n_api_calls} API calls."
    )


if __name__ == "__main__":
    sys.exit(main())

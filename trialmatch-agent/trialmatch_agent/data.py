"""Reading data files, plus converting raw TREC / ClinicalTrials.gov files (free, local).

A trial is a dict: nct_id, title, conditions, interventions, brief_summary,
inclusion (list), exclusion (list), gender, min_age, max_age.
"""
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path


# ------------------------------------------------------------------ helpers
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_sentences(text):
    text = " ".join((text or "").split())
    return [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"(\[])', text) if s] if text else []


def truncate(text, n):
    text = text or ""
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


# -------------------------------------------------------------------- loading
def load_trials(path):
    return {t["nct_id"]: t for t in read_jsonl(path)}


def load_topics(path):
    """{topic_id: patient text}, from .jsonl or a TREC topics .xml."""
    rows = parse_topics_xml(path) if str(path).endswith(".xml") else read_jsonl(path)
    return {str(r["topic_id"]): r["text"] for r in rows}


def load_qrels(path):
    """TREC qrels lines `topic 0 NCT grade` -> {topic: {nct: grade}}."""
    qrels = defaultdict(dict)
    if path and Path(path).exists():
        for line in open(path, encoding="utf-8"):
            p = line.split()
            if len(p) >= 4:
                qrels[p[0]][p[2]] = int(p[3])
    return dict(qrels)


def load_run(path):
    """TREC run lines `topic Q0 NCT rank score tag` -> {topic: [nct, ...]}."""
    rows = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 5:
            rows[p[0]].append((int(p[3]), p[2]))
    return {t: [n for _, n in sorted(v)] for t, v in rows.items()}


# ---------------------------------------------------------- raw data parsing
_INC = re.compile(r"inclusion\s+criteria\s*:?", re.I)
_EXC = re.compile(r"exclusion\s+criteria\s*:?", re.I)
_BULLET = re.compile(r"^\s*(?:[-*\u2022\u00b7]|\d{1,2}[.)]|\(\d{1,2}\)|[a-zA-Z][.)])\s+")


def _clean(t):
    return " ".join((t or "").split())


def _items(text):
    items, cur = [], []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or _BULLET.match(s):
            if cur:
                items.append(" ".join(cur))
            cur = [_BULLET.sub("", s, count=1)] if s else []
        else:
            cur.append(s)
    if cur:
        items.append(" ".join(cur))
    return [_clean(i) for i in items if len(_clean(i)) > 3]


def split_criteria(raw):
    """Eligibility text block -> (inclusion list, exclusion list)."""
    raw = raw or ""
    m_inc, m_exc = _INC.search(raw), _EXC.search(raw)
    if m_exc and (not m_inc or m_exc.start() > m_inc.start()):
        return _items(raw[(m_inc.end() if m_inc else 0): m_exc.start()]), _items(raw[m_exc.end():])
    if m_exc and m_inc:  # exclusion section listed first
        return _items(raw[m_inc.end():]), _items(raw[m_exc.end(): m_inc.start()])
    return _items(raw[m_inc.end():] if m_inc else raw), []


def parse_trial_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    get = lambda el, p: _clean(el.find(p).text) if el.find(p) is not None and el.find(p).text else ""
    crit = root.find("eligibility/criteria/textblock")
    inc, exc = split_criteria(crit.text if crit is not None else "")
    return {
        "nct_id": get(root, "id_info/nct_id"),
        "title": get(root, "brief_title") or get(root, "official_title"),
        "conditions": [_clean(c.text) for c in root.findall("condition") if c.text],
        "interventions": [get(i, "intervention_name") for i in root.findall("intervention")
                          if get(i, "intervention_name")],
        "brief_summary": get(root, "brief_summary/textblock"),
        "inclusion": inc, "exclusion": exc,
        "gender": get(root, "eligibility/gender"),
        "min_age": get(root, "eligibility/minimum_age"),
        "max_age": get(root, "eligibility/maximum_age"),
    }


def _iter_xml(path):
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in (".xml", ".zip"):
                yield from _iter_xml(p)
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".xml"):
                    yield zf.read(name)
    elif path.suffix.lower() == ".xml":
        yield path.read_bytes()


def prepare_trials(src, out_path, keep_ids=None):
    """Convert ClinicalTrials.gov XML (file, .zip, or folder) to trials.jsonl."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    seen, bad = set(), 0
    with open(out_path, "w", encoding="utf-8") as out:
        for xml_bytes in _iter_xml(src):
            try:
                t = parse_trial_xml(xml_bytes)
            except ET.ParseError:
                bad += 1
                continue
            n = t["nct_id"]
            if n and n not in seen and (keep_ids is None or n in keep_ids):
                seen.add(n)
                out.write(json.dumps(t, ensure_ascii=False) + "\n")
                if len(seen) % 20000 == 0:
                    print(f"  {len(seen)} trials...")
    return len(seen), bad


def parse_topics_xml(path):
    root = ET.parse(path).getroot()
    return [{"topic_id": str(t.get("number")), "text": _clean("".join(t.itertext()))}
            for t in root.iter("topic")]

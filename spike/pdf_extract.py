"""Scientific-PDF extraction for TTS -- structured "read mode" output.

Pipeline:
1. Redact margin line-number rails (digit-only words clustered at a
   stable x in the margins -- the PDFBoT vertical-sweep idea).
2. Detect column boxes with the vendored PyMuPDF `column_boxes`.
3. Extract dict blocks per column, in reading order, classifying each as
   title / heading / paragraph from font size + weight + section names.
4. Drop running headers/footers: short blocks whose digit-normalized
   text repeats across pages (catches journal running titles that sit
   below the fixed header margin).
5. Dehyphenate, unwrap, and merge paragraph continuations across
   columns/pages (a paragraph that ends mid-sentence flows into the
   next block).

extract_blocks() -> [{"type": "title"|"heading"|"paragraph", "text": str}]
extract()        -> plain text (blocks joined with blank lines)

Usage:  python spike/pdf_extract.py <pdf> [max_pages]
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
from multi_column import column_boxes

MARGIN_FRACTION = 0.15
MIN_RAIL_SIZE = 5
X_CLUSTER_TOL = 4.0  # pt

SECTION_RE = re.compile(
    r"^(\d+(\.\d+)*\.?\s+)?(abstract|introduction|method(s|ology)?|results?"
    r"|discussion|conclusions?|references|acknowledg\w+|appendix\w*"
    r"|related work|background|data availability|supplementary\b.*)\s*$",
    re.IGNORECASE,
)
NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")
BOLD_FLAG = 1 << 4
# named-bold fonts don't set the synthetic-bold flag (Nature's HardingText-Bold,
# SRL's AdvOT….B); recognize them by PostScript name
BOLD_NAME_RE = re.compile(r"bold|semibold|black|heavy|\.b$|-b$", re.IGNORECASE)

# figure/table captions ("Table 1 | …", "Fig. 3 | …", "Extended Data Table 1")
# and journal article-type labels / branding that shouldn't be read or headed
CAPTION_RE = re.compile(
    r"^(?:extended\s+data\s+|sup(?:plementary|pl?\.?)\s+)?"
    r"(?:table|fig(?:ure|s)?\.?)\s*\d+\b",
    re.IGNORECASE,
)
ARTICLE_LABELS = {
    "technical report", "article", "letter", "review", "perspective",
    "brief communication", "resource", "analysis", "matters arising",
    "research article", "research", "report",
}


def _is_bold_span(span) -> bool:
    return bool(span["flags"] & BOLD_FLAG) or bool(BOLD_NAME_RE.search(span["font"]))


# ---------------------------------------------------------------- rails

def find_line_number_rails(page: fitz.Page) -> list[fitz.Rect]:
    """Return bboxes of margin line numbers on this page."""
    words = page.get_text("words")
    digit_words = [w for w in words if re.fullmatch(r"\d{1,4}", w[4])]

    clusters: dict[float, list] = defaultdict(list)
    for w in digit_words:
        for cx in clusters:
            if abs(w[0] - cx) <= X_CLUSTER_TOL:
                clusters[cx].append(w)
                break
        else:
            clusters[w[0]].append(w)

    body = [w for w in words if w not in digit_words]
    if not body:
        return []
    body_x0 = min(w[0] for w in body)
    body_x1 = max(w[2] for w in body)

    rails = []
    page_w = page.rect.width
    for cx, members in clusters.items():
        if len(members) < MIN_RAIL_SIZE:
            continue
        in_outer_margin = cx < page_w * MARGIN_FRACTION or cx > page_w * (1 - MARGIN_FRACTION)
        left_of_body = max(m[2] for m in members) <= body_x0 + 2
        right_of_body = min(m[0] for m in members) >= body_x1 - 2
        if in_outer_margin or left_of_body or right_of_body or _is_isolated_rail(members, words):
            rails.extend(fitz.Rect(m[:4]) for m in members)
    return rails


def _is_isolated_rail(members, words) -> bool:
    for m in members:
        mx0, my0, mx1, my1 = m[:4]
        for w in words:
            if w is m or re.fullmatch(r"\d{1,4}", w[4]):
                continue
            same_line = not (w[3] < my0 or w[1] > my1)
            if same_line and 0 <= mx0 - w[2] < 12:
                return False
    return True


# ------------------------------------------------------------- cleanup

# links read as garbage -- remove them entirely
URL_RE = re.compile(
    r"(?:https?://|www\.|doi\.org/|doi:\s*10\.)\S+|\b10\.\d{4,}/\S+", re.IGNORECASE
)

# in-text citations read as noise -- strip every common style
CITE_BRACKET_NUM = re.compile(r"\s*\[\d{1,3}(\s*[,–—-]\s*\d{1,3})*\]")
# any parenthetical containing a year: (Peters et al., 2018a; Radford, 2019)
CITE_AUTHOR_YEAR = re.compile(r"\s*\(\s*[^()]*\b(?:19|20)\d{2}[a-z]?[^()]*\)")
# parenthetical numerics -- but keep "Eq. (3)" / "Figure (2)" references
CITE_PAREN_NUM = re.compile(
    r"(\b(?:eq|eqs|equation|fig|figs|figure|table|sec|section|step|item)s?\.?\s*)?"
    r"\(\s*\d{1,3}(?:\s*[,–—-]\s*\d{1,3})*\s*\)",
    re.IGNORECASE,
)


def _strip_citations(text: str) -> str:
    text = URL_RE.sub("", text)
    text = CITE_BRACKET_NUM.sub("", text)
    text = CITE_AUTHOR_YEAR.sub("", text)
    text = CITE_PAREN_NUM.sub(lambda m: m.group(0) if m.group(1) else "", text)
    text = re.sub(r"\(\s*\)", "", text)  # parens emptied by URL removal
    return re.sub(r"\s+([.,;:!?])", r"\1", text)  # tidy "word ," leftovers


def _clean_block_text(text: str) -> str:
    # decomposed-ligature and hyphenation cleanup, then unwrap into prose
    text = re.sub(r"(?<![-\w])(\w+)-\n[ \t]*([a-z]\w*)", r"\1\2", text)
    text = re.sub(r"(\w)-\n[ \t]*(\w)", r"\1-\2", text)
    text = re.sub(r"\s*\n\s*", " ", text).strip()
    text = _strip_citations(text)
    return re.sub(r" {2,}", " ", text)


# ---------------------------------------------------------- extraction

def _figure_table_rects(page) -> list[fitz.Rect]:
    """Bounding boxes of figures (raster images + dense vector-drawing
    clusters) and tables. Text inside these is dropped."""
    rects = []
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"])
        if r.get_area() > 2000:
            rects.append(r)
    try:
        for t in page.find_tables():
            rects.append(fitz.Rect(t.bbox))
    except Exception:
        pass
    # vector figures: union up drawing paths, keep clusters of real size
    cluster = None
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"])
        if r.get_area() < 50 or r.width > page.rect.width * 0.95:
            continue
        if cluster and (cluster & r or (cluster | r).get_area()
                        < cluster.get_area() + r.get_area() + 8000):
            cluster |= r
        else:
            if cluster and cluster.get_area() > 12000:
                rects.append(fitz.Rect(cluster))
            cluster = r
    if cluster and cluster.get_area() > 12000:
        rects.append(fitz.Rect(cluster))
    return rects


def _in_figure(bbox, fig_rects) -> bool:
    r = fitz.Rect(bbox)
    if r.is_empty:
        return False
    for f in fig_rects:
        if (r & f).get_area() > 0.5 * r.get_area():
            return True
    return False


def _doc_body_size(doc, n_pages) -> float:
    """Char-weighted dominant font size across the document body."""
    sizes = Counter()
    for pno in range(n_pages):
        for blk in doc[pno].get_text("dict")["blocks"]:
            if blk.get("type"):
                continue
            for ln in blk["lines"]:
                for s in ln["spans"]:
                    n = len(s["text"].strip())
                    if n:
                        sizes[round(s["size"] * 2) / 2] += n
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _collect_raw_blocks(doc, max_pages=None):
    """Per page, per column box: dict blocks with dominant font info."""
    raw = []  # {page, text, size, bold, is_heading, y0}
    n_rails = 0
    n_fig = 0
    n_pages = min(len(doc), max_pages) if max_pages else len(doc)
    body_size = _doc_body_size(doc, n_pages)
    for pno in range(n_pages):
        page = doc[pno]
        rails = find_line_number_rails(page)
        n_rails += len(rails)
        if rails:
            for r in rails:
                page.add_redact_annot(r)
            page.apply_redactions()

        fig_rects = _figure_table_rects(page)
        boxes = column_boxes(page, footer_margin=50, header_margin=50, no_image_text=True)
        for box in boxes:
            d = page.get_text(
                "dict", clip=box, sort=True,
                flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES,
            )
            for blk in d["blocks"]:
                if blk.get("type") != 0:
                    continue
                if _in_figure(blk["bbox"], fig_rects):
                    n_fig += 1
                    continue
                # per-line records so a heading line that shares a block with
                # its following paragraph (Nature/SRL layout) is detected, not
                # averaged away into a body-sized paragraph
                for seg in _segment_block(blk, body_size):
                    seg["page"] = pno
                    seg["y0"] = blk["bbox"][1]
                    raw.append(seg)
    return raw, n_pages, n_rails, n_fig


def _line_record(ln):
    """(text, dominant_size, bold_fraction) for one line, dropping
    superscript numeric citation/footnote markers."""
    spans = [
        s for s in ln["spans"]
        if not (s["flags"] & 1 and re.fullmatch(r"[\d,\s–-]+", s["text"]))
    ]
    text = "".join(s["text"] for s in spans)
    if not text.strip():
        return None
    sizes, bold_chars, total = Counter(), 0, 0
    for s in spans:
        n = len(s["text"].strip())
        if not n:
            continue
        sizes[round(s["size"] * 2) / 2] += n
        total += n
        if _is_bold_span(s):
            bold_chars += n
    if not total:
        return None
    return {"text": text, "size": sizes.most_common(1)[0][0],
            "bold": bold_chars / total > 0.6}


def _line_is_heading(rec, body_size) -> bool:
    t = rec["text"].strip()
    if not t or len(t) > 100 or t.endswith((".", ",", ";", ":")):
        # section names may end in ':' — allow those explicitly below
        if not (t.endswith(":") and len(t) < 40):
            return False
    if CAPTION_RE.match(t) or t.lower() in ARTICLE_LABELS or t.islower():
        return False  # captions, article-type labels, journal branding
    if _is_reference_entry(t):
        return False  # numbered bibliography entries look numbered-heading-ish
    # numbered section heading: small number, short title ("3.2 Fine-tuning")
    numbered = bool(NUMBERED_HEADING_RE.match(t)) and len(t) < 60 and (
        int(re.match(r"\d+", t).group()) <= 40
    )
    return (
        rec["size"] >= body_size * 1.09
        or bool(SECTION_RE.match(t))
        or numbered
        or (rec["bold"] and rec["size"] >= body_size * 1.03 and len(t) < 70)
    )


def _segment_block(blk, body_size):
    """Split a text block into heading / paragraph segments by walking its
    lines and grouping runs of the same kind. A caption line drops the whole
    block (figure/table caption bodies aren't read)."""
    recs = [r for ln in blk["lines"] if (r := _line_record(ln))]
    if not recs:
        return []
    if CAPTION_RE.match(recs[0]["text"].strip()):
        return []

    segs, cur, cur_head = [], [], None
    for r in recs:
        is_head = _line_is_heading(r, body_size)
        # break on kind change, or between two stacked headings of different
        # size (major section + subsection) so they don't concatenate
        size_break = (
            is_head and cur_head and cur
            and abs(r["size"] - cur[-1]["size"]) > 1.0
        )
        if cur and (is_head != cur_head or size_break):
            segs.append((cur_head, cur))
            cur = []
        cur.append(r)
        cur_head = is_head
    if cur:
        segs.append((cur_head, cur))

    out = []
    for is_head, group in segs:
        text = _clean_block_text("\n".join(r["text"] for r in group))
        if not text:
            continue
        size = Counter(
            {s: sum(len(r["text"]) for r in group if r["size"] == s)
             for s in {r["size"] for r in group}}
        ).most_common(1)[0][0]
        out.append({
            "text": text,
            "size": size,
            "bold": sum(r["bold"] for r in group) / len(group) > 0.5,
            "is_heading": is_head,
        })
    return out


def _drop_repeated_furniture(raw, n_pages):
    """Remove running headers/footers: short digit-normalized text that
    repeats across pages (catches the title fragments journals repeat)."""
    if n_pages < 3:
        return raw, 0
    norm = lambda t: re.sub(r"\d+", "#", t).strip().lower()
    pages_with = defaultdict(set)
    for b in raw:
        if len(b["text"]) < 120:
            pages_with[norm(b["text"])].add(b["page"])
    threshold = max(2, round(n_pages * 0.4))
    furniture = {t for t, ps in pages_with.items() if len(ps) >= threshold}
    kept = [b for b in raw if not (len(b["text"]) < 120 and norm(b["text"]) in furniture)]
    return kept, len(raw) - len(kept)


def _body_size(raw) -> float:
    sizes = Counter()
    for b in raw:
        sizes[b["size"]] += len(b["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _is_prose(text: str) -> bool:
    """Reject figure/diagram text: streams of labels, symbols, axis ticks."""
    tokens = text.split()
    if not tokens:
        return False
    wordy = sum(
        1 for t in tokens
        if re.fullmatch(r"[A-Za-z][a-z’']+[.,;:!?)\"”]*", t)
    )
    return wordy / len(tokens) >= 0.4


def _is_reference_entry(text: str) -> bool:
    """Bibliography entries that escaped section-level removal."""
    t = text.strip()
    initials = len(re.findall(r"\b[A-Z]\.", t))
    ranges = len(re.findall(r"\d+,\s*\d+\s*[-–]\s*\d+", t))
    if initials >= 3 and (ranges >= 1 or initials >= 6):
        return True
    # numbered entry: "12. Herff, C. et al. …" / "12  Herff, C. et al …"
    if re.match(r"^\d{1,3}[.\s]\s*[A-Z]", t) and (
        "et al" in t or initials >= 2 or re.search(r"\b(19|20)\d{2}\b", t)
    ):
        return True
    # single-author numbered entry: "31. Mermelstein, P. Articulatory …"
    if re.match(r"^\d{1,3}[.\s]\s*[A-Z][A-Za-z’'-]+,\s+[A-Z]\.", t):
        return True
    return False


def _drop_non_prose(raw, body_size):
    """Drop figure text, footnotes, and stray reference entries before
    paragraph merging, so prose that flows around them stays adjacent."""
    kept = []
    dropped = 0
    for b in raw:
        t = b["text"].strip()
        # journal masthead / article-type label (page-1 furniture)
        is_furniture = t.lower() in ARTICLE_LABELS or (
            b["page"] == 0 and t.islower() and len(t) < 40
        )
        # footnotes run ~2pt below body; abstracts only ~1pt (keep those)
        is_footnote = b["size"] <= body_size - 1.6
        heading_like = b.get("is_heading") or SECTION_RE.match(t) or (
            NUMBERED_HEADING_RE.match(t) and len(t) < 80
        )
        if is_furniture or is_footnote or _is_reference_entry(t) or not (
            _is_prose(t) or heading_like
        ):
            dropped += 1
            continue
        kept.append(b)
    return kept, dropped


def _classify(raw, body_size):
    """Assign title/heading/paragraph from font size, weight, names."""
    if not raw:
        return []

    p1 = [b for b in raw if b["page"] == 0]
    title_size = max((b["size"] for b in p1), default=0)

    blocks = []
    for b in raw:
        text, size = b["text"], b["size"]
        if (b["page"] == 0 and title_size > body_size * 1.2
                and size >= title_size - 0.5 and b["y0"] < 350
                and not b.get("is_heading_forced_paragraph")):
            btype = "title"
        elif b.get("is_heading") and len(text) < 120:
            # trust the line-level heading detection (handles Nature/SRL where
            # the heading shares a block with the body paragraph)
            btype = "heading"
        elif len(text) < 120 and not text.endswith((".", ",", ";")) and (
            size >= body_size * 1.12
            or SECTION_RE.match(text)
            or (b["bold"] and NUMBERED_HEADING_RE.match(text))
        ):
            btype = "heading"
        else:
            btype = "paragraph"
        blocks.append({"type": btype, "text": text})

    # merge consecutive same-type title blocks (multi-line titles)
    merged = []
    for blk in blocks:
        if merged and blk["type"] == "title" and merged[-1]["type"] == "title":
            merged[-1]["text"] += " " + blk["text"]
        else:
            merged.append(blk)
    return merged


def _merge_continuations(blocks):
    """A paragraph that ends mid-sentence flows into the next paragraph
    (column/page break in the middle of a sentence)."""
    out = []
    for blk in blocks:
        prev = out[-1] if out else None
        if (prev and blk["type"] == "paragraph" and prev["type"] == "paragraph"
                and prev["text"] and not prev["text"].endswith((".", "!", "?", ":", '"', "”"))):
            joiner = "" if prev["text"].endswith("-") else " "
            if prev["text"].endswith("-") and blk["text"][:1].islower():
                prev["text"] = prev["text"][:-1]
            prev["text"] += joiner + blk["text"]
        else:
            out.append(blk)
    return out


SKIP_SECTION_RE = re.compile(
    r"^(\d+(\.\d+)*\.?\s+)?(references|bibliography|acknowledg\w+)\s*$", re.IGNORECASE
)


def _drop_skip_sections(blocks):
    """Remove the References/Acknowledgments sections wholesale -- nobody
    wants reference lists read aloud."""
    out, skipping = [], False
    for b in blocks:
        if b["type"] in ("heading", "title"):
            skipping = bool(SKIP_SECTION_RE.match(b["text"]))
            if skipping:
                continue
        if not skipping:
            out.append(b)
    return out


def extract_blocks(pdf, max_pages: int | None = None) -> list[dict]:
    if isinstance(pdf, (bytes, bytearray)):
        doc = fitz.open(stream=pdf, filetype="pdf")
        name = "<bytes>"
    else:
        doc = fitz.open(pdf)
        name = Path(pdf).name

    raw, n_pages, n_rails, n_fig = _collect_raw_blocks(doc, max_pages)
    raw, n_furniture = _drop_repeated_furniture(raw, n_pages)
    body_size = _body_size(raw)
    raw, n_nonprose = _drop_non_prose(raw, body_size)
    blocks = _drop_skip_sections(_merge_continuations(_classify(raw, body_size)))
    print(f"[{name}] {n_pages} pages, {n_rails} margin line-numbers stripped, "
          f"{n_furniture} running header/footer blocks dropped, "
          f"{n_fig} figure/table-region blocks dropped, "
          f"{n_nonprose} figure/footnote/reference blocks dropped, "
          f"{len(blocks)} blocks", file=sys.stderr)
    return blocks


def extract(pdf, max_pages: int | None = None) -> str:
    return "\n\n".join(b["text"] for b in extract_blocks(pdf, max_pages))


if __name__ == "__main__":
    pdf = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for b in extract_blocks(pdf, max_pages):
        tag = {"title": "T", "heading": "H", "paragraph": "P"}[b["type"]]
        print(f"[{tag}] {b['text'][:110]}")

"""Generate a synthetic two-column paper with margin line numbers.

Mimics the geometry of journal/IOS-Press LaTeX review templates
(lineno.sty): running header, centered footer page number, and line
numbers at fixed x in the left margin of each column block.
Deterministic test fixture for the margin-stripping heuristic.
"""

from pathlib import Path

import fitz

PAGE_W, PAGE_H = 595, 842  # A4
MARGIN = 72
COL_GAP = 24
COL_W = (PAGE_W - 2 * MARGIN - COL_GAP) / 2
LINE_H = 14
FONT = "helv"

LEFT_COL_SENTENCES = [
    "This is the first sentence of the left column and it belongs first.",
    "The second sentence continues the left column argument.",
    "A third sentence concludes the opening paragraph of the study.",
    "We then describe the experimental setup in careful detail.",
    "Each trial was repeated twelve times to reduce variance.",
]

RIGHT_COL_SENTENCES = [
    "The right column must be read only after the left column ends.",
    "Results indicate a strong effect across all twelve trials.",
    "We discuss limitations and future work in the final section.",
    "These findings replicate earlier reports with larger samples.",
    "In conclusion, the method is both simple and effective.",
]


def wrap(text: str, width_chars: int = 38):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main():
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)

    # running header + footer page number (should be stripped)
    page.insert_text((MARGIN, 40), "Journal of Synthetic Examples  Vol. 1", fontsize=8, fontname=FONT)
    page.insert_text((PAGE_W / 2 - 5, PAGE_H - 30), "17", fontsize=9, fontname=FONT)

    # title spanning both columns
    page.insert_text((MARGIN + 60, 100), "A Synthetic Two-Column Test Article", fontsize=14, fontname=FONT)

    lineno = 1
    for col, sentences in ((0, LEFT_COL_SENTENCES), (1, RIGHT_COL_SENTENCES)):
        x_text = MARGIN + col * (COL_W + COL_GAP)
        # line numbers sit in the margin left of each column, lineno.sty-style
        x_num = x_text - 26
        y = 150
        for sent in sentences:
            for line in wrap(sent):
                page.insert_text((x_num, y), str(lineno), fontsize=7, fontname=FONT)
                page.insert_text((x_text, y), line, fontsize=10, fontname=FONT)
                lineno += 1
                y += LINE_H
            y += 4

    out = Path(__file__).parent / "papers" / "synthetic_lineno.pdf"
    doc.save(out)
    print(f"wrote {out} ({lineno - 1} numbered lines)")


if __name__ == "__main__":
    main()

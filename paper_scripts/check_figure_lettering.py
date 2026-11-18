#!/usr/bin/env python3
"""check_figure_lettering.py -- printed size of every glyph of the figures (issue O53, QD221).

    python3 Scripts/check_figure_lettering.py Main.pdf [--min 7] [--json Scripts/figure_lettering_report.json]
    python3 Scripts/check_figure_lettering.py Images/Figure_5.pdf Images/Figure_6.pdf ...

(run from paper/Springer/).  For a manuscript built by pdfTeX, the glyphs are grouped by the included figure file
(/PTEX.FileName of the form XObject), so the sizes are the printed sizes after the scaling of \\includegraphics.  For a
figure PDF, the sizes are those of the file itself.  The size of a glyph is the font size times the length of the image
of the vertical unit vector of the glyph under the text rendering matrix, so rotated text is measured like upright text
(the 'size' of pdfplumber is the height of the bounding box, which for rotated text is the advance width of the glyph).
Sub- and superscripts count like any other glyph.  Exit status 1 when a glyph prints below --min.
"""
import argparse
import json
import math
import statistics
import sys

from pdfminer.pdfdevice import PDFTextDevice
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdffont import PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import resolve1, stream_value
from pdfminer.psparser import literal_name

TOL = 0.02  # pt; rounding of the font matrices


class _Device(PDFTextDevice):
    def __init__(self, rsrcmgr):
        super().__init__(rsrcmgr)
        self.files = []
        self.glyphs = []
        self.page = 0

    def render_char(self, matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate):
        try:
            text = font.to_unichr(cid)
        except PDFUnicodeNotDefined:
            text = "?"
        a, b, c, d, e, f = matrix
        fname = getattr(font, "fontname", "?")
        if isinstance(fname, bytes):
            fname = fname.decode("latin-1")
        self.glyphs.append({"file": self.files[0] if self.files else None, "page": self.page, "text": text,
                            "size": abs(fontsize) * math.hypot(c, d), "font": str(fname).split("+")[-1],
                            "x": round(e, 1), "y": round(f, 1)})
        return font.char_width(cid) * fontsize * scaling


class _Interpreter(PDFPageInterpreter):
    def do_Do(self, xobjid_arg):
        name = None
        try:
            xobj = stream_value(self.xobjmap[literal_name(xobjid_arg)])
            fn = resolve1(xobj.get("PTEX.FileName"))
            if fn is not None:
                name = fn.decode("latin-1") if isinstance(fn, bytes) else str(fn)
                name = name[2:] if name.startswith("./") else name
        except Exception:
            name = None
        if name:
            self.device.files.append(name)
        try:
            super().do_Do(xobjid_arg)
        finally:
            if name:
                self.device.files.pop()


def glyphs_of(path):
    """All glyphs of a PDF, with the outermost included file (or None) and the printed size in pt."""
    rsrc = PDFResourceManager()
    dev = _Device(rsrc)
    with open(path, "rb") as fh:
        doc = PDFDocument(PDFParser(fh))
        interp = _Interpreter(rsrc, dev)
        for i, page in enumerate(PDFPage.create_pages(doc), 1):
            dev.page = i
            interp.process_page(page)
    return dev.glyphs


def summarise(glyphs, min_pt=7.0, context=8):
    """Per figure: number of glyphs, min, median, share below min_pt, and every glyph below min_pt with its context."""
    out = {}
    groups = {}
    for i, g in enumerate(glyphs):
        if g["text"].strip():
            groups.setdefault(g["file"], []).append(i)
    for key, idx in groups.items():
        sizes = [glyphs[i]["size"] for i in idx]
        small = [i for i in idx if glyphs[i]["size"] < min_pt - TOL]
        items = []
        for i in small:
            lo, hi = max(0, i - context), min(len(glyphs), i + context + 1)
            ctx = "".join(("[%s]" % glyphs[j]["text"]) if j == i else glyphs[j]["text"]
                          for j in range(lo, hi) if glyphs[j]["file"] == key)
            items.append({"text": glyphs[i]["text"], "size": round(glyphs[i]["size"], 2), "font": glyphs[i]["font"],
                          "page": glyphs[i]["page"], "x": glyphs[i]["x"], "y": glyphs[i]["y"], "context": ctx})
        out[key or "(page text)"] = {"glyphs": len(idx), "min_pt": round(min(sizes), 2),
                                     "median_pt": round(statistics.median(sizes), 2),
                                     "below_min": len(small), "share_below_min": round(len(small) / len(idx), 4),
                                     "pages": sorted({glyphs[i]["page"] for i in idx}), "small": items}
    return out


def check_pdf(path, min_pt=7.0, figures_only=None):
    """Summary of one PDF.  figures_only: keep only the groups whose file name contains this text (e.g. 'Images/')."""
    s = summarise(glyphs_of(path), min_pt)
    if figures_only is not None:
        s = {k: v for k, v in s.items() if figures_only in k}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--min", type=float, default=7.0)
    ap.add_argument("--json")
    ap.add_argument("--show", type=int, default=6, help="glyphs below the minimum listed per figure")
    a = ap.parse_args()
    report, bad = {}, 0
    for path in a.pdfs:
        s = summarise(glyphs_of(path), a.min)
        if any(k != "(page text)" for k in s):
            s = {k: v for k, v in s.items() if k != "(page text)"}
        else:
            s = {path: s.pop("(page text)")} if s else {}
        for key in sorted(s, key=lambda k: (s[k]["pages"][0], k)):
            v = s[key]
            flag = "OK " if v["below_min"] == 0 else "LOW"
            bad += v["below_min"] > 0
            print("%s %-34s p.%-7s glyphs %5d  min %5.2f  median %5.2f  below %.1f pt: %d"
                  % (flag, key, ",".join(map(str, v["pages"])), v["glyphs"], v["min_pt"], v["median_pt"], a.min,
                     v["below_min"]))
            for it in v["small"][:a.show]:
                print("      %5.2f pt  %-24s %s" % (it["size"], it["font"][:24], it["context"]))
        report[path] = s
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"min_pt": a.min, "report": report}, fh, indent=1, ensure_ascii=False)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

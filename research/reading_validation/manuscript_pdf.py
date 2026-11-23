"""Typeset the integrated English research report and its measured figure."""

import re
from argparse import ArgumentParser
from html import escape
from pathlib import Path

from PIL import Image as Raster
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepInFrame,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

SOURCE = Path(__file__).with_name("manuscript.md")
FIGURE = (
    SOURCE.parent.parent
    / "results/reading-validation-manuscript/observed-reading-changes.png"
)
TARGET = Path("output/pdf/CogLLM-reading-study.pdf")
LINK = re.compile(r"\[([^\[\]]+)\]\(([^)]+)\)")
INK = "#172629"
MUTED = "#415456"
TEAL = "#11675E"
SOURCE_REFERENCES = (
    ("**A reader population", (1, 2, 3)),
    ("A second observer", (1, 2, 3)),
    ("**The earlier passage", (4,)),
    ("**A professional reader", (5,)),
    ("**Which textual changes", (6, 13, 14)),
    ("The same narrative feature", (6,)),
    ("**What reader context", (1, 3, 11)),
    ("In the controlled English", (7,)),
    ("The most direct available", (8,)),
    ("An experimental change of goal", (9, 10)),
    ("Changing the order", (12, 13)),
    ("Holding the body", (12, 13)),
    ("A separate study using headings", (15,)),
    ("The archive's 48-dimensional", (7,)),
    ("**Figure 2", (1, 2, 3)),
)


def fonts(directory):
    for name, file in (
        ("ResearchText", "DejaVuSerif"),
        ("ResearchBold", "DejaVuSerif-Bold"),
        ("ResearchItalic", "DejaVuSerif-Italic"),
        ("ResearchSans", "DejaVuSans"),
        ("ResearchSansBold", "DejaVuSans-Bold"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(directory / (file + ".ttf"))))
    pdfmetrics.registerFontFamily(
        "ResearchText",
        normal="ResearchText",
        bold="ResearchBold",
        italic="ResearchItalic",
        boldItalic="ResearchBold",
    )


def style(name, family, size, leading, color=INK, **kwargs):
    return ParagraphStyle(
        name,
        fontName=family,
        fontSize=size,
        leading=leading,
        textColor=HexColor(color),
        alignment=TA_LEFT,
        allowWidows=0,
        allowOrphans=0,
        **kwargs,
    )


def rich(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(s))
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)


def prose(s):
    first = LINK.search(s)
    value = rich(s[: first.start()].rstrip() if first else s)
    for prefix, numbers in SOURCE_REFERENCES:
        if s.startswith(prefix):
            refs = ", ".join(
                f'<link href="#study-{n}" color="{TEAL}">{n}</link>' for n in numbers
            )
            return value + f" <super>[{refs}]</super>"
    return value


def make(font_dir, output):
    fonts(font_dir)
    blocks = SOURCE.read_text().strip().split("\n\n")
    title = blocks[0].removeprefix("# ")
    reference_start = blocks.index("## Selected primary studies")
    refs = blocks[reference_start + 1].splitlines()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=58,
        rightMargin=58,
        topMargin=57,
        bottomMargin=55,
        title=title,
        author="CogLLM research project",
    )
    body = style("Body", "ResearchText", 10.1, 14.9, spaceAfter=8)
    small = style("Caption", "ResearchText", 8.4, 11.8, MUTED)
    heading = style(
        "Heading",
        "ResearchSansBold",
        16.2,
        20.5,
        TEAL,
        spaceBefore=8,
        spaceAfter=9,
        keepWithNext=True,
    )
    reference = style(
        "Reference",
        "ResearchText",
        9,
        12.2,
        firstLineIndent=-16,
        leftIndent=16,
        spaceAfter=4.5,
    )
    figure_width = doc.width - 34
    fig = Image(str(FIGURE), width=figure_width, height=figure_width * 1367 / 2060)
    cover = [
        Paragraph("RESEARCH REPORT", style("Kind", "ResearchSansBold", 9, 13, TEAL)),
        Spacer(1, 7),
        Paragraph(rich(title), style("Title", "ResearchBold", 24, 29)),
        Spacer(1, 7),
        Paragraph(
            "CogLLM  ·  23 November 2026", style("Date", "ResearchSans", 9, 12, MUTED)
        ),
        Spacer(1, 13),
        Paragraph(prose(blocks[2]), style("Abstract", "ResearchText", 10, 14.3)),
        Spacer(1, 10),
        fig,
        Spacer(1, 8),
        Paragraph(prose(blocks[4]), small),
    ]
    flow = [
        KeepInFrame(doc.width - 12, doc.height - 15, cover, mode="shrink"),
        PageBreak(),
    ]
    body_blocks = iter(blocks[5:reference_start])
    for block in body_blocks:
        if block.startswith("## "):
            flow.append(Paragraph(rich(block[3:]), heading))
        elif block.startswith("!["):
            path = SOURCE.parent / LINK.search(block)[2]
            with Raster.open(path) as im:
                w, h = im.size
            width = doc.width - 15
            caption = next(body_blocks)
            if not caption.startswith("**Figure"):
                raise ValueError("Each empirical figure requires a caption")
            flow.append(
                KeepTogether(
                    [
                        Spacer(1, 8),
                        Image(str(path), width=width, height=width * h / w),
                        Spacer(1, 7),
                        Paragraph(prose(caption), small),
                        Spacer(1, 13),
                    ]
                )
            )
        else:
            flow.append(Paragraph(prose(block), body))
    flow.extend((PageBreak(), Paragraph("Selected primary studies", heading)))
    for number, s in enumerate(refs, 1):
        match = LINK.search(s)
        if match is None or not match[2].startswith("https://"):
            raise ValueError("Each printed reference needs a verified primary link")
        label = match[2].removeprefix("https://").rstrip("/")
        if len(label) > 65:
            label = match[1]
        link = f'<link href="{escape(match[2], quote=True)}" color="{TEAL}">{escape(label)}</link>'
        flow.append(
            Paragraph(f'<a name="study-{number}"/>' + prose(s) + " " + link, reference)
        )
    flow += [
        Spacer(1, 12),
        Paragraph("Reproducibility and materials", heading),
        Paragraph(
            "The companion repository contains the Markdown version of this report with links to study-specific "
            "methods, fit diagnostics, numerical results, and data-source access instructions. Open "
            "<b>research/reading_validation/reproduction-index.md</b> at the repository root. The "
            "original studies retain their own data and text reuse terms.",
            body,
        ),
    ]

    def frame(canvas, doc):
        w, h = A4
        canvas.setStrokeColor(HexColor("#a5b7b3"))
        canvas.setLineWidth(0.45)
        canvas.line(58, h - 39, w - 58, h - 39)
        canvas.line(58, 39, w - 58, 39)
        canvas.setFillColor(HexColor(MUTED))
        canvas.setFont("ResearchSans", 8)
        canvas.drawString(58, h - 31, "CogLLM")
        canvas.drawRightString(w - 58, h - 31, "Human reading responses")
        canvas.drawString(58, 25, "Research report   ·   November 2026")
        canvas.drawRightString(w - 58, 25, str(doc.page))

    doc.build(flow, onFirstPage=frame, onLaterPages=frame)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--font-directory",
        type=Path,
        required=True,
        help="Directory containing DejaVu serif and sans TTFs",
    )
    parser.add_argument("--output", type=Path, default=TARGET)
    args = parser.parse_args()
    make(args.font_directory, args.output)

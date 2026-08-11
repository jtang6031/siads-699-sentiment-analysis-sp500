"""Build the final searchable PDF directly from the accepted report Markdown."""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterator

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BLUE = colors.HexColor("#2E74B5")
DARK_BLUE = colors.HexColor("#1F4D78")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5B6570")
TABLE_FILL = colors.HexColor("#F4F6F9")
FIGURE_WIDTH = 6.25 * inch


def _font_path(name: str) -> Path:
    font_root = Path(
        "C:/Users/dongx/.cache/codex-runtimes/codex-primary-runtime/"
        "dependencies/native/poppler/Library/share/fonts"
    )
    candidate = font_root / name
    if not candidate.is_file():
        raise FileNotFoundError(f"required embedded font is unavailable: {candidate}")
    return candidate


def _register_fonts() -> None:
    for name, filename in (
        ("DejaVuSans", "DejaVuSans.ttf"),
        ("DejaVuSans-Bold", "DejaVuSans-Bold.ttf"),
        ("DejaVuSans-Oblique", "DejaVuSans-Oblique.ttf"),
    ):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(_font_path(filename))))
    pdfmetrics.registerFontFamily(
        "DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold", italic="DejaVuSans-Oblique", boldItalic="DejaVuSans-Bold"
    )


def _strip_markdown(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return text.replace("**", "").replace("*", "").replace("`", "").strip()


def _paragraph_markup(text: str) -> str:
    """Translate the small Markdown subset used by the accepted report to Paragraph XML."""
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"\1", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*", r"<i>\1</i>", escaped)
    url_pattern = re.compile(r"(?<![\"=&gt;])(https://[^\s&lt;]+)")
    return url_pattern.sub(lambda match: f'<link href="{match.group(1)}" color="#2E74B5">{match.group(1)}</link>', escaped)


def _tokens(markdown_path: Path) -> Iterator[tuple[str, object]]:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            yield "heading", (len(heading.group(1)), _strip_markdown(heading.group(2)))
            index += 1
            continue
        image = re.match(r"^!\[([^\]]+)\]\(([^)]+)\)$", line)
        if image:
            yield "image", (image.group(1).strip(), image.group(2).strip())
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|(?:\s*:?-{3,}:?\s*\|)+$", lines[index + 1].strip()):
            rows = [[cell.strip() for cell in line.strip("|").split("|")]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append([_strip_markdown(cell.strip()) for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            yield "table", rows
            continue
        if line.startswith("- "):
            yield "bullet", _strip_markdown(line[2:])
            index += 1
            continue
        paragraph = [line]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate or candidate.startswith(("#", "!", "|", "- ")):
                break
            paragraph.append(candidate)
            index += 1
        yield "paragraph", " ".join(paragraph)


def _styles() -> dict[str, ParagraphStyle]:
    _register_fonts()
    sample = getSampleStyleSheet()
    return {
        "body": ParagraphStyle("ReportBody", parent=sample["BodyText"], fontName="DejaVuSans", fontSize=10.4, leading=14.0, textColor=INK, spaceAfter=8),
        "h1": ParagraphStyle("ReportH1", parent=sample["Heading1"], fontName="DejaVuSans-Bold", fontSize=15.5, leading=19, textColor=BLUE, spaceBefore=16, spaceAfter=8, keepWithNext=True),
        "h2": ParagraphStyle("ReportH2", parent=sample["Heading2"], fontName="DejaVuSans-Bold", fontSize=12.5, leading=15.5, textColor=DARK_BLUE, spaceBefore=11, spaceAfter=6, keepWithNext=True),
        "caption": ParagraphStyle("ReportCaption", parent=sample["BodyText"], fontName="DejaVuSans-Oblique", fontSize=8.8, leading=11.2, textColor=MUTED, spaceAfter=9, keepWithNext=True),
        "bullet": ParagraphStyle("ReportBullet", parent=sample["BodyText"], fontName="DejaVuSans", fontSize=10.2, leading=13.2, leftIndent=18, firstLineIndent=-9, bulletIndent=0, textColor=INK, spaceAfter=3),
        "reference": ParagraphStyle("Reference", parent=sample["BodyText"], fontName="DejaVuSans", fontSize=9.5, leading=12.5, leftIndent=18, firstLineIndent=-18, textColor=INK, spaceAfter=5),
        "table": ParagraphStyle("TableCell", parent=sample["BodyText"], fontName="DejaVuSans", fontSize=8.25, leading=10.1, textColor=INK),
        "table_head": ParagraphStyle("TableHead", parent=sample["BodyText"], fontName="DejaVuSans-Bold", fontSize=8.25, leading=10.1, textColor=INK),
    }


def _figure(markdown_path: Path, alt_text: str, relative_path: str, caption: str | None, styles: dict[str, ParagraphStyle]):
    path = (markdown_path.parent / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required report figure is missing: {path}")
    image = Image(str(path))
    image.drawWidth = FIGURE_WIDTH
    image.drawHeight = image.imageHeight * FIGURE_WIDTH / image.imageWidth
    image.hAlign = "CENTER"
    items = [Spacer(1, 4), image]
    if caption:
        items.append(Spacer(1, 3))
        items.append(Paragraph(_paragraph_markup(caption), styles["caption"]))
    return KeepTogether(items)


def _table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    columns = len(rows[0])
    width = 6.5 * inch / columns
    values = []
    for row_number, row in enumerate(rows):
        values.append([Paragraph(_paragraph_markup(value), styles["table_head"] if row_number == 0 else styles["table"]) for value in row])
    table = Table(values, colWidths=[width] * columns, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_FILL),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C2CC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _cover(story: list[object], styles: dict[str, ParagraphStyle]) -> None:
    cover_label = ParagraphStyle("CoverLabel", fontName="DejaVuSans-Bold", fontSize=10, leading=12, textColor=BLUE, alignment=TA_CENTER)
    cover_title = ParagraphStyle("CoverTitle", fontName="DejaVuSans-Bold", fontSize=27, leading=32, textColor=DARK_BLUE, alignment=TA_CENTER)
    cover_subtitle = ParagraphStyle("CoverSubtitle", fontName="DejaVuSans", fontSize=13.5, leading=17, textColor=MUTED, alignment=TA_CENTER)
    cover_team = ParagraphStyle("CoverTeam", fontName="DejaVuSans-Bold", fontSize=10.7, leading=14, textColor=INK, alignment=TA_CENTER)
    cover_note = ParagraphStyle("CoverNote", fontName="DejaVuSans-Oblique", fontSize=9.5, leading=12, textColor=MUTED, alignment=TA_CENTER)
    story.extend([
        Spacer(1, 2.0 * inch),
        Paragraph("SIADS 699 CAPSTONE REPORT", cover_label), Spacer(1, 12),
        Paragraph("Financial News and<br/>Sector ETF Returns", cover_title), Spacer(1, 22),
        Paragraph("Timing, sentiment, and out-of-sample evidence", cover_subtitle), Spacer(1, 25),
        Paragraph("Jeremy Tang | Christian Goelz | Dongxin Liang", cover_team), Spacer(1, 5),
        Paragraph("Final report - August 10, 2026", cover_subtitle), Spacer(1, 20),
        Paragraph("Academic research only. This report is not investment advice.", cover_note),
        PageBreak(),
    ])


def _later_page(canvas, doc) -> None:
    if doc.page <= 1:
        return
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5DCE3"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, letter[1] - 0.68 * inch, letter[0] - doc.rightMargin, letter[1] - 0.68 * inch)
    canvas.setFont("DejaVuSans-Bold", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, letter[1] - 0.57 * inch, "FINANCIAL NEWS AND SECTOR ETF RETURNS")
    canvas.setFont("DejaVuSans", 8.5)
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.55 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_report_pdf(markdown_path: Path, destination: Path) -> Path:
    """Create the final searchable PDF from the accepted Markdown source."""
    markdown_path = Path(markdown_path).resolve()
    destination = Path(destination).resolve()
    styles = _styles()
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination), pagesize=letter,
        leftMargin=inch, rightMargin=inch, topMargin=0.9 * inch, bottomMargin=0.85 * inch,
        title="Financial News and Sector ETF Returns", author="SIADS 699 Capstone Team",
        subject="SIADS 699 capstone report",
    )
    story: list[object] = []
    _cover(story, styles)
    in_references = False
    pending_image: tuple[str, str] | None = None
    for token_type, value in _tokens(markdown_path):
        if token_type == "heading":
            level, text = value  # type: ignore[misc]
            if level == 1:
                continue
            if pending_image:
                story.append(_figure(markdown_path, pending_image[0], pending_image[1], None, styles))
                pending_image = None
            story.append(Paragraph(_paragraph_markup(text), styles["h1"] if level == 2 else styles["h2"]))
            in_references = text == "References"
        elif token_type == "image":
            pending_image = value  # type: ignore[assignment]
        elif token_type == "paragraph":
            text = str(value)
            if pending_image:
                caption = text[1:-1] if text.startswith("*Figure ") and text.endswith("*") else None
                story.append(_figure(markdown_path, pending_image[0], pending_image[1], caption, styles))
                pending_image = None
                if caption:
                    continue
            style = styles["reference"] if in_references else (styles["caption"] if text.startswith("*Figure ") and text.endswith("*") else styles["body"])
            story.append(Paragraph(_paragraph_markup(text[1:-1] if style is styles["caption"] else text), style))
        elif token_type == "bullet":
            story.append(Paragraph(_paragraph_markup(str(value)), styles["bullet"], bulletText="•"))
        elif token_type == "table":
            story.extend([Spacer(1, 2), _table(value, styles), Spacer(1, 7)])  # type: ignore[arg-type]
    if pending_image:
        story.append(_figure(markdown_path, pending_image[0], pending_image[1], None, styles))
    document.build(story, onFirstPage=_later_page, onLaterPages=_later_page)
    return destination


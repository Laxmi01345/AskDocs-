from io import BytesIO
from pypdf import PdfReader
from docx import Document


def parse_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def parse_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text).strip()


def parse_docx(data: bytes) -> str:
    doc = Document(BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs).strip()
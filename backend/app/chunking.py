def chunk_text(text: str):
    chunks = []
    buf = ""
    for ch in text:
        if ch != ".":
            buf += ch
        else:
            buf += "."
            chunks.append(buf.strip())
            buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    return chunks
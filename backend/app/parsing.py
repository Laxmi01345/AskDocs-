from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader


def load_document(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    loader_map = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
    }

    loader_class = loader_map.get(suffix)
    if loader_class is None:
        raise ValueError(f"Unsupported file type: {suffix}")

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / filename
        temp_path.write_bytes(data)
        if suffix == ".txt":
            documents = loader_class(str(temp_path), encoding="utf-8").load()
        else:
            documents = loader_class(str(temp_path)).load()
        return "\n".join(doc.page_content for doc in documents).strip()
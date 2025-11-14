"""Native file-system based acquirer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from ...interfaces import Document, IAcquirer, LLMRequest


class FileSystemAcquirer(IAcquirer):
    """Loads UTF-8 text files from disk as `Document` objects."""

    def __init__(self, root: str | Path, glob: str = "**/*.txt") -> None:
        self.root = Path(root)
        self.glob = glob

    def acquire(self, request: LLMRequest) -> List[Document]:
        patterns: Iterable[str] = request.metadata.get("file_patterns", [self.glob])
        documents: List[Document] = []
        for pattern in patterns:
            for path in self.root.glob(pattern):
                if not path.is_file():
                    continue
                content = path.read_text(encoding="utf-8")
                documents.append(
                    Document(
                        id=str(path.relative_to(self.root)),
                        content=content,
                        metadata={"path": str(path)},
                    )
                )
        return documents

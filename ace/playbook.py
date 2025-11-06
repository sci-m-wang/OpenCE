"""Playbook storage and mutation logic for ACE."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from .delta import DeltaBatch, DeltaOperation
from .deduplication import Deduplicator


@dataclass
class Bullet:
    """Single playbook entry."""

    id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def apply_metadata(self, metadata: Dict[str, int]) -> None:
        for key, value in metadata.items():
            if hasattr(self, key):
                setattr(self, key, int(value))

    def tag(self, tag: str, increment: int = 1) -> None:
        if tag not in ("helpful", "harmful", "neutral"):
            raise ValueError(f"Unsupported tag: {tag}")
        current = getattr(self, tag)
        setattr(self, tag, current + increment)
        self.updated_at = datetime.now(timezone.utc).isoformat()


class Playbook:
    """Structured context store as defined by ACE."""

    def __init__(self) -> None:
        self._bullets: Dict[str, Bullet] = {}
        self._sections: Dict[str, List[str]] = {}
        self._next_id = 0

    # ------------------------------------------------------------------ #
    # CRUD utils
    # ------------------------------------------------------------------ #
    def add_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Bullet:
        bullet_id = bullet_id or self._generate_id(section)
        metadata = metadata or {}
        bullet = Bullet(id=bullet_id, section=section, content=content)
        bullet.apply_metadata(metadata)
        self._bullets[bullet_id] = bullet
        self._sections.setdefault(section, []).append(bullet_id)
        return bullet

    def update_bullet(
        self,
        bullet_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Optional[Bullet]:
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        if content is not None:
            bullet.content = content
        if metadata:
            bullet.apply_metadata(metadata)
        bullet.updated_at = datetime.now(timezone.utc).isoformat()
        return bullet

    def tag_bullet(self, bullet_id: str, tag: str, increment: int = 1) -> Optional[Bullet]:
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        bullet.tag(tag, increment=increment)
        return bullet

    def remove_bullet(self, bullet_id: str) -> None:
        bullet = self._bullets.pop(bullet_id, None)
        if bullet is None:
            return
        section_list = self._sections.get(bullet.section)
        if section_list:
            self._sections[bullet.section] = [
                bid for bid in section_list if bid != bullet_id
            ]
            if not self._sections[bullet.section]:
                del self._sections[bullet.section]

    def get_bullet(self, bullet_id: str) -> Optional[Bullet]:
        return self._bullets.get(bullet_id)

    def bullets(self) -> List[Bullet]:
        return list(self._bullets.values())

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, object]:
        return {
            "bullets": {bullet_id: asdict(bullet) for bullet_id, bullet in self._bullets.items()},
            "sections": self._sections,
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Playbook":
        instance = cls()
        bullets_payload = payload.get("bullets", {})
        if isinstance(bullets_payload, dict):
            for bullet_id, bullet_value in bullets_payload.items():
                if isinstance(bullet_value, dict):
                    instance._bullets[bullet_id] = Bullet(**bullet_value)
        sections_payload = payload.get("sections", {})
        if isinstance(sections_payload, dict):
            instance._sections = {
                section: list(ids) if isinstance(ids, Iterable) else []
                for section, ids in sections_payload.items()
            }
        instance._next_id = int(payload.get("next_id", 0))
        return instance

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def loads(cls, data: str) -> "Playbook":
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Playbook serialization must be a JSON object.")
        return cls.from_dict(payload)

    # ------------------------------------------------------------------ #
    # Delta application
    # ------------------------------------------------------------------ #
    def apply_delta(self, delta: DeltaBatch) -> None:
        for operation in delta.operations:
            self._apply_operation(operation)

    def _apply_operation(self, operation: DeltaOperation) -> None:
        op_type = operation.type.upper()
        if op_type == "ADD":
            self.add_bullet(
                section=operation.section,
                content=operation.content or "",
                bullet_id=operation.bullet_id,
                metadata=operation.metadata,
            )
        elif op_type == "UPDATE":
            if operation.bullet_id is None:
                return
            self.update_bullet(
                operation.bullet_id,
                content=operation.content,
                metadata=operation.metadata,
            )
        elif op_type == "TAG":
            if operation.bullet_id is None:
                return
            for tag, increment in operation.metadata.items():
                self.tag_bullet(operation.bullet_id, tag, increment)
        elif op_type == "REMOVE":
            if operation.bullet_id is None:
                return
            self.remove_bullet(operation.bullet_id)

    def deduplicate(self, deduplicator: Deduplicator, bullet_ids: List[str]) -> List[str]:
        """
        Finds and removes duplicate bullets from the playbook.

        Args:
            deduplicator: The Deduplicator instance.
            bullet_ids: A list of bullet IDs to check for duplicates.

        Returns:
            A list of bullet IDs that were removed.
        """
        new_bullets = {
            bullet_id: self._bullets[bullet_id].content
            for bullet_id in bullet_ids
            if bullet_id in self._bullets
        }
        existing_bullets = {
            bullet_id: bullet.content
            for bullet_id, bullet in self._bullets.items()
            if bullet_id not in new_bullets
        }

        duplicate_ids = deduplicator.find_duplicates(new_bullets, existing_bullets)

        for bullet_id in duplicate_ids:
            self.remove_bullet(bullet_id)

        return duplicate_ids

    # ------------------------------------------------------------------ #
    # Presentation helpers
    # ------------------------------------------------------------------ #
    def as_prompt(self) -> str:
        """Return a human-readable playbook string for prompting LLMs."""
        parts: List[str] = []
        for section, bullet_ids in sorted(self._sections.items()):
            parts.append(f"## {section}")
            for bullet_id in bullet_ids:
                bullet = self._bullets[bullet_id]
                counters = (
                    f"(helpful={bullet.helpful}, harmful={bullet.harmful}, neutral={bullet.neutral})"
                )
                parts.append(f"- [{bullet.id}] {bullet.content} {counters}")
        return "\n".join(parts)

    def stats(self) -> Dict[str, object]:
        return {
            "sections": len(self._sections),
            "bullets": len(self._bullets),
            "tags": {
                "helpful": sum(b.helpful for b in self._bullets.values()),
                "harmful": sum(b.harmful for b in self._bullets.values()),
                "neutral": sum(b.neutral for b in self._bullets.values()),
            },
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _generate_id(self, section: str) -> str:
        self._next_id += 1
        section_prefix = section.split()[0].lower()
        return f"{section_prefix}-{self._next_id:05d}"

"""JSON-backed prompt version store implementation."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, cast

from dynamic_prompt_core.domain.prompts.fixed import CLASSIFICATION_PROMPT_V0

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_STORE_PATH = "data/prompt_store.json"
DEFAULT_MAX_VERSIONS: int | None = None
DEFAULT_ATOMIC_WRITES = True
DEFAULT_LOG_PATH = "prompt_store.jsonl"

SCHEMA_VERSION = 1

REQUIRED_RECORD_FIELDS = (
    "version",
    "text",
    "hash",
    "rules",
    "source_candidates",
    "base_version",
    "created_at",
    "active",
    "archived",
    "reason",
)

MUTABLE_FIELDS = frozenset({"active", "archived"})


class PromptStoreError(ValueError):
    """Raised when a store operation fails or the store is invalid."""


@dataclass
class PromptStoreConfig:
    store_path: str = DEFAULT_STORE_PATH
    max_versions: int | None = DEFAULT_MAX_VERSIONS
    atomic_writes: bool = DEFAULT_ATOMIC_WRITES
    log_path: str = DEFAULT_LOG_PATH

    @classmethod
    def from_config(
        cls, config_path: str = DEFAULT_CONFIG_PATH
    ) -> PromptStoreConfig:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        s = config.get("prompt_store", {})
        store_path = s.get("store_path")
        if not store_path:
            raise PromptStoreError(
                "prompt store config missing required key: store_path"
            )
        max_versions = s.get("max_versions")
        if max_versions is not None:
            max_versions = int(max_versions)
        return cls(
            store_path=store_path,
            max_versions=max_versions,
            atomic_writes=bool(s.get("atomic_writes", DEFAULT_ATOMIC_WRITES)),
            log_path=s.get("log_path", DEFAULT_LOG_PATH),
        )


@dataclass
class PromptVersionRecord:
    version: int
    text: str
    hash: str
    rules: list[Any]
    source_candidates: list[int]
    base_version: int | None
    created_at: str
    active: bool = False
    archived: bool = False
    reason: str = ""


def _sha256_hex(text: str) -> str:
    """Full sha256 hexdigest of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_record(record: dict[str, Any]) -> None:
    """Validate that *record* has all required fields and a correct hash."""
    for name in REQUIRED_RECORD_FIELDS:
        if name not in record:
            raise PromptStoreError(
                f"prompt version record missing required field: {name}"
            )
    expected_hash = _sha256_hex(record["text"])
    if record["hash"] != expected_hash:
        raise PromptStoreError(
            f"prompt version record hash mismatch for version "
            f"{record['version']}: expected {expected_hash}, "
            f"got {record['hash']}"
        )


def _normalize_input(
    prompt_version_dict: dict[str, Any],
    reason: str,
    base_version: int | None = None,
) -> dict[str, Any]:
    """Convert a composer-produced prompt-version dict into a store record dict."""
    raw_rules = prompt_version_dict.get("rules", [])
    rules: list[Any] = []
    for r in raw_rules:
        if isinstance(r, dict):
            rules.append(r.get("rule", str(r)))
        else:
            rules.append(r)

    text = prompt_version_dict.get("text", "")
    created_at = prompt_version_dict.get("created_at")
    if not created_at:
        created_at = datetime.now(UTC).isoformat()

    record = PromptVersionRecord(
        version=-1,
        text=text,
        hash=_sha256_hex(text),
        rules=rules,
        source_candidates=list(prompt_version_dict.get("source_candidates", [])),
        base_version=base_version,
        created_at=created_at,
        active=False,
        archived=False,
        reason=reason,
    )
    return asdict(record)


class PromptStore:
    """Persistent prompt-version store backed by a single JSON file."""

    def __init__(self, config: PromptStoreConfig) -> None:
        self.config = config
        self._versions: dict[str, dict[str, Any]] = self.load()

    def load(self) -> dict[str, dict[str, Any]]:
        """Load the store file and return the in-memory versions dict."""
        path = self.config.store_path
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise PromptStoreError(
                f"corrupted prompt store file {path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise PromptStoreError(
                f"corrupted prompt store file {path}: top-level is not a JSON object"
            )
        if not isinstance(data.get("versions"), dict):
            raise PromptStoreError(
                f"corrupted prompt store file {path}: missing or invalid 'versions' dict"
            )
        self._log("reload", None, {"count": len(data["versions"])})
        return cast(dict[str, dict[str, Any]], data["versions"])

    def _flush(self, versions: dict[str, dict[str, Any]]) -> None:
        """Atomically write the full store to disk."""
        path = self.config.store_path
        payload = json.dumps(
            {"schema_version": SCHEMA_VERSION, "versions": versions},
            ensure_ascii=False,
            indent=2,
        )
        if self.config.atomic_writes:
            directory = os.path.dirname(path) or "."
            tmp = os.path.join(directory, f".{os.path.basename(path)}.tmp.{os.getpid()}")
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp, path)
            except Exception:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                raise
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)

    def _seed_baseline(self) -> None:
        """Seed version 0 from CLASSIFICATION_PROMPT_V0."""
        text = CLASSIFICATION_PROMPT_V0.text
        record = asdict(
            PromptVersionRecord(
                version=0,
                text=text,
                hash=_sha256_hex(text),
                rules=list(CLASSIFICATION_PROMPT_V0.layers.rules)
                if CLASSIFICATION_PROMPT_V0.layers
                else [],
                source_candidates=[],
                base_version=None,
                created_at=datetime.now(UTC).isoformat(),
                active=True,
                archived=False,
                reason="baseline",
            )
        )
        self._versions["0"] = record
        self._flush(self._versions)
        self._log("save", 0, {"reason": "baseline"})

    def save(
        self,
        prompt_version_dict: dict[str, Any],
        reason: str = "composed",
        base_version: int | None = None,
    ) -> dict[str, Any]:
        """Persist a new prompt version with a store-assigned monotonic number."""
        if base_version is None:
            try:
                base_version = self.get_active()["version"]
            except PromptStoreError:
                base_version = None

        record = _normalize_input(prompt_version_dict, reason, base_version)

        existing = {int(k) for k in self._versions}
        if existing:
            new_version = max(existing) + 1
        else:
            new_version = 0
        record["version"] = new_version

        key = str(new_version)
        if key in self._versions:
            raise PromptStoreError(
                f"version collision: version {new_version} already exists"
            )

        _validate_record(record)
        self._versions[key] = record
        self._flush(self._versions)
        self._log("save", new_version, {"reason": reason})
        self.prune()
        return record

    def get(self, version_number: int) -> dict[str, Any]:
        """Return the full record for *version_number*."""
        key = str(version_number)
        if key not in self._versions:
            raise PromptStoreError(
                f"version not found: {version_number}"
            )
        record = self._versions[key]
        self._log("get", version_number, {})
        return record

    def get_active(self) -> dict[str, Any]:
        """Return the record with active=True."""
        for record in self._versions.values():
            if record.get("active"):
                self._log("get_active", record["version"], {})
                return record
        raise PromptStoreError("no active prompt version found in the store")

    def activate(self, version_number: int) -> dict[str, Any]:
        """Mark *version_number* as active, archiving the previous active."""
        key = str(version_number)
        if key not in self._versions:
            raise PromptStoreError(
                f"cannot activate: version not found: {version_number}"
            )

        target = self._versions[key]
        if target.get("active"):
            self._log("activate", version_number, {"noop": True})
            return target

        previous_version: int | None = None
        for k, rec in self._versions.items():
            if rec.get("active") and k != key:
                rec["active"] = False
                rec["archived"] = True
                previous_version = rec["version"]

        target["active"] = True
        target["archived"] = False
        self._flush(self._versions)
        self._log("activate", version_number, {"previous": previous_version})
        if previous_version is not None:
            self._log("archive", previous_version, {"by": version_number})
        return target

    def list_versions(self) -> list[dict[str, Any]]:
        """Return metadata dicts sorted by version number, excluding ``text``."""
        result: list[dict[str, Any]] = []
        for key in sorted(self._versions, key=lambda k: int(k)):
            rec = self._versions[key]
            result.append(
                {
                    "version": rec["version"],
                    "hash": rec["hash"],
                    "active": rec["active"],
                    "archived": rec["archived"],
                    "created_at": rec["created_at"],
                    "base_version": rec["base_version"],
                }
            )
        self._log("list", None, {"count": len(result)})
        return result

    def lineage(self, version_number: int) -> list[dict[str, Any]]:
        """Return the ordered ancestry from oldest ancestor to *version_number*."""
        chain: list[dict[str, Any]] = []
        current = version_number
        visited: set[int] = set()
        while True:
            key = str(current)
            if key not in self._versions:
                raise PromptStoreError(
                    f"lineage reconstruction failed: missing ancestor version {current}"
                )
            if current in visited:
                raise PromptStoreError(
                    f"lineage reconstruction failed: cycle detected at version {current}"
                )
            visited.add(current)
            rec = self._versions[key]
            chain.append(rec)
            base = rec.get("base_version")
            if base is None:
                break
            current = int(base)
        chain.reverse()
        self._log("lineage", version_number, {"length": len(chain)})
        return chain

    def _protected_versions(self) -> set[int]:
        """Return the active version plus all its ancestors."""
        try:
            active = self.get_active()
        except PromptStoreError:
            return set()
        protected: set[int] = set()
        for rec in self.lineage(active["version"]):
            protected.add(rec["version"])
        return protected

    def prune(self) -> list[int]:
        """Delete oldest archived versions not in the protected set."""
        if self.config.max_versions is None:
            return []
        protected = self._protected_versions()
        deleted: list[int] = []
        while len(self._versions) > self.config.max_versions:
            candidates = sorted(
                (
                    int(k)
                    for k, rec in self._versions.items()
                    if rec.get("archived") and int(k) not in protected
                ),
            )
            if not candidates:
                break
            victim = candidates[0]
            del self._versions[str(victim)]
            deleted.append(victim)
            self._log("prune", victim, {})
        if deleted:
            self._flush(self._versions)
        return deleted

    def _set_field(
        self, version_number: int, field_name: str, value: Any
    ) -> None:
        """Set a single field on a record, enforcing the mutable allowlist."""
        if field_name not in MUTABLE_FIELDS:
            raise PromptStoreError(
                f"cannot modify immutable field: {field_name}"
            )
        key = str(version_number)
        if key not in self._versions:
            raise PromptStoreError(
                f"version not found: {version_number}"
            )
        self._versions[key][field_name] = value
        self._flush(self._versions)

    def _log(
        self,
        operation: str,
        version: int | None,
        details: dict[str, Any],
    ) -> None:
        """Append one JSON line to the operation log."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": operation,
            "version": version,
            "details": details,
        }
        try:
            with open(self.config.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(
                f"prompt_store: failed to write log: {exc}",
                file=sys.stderr,
            )


def init_store(config: PromptStoreConfig) -> PromptStore:
    """Create a PromptStore, seeding v0 when empty."""
    store = PromptStore(config)
    if not store._versions:
        store._seed_baseline()
    return store

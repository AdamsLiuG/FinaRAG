"""SQLite persistence for LangGraph checkpoints and task writes."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import stat
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import ormsgpack
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    RunnableConfig,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde._msgpack import SAFE_MSGPACK_METHODS, SAFE_MSGPACK_TYPES
from langgraph.checkpoint.serde.jsonplus import (
    EXT_CONSTRUCTOR_KW_ARGS,
    EXT_CONSTRUCTOR_POS_ARGS,
    EXT_CONSTRUCTOR_SINGLE_ARG,
    EXT_METHOD_SINGLE_ARG,
    EXT_PYDANTIC_V1,
    EXT_PYDANTIC_V2,
    JsonPlusSerializer,
)
from langgraph.checkpoint.serde.types import INTERRUPT

from src.storage.base import StorageError, validate_safe_path

_SCHEMA_VERSION = 1
_PAGE_SIZE = 64
_FINARAG_TYPES = {("src.query_plan", "QueryPlan"), ("src.retrieval_filters", "RetrievalFilters")}
_CONSTRUCTOR_CODES = {
    EXT_CONSTRUCTOR_SINGLE_ARG, EXT_CONSTRUCTOR_POS_ARGS, EXT_CONSTRUCTOR_KW_ARGS,
    EXT_PYDANTIC_V1, EXT_PYDANTIC_V2,
}


_CHECKPOINT_COLUMNS = (
    "thread_id, checkpoint_ns, checkpoint_id, parent_id, checkpoint_type, checkpoint_blob, "
    "metadata_type, metadata_blob, insertion_id"
)
_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
        parent_id TEXT, checkpoint_type TEXT NOT NULL, checkpoint_blob BLOB NOT NULL,
        metadata_type TEXT NOT NULL, metadata_blob BLOB NOT NULL,
        insertion_id INTEGER NOT NULL UNIQUE,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    )""",
    """CREATE TABLE IF NOT EXISTS checkpoint_sequence (last_id INTEGER NOT NULL)""",
    """INSERT INTO checkpoint_sequence SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM checkpoint_sequence)""",
    """CREATE INDEX IF NOT EXISTS checkpoint_order
        ON checkpoints (checkpoint_id DESC, thread_id DESC, checkpoint_ns DESC)""",
    """CREATE TABLE IF NOT EXISTS writes (
        thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
        task_id TEXT NOT NULL, idx INTEGER NOT NULL, channel TEXT NOT NULL,
        value_type TEXT NOT NULL, value_blob BLOB NOT NULL, task_path TEXT NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    )""",
    """CREATE TABLE IF NOT EXISTS checkpoint_forks (
        target_thread_id TEXT PRIMARY KEY, source_thread_id TEXT NOT NULL,
        source_checkpoint_ns TEXT NOT NULL, source_checkpoint_id TEXT NOT NULL
    )""",
)


def _validate_extension(code, data):
    # JsonPlus intentionally treats blocked extensions as plain data. Check their
    # type tags before decoding so checkpoint state cannot change type silently.
    fields = ormsgpack.unpackb(data, ext_hook=_validate_extension, option=ormsgpack.OPT_NON_STR_KEYS)
    symbol = tuple(fields[:2])
    if len(fields) == 4 and code == EXT_METHOD_SINGLE_ARG:
        if (*symbol, fields[3]) in SAFE_MSGPACK_METHODS:
            return None
    elif len(fields) >= 3 and code in _CONSTRUCTOR_CODES and symbol in SAFE_MSGPACK_TYPES | _FINARAG_TYPES:
        return None
    raise ValueError("Unsupported msgpack extension")


def _validate_typed(kind, payload):
    if not isinstance(payload, (bytes, bytearray)) or kind not in {"msgpack", "null", "bytes", "bytearray"}:
        raise ValueError("Invalid checkpoint encoding")
    if kind == "null" and payload:
        raise ValueError("Invalid null encoding")
    if kind == "msgpack":
        ormsgpack.unpackb(payload, ext_hook=_validate_extension, option=ormsgpack.OPT_NON_STR_KEYS)


def _identifier(value, *, empty=False):
    if (not isinstance(value, str) or (not value and not empty) or
            len(value) > 4096 or "\0" in value):
        raise ValueError("Invalid checkpoint configuration")
    return value


def checkpoint_config(thread_id, checkpoint_ns="", checkpoint_id=None) -> RunnableConfig:
    fields = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
    if checkpoint_id is not None:
        fields["checkpoint_id"] = checkpoint_id
    return {"configurable": fields}


def _scope(config):
    if not isinstance(config, dict) or not isinstance(config.get("configurable"), dict):
        raise ValueError("Invalid checkpoint configuration")
    fields = config["configurable"]
    thread = _identifier(fields.get("thread_id"))
    namespace = fields.get("checkpoint_ns")
    if "checkpoint_ns" in fields:
        _identifier(namespace, empty=True)
    checkpoint = get_checkpoint_id(config)
    if "checkpoint_id" in fields and checkpoint is not None:
        _identifier(checkpoint)
    return thread, namespace, checkpoint


def _private_file(path: Path) -> int:
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise StorageError("Invalid checkpoint storage")
        os.fchmod(fd, 0o600)
        return fd
    except BaseException:
        os.close(fd)
        raise


class SqliteCheckpointSaver(BaseCheckpointSaver[int]):
    def __init__(self, db_path, *, allowed_base_dir, serde=None, busy_timeout_ms=5000):
        if not allowed_base_dir or type(busy_timeout_ms) is not int or not 0 < busy_timeout_ms <= 60000:
            raise ValueError("Invalid checkpoint storage configuration")
        if serde is None:
            serde = JsonPlusSerializer(
                pickle_fallback=False,
                allowed_msgpack_modules=_FINARAG_TYPES,
            )
        if getattr(serde, "pickle_fallback", False):
            raise ValueError("Pickle is disabled for checkpoints")
        super().__init__(serde=serde)
        self.root = Path(allowed_base_dir).expanduser().resolve()
        raw_path = Path(db_path).expanduser().absolute()
        self.path = validate_safe_path(raw_path, self.root)
        if raw_path != self.path:
            raise StorageError("Invalid checkpoint storage")
        self.busy_timeout_ms = busy_timeout_ms
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            for directory in {self.path.parent, self.root}:
                if directory.stat().st_uid != os.geteuid():
                    raise StorageError("Invalid checkpoint storage")
                directory.chmod(0o700)
            os.close(_private_file(self.path))
            with self._connection(initialize=True) as conn:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, _SCHEMA_VERSION):
                    raise StorageError("Unsupported checkpoint schema")
                if version == 0:
                    for statement in _SCHEMA:
                        conn.execute(statement)
                    conn.execute("PRAGMA user_version = 1")
        except OSError:
            raise StorageError("Checkpoint storage unavailable") from None

    @contextmanager
    def _connection(self, *, readonly=False, initialize=False):
        conn = None
        try:
            for suffix in ("", "-wal", "-shm", "-journal"):
                path = Path(f"{self.path}{suffix}")
                if validate_safe_path(path, self.root) != path:
                    raise StorageError("Invalid checkpoint storage")
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
                    raise StorageError("Invalid checkpoint storage")
            mode = "ro" if readonly else "rw"
            conn = sqlite3.connect(
                f"{self.path.as_uri()}?mode={mode}", uri=True, isolation_level=None,
                timeout=self.busy_timeout_ms / 1000,
            )
            conn.row_factory = sqlite3.Row
            if readonly:
                conn.execute("PRAGMA query_only = ON")
            else:
                if initialize and conn.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
                    raise StorageError("Checkpoint WAL unavailable")
                conn.execute("PRAGMA synchronous = FULL")
            conn.execute("BEGIN" if readonly else "BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except (sqlite3.Error, OSError):
            raise StorageError("Checkpoint storage unavailable") from None
        finally:
            if conn is not None:
                conn.close()

    def _dump(self, value):
        try:
            kind, payload = self.serde.dumps_typed(value)
            _validate_typed(kind, payload)
            return kind, payload
        except Exception:
            raise StorageError("Unsupported checkpoint value") from None

    def _load(self, kind, payload):
        try:
            _validate_typed(kind, payload)
            value = self.serde.loads_typed((kind, payload))
            if value is None and kind != "null":
                raise ValueError("Invalid checkpoint encoding")
            return value
        except Exception:
            raise StorageError("Corrupt checkpoint data") from None

    def _tuple(self, conn, row, metadata):
        thread, namespace, checkpoint_id = row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"]
        checkpoint = self._load(row["checkpoint_type"], row["checkpoint_blob"])
        if (not isinstance(checkpoint, dict) or checkpoint.get("id") != checkpoint_id or
                not all(isinstance(checkpoint.get(key), dict) for key in
                        ("channel_values", "channel_versions", "versions_seen"))):
            raise StorageError("Corrupt checkpoint data")
        writes = conn.execute(
            """SELECT task_id, channel, value_type, value_blob FROM writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            ORDER BY task_path, task_id, idx""", (thread, namespace, checkpoint_id),
        )
        return CheckpointTuple(
            checkpoint_config(thread, namespace, checkpoint_id), checkpoint, metadata,
            checkpoint_config(thread, namespace, row["parent_id"]) if row["parent_id"] else None,
            [(item["task_id"], item["channel"], self._load(item["value_type"], item["value_blob"]))
             for item in writes],
        )

    def _metadata(self, row):
        value = self._load(row["metadata_type"], row["metadata_blob"])
        if not isinstance(value, dict):
            raise StorageError("Corrupt checkpoint data")
        return value

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread, namespace, checkpoint_id = _scope(config)
        with self._connection(readonly=True) as conn:
            row = conn.execute(
                """SELECT * FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?
                AND (? IS NULL OR checkpoint_id = ?) ORDER BY checkpoint_id DESC LIMIT 1""",
                (thread, namespace or "", checkpoint_id, checkpoint_id),
            ).fetchone()
            return self._tuple(conn, row, self._metadata(row)) if row else None

    @staticmethod
    def _list_query(config, before):
        clauses, values = [], []
        if config is not None:
            for column, value in zip(("thread_id", "checkpoint_ns", "checkpoint_id"), _scope(config), strict=True):
                if value is not None:
                    clauses.append(f"{column} = ?")
                    values.append(value)
        if before is not None:
            if not isinstance(before, dict) or not isinstance(before.get("configurable"), dict):
                raise ValueError("Invalid checkpoint configuration")
            cursor = get_checkpoint_id(before)
            if cursor is not None:
                clauses.append("checkpoint_id < ?")
                values.append(_identifier(cursor))
        return " AND ".join(clauses) or "1 = 1", tuple(values)

    def _page(self, conn, query, filter, cursor, maximum):
        where, values = query
        ordering = "(checkpoint_id, thread_id, checkpoint_ns)"
        if cursor is not None:
            where += f" AND {ordering} < (?, ?, ?)"
            values += cursor
        rows = conn.execute(
            f"""SELECT * FROM checkpoints WHERE {where}
            ORDER BY checkpoint_id DESC, thread_id DESC, checkpoint_ns DESC LIMIT ?""",
            (*values, _PAGE_SIZE),
        ).fetchall()
        results = []
        for row in rows:
            cursor = tuple(row[field] for field in ("checkpoint_id", "thread_id", "checkpoint_ns"))
            metadata = self._metadata(row)
            if filter and not all(metadata.get(key) == value for key, value in filter.items()):
                continue
            results.append(self._tuple(conn, row, metadata))
            if len(results) == maximum:
                return results, None
        return results, cursor if len(rows) == _PAGE_SIZE else None

    def list(
        self, config: RunnableConfig | None, *, filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None, limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if limit is not None and limit <= 0:
            return
        query = self._list_query(config, before)
        cursor = None
        with self._connection(readonly=True) as conn:
            while True:
                items, cursor = self._page(conn, query, filter, cursor, limit)
                yield from items
                if cursor is None:
                    break
                if limit is not None:
                    limit -= len(items)

    @staticmethod
    def _next_insertion_id(conn):
        return conn.execute("UPDATE checkpoint_sequence SET last_id = last_id + 1 RETURNING last_id").fetchone()[0]

    def put(
        self, config: RunnableConfig, checkpoint: Checkpoint,
        metadata: CheckpointMetadata, new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread, namespace, parent = _scope(config)
        namespace = namespace or ""
        checkpoint_id = _identifier(checkpoint.get("id"))
        snapshot = self._dump(checkpoint)
        meta = self._dump(get_checkpoint_metadata(config, metadata))
        with self._connection() as conn:
            conn.execute(
                f"""INSERT INTO checkpoints ({_CHECKPOINT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO UPDATE SET
                    parent_id = excluded.parent_id, checkpoint_type = excluded.checkpoint_type,
                    checkpoint_blob = excluded.checkpoint_blob, metadata_type = excluded.metadata_type,
                    metadata_blob = excluded.metadata_blob""",
                (thread, namespace, checkpoint_id, parent, *snapshot, *meta, self._next_insertion_id(conn)),
            )
        return checkpoint_config(thread, namespace, checkpoint_id)

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "",
    ) -> None:
        thread, namespace, checkpoint_id = _scope(config)
        _identifier(checkpoint_id)
        _identifier(task_id)
        records = [
            (thread, namespace or "", checkpoint_id, task_id, WRITES_IDX_MAP.get(channel, index),
             channel, *self._dump(value), task_path)
            for index, (channel, value) in enumerate(writes)
        ]
        with self._connection() as conn:
            conn.executemany(
                """INSERT INTO writes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
                    channel = excluded.channel, value_type = excluded.value_type,
                    value_blob = excluded.value_blob, task_path = excluded.task_path
                WHERE excluded.idx < 0""", records,
            )

    def delete_thread(self, thread_id: str) -> None:
        _identifier(thread_id)
        with self._connection() as conn:
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_forks WHERE target_thread_id = ?", (thread_id,))

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self, config: RunnableConfig | None, *, filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None, limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if limit is not None and limit <= 0:
            return
        where, values = self._list_query(config, before)
        cursor = high_water = None
        while True:
            def read_page():
                with self._connection(readonly=True) as conn:
                    ceiling = high_water
                    if ceiling is None:
                        ceiling = conn.execute("SELECT last_id FROM checkpoint_sequence").fetchone()[0]
                    page_query = (f"{where} AND insertion_id <= ?", (*values, ceiling))
                    return (*self._page(conn, page_query, filter, cursor, limit), ceiling)

            items, cursor, high_water = await asyncio.to_thread(read_page)
            for item in items:
                yield item
            if cursor is None:
                break
            if limit is not None:
                limit -= len(items)

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint,
        metadata: CheckpointMetadata, new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)

    def inspect_checkpoint(self, thread_id, checkpoint_id=None, *, checkpoint_ns=""):
        return self.get_tuple(checkpoint_config(thread_id, checkpoint_ns, checkpoint_id))

    def fork_checkpoint(self, source_config, target_thread_id):
        thread, namespace, selected = _scope(source_config)
        _identifier(target_thread_id)
        namespace = namespace or ""
        with self._connection() as conn:
            return self._fork(conn, thread, namespace, selected, target_thread_id)

    def _fork(self, conn, thread, namespace, selected, target):
        row = conn.execute(
            """SELECT * FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?
            AND (? IS NULL OR checkpoint_id = ?) ORDER BY checkpoint_id DESC LIMIT 1""",
            (thread, namespace, selected, selected),
        ).fetchone()
        if row is None:
            raise ValueError("Unknown checkpoint")
        entry = self._tuple(conn, row, self._metadata(row))
        channels = entry.checkpoint["channel_values"]
        if (any(channel in {"__start__", "__pregel_tasks"} or channel.startswith("branch:to:")
                for channel in channels) or entry.checkpoint.get("pending_sends") or
                channels.get("approval_status") == "pending" or
                any(channel == INTERRUPT for _, channel, _ in entry.pending_writes)):
            raise ValueError("Checkpoint has pending execution")
        if conn.execute(
            """SELECT 1 FROM checkpoints WHERE thread_id = ?
            UNION ALL SELECT 1 FROM writes WHERE thread_id = ?
            UNION ALL SELECT 1 FROM checkpoint_forks WHERE target_thread_id = ? LIMIT 1""",
            (target, target, target),
        ).fetchone():
            raise ValueError("Fork target already exists")
        selected = row["checkpoint_id"]
        conn.execute("INSERT INTO checkpoint_forks VALUES (?, ?, ?, ?)", (target, thread, namespace, selected))
        seen = set()
        while row is not None:
            current = row["checkpoint_id"]
            if current in seen:
                raise StorageError("Corrupt checkpoint ancestry")
            seen.add(current)
            conn.execute(
                f"""INSERT INTO checkpoints ({_CHECKPOINT_COLUMNS})
                SELECT ?, checkpoint_ns, checkpoint_id, parent_id, checkpoint_type, checkpoint_blob,
                    metadata_type, metadata_blob, ? FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?""",
                (target, self._next_insertion_id(conn), thread, namespace, current),
            )
            if current != selected:
                conn.execute(
                    """INSERT INTO writes SELECT ?, checkpoint_ns, checkpoint_id, task_id, idx, channel,
                    value_type, value_blob, task_path FROM writes
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?""",
                    (target, thread, namespace, current),
                )
            if row["parent_id"] is None:
                break
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                (thread, namespace, row["parent_id"]),
            ).fetchone()
            if row is None:
                raise StorageError("Corrupt checkpoint ancestry")
        return checkpoint_config(target, namespace, selected)

"""SQLite access for the lake temperature archive.

Design invariant
----------------
The restored temperature (``restored_mc``) of every frame is persisted and is
authoritative.  A frame's *stored* representation is derived from it:

    absolute frame : stored = restored
    relative frame : stored = restored - restored(base_seq)

Consequently, correcting a base frame

  * changes the restored temperature of that frame only;
  * leaves every other frame's restored temperature and base reference intact;
  * only changes the derived stored difference of frames that reference it.

A correction reads and writes just the addressed frame (plus, for a relative
frame, its base's restored value).

Stale-page protection
---------------------
A correction carries ``seen_restored_mc``: the temperature the page displayed.
If the archive's current restored value differs, the submission is rejected
with 409 and nothing is written.  A submission whose restored value is
unchanged is accepted even when the stored representation has drifted (e.g.
its base frame was corrected in the meantime).
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from . import csvio

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


class ArchiveError(Exception):
    """Domain error carrying an HTTP status code for the UI."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


class ConflictError(ArchiveError):
    def __init__(self, message: str):
        super().__init__(message, status=409)


class PackageError(ArchiveError):
    """A frame cannot fit into a package even on its own."""

    def __init__(self, seq: int, frame_size: int, byte_limit: int):
        super().__init__(
            "帧 %d 单独成行就有 %d 字节，超过每包 %d 字节的上限，无法装包。"
            "请调大字节上限后重试。" % (seq, frame_size, byte_limit),
            status=422,
        )
        self.seq = seq
        self.frame_size = frame_size
        self.byte_limit = byte_limit


@dataclass(frozen=True)
class FrameView:
    seq: int
    sampled_at: str
    sampled_at_raw: str
    restored_mc: int
    base_seq: int | None
    stored_kind: str
    stored_mc: int
    kind_raw: str = ""
    extras: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchiveSummary:
    id: int
    name: str
    source_filename: str
    frame_count: int
    created_at: str


@dataclass(frozen=True)
class PackageView:
    package_no: int
    filename: str
    first_seq: int
    last_seq: int
    frame_count: int
    size_bytes: int


@dataclass(frozen=True)
class BatchView:
    id: int
    byte_limit: int
    frame_count: int
    package_count: int
    created_at: str
    packages: tuple[PackageView, ...] = ()


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as fh:
            conn.executescript(fh.read())
    finally:
        conn.close()


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _stored(restored_mc: int, base_restored_mc: int | None) -> tuple[str, int]:
    if base_restored_mc is None:
        return "absolute", restored_mc
    return "delta", restored_mc - base_restored_mc


def _view(row: sqlite3.Row, base_restored_mc: int | None) -> FrameView:
    kind, stored = _stored(row["restored_mc"], base_restored_mc)
    return FrameView(
        seq=row["seq"],
        sampled_at=row["sampled_at"],
        sampled_at_raw=row["sampled_at_raw"],
        restored_mc=row["restored_mc"],
        base_seq=row["base_seq"],
        stored_kind=kind,
        stored_mc=stored,
        kind_raw=row["kind_raw"],
        extras=tuple(json.loads(row["extras_json"])) if row["extras_json"] else (),
    )


def _fetch_base_restored(conn, archive_id: int, base_seq: int | None) -> int | None:
    if base_seq is None:
        return None
    row = conn.execute(
        "SELECT restored_mc FROM frame WHERE archive_id = ? AND seq = ?",
        (archive_id, base_seq),
    ).fetchone()
    if row is None:
        raise ArchiveError("base frame %d is missing from the archive" % base_seq, status=500)
    return row["restored_mc"]


def create_archive(db_path: str, name: str, filename: str, payload: bytes) -> int:
    """Import a logger CSV as a new archive; returns the archive id."""
    rows, layout = csvio.parse_csv(payload)
    restored: dict[int, int] = {}
    for fr in rows:  # ordered by seq, and base_seq < seq
        if fr.stored_kind == "absolute":
            restored[fr.seq] = fr.stored_mc
        else:
            restored[fr.seq] = fr.stored_mc + restored[fr.base_seq]

    conn = connect(db_path)
    try:
        with _transaction(conn):
            try:
                cur = conn.execute(
                    "INSERT INTO archive (name, source_filename, layout_json)"
                    " VALUES (?, ?, ?)",
                    (name, filename, layout.to_json()),
                )
            except sqlite3.IntegrityError as exc:
                if "archive.name" in str(exc):
                    raise ConflictError(
                        "an archive named %r already exists" % name) from exc
                raise
            archive_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO frame (archive_id, seq, sampled_at, sampled_at_raw,"
                " restored_mc, base_seq, kind_raw, extras_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(archive_id, fr.seq, fr.sampled_at, fr.sampled_at_raw,
                  restored[fr.seq], fr.base_seq, fr.kind_raw,
                  json.dumps(list(fr.extras), ensure_ascii=False)) for fr in rows],
            )
        return archive_id
    finally:
        conn.close()


def list_archives(db_path: str) -> list[ArchiveSummary]:
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT a.id, a.name, a.source_filename, a.created_at, COUNT(f.seq) AS n"
            " FROM archive a LEFT JOIN frame f ON f.archive_id = a.id"
            " GROUP BY a.id ORDER BY a.id"
        ).fetchall()
        return [
            ArchiveSummary(id=r["id"], name=r["name"],
                           source_filename=r["source_filename"],
                           frame_count=r["n"], created_at=r["created_at"])
            for r in rows
        ]
    finally:
        conn.close()


def _archive_row(conn, archive_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM archive WHERE id = ?", (archive_id,)).fetchone()
    if row is None:
        raise ArchiveError("archive not found", status=404)
    return row


def get_archive(db_path: str, archive_id: int) -> ArchiveSummary:
    conn = connect(db_path)
    try:
        row = _archive_row(conn, archive_id)
        n = conn.execute(
            "SELECT COUNT(*) FROM frame WHERE archive_id = ?", (archive_id,)
        ).fetchone()[0]
        return ArchiveSummary(id=row["id"], name=row["name"],
                              source_filename=row["source_filename"],
                              frame_count=n, created_at=row["created_at"])
    finally:
        conn.close()


def get_frame(dbp: str, archive_id: int, seq: int) -> FrameView:
    """Read just the addressed frame (and its base's restored value).

    Only rows of this archive are read.
    """
    conn = connect(dbp)
    try:
        _archive_row(conn, archive_id)
        row = conn.execute(
            "SELECT * FROM frame WHERE archive_id = ? AND seq = ?", (archive_id, seq)
        ).fetchone()
        if row is None:
            raise ArchiveError("frame %d not found in this archive" % seq, status=404)
        return _view(row, _fetch_base_restored(conn, archive_id, row["base_seq"]))
    finally:
        conn.close()


def get_frame_bundle(dbp: str, archive_id: int, seq: int):
    """Read the addressed frame, its base, and the frames that cite it as base.

    Used only to render the single-frame page.  The correction path itself
    reads and writes nothing but the addressed frame (and the base value it is
    relative to); this listing merely warns the operator which stored
    differences will move when a base frame is corrected.
    """
    conn = connect(dbp)
    try:
        _archive_row(conn, archive_id)
        row = conn.execute(
            "SELECT * FROM frame WHERE archive_id = ? AND seq = ?", (archive_id, seq)
        ).fetchone()
        if row is None:
            raise ArchiveError("frame %d not found in this archive" % seq, status=404)
        base_val = _fetch_base_restored(conn, archive_id, row["base_seq"])
        base = None
        if row["base_seq"] is not None:
            base_row = conn.execute(
                "SELECT * FROM frame WHERE archive_id = ? AND seq = ?",
                (archive_id, row["base_seq"]),
            ).fetchone()
            base = _view(base_row, _fetch_base_restored(conn, archive_id, base_row["base_seq"]))

        ref_rows = conn.execute(
            "SELECT * FROM frame WHERE archive_id = ? AND base_seq = ? ORDER BY seq",
            (archive_id, seq),
        ).fetchall()
        referrers = [_view(r, row["restored_mc"]) for r in ref_rows]
        return _view(row, base_val), base, referrers
    finally:
        conn.close()


def list_frames(db_path: str, archive_id: int) -> list[FrameView]:
    conn = connect(db_path)
    try:
        _archive_row(conn, archive_id)
        rows = conn.execute(
            "SELECT * FROM frame WHERE archive_id = ? ORDER BY seq", (archive_id,)
        ).fetchall()
        by_seq = {r["seq"]: r["restored_mc"] for r in rows}
        return [
            _view(r, by_seq.get(r["base_seq"]) if r["base_seq"] is not None else None)
            for r in rows
        ]
    finally:
        conn.close()


def correct_frame(db_path: str, archive_id: int, seq: int, *,
                  seen_restored_mc: int, new_restored_mc: int) -> FrameView:
    """Apply a correction to one frame; return the updated view.

    Raises :class:`ConflictError` when the temperature the submitter saw no
    longer matches the stored restored value.  Nothing is written in that case.
    """
    if new_restored_mc < -1_000_000 or new_restored_mc > 1_000_000:
        raise ArchiveError("restored temperature is outside the supported range")

    conn = connect(db_path)
    try:
        with _transaction(conn):
            _archive_row(conn, archive_id)
            row = conn.execute(
                "SELECT * FROM frame WHERE archive_id = ? AND seq = ?",
                (archive_id, seq),
            ).fetchone()
            if row is None:
                raise ArchiveError("frame %d not found in this archive" % seq, status=404)

            old_restored = row["restored_mc"]
            old_base = row["base_seq"]
            old_base_restored = _fetch_base_restored(conn, archive_id, old_base)
            old_kind, old_stored = _stored(old_restored, old_base_restored)

            if seen_restored_mc != old_restored:
                raise ConflictError(
                    "页面显示的温度是 %d m°C，但该帧当前的还原值是 %d m°C；"
                    "更正已拒绝，现档保持不变。" % (seen_restored_mc, old_restored)
                )

            new_kind, new_stored = _stored(new_restored_mc, old_base_restored)

            if new_restored_mc != old_restored:
                conn.execute(
                    "UPDATE frame SET restored_mc = ? WHERE archive_id = ? AND seq = ?",
                    (new_restored_mc, archive_id, seq),
                )
                conn.execute(
                    "INSERT INTO revision (archive_id, seq, old_restored_mc, old_base_seq,"
                    " old_stored_kind, old_stored_mc, new_restored_mc, new_base_seq,"
                    " new_stored_kind, new_stored_mc, seen_restored_mc)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (archive_id, seq, old_restored, old_base, old_kind, old_stored,
                     new_restored_mc, old_base, new_kind, new_stored, seen_restored_mc),
                )

            return FrameView(
                seq=row["seq"], sampled_at=row["sampled_at"],
                sampled_at_raw=row["sampled_at_raw"],
                restored_mc=new_restored_mc, base_seq=old_base,
                stored_kind=new_kind, stored_mc=new_stored,
                kind_raw=row["kind_raw"],
                extras=tuple(json.loads(row["extras_json"])),
            )
    finally:
        conn.close()


def export_csv(db_path: str, archive_id: int) -> bytes:
    """Render every frame of one archive from a single consistent snapshot.

    The whole read runs inside one transaction, so the file reflects one
    committed state; a correction cannot be half-visible in a download.
    """
    conn = connect(db_path)
    try:
        conn.execute("BEGIN")
        try:
            row = _archive_row(conn, archive_id)
            layout = csvio.CsvLayout.from_json(row["layout_json"])
            rows = conn.execute(
                "SELECT * FROM frame WHERE archive_id = ? ORDER BY seq", (archive_id,)
            ).fetchall()
            by_seq = {r["seq"]: r["restored_mc"] for r in rows}
            frames = [
                _view(r, by_seq.get(r["base_seq"]) if r["base_seq"] is not None else None)
                for r in rows
            ]
        finally:
            conn.execute("COMMIT")
    finally:
        conn.close()
    return csvio.render_csv(frames, layout)


# ---------------------------------------------------------------------------
# Size-limited export batches
# ---------------------------------------------------------------------------

def _absolute_kind_token(conn, archive_id: int) -> str:
    """Pick a kind token spelling 'absolute' used by this archive, if any.

    A delta frame rewritten as absolute must keep the kind column parseable;
    prefer a token the source file itself already used, else the canonical
    English token.  When the layout has no kind column the token is unused.
    """
    row = conn.execute(
        "SELECT kind_raw FROM frame WHERE archive_id = ? AND base_seq IS NULL"
        " AND kind_raw <> '' LIMIT 1",
        (archive_id,),
    ).fetchone()
    return row["kind_raw"] if row is not None else "absolute"


def _as_absolute(frame: FrameView, kind_token: str) -> FrameView:
    """Return the frame rendered as an absolute value frame.

    Used when a delta frame's base is not in the same package: the package
    must restore its measurements on its own, so the frame is emitted with
    its restored temperature and without any base reference.
    """
    return FrameView(
        seq=frame.seq,
        sampled_at=frame.sampled_at,
        sampled_at_raw=frame.sampled_at_raw,
        restored_mc=frame.restored_mc,
        base_seq=None,
        stored_kind="absolute",
        stored_mc=frame.restored_mc,
        kind_raw=kind_token,
        extras=frame.extras,
    )


def _plan_packages(frames: list[FrameView], layout: csvio.CsvLayout,
                   byte_limit: int, kind_token: str):
    """Split ordered frames into packages that each fit within byte_limit.

    Frames are visited in seq order and added to the current package while
    the *rendered* file size (header and line terminators included) stays at
    or under the limit; the frame that would overflow opens the next package.
    A delta frame whose base is not present in the package it is about to
    join is rewritten as an absolute frame, so every package restores all of
    its measurements independently.  Returns a list of ``(frames, payload)``
    tuples; raises :class:`PackageError` if a frame alone cannot fit.
    """
    header = csvio.header_bytes(layout)
    packages: list[tuple[list[FrameView], bytes]] = []

    current: list[FrameView] = []
    current_size = len(header)
    current_bases: set[int] = set()

    for fr in frames:
        if fr.base_seq is not None and fr.base_seq not in current_bases:
            candidate = _as_absolute(fr, kind_token)
        else:
            candidate = fr
        row_bytes = len(csvio.render_frame_row(candidate, layout))

        if current and current_size + row_bytes > byte_limit:
            # The current package is full; seal it and open the next one.
            packages.append((current, header + b"".join(
                csvio.render_frame_row(x, layout) for x in current)))
            current = []
            current_size = len(header)
            current_bases = set()
            candidate = _as_absolute(fr, kind_token) if fr.base_seq is not None else fr
            row_bytes = len(csvio.render_frame_row(candidate, layout))

        if len(header) + row_bytes > byte_limit:
            raise PackageError(fr.seq, row_bytes, byte_limit)

        current.append(candidate)
        current_size += row_bytes
        current_bases.add(fr.seq)

    if current:
        packages.append((current, header + b"".join(
            csvio.render_frame_row(x, layout) for x in current)))
    return packages


def create_export_batch(dbp: str, archive_id: int, byte_limit: int) -> int:
    """Render a size-limited, immutable export batch from one committed state.

    All frames are read inside one ``BEGIN IMMEDIATE`` transaction, so the
    whole batch reflects a single committed state (a concurrent correction is
    serialized before or after it, never half-visible).  Every package is
    rendered in memory first; the batch rows are inserted only once all
    packages fit, and any failure rolls the transaction back without leaving
    a half batch behind.
    """
    if not isinstance(byte_limit, int) or byte_limit <= 0:
        raise ArchiveError("每包字节上限必须是正整数。")

    conn = connect(dbp)
    try:
        with _transaction(conn):
            _archive_row(conn, archive_id)
            layout = csvio.CsvLayout.from_json(conn.execute(
                "SELECT layout_json FROM archive WHERE id = ?", (archive_id,)
            ).fetchone()["layout_json"])
            rows = conn.execute(
                "SELECT * FROM frame WHERE archive_id = ? ORDER BY seq", (archive_id,)
            ).fetchall()
            if not rows:
                raise ArchiveError("该档案没有任何帧，无法生成导出包。", status=404)
            by_seq = {r["seq"]: r["restored_mc"] for r in rows}
            frames = [
                _view(r, by_seq.get(r["base_seq"]) if r["base_seq"] is not None else None)
                for r in rows
            ]
            kind_token = _absolute_kind_token(conn, archive_id)

            # Pure in-memory planning: PackageError here aborts before any
            # insert, and the transaction rolls back untouched.
            planned = _plan_packages(frames, layout, byte_limit, kind_token)

            cur = conn.execute(
                "INSERT INTO export_batch (archive_id, byte_limit, frame_count,"
                " package_count) VALUES (?, ?, ?, ?)",
                (archive_id, byte_limit, len(frames), len(planned)),
            )
            batch_id = cur.lastrowid
            source = conn.execute(
                "SELECT source_filename FROM archive WHERE id = ?", (archive_id,)
            ).fetchone()["source_filename"]
            base_name, dot, ext = source.rpartition(".")
            if dot and base_name:
                suffix = "." + ext
            else:
                suffix = ".csv"

            for no, (pkg_frames, payload) in enumerate(planned, start=1):
                filename = "archive-%d-batch-%d-p%d%s" % (
                    archive_id, batch_id, no, suffix)
                conn.execute(
                    "INSERT INTO export_package (batch_id, package_no, filename,"
                    " first_seq, last_seq, frame_count, size_bytes, payload_bytes)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (batch_id, no, filename, pkg_frames[0].seq, pkg_frames[-1].seq,
                     len(pkg_frames), len(payload), payload),
                )
        return batch_id
    finally:
        conn.close()


def list_batches(dbp: str, archive_id: int) -> list[BatchView]:
    conn = connect(dbp)
    try:
        _archive_row(conn, archive_id)
        batch_rows = conn.execute(
            "SELECT * FROM export_batch WHERE archive_id = ? ORDER BY id", (archive_id,)
        ).fetchall()
        out: list[BatchView] = []
        for b in batch_rows:
            pkg_rows = conn.execute(
                "SELECT package_no, filename, first_seq, last_seq, frame_count,"
                " size_bytes FROM export_package WHERE batch_id = ? ORDER BY package_no",
                (b["id"],),
            ).fetchall()
            out.append(BatchView(
                id=b["id"], byte_limit=b["byte_limit"],
                frame_count=b["frame_count"], package_count=b["package_count"],
                created_at=b["created_at"],
                packages=tuple(
                    PackageView(package_no=p["package_no"], filename=p["filename"],
                                first_seq=p["first_seq"], last_seq=p["last_seq"],
                                frame_count=p["frame_count"], size_bytes=p["size_bytes"])
                    for p in pkg_rows),
            ))
        return out
    finally:
        conn.close()


def get_batch(dbp: str, archive_id: int, batch_id: int) -> BatchView:
    conn = connect(dbp)
    try:
        _archive_row(conn, archive_id)
        b = conn.execute(
            "SELECT * FROM export_batch WHERE id = ? AND archive_id = ?",
            (batch_id, archive_id),
        ).fetchone()
        if b is None:
            raise ArchiveError("export batch not found", status=404)
        pkg_rows = conn.execute(
            "SELECT package_no, filename, first_seq, last_seq, frame_count,"
            " size_bytes FROM export_package WHERE batch_id = ? ORDER BY package_no",
            (batch_id,),
        ).fetchall()
        return BatchView(
            id=b["id"], byte_limit=b["byte_limit"],
            frame_count=b["frame_count"], package_count=b["package_count"],
            created_at=b["created_at"],
            packages=tuple(
                PackageView(package_no=p["package_no"], filename=p["filename"],
                            first_seq=p["first_seq"], last_seq=p["last_seq"],
                            frame_count=p["frame_count"], size_bytes=p["size_bytes"])
                for p in pkg_rows),
        )
    finally:
        conn.close()


def download_package(dbp: str, archive_id: int, batch_id: int,
                     package_no: int) -> tuple[str, bytes]:
    """Return ``(filename, payload)`` for one frozen package file."""
    conn = connect(dbp)
    try:
        _archive_row(conn, archive_id)
        row = conn.execute(
            "SELECT p.filename, p.payload_bytes, p.size_bytes FROM export_package p"
            " JOIN export_batch b ON b.id = p.batch_id"
            " WHERE b.id = ? AND b.archive_id = ? AND p.package_no = ?",
            (batch_id, archive_id, package_no),
        ).fetchone()
        if row is None:
            raise ArchiveError("package not found", status=404)
        payload = bytes(row["payload_bytes"])
        if len(payload) != row["size_bytes"]:
            raise ArchiveError("stored package size is inconsistent", status=500)
        return row["filename"], payload
    finally:
        conn.close()

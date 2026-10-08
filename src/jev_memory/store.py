"""SQLite owns immutable history; model calls never hold write transactions."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .models import Episode, canonical, digest


class Conflict(RuntimeError):
    """The assessed snapshot changed, or an immutable identity was reused."""


class Store:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        path.chmod(0o600)  # Restrict permissions before SQLite creates WAL sidecars.
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA foreign_keys=ON;
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS episodes (
          namespace TEXT, id TEXT, hash TEXT NOT NULL, body TEXT NOT NULL,
          PRIMARY KEY(namespace,id));
        CREATE TABLE IF NOT EXISTS generations (namespace TEXT PRIMARY KEY, value INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS revisions (
          namespace TEXT, id TEXT, revision INTEGER, status TEXT NOT NULL,
          body TEXT NOT NULL, identity TEXT NOT NULL, validation TEXT NOT NULL DEFAULT 'source',
          PRIMARY KEY(namespace,id,revision));
        CREATE TABLE IF NOT EXISTS evidence (
          namespace TEXT, lesson_id TEXT, episode_id TEXT, observation_id TEXT, hash TEXT,
          PRIMARY KEY(namespace,lesson_id,episode_id,observation_id),
          FOREIGN KEY(namespace,episode_id) REFERENCES episodes(namespace,id));
        CREATE TABLE IF NOT EXISTS candidates (
          namespace TEXT, episode_id TEXT, fingerprint TEXT, outcome TEXT NOT NULL,
          PRIMARY KEY(namespace,episode_id,fingerprint));
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY, created TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          namespace TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
          namespace UNINDEXED, id UNINDEXED, revision UNINDEXED, text);
        """)
        path.chmod(0o600)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def episode(self, episode: Episode):
        with self.transaction():
            row = self.db.execute(
                "SELECT hash FROM episodes WHERE namespace=? AND id=?",
                (episode.namespace, episode.id),
            ).fetchone()
            if row and row[0] != episode.trace_hash:
                raise Conflict("episode ID already belongs to a different trace")
            self.db.execute(
                "INSERT OR IGNORE INTO episodes VALUES (?,?,?,?)",
                (episode.namespace, episode.id, episode.trace_hash, canonical(episode)),
            )

    def event(self, namespace, kind, body):
        with self.transaction():
            self.db.execute(
                "INSERT INTO events(namespace,kind,body) VALUES (?,?,?)",
                (namespace, kind, canonical(body)),
            )

    def events(self, namespace):
        return [
            dict(r) | {"body": json.loads(r["body"])}
            for r in self.db.execute(
                "SELECT * FROM events WHERE namespace=? ORDER BY id", (namespace,)
            )
        ]

    def generation(self, namespace):
        row = self.db.execute(
            "SELECT value FROM generations WHERE namespace=?", (namespace,)
        ).fetchone()
        return row[0] if row else 0

    def _bump(self, namespace):
        self.db.execute(
            "INSERT INTO generations VALUES (?,1) ON CONFLICT(namespace) DO UPDATE SET value=value+1",
            (namespace,),
        )

    def lessons(self, namespace, *, active=True):
        rows = self.db.execute(
            """SELECT r.* FROM revisions r WHERE namespace=? AND revision=(
          SELECT max(revision) FROM revisions s WHERE s.namespace=r.namespace AND s.id=r.id)
          ORDER BY id""",
            (namespace,),
        )
        return [
            dict(r) | {"body": json.loads(r["body"])}
            for r in rows
            if not active or r["status"] == "active"
        ]

    def snapshot(self, namespace):
        # Read generation and lessons in the same SQLite snapshot.
        self.db.execute("BEGIN")
        try:
            return self.generation(namespace), self.lessons(namespace)
        finally:
            self.db.rollback()

    def search(self, namespace, text, limit):
        terms = list(dict.fromkeys(re.findall(r"\w+", text.lower())))[:64]
        if not terms:
            return []
        query = " OR ".join('"' + t + '"' for t in terms)
        hits = self.db.execute(
            """SELECT id FROM search WHERE search MATCH ? AND namespace=?
                                  ORDER BY bm25(search),id LIMIT ?""",
            (query, namespace, limit),
        ).fetchall()
        by_id = {r["id"]: r for r in self.lessons(namespace)}
        return [by_id[r[0]] for r in hits if r[0] in by_id]

    def outcome(self, episode, candidate):
        row = self.db.execute(
            "SELECT outcome FROM candidates WHERE namespace=? AND episode_id=? AND fingerprint=?",
            (episode.namespace, episode.id, digest(candidate)),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save(
        self,
        episode,
        candidate,
        outcome,
        generation,
        *,
        duplicate_id=None,
        replace_id=None,
        expected_revision=None,
    ):
        with self.transaction():
            if self.generation(episode.namespace) != generation:
                raise Conflict("memory changed during assessment; retry against a fresh snapshot")
            old = self.outcome(episode, candidate)
            if old:
                return old
            result = dict(outcome)
            if result["status"] == "active":
                id = duplicate_id or replace_id or candidate.identity
                latest = next(
                    (r for r in self.lessons(episode.namespace, active=False) if r["id"] == id),
                    None,
                )
                if replace_id and (latest is None or latest["revision"] != expected_revision):
                    raise Conflict("revision mismatch")
                if duplicate_id and (latest is None or latest["status"] != "active"):
                    raise Conflict("duplicate target no longer active")
                revision = (
                    latest["revision"]
                    if duplicate_id
                    else (latest["revision"] + 1 if latest else 1)
                )
                if not duplicate_id:
                    self.db.execute(
                        "INSERT INTO revisions(namespace,id,revision,status,body,identity) VALUES (?,?,?,?,?,?)",
                        (
                            episode.namespace,
                            id,
                            revision,
                            "active",
                            canonical(candidate),
                            candidate.identity,
                        ),
                    )
                    self.db.execute(
                        "DELETE FROM search WHERE namespace=? AND id=?", (episode.namespace, id)
                    )
                    self.db.execute(
                        "INSERT INTO search VALUES (?,?,?,?)",
                        (
                            episode.namespace,
                            id,
                            revision,
                            " ".join(
                                [
                                    candidate.trigger,
                                    candidate.lesson,
                                    *candidate.preconditions,
                                    *candidate.exceptions,
                                ]
                            ),
                        ),
                    )
                for observation_id, hash in candidate.evidence.items():
                    self.db.execute(
                        "INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?)",
                        (episode.namespace, id, episode.id, observation_id, hash),
                    )
                self._bump(episode.namespace)
                result.update(
                    id=id, revision=revision, duplicate=bool(duplicate_id), validation="source"
                )
            self.db.execute(
                "INSERT INTO candidates VALUES (?,?,?,?)",
                (episode.namespace, episode.id, digest(candidate), canonical(result)),
            )
            return result

    def transition(self, namespace, id, expected_revision, status):
        if status not in {"stale", "archived"}:
            raise ValueError("only stale or archived transitions are allowed")
        with self.transaction():
            old = next((r for r in self.lessons(namespace, active=False) if r["id"] == id), None)
            if old is None or old["revision"] != expected_revision:
                raise Conflict("revision mismatch")
            self.db.execute(
                "INSERT INTO revisions VALUES (?,?,?,?,?,?,?)",
                (
                    namespace,
                    id,
                    expected_revision + 1,
                    status,
                    canonical(old["body"]),
                    old["identity"],
                    old["validation"],
                ),
            )
            self.db.execute("DELETE FROM search WHERE namespace=? AND id=?", (namespace, id))
            self._bump(namespace)

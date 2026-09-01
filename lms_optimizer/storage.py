"""SQLite persistence for all manual LMS workflow records."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TypeVar
from .models import Entry, FamilyMember, Fixture, OddsQuote, Player, Round, Season, WiderFieldSnapshot

T = TypeVar("T")

class Repository:
    TABLES = ("seasons", "rounds", "fixtures", "odds_quotes", "players", "family_members", "entries", "selections", "wider_field", "raw_imports", "audit_log")

    def __init__(self, path: str | Path = "data/lms.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS seasons (season TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS rounds (season TEXT NOT NULL, round_number INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(season, round_number));
            CREATE TABLE IF NOT EXISTS fixtures (fixture_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS odds_quotes (id INTEGER PRIMARY KEY AUTOINCREMENT, fixture_id TEXT NOT NULL, bookmaker TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(fixture_id, bookmaker));
            CREATE TABLE IF NOT EXISTS players (player_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS family_members (member_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS entries (entry_id TEXT PRIMARY KEY, player_id TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS selections (id INTEGER PRIMARY KEY AUTOINCREMENT, entry_id TEXT NOT NULL, round_number INTEGER NOT NULL, team TEXT NOT NULL, is_backup INTEGER NOT NULL, payload TEXT NOT NULL, UNIQUE(entry_id, round_number, team, is_backup));
            CREATE TABLE IF NOT EXISTS wider_field (season TEXT NOT NULL, round_number INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(season, round_number));
            CREATE TABLE IF NOT EXISTS raw_imports (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, collected_at TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL);
        """)
        self.connection.commit()
        # A non-destructive migration: old players remain untouched and are
        # copied into the explicit family-member table only when missing.
        for row in self.connection.execute("SELECT player_id, payload FROM players").fetchall():
            self.connection.execute("INSERT OR IGNORE INTO family_members(member_id, payload) VALUES (?, ?)", (row[0], row[1]))
        self.connection.commit()

    def _save(self, table: str, key_sql: str, values: tuple[object, ...]) -> None:
        self.connection.execute(f"INSERT OR REPLACE INTO {table} {key_sql}", values)
        self.connection.commit()

    def save_season(self, season: Season) -> None:
        self._save("seasons", "VALUES (?, ?)", (season.season, season.model_dump_json()))

    def save_round(self, round_: Round) -> None:
        self._save("rounds", "VALUES (?, ?, ?)", (round_.season, round_.round_number, round_.model_dump_json()))

    def save_fixtures(self, fixtures: Iterable[Fixture]) -> None:
        for fixture in fixtures:
            self._save("fixtures", "VALUES (?, ?)", (fixture.fixture_id, fixture.model_dump_json()))

    def save_odds(self, quotes: Iterable[OddsQuote]) -> None:
        for quote in quotes:
            self.connection.execute("INSERT OR REPLACE INTO odds_quotes(fixture_id, bookmaker, payload) VALUES (?, ?, ?)", (quote.fixture_id, quote.bookmaker, quote.model_dump_json()))
        self.connection.commit()

    def save_player(self, player: Player) -> None:
        self._save("players", "VALUES (?, ?)", (player.player_id, player.model_dump_json()))

    def save_family_member(self, member: FamilyMember) -> None:
        self._save("family_members", "VALUES (?, ?)", (member.member_id, member.model_dump_json()))
        # Keep legacy readers and old snapshots working.
        self._save("players", "VALUES (?, ?)", (member.member_id, Player(player_id=member.member_id, name=member.name, is_sample=member.is_sample).model_dump_json()))

    def save_entry(self, entry: Entry) -> None:
        owner = entry.member_id or entry.player
        self._save("entries", "VALUES (?, ?, ?)", (entry.entry_id, owner, entry.model_dump_json()))

    def save_selection(self, selection: dict[str, object]) -> None:
        self.connection.execute("INSERT INTO selections(entry_id, round_number, team, is_backup, payload) VALUES (?, ?, ?, ?, ?)", (selection["entry_id"], selection["round_number"], selection["team"], int(bool(selection.get("is_backup", False))), json.dumps(selection)))
        self.connection.commit()

    def save_wider_field(self, snapshot: WiderFieldSnapshot) -> None:
        self._save("wider_field", "VALUES (?, ?, ?)", (snapshot.season, snapshot.round_number, snapshot.model_dump_json()))

    def list_payloads(self, table: str) -> list[dict[str, object]]:
        if table not in self.TABLES:
            raise ValueError("unknown table")
        return [json.loads(row["payload"]) for row in self.connection.execute(f"SELECT payload FROM {table} ORDER BY rowid")]

    def record_raw(self, kind: str, collected_at: str, payload: object) -> None:
        self.connection.execute("INSERT INTO raw_imports(kind, collected_at, payload) VALUES (?, ?, ?)", (kind, collected_at, json.dumps(payload, default=str)))
        self.connection.commit()

    def audit(self, event: str, payload: object, created_at: str | None = None) -> None:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        self.connection.execute("INSERT INTO audit_log(event, created_at, payload) VALUES (?, ?, ?)", (event, timestamp, json.dumps(payload, default=str)))
        self.connection.commit()

    def count(self, table: str) -> int:
        if table not in self.TABLES:
            raise ValueError("unknown table")
        return int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def close(self) -> None:
        self.connection.close()

"""Reusable weekly recommendation snapshots and explicit locking."""
from datetime import datetime, timezone
from pathlib import Path
import json
from pydantic import BaseModel, Field

class RecommendationSnapshot(BaseModel):
    version: str = Field(min_length=1)
    created_at: datetime
    season: str
    round_number: int
    odds_snapshot_version: str
    forecast_snapshot_version: str
    active_entries: list[str]
    used_teams: dict[str, list[str]]
    objective_weights: dict[str, float]
    exposure_limits: dict[str, object]
    simulation_settings: dict[str, object]
    seed: int
    optimiser_version: str
    allocation: dict[str, str]
    risk_estimates: dict[str, float]
    locked: bool = False

class WeeklyStore:
    def __init__(self, directory: str | Path = "data/recommendations"):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
    def save(self, snapshot: RecommendationSnapshot) -> Path:
        path = self.directory / f"{snapshot.version}.json"
        if path.exists(): raise FileExistsError("recommendation versions are immutable")
        path.write_text(snapshot.model_dump_json(indent=2)); return path
    def lock(self, version: str) -> Path:
        path = self.directory / f"{version}.json"
        if not path.exists(): raise FileNotFoundError(version)
        data = json.loads(path.read_text())
        if data.get("locked"): return path
        data["locked"] = True; data["locked_at"] = datetime.now(timezone.utc).isoformat()
        locked = path.with_name(path.stem + "-locked.json")
        if locked.exists(): raise FileExistsError("locked recommendation already exists")
        locked.write_text(json.dumps(data, indent=2)); return locked
    @staticmethod
    def whatsapp_message(snapshot: RecommendationSnapshot) -> str:
        lines = [f"LMS round {snapshot.round_number} — recommendation {snapshot.version}"]
        lines += [f"{entry}: {team}" for entry, team in snapshot.allocation.items()]
        lines.append("Backups and risk estimates are in the saved local snapshot.")
        return "\\n".join(lines)


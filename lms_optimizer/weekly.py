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
    backups: dict[str, str | None] = Field(default_factory=dict)
    odds_snapshot: dict[str, object] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(default_factory=dict)
    exact_risk: dict[str, object] = Field(default_factory=dict)
    previous_version: str | None = None
    unlock_reason: str | None = None
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
        data["locked"] = True; data["locked_at"] = datetime.now(timezone.utc).isoformat(); data["previous_version"] = version; data["version"] = f"{version}-locked"
        locked = path.with_name(path.stem + "-locked.json")
        if locked.exists(): raise FileExistsError("locked recommendation already exists")
        locked.write_text(json.dumps(data, indent=2)); return locked
    def versions(self) -> list[RecommendationSnapshot]:
        return [RecommendationSnapshot.model_validate(json.loads(path.read_text())) for path in sorted(self.directory.glob("*.json"))]

    def unlock(self, version: str, reason: str, user_action_time: datetime | None = None) -> RecommendationSnapshot:
        if not reason.strip(): raise ValueError("an unlock reason is required")
        source = next((item for item in self.versions() if item.version == version), None)
        if source is None: raise FileNotFoundError(version)
        if not source.locked: raise ValueError("only a locked recommendation can be unlocked")
        created = user_action_time or datetime.now(timezone.utc)
        new_version = f"{created.strftime('%Y%m%dT%H%M%S%fZ')}-{version}-unlocked"
        new = source.model_copy(update={"version": new_version, "created_at": created, "locked": False, "previous_version": version, "unlock_reason": reason})
        self.save(new)
        return new

    @staticmethod
    def compare(left: RecommendationSnapshot, right: RecommendationSnapshot) -> dict[str, object]:
        def changes(a, b): return {key: {"before": a.get(key), "after": b.get(key)} for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)}
        return {"odds": changes(left.odds_snapshot, right.odds_snapshot), "probabilities": changes(left.probabilities, right.probabilities), "allocations": changes(left.allocation, right.allocation), "backups": changes(left.backups, right.backups), "exact_risk": changes(left.exact_risk, right.exact_risk), "strategy_settings": changes(left.objective_weights, right.objective_weights) | changes(left.exposure_limits, right.exposure_limits) | changes(left.simulation_settings, right.simulation_settings), "left_version": left.version, "right_version": right.version}
    @staticmethod
    def whatsapp_message(snapshot: RecommendationSnapshot) -> str:
        lines = [f"LMS round {snapshot.round_number} — recommendation {snapshot.version}"]
        lines += [f"{entry}: {team}" for entry, team in snapshot.allocation.items()]
        lines.append("Backups and risk estimates are in the saved local snapshot.")
        return "\\n".join(lines)

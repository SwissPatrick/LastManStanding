"""Immutable, leakage-safe dated forecast snapshots."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json
from pydantic import BaseModel, Field

class ForecastSnapshot(BaseModel):
    version: str = Field(min_length=1)
    created_at: datetime
    information_cutoff: datetime
    training_cutoff: datetime
    raw_manifest_checksum: str = Field(min_length=1)
    model_name: str
    model_version: str
    model_config_data: dict[str, object]
    fixtures: list[dict[str, object]]
    validation_status: str
    provenance: str

class ForecastStore:
    def __init__(self, directory: str | Path = "data/forecasts"):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
    def save(self, snapshot: ForecastSnapshot) -> Path:
        path = self.directory / f"{snapshot.version}.json"
        if path.exists(): raise FileExistsError(f"forecast snapshot is immutable: {path}")
        path.write_text(snapshot.model_dump_json(indent=2)); return path
    def create(self, information_cutoff: datetime, training_cutoff: datetime, manifest_path: str | Path, model_name: str, model_version: str, fixtures: list[dict[str, object]], model_config: dict[str, object] | None = None) -> ForecastSnapshot:
        manifest_checksum = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + manifest_checksum[:8]
        return ForecastSnapshot(version=version, created_at=datetime.now(timezone.utc), information_cutoff=information_cutoff, training_cutoff=training_cutoff, raw_manifest_checksum=manifest_checksum, model_name=model_name, model_version=model_version, model_config_data=model_config or {}, fixtures=fixtures, validation_status="unvalidated", provenance="local historical data before information cutoff")

    def create_manual(self, information_cutoff: datetime, training_cutoff: datetime, manifest_text: str, model_name: str, model_version: str, fixtures: list[dict[str, object]], validation_status: str = "validated", provenance: str = "manual local forecast input") -> ForecastSnapshot:
        """Create a deterministic immutable snapshot from guided manual input."""
        checksum = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        version = f"manual-{checksum[:16]}"
        return ForecastSnapshot(version=version, created_at=datetime.now(timezone.utc), information_cutoff=information_cutoff, training_cutoff=training_cutoff, raw_manifest_checksum=checksum, model_name=model_name, model_version=model_version, model_config_data={}, fixtures=fixtures, validation_status=validation_status, provenance=provenance)

    def get(self, version: str) -> ForecastSnapshot:
        path = self.directory / f"{version}.json"
        if not path.exists(): raise FileNotFoundError(version)
        return ForecastSnapshot.model_validate_json(path.read_text())

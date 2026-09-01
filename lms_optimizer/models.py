"""Validated domain objects."""
from datetime import datetime
from enum import StrEnum
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

class FixtureStatus(StrEnum):
    SCHEDULED = "scheduled"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"
    PLAYED = "played"

class Season(BaseModel):
    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    name: str = Field(min_length=1)
    is_sample: bool = False

class Round(BaseModel):
    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    round_number: int = Field(ge=1)
    selection_deadline: datetime
    is_sample: bool = False

class Player(BaseModel):
    player_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    is_sample: bool = False


class FamilyMember(BaseModel):
    """One of the five permanent cartel members.

    ``member_id`` is an internal key; presentation code must use ``name``.
    """
    member_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    position: int = Field(ge=1, le=5)
    is_sample: bool = False

class Fixture(BaseModel):
    fixture_id: str = Field(min_length=1)
    provider_event_id: Optional[str] = None
    competition: Literal["Premier League"] = "Premier League"
    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    round_number: int = Field(ge=1)
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    kickoff: datetime
    status: FixtureStatus = FixtureStatus.SCHEDULED
    home_goals: Optional[int] = Field(default=None, ge=0)
    away_goals: Optional[int] = Field(default=None, ge=0)
    data_source: str = "manual"
    collected_at: datetime
    market_timestamp: Optional[datetime] = None
    is_sample: bool = False

    @field_validator("away_team")
    @classmethod
    def teams_must_differ(cls, value: str, info):
        if value == info.data.get("home_team"):
            raise ValueError("home and away teams must differ")
        return value

class OddsQuote(BaseModel):
    fixture_id: str
    bookmaker: str = Field(min_length=1)
    home: float = Field(gt=1.0)
    draw: float = Field(gt=1.0)
    away: float = Field(gt=1.0)
    collected_at: datetime
    market_timestamp: datetime
    data_source: str = "manual"
    is_sample: bool = False

class Entry(BaseModel):
    entry_id: str = Field(min_length=1)
    # player is retained for backwards-compatible database payloads.
    player: Optional[str] = None
    member_id: Optional[str] = None
    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    active: bool = True
    is_sample: bool = False

    @field_validator("member_id", mode="after")
    @classmethod
    def owner_is_present(cls, value, info):
        if not value and not info.data.get("player"):
            raise ValueError("an entry must belong to a family member")
        return value


class WiderFieldSnapshot(BaseModel):
    """Manual, round-stamped information about the private wider competition."""
    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    round_number: int = Field(ge=1)
    starting_entries: int = Field(ge=0)
    surviving_entries: int = Field(ge=0)
    known_selections: Optional[dict[str, int]] = None
    recorded_at: datetime

    @field_validator("surviving_entries")
    @classmethod
    def survivors_cannot_exceed_start(cls, value, info):
        if "starting_entries" in info.data and value > info.data["starting_entries"]:
            raise ValueError("surviving entries cannot exceed starting entries")
        return value

class Selection(BaseModel):
    entry_id: str
    round_number: int = Field(ge=1)
    team: str = Field(min_length=1)
    selected_at: datetime
    automatic: bool = False
    result: Optional[str] = None

class HistoricalMatch(BaseModel):
    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    match_date: datetime
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    full_time_home_goals: int = Field(ge=0)
    full_time_away_goals: int = Field(ge=0)
    closing_home_odds: float = Field(gt=1.0)
    closing_draw_odds: float = Field(gt=1.0)
    closing_away_odds: float = Field(gt=1.0)
    expected_home_goals: Optional[float] = Field(default=None, ge=0)
    expected_away_goals: Optional[float] = Field(default=None, ge=0)
    data_source: str = Field(min_length=1)
    collected_at: datetime
    is_sample: bool = False
    odds_timing: Literal["opening", "closing", "intermediate", "timestamp-unknown"] = "timestamp-unknown"
    odds_method: str = "market-average"

    @field_validator("away_team")
    @classmethod
    def historical_teams_must_differ(cls, value: str, info):
        if value == info.data.get("home_team"):
            raise ValueError("home and away teams must differ")
        return value

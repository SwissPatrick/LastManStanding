"""Guided manual weekly LMS dashboard; business rules stay in LMSWorkflow."""
from datetime import datetime, time, timezone, timedelta
from pathlib import Path
import csv
import io
import uuid

def run() -> None:
    try:
        import pandas as pd
        import streamlit as st
    except ImportError:
        print("Install dependencies with: pip install -r requirements.txt")
        return
    from .models import Entry, Fixture, FixtureStatus, OddsQuote, Player, Round, Season
    from .rules import eligible_fixtures
    from .storage import Repository
    from .weekly import RecommendationSnapshot, WeeklyStore
    from .workflow import LMSWorkflow
    st.set_page_config(page_title="LMS Weekly Manager", layout="wide")
    st.title("Premier League Last Man Standing")
    st.caption("Guided manual weekly workflow · local only · no live APIs or automatic betting")
    repo = Repository(Path("data/lms.sqlite3")); service = LMSWorkflow(repo)
    steps = ["Season and round", "Fixtures and odds", "Players and entries", "Validate round", "Analyse", "Review selections", "Lock and share", "Record results"]
    st.session_state.setdefault("step", 0); st.session_state.setdefault("analysis", None); st.session_state.setdefault("locked", False)
    season = st.session_state.get("season", "2026/27"); round_number = int(st.session_state.get("round_number", 1))
    st.progress((st.session_state.step + 1) / len(steps), text=f"Step {st.session_state.step + 1} of {len(steps)} · {steps[st.session_state.step]}")
    st.write("Completed:", " · ".join(steps[:st.session_state.step]) or "none")
    try: gate = service.validate_round(season, round_number, st.session_state.get("strategy", "concentrated_favourite"))
    except Exception as exc: gate = {"valid": False, "errors": [str(exc)]}
    with st.expander("Blocking problems", expanded=True):
        if gate.get("valid"): st.success("Current prerequisites are satisfied.")
        else:
            for problem in gate.get("errors", []): st.error(problem)
    tabs = st.tabs(steps)
    def go(index): st.session_state.step = index; st.rerun()
    with tabs[0]:
        st.header("1. Season and round")
        with st.form("season_form"):
            season = st.text_input("Season", season); season_name = st.text_input("Season name", "Premier League LMS")
            if st.form_submit_button("Create / update season"):
                try: service.create_season(Season(season=season, name=season_name)); st.session_state.season = season; st.success("Season saved.")
                except Exception as exc: st.error(str(exc))
        with st.form("round_form"):
            round_number = int(st.number_input("Round number", min_value=1, value=round_number)); deadline_date = st.date_input("Selection deadline date"); deadline_time = st.time_input("Selection deadline time", value=time(12, 0))
            if st.form_submit_button("Create / update round"):
                try: service.create_round(Round(season=season, round_number=round_number, selection_deadline=datetime.combine(deadline_date, deadline_time, tzinfo=timezone.utc))); st.session_state.update(season=season, round_number=round_number); st.success("Round saved.")
                except Exception as exc: st.error(str(exc))
        st.info("Configured local timezone: UTC. Deadlines are timezone-aware.")
        if st.button("Load deterministic historical dry-run"):
            try:
                demo_season = "2020/21"; demo_day = datetime(2020, 8, 1, tzinfo=timezone.utc)
                service.create_season(Season(season=demo_season, name="Historical dry run", is_sample=True))
                service.create_round(Round(season=demo_season, round_number=1, selection_deadline=demo_day + timedelta(days=1), is_sample=True))
                demo_fixtures = [Fixture(fixture_id=f"dry-{i}", season=demo_season, round_number=1, home_team=f"Home {i}", away_team=f"Away {i}", kickoff=demo_day + timedelta(days=2, minutes=i), collected_at=demo_day, data_source="deterministic-dry-run", is_sample=True) for i in range(6)]
                service.add_fixtures(demo_fixtures); service.add_odds([OddsQuote(fixture_id=f.fixture_id, bookmaker="dry-run-bookmaker", home=1.5, draw=4.0, away=6.0, collected_at=demo_day, market_timestamp=demo_day, data_source="deterministic-dry-run", is_sample=True) for f in demo_fixtures]); service.add_player(Player(player_id="dry-player", name="Dry Run Player", is_sample=True)); service.add_entry(Entry(entry_id="dry-entry", player="dry-player", season=demo_season, is_sample=True)); st.session_state.update(season=demo_season, round_number=1); st.success("Historical dry-run data loaded. It is demonstration data and does not predict future performance.")
            except Exception as exc: st.error(str(exc))
        if st.button("Continue to fixtures", key="next0"): go(1)
    with tabs[1]:
        st.header("2. Fixtures and odds"); st.caption("Manual and timestamp-unknown odds are never labelled live.")
        with st.form("fixture_form"):
            fixture_id = st.text_input("Fixture identifier", str(uuid.uuid4())[:8]); home = st.text_input("Home team"); away = st.text_input("Away team"); kickoff_date = st.date_input("Date and kickoff date"); kickoff_time = st.time_input("Kickoff time"); status = st.selectbox("Fixture status", list(FixtureStatus), format_func=lambda x: x.value)
            if st.form_submit_button("Add fixture"):
                try: service.add_fixtures([Fixture(fixture_id=fixture_id, season=season, round_number=round_number, home_team=home, away_team=away, kickoff=datetime.combine(kickoff_date, kickoff_time, tzinfo=timezone.utc), status=status, data_source="manual", collected_at=datetime.now(timezone.utc))]); st.success("Fixture saved.")
                except Exception as exc: st.error(str(exc))
        st.subheader("CSV paste/import"); st.caption("Columns: fixture_id,home_team,away_team,kickoff,bookmaker,home_odds,draw_odds,away_odds,market_timestamp")
        pasted = st.text_area("Fixture and odds CSV")
        if st.button("Import CSV paste") and pasted:
            try:
                fixtures, quotes = [], []
                for row in csv.DictReader(io.StringIO(pasted)):
                    fid = row["fixture_id"]
                    if not any(f.fixture_id == fid for f in service.fixtures() + fixtures): fixtures.append(Fixture(fixture_id=fid, season=season, round_number=round_number, home_team=row["home_team"], away_team=row["away_team"], kickoff=datetime.fromisoformat(row["kickoff"]), data_source="manual-csv", collected_at=datetime.now(timezone.utc)))
                    quotes.append(OddsQuote(fixture_id=fid, bookmaker=row["bookmaker"], home=float(row["home_odds"]), draw=float(row["draw_odds"]), away=float(row["away_odds"]), collected_at=datetime.now(timezone.utc), market_timestamp=datetime.fromisoformat(row["market_timestamp"]), data_source="manual-csv"))
                if fixtures: service.add_fixtures(fixtures)
                service.add_odds(quotes); st.success(f"Imported {len(fixtures)} fixtures and {len(quotes)} odds quotes.")
            except Exception as exc: st.error(f"CSV import error: {exc}")
        rows = [f.model_dump() for f in service.fixtures() if f.season == season and f.round_number == round_number]
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True)
        eligible = eligible_fixtures(service.fixtures(), round_number); st.metric("Eligible fixtures", len(eligible)); (st.success("Six-match rule satisfied.") if len(eligible) >= 6 else st.warning("Six-match rule is not satisfied."))
        if st.button("Continue to players", key="next1"): go(2)
    with tabs[2]:
        st.header("3. Players and entries")
        with st.form("player_form"):
            player_id = st.text_input("Player ID"); player_name = st.text_input("Player name")
            if st.form_submit_button("Create player"):
                try: service.add_player(Player(player_id=player_id, name=player_name)); st.success("Player saved.")
                except Exception as exc: st.error(str(exc))
        with st.form("entry_form"):
            entry_id = st.text_input("Entry ID"); entry_player = st.text_input("Player ID for this paid entry")
            if st.form_submit_button("Create paid entry"):
                try: service.add_entry(Entry(entry_id=entry_id, player=entry_player, season=season)); st.success("Entry saved.")
                except Exception as exc: st.error(str(exc))
        entries = [e for e in service.entries() if e.season == season]
        if entries: st.dataframe(pd.DataFrame([{"Entry": e.entry_id, "Player": e.player, "Active": e.active, "Used teams": ", ".join(service.used_teams(e.entry_id)), "Available teams": ", ".join(service.available_teams(e.entry_id, round_number))} for e in entries]), use_container_width=True)
        if st.button("Continue to validation", key="next2"): go(3)
    labels = {"concentrated_favourite": "Concentrated market favourite — Validated default", "independent_greedy": "Independent greedy — validated alternative", "max_expected_survivors": "Maximum expected survivors — validated alternative", "protect_one": "Protect One — experimental", "bellman": "Bellman — experimental", "equal_diversification": "Equal diversification — experimental", "balanced": "Balanced — experimental"}
    with tabs[3]:
        st.header("4. Validate round"); strategy = st.selectbox("Strategy", list(labels), format_func=lambda x: labels[x], key="strategy"); gate = service.validate_round(season, round_number, strategy); st.json(gate)
        if gate["valid"]: st.success("Validation passed; analysis is unlocked."); st.session_state.validation = True
        else: st.error("Resolve the blocking problems before analysis.")
        if st.button("Continue to analysis", key="next3", disabled=not gate["valid"]): go(4)
    with tabs[4]:
        st.header("5. Analyse")
        if st.button("Run exact analysis", disabled=not st.session_state.get("validation", False)):
            try: st.session_state.analysis = service.analyse_round(season, round_number, st.session_state.get("strategy", "concentrated_favourite")); st.success("Exact conditional analysis complete.")
            except Exception as exc: st.error(str(exc))
        analysis = st.session_state.analysis
        if analysis:
            risk = analysis["risk"]; st.dataframe(pd.DataFrame(analysis["probabilities"]), use_container_width=True); st.json({k: risk[k] for k in ("expected_survivors", "probability_at_least_one", "wipeout_probability", "cvar", "survivor_counts", "probabilities")}); st.dataframe(pd.DataFrame([{"Entry": e, "Recommended": t, "Backup": analysis["backups"].get(e)} for e, t in analysis["allocation"].items()]), use_container_width=True); st.caption("Risk is exact conditional on supplied fair probabilities; future values are model forecasts.")
        if st.button("Continue to review", key="next4", disabled=analysis is None): go(5)
    with tabs[5]:
        st.header("6. Review selections — Selections and backups"); analysis = st.session_state.analysis
        if analysis: st.dataframe(pd.DataFrame([{"Entry": e, "Primary pick": t, "Backup": analysis["backups"].get(e)} for e, t in analysis["allocation"].items()]), use_container_width=True)
        st.checkbox("I reviewed every pick and backup", key="reviewed")
        if st.button("Continue to lock", key="next5", disabled=not st.session_state.get("reviewed", False)): go(6)
    with tabs[6]:
        st.header("7. Lock and share"); analysis = st.session_state.analysis
        if analysis:
            if st.button("Save recommendation snapshot", disabled=st.session_state.locked):
                try:
                    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"); snap = RecommendationSnapshot(version=version, created_at=datetime.now(timezone.utc), season=season, round_number=round_number, odds_snapshot_version="manual-odds", forecast_snapshot_version="not-selected", active_entries=list(analysis["allocation"]), used_teams={e: service.used_teams(e) for e in analysis["allocation"]}, objective_weights=analysis["objective_weights"], exposure_limits={}, simulation_settings={}, seed=7, optimiser_version="weekly-service", allocation=analysis["allocation"], risk_estimates={"expected_survivors": float(analysis["risk"]["expected_survivors"])}); WeeklyStore().save(snap); st.session_state.snapshot = snap; st.success("Immutable recommendation snapshot saved.")
                except Exception as exc: st.error(str(exc))
            if st.button("Lock picks", disabled=st.session_state.locked or "snapshot" not in st.session_state):
                try:
                    service.save_recommendation_selections(analysis["allocation"], analysis["backups"], round_number); st.session_state.locked = True; repo.audit("recommendation_locked", {"season": season, "round": round_number}); st.success("Picks locked.")
                except Exception as exc: st.error(str(exc))
            if st.session_state.locked: st.code(WeeklyStore.whatsapp_message(st.session_state.snapshot), language=None)
        if st.button("Continue to results", key="next6", disabled=not st.session_state.locked): go(7)
    with tabs[7]:
        st.header("8. Record results"); fixtures = [f for f in service.fixtures() if f.season == season and f.round_number == round_number]
        if fixtures:
            chosen = st.selectbox("Fixture to update", [f"{f.fixture_id}: {f.home_team} v {f.away_team}" for f in fixtures]); target = next(f for f in fixtures if f.fixture_id == chosen.split(":")[0]); result_status = st.selectbox("Result status", list(FixtureStatus), format_func=lambda x: x.value); home_goals = st.number_input("Home goals", min_value=0, value=0); away_goals = st.number_input("Away goals", min_value=0, value=0)
            if st.button("Finalise result", disabled=not st.session_state.locked):
                try: st.session_state.survival = service.record_results_and_advance(target.fixture_id, result_status, home_goals if result_status == FixtureStatus.PLAYED else None, away_goals if result_status == FixtureStatus.PLAYED else None); st.success("Result recorded and survivors advanced.")
                except Exception as exc: st.error(str(exc))
        if st.session_state.get("survival"): st.dataframe(pd.DataFrame([{"Entry": k, "Status": v} for k, v in st.session_state.survival.items()]), use_container_width=True)

if __name__ == "__main__": run()

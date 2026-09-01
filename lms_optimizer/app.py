"""Local Streamlit dashboard. All business operations are delegated to LMSWorkflow."""
from datetime import datetime, time, timezone
from pathlib import Path
import uuid
import json

def run() -> None:
    try:
        import pandas as pd
        import plotly.express as px
        import streamlit as st
    except ImportError:
        print("Install dependencies with: pip install -r requirements.txt")
        return
    from .models import Entry, Fixture, FixtureStatus, OddsQuote, Player, Round, Season
    from .storage import Repository
    from .workflow import LMSWorkflow
    from .optimizer import DynamicProgram
    from .rules import eligible_fixtures

    st.set_page_config(page_title="LMS Cartel Manager", layout="wide")
    st.title("Premier League Last Man Standing")
    st.caption("Local manual cartel manager • no external APIs • no automated betting")
    st.info("Only records you enter are used. Each record carries a manual source and sample data is never created automatically.")
    repo = Repository(Path("data/lms.sqlite3"))
    service = LMSWorkflow(repo)

    tabs = st.tabs(["Season & round", "Fixtures & odds", "Players & entries", "Selections", "Results", "Analysis", "Historical models", "Optimiser"])
    with tabs[0]:
        st.header("Season and round")
        with st.form("season_form"):
            season = st.text_input("Season", "2026/27")
            season_name = st.text_input("Season name", "Premier League LMS")
            season_sample = st.checkbox("This is demonstration/sample data")
            if st.form_submit_button("Create / update season"):
                try:
                    service.create_season(Season(season=season, name=season_name, is_sample=season_sample))
                    st.success("Season saved.")
                except Exception as exc: st.error(str(exc))
        with st.form("round_form"):
            round_season = st.text_input("Round season", "2026/27")
            round_number = st.number_input("Round number", min_value=1, value=1)
            deadline_date = st.date_input("Selection deadline date")
            deadline_time = st.time_input("Selection deadline time", value=time(12, 0))
            round_sample = st.checkbox("Round is demonstration/sample data")
            if st.form_submit_button("Create / update round"):
                try:
                    deadline = datetime.combine(deadline_date, deadline_time, tzinfo=timezone.utc)
                    service.create_round(Round(season=round_season, round_number=round_number, selection_deadline=deadline, is_sample=round_sample))
                    st.success("Round saved.")
                except Exception as exc: st.error(str(exc))
        st.metric("Stored audit events", repo.count("audit_log"))

    with tabs[1]:
        st.header("Fixtures and bookmaker odds")
        with st.form("fixture_form"):
            f_season = st.text_input("Fixture season", "2026/27")
            f_round = st.number_input("Fixture round", min_value=1, value=1)
            fixture_id = st.text_input("Fixture identifier", str(uuid.uuid4())[:8])
            home = st.text_input("Home team")
            away = st.text_input("Away team")
            kickoff_date = st.date_input("Kickoff date")
            kickoff_time = st.time_input("Kickoff time")
            source = st.text_input("Fixture source", "manual")
            is_sample = st.checkbox("Fixture is demonstration/sample data")
            if st.form_submit_button("Add fixture"):
                try:
                    fixture = Fixture(fixture_id=fixture_id, season=f_season, round_number=f_round, home_team=home, away_team=away, kickoff=datetime.combine(kickoff_date, kickoff_time, tzinfo=timezone.utc), data_source=source, collected_at=datetime.now(timezone.utc), is_sample=is_sample)
                    service.add_fixtures([fixture]); st.success("Fixture saved.")
                except Exception as exc: st.error(str(exc))
        fixture_rows = [f.model_dump() for f in service.fixtures()]
        if fixture_rows: st.dataframe(pd.DataFrame(fixture_rows), use_container_width=True)
        st.subheader("Add bookmaker odds")
        fixtures = service.fixtures()
        if fixtures:
            choices = {f"{f.fixture_id}: {f.home_team} v {f.away_team}": f for f in fixtures}
            label = st.selectbox("Fixture", list(choices))
            with st.form("odds_form"):
                bookmaker = st.text_input("Bookmaker")
                home_odds = st.number_input("Home decimal odds", min_value=1.01, value=2.0)
                draw_odds = st.number_input("Draw decimal odds", min_value=1.01, value=3.4)
                away_odds = st.number_input("Away decimal odds", min_value=1.01, value=3.6)
                market_date = st.date_input("Market timestamp date")
                market_clock = st.time_input("Market timestamp time")
                odds_sample = st.checkbox("Odds are demonstration/sample data")
                if st.form_submit_button("Add odds"):
                    try:
                        market_time = datetime.combine(market_date, market_clock, tzinfo=timezone.utc)
                        service.add_odds([OddsQuote(fixture_id=choices[label].fixture_id, bookmaker=bookmaker, home=home_odds, draw=draw_odds, away=away_odds, collected_at=datetime.now(timezone.utc), market_timestamp=market_time, is_sample=odds_sample)])
                        st.success("Odds saved.")
                    except Exception as exc: st.error(str(exc))
        else: st.caption("Add a fixture first.")

    with tabs[2]:
        st.header("Players and paid entries")
        with st.form("player_form"):
            player_id = st.text_input("Player ID")
            player_name = st.text_input("Player name")
            player_sample = st.checkbox("Player is demonstration/sample data")
            if st.form_submit_button("Create player"):
                try: service.add_player(Player(player_id=player_id, name=player_name, is_sample=player_sample)); st.success("Player saved.")
                except Exception as exc: st.error(str(exc))
        with st.form("entry_form"):
            entry_id = st.text_input("Entry ID")
            entry_player = st.text_input("Player ID for this paid entry")
            entry_season = st.text_input("Entry season", "2026/27")
            entry_sample = st.checkbox("Entry is demonstration/sample data")
            if st.form_submit_button("Create paid entry"):
                try: service.add_entry(Entry(entry_id=entry_id, player=entry_player, season=entry_season, is_sample=entry_sample)); st.success("Entry saved.")
                except Exception as exc: st.error(str(exc))
        entries = service.entries()
        if entries:
            matrix = []
            for entry in entries:
                matrix.append({"Entry": entry.entry_id, "Player": entry.player, "Used teams": ", ".join(service.used_teams(entry.entry_id)), "Available teams": ", ".join(service.available_teams(entry.entry_id, int(st.session_state.get("selected_round", 1))))})
            st.dataframe(pd.DataFrame(matrix), use_container_width=True)

    with tabs[3]:
        st.header("Selections and backups")
        selection_round = st.number_input("Selection round", min_value=1, value=1, key="selection_round")
        st.session_state["selected_round"] = int(selection_round)
        entries = service.entries()
        if entries:
            entry = st.selectbox("Entry", [e.entry_id for e in entries])
            available = service.available_teams(entry, int(selection_round))
            st.write("Available teams:", ", ".join(available) if available else "None")
            with st.form("selection_form"):
                team = st.selectbox("Team", available) if available else st.text_input("Team", disabled=True)
                backup = st.checkbox("Record as backup")
                if st.form_submit_button("Record selection"):
                    try: service.record_selection(entry, int(selection_round), team, backup); st.success("Selection recorded.")
                    except Exception as exc: st.error(str(exc))
            fallback = service.fallback_preview(entry, int(selection_round))
            st.info(f"Lowest-ranked unused-team fallback preview: {fallback or 'not available'} (preview only; not applied)")
            selection_rows = repo.list_payloads("selections")
            if selection_rows: st.dataframe(pd.DataFrame(selection_rows), use_container_width=True)
        else: st.caption("Create a player and entry first.")

    with tabs[4]:
        st.header("Results, postponements and cancellations")
        fixtures = service.fixtures()
        if fixtures:
            chosen = st.selectbox("Fixture to update", [f"{f.fixture_id}: {f.home_team} v {f.away_team}" for f in fixtures], key="result_fixture")
            selected_fixture = next(f for f in fixtures if f.fixture_id == chosen.split(":")[0])
            status = st.selectbox("Status", list(FixtureStatus), format_func=lambda x: x.value)
            goals_home = st.number_input("Home goals", min_value=0, value=0)
            goals_away = st.number_input("Away goals", min_value=0, value=0)
            if st.button("Record fixture status"):
                try: service.record_fixture_status(selected_fixture.fixture_id, status, goals_home if status == FixtureStatus.PLAYED else None, goals_away if status == FixtureStatus.PLAYED else None); st.success("Fixture status saved.")
                except Exception as exc: st.error(str(exc))
        survival = service.survival()
        if survival: st.dataframe(pd.DataFrame([{"Entry": k, "Status": v} for k, v in survival.items()]), use_container_width=True)

    with tabs[5]:
        st.header("Cartel analysis")
        analysis_round = st.number_input("Analysis round", min_value=1, value=1, key="analysis_round")
        probabilities = service.team_probabilities(int(analysis_round))
        if probabilities:
            prob_df = pd.DataFrame([p.__dict__ for p in probabilities])
            st.plotly_chart(px.bar(prob_df, x="team", y="shin", color="fixture_id", title="Fair win probability by eligible team"), use_container_width=True)
            comparison = prob_df.melt(id_vars=["team"], value_vars=["proportional", "additive", "power", "shin"], var_name="method", value_name="probability")
            st.plotly_chart(px.bar(comparison, x="team", y="probability", color="method", barmode="group", title="Margin-removal comparison"), use_container_width=True)
            st.plotly_chart(px.bar(prob_df, x="team", y="overround", color="fixture_id", title="Bookmaker overround"), use_container_width=True)
            st.plotly_chart(px.bar(prob_df, x="team", y="disagreement", color="fixture_id", title="Bookmaker disagreement"), use_container_width=True)
            st.plotly_chart(px.bar(pd.DataFrame(list(service.exposure(int(analysis_round))).items(), columns=["team", "entries"]), x="team", y="entries", title="Current cartel exposure by team"), use_container_width=True)
        else: st.caption("Add eligible fixtures and bookmaker odds to see fair-probability charts.")
        selections = repo.list_payloads("selections")
        entries = service.entries()
        if entries:
            matrix = pd.DataFrame({e.entry_id: [team in service.used_teams(e.entry_id) for team in sorted({x["team"] for x in selections})] for e in entries}, index=sorted({x["team"] for x in selections}))
            if not matrix.empty: st.plotly_chart(px.imshow(matrix, text_auto=True, aspect="auto", title="Used-team matrix"), use_container_width=True)
        st.caption("Disagreement is measured from each bookmaker's normalized market probabilities and is available in the service layer. Exact values require at least two bookmaker quotes.")

    with tabs[6]:
        st.header("Historical modelling")
        stored_report = Path("data/real_validation_report.json")
        if stored_report.exists():
            real = json.loads(stored_report.read_text())
            st.success(f"Genuine Football-Data archive available: {len(real.get('data_audit', {}).get('matches_per_season', {}))} seasons. Final holdouts: 2024/25 and 2025/26.")
            st.dataframe(pd.DataFrame(real.get("data_audit", {}).get("matches_per_season", {}).items(), columns=["season", "matches"]), use_container_width=True)
            for season, holdout in real.items():
                if season in ("2024/25", "2025/26"):
                    metrics = pd.DataFrame(holdout["metrics"]).T.reset_index(names="model")
                    st.subheader(f"Real-data holdout metrics · {season}")
                    st.dataframe(metrics, use_container_width=True)
        st.caption("Upload historical CSV data for research/backtesting. Uploaded data is not current Premier League information unless you provide it and verify its source.")
        historical_file = st.file_uploader("Historical match CSV", type=["csv"])
        if historical_file is not None and st.button("Import and fit historical models"):
            from .data import import_historical_csv
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
                handle.write(historical_file.getvalue()); temp_path = handle.name
            report = import_historical_csv(temp_path)
            st.session_state["historical_report"] = report
            st.success(f"Imported {len(report.matches)} matches; {report.errors} rejected rows. Raw data preserved at {report.raw_path}.")
        report = st.session_state.get("historical_report")
        if report and report.matches:
            from .elo import EloModel
            from .modeling import DixonColesModel
            try:
                dc = DixonColesModel().fit(report.matches)
                elo = EloModel().fit(report.matches)
                rating_df = pd.DataFrame({"team": dc.teams, "attack": [dc.attack[t] for t in dc.teams], "defence": [dc.defence[t] for t in dc.teams], "elo": [elo.ratings.get(t, elo.initial) for t in dc.teams]})
                st.plotly_chart(px.bar(rating_df.melt(id_vars="team", value_vars=["attack", "defence"]), x="team", y="value", color="variable", title="Dixon–Coles attack and defence ratings"), use_container_width=True)
                st.dataframe(rating_df, use_container_width=True)
                if len(dc.teams) >= 2:
                    prediction = dc.predict(dc.teams[0], dc.teams[1])
                    score_df = pd.DataFrame(prediction.scoreline, index=range(dc.max_goals + 1), columns=range(dc.max_goals + 1))
                    st.plotly_chart(px.imshow(score_df, labels={"x": "Away goals", "y": "Home goals", "color": "Probability"}, title="Expected-goals scoreline probability matrix"), use_container_width=True)
                    st.bar_chart(pd.DataFrame({"Dixon–Coles": prediction.outcome, "Elo": elo.probabilities(dc.teams[0], dc.teams[1])}, index=["Home", "Draw", "Away"]))
                if len(report.matches) > 20:
                    from .backtest import expanding_backtest, market_predictor
                    backtest = expanding_backtest(report.matches, market_predictor, min_train=min(20, len(report.matches) - 1))
                    st.subheader("Chronological market benchmark backtest")
                    st.dataframe(pd.DataFrame([backtest.metrics]), use_container_width=True)
                    band = pd.DataFrame({"probability": backtest.predictions.max(axis=1), "correct": (backtest.predictions.argmax(axis=1) == backtest.outcomes).astype(int)})
                    band["band"] = pd.cut(band["probability"], bins=[0, .4, .6, .8, 1], include_lowest=True)
                    reliability = band.groupby("band", observed=True).agg(predicted=("probability", "mean"), observed=("correct", "mean")).reset_index()
                    st.plotly_chart(px.line(reliability, x="predicted", y="observed", markers=True, title="Reliability diagram (market benchmark)"), use_container_width=True)
                st.caption(f"Fitted on {len(report.matches)} dated matches; decay={dc.decay_rate}; synthetic/demo status is controlled by the imported records.")
            except Exception as exc: st.error(f"Model fitting failed: {exc}")

    with tabs[7]:
        st.header("Future value and cartel optimiser")
        st.caption("Optimisation uses current proportional market probabilities; future forecasts must be supplied by a dated, leakage-safe model run. No bets are placed.")
        opt_round = int(st.number_input("Optimiser round", min_value=1, value=1, key="opt_round"))
        entries = service.entries(); probabilities = service.team_probabilities(opt_round)
        if entries and probabilities:
            col1, col2 = st.columns(2)
            with col1: expected_weight = st.number_input("Expected survivors weight", min_value=0.0, value=1.0, key="ow1")
            with col2: at_least_weight = st.number_input("At-least-one weight", min_value=0.0, value=1.0, key="ow2")
            cap = st.number_input("Optional exposure cap (0 = none)", min_value=0, value=0, key="ocap")
            candidate_map = {e.entry_id: service.available_teams(e.entry_id, opt_round) for e in entries}
            forecast = {p.team: p.proportional for p in probabilities}
            available = {opt_round: sorted(set(forecast))}
            dp = DynamicProgram({opt_round: forecast}, available)
            st.dataframe(pd.DataFrame([v.__dict__ for v in dp.solve()]), use_container_width=True)
            quotes = service.odds(); fixture_map = {f.fixture_id: f for f in eligible_fixtures(service.fixtures(), opt_round)}
            scenarios = []
            rng = np.random.default_rng(7)
            for _ in range(2000):
                state = {}
                for fixture_id, fixture in fixture_map.items():
                    market = [q for q in quotes if q.fixture_id == fixture_id]
                    if not market: continue
                    q = market[-1]; outcome = rng.choice(3, p=proportional([q.home, q.draw, q.away]))
                    if outcome == 0: state[fixture.home_team] = True
                    elif outcome == 2: state[fixture.away_team] = True
                scenarios.append(state)
            try:
                from .optimizer import PortfolioOptimizer, PortfolioWeights
                allocation, score = PortfolioOptimizer(candidate_map, scenarios, PortfolioWeights(expected_survivors=expected_weight, at_least_one=at_least_weight), int(cap) or None).optimize()
                st.subheader("Selected allocation")
                st.dataframe(pd.DataFrame(list(allocation.items()), columns=["Entry", "Team"]), use_container_width=True)
                st.json(score)
                st.caption("Scenario estimates use 2,000 reproducible current-round outcome simulations; they are not future-season forecasts.")
            except ValueError as exc: st.error(str(exc))
        else: st.info("Create entries, eligible fixtures, and odds to activate optimisation.")

if __name__ == "__main__":
    run()

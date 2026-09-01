"""Simple weekly Streamlit journey; business operations stay in LMSWorkflow."""
from datetime import datetime, timezone
from pathlib import Path
import uuid


def _error(st, exc):
    text = str(exc).lower()
    st.error("We couldn't retrieve or save that information. Please try again or use the manual option." if any(x in text for x in ("api", "provider", "timeout")) else "We couldn't complete that step. Please check the information and try again.")


def _state(st, service):
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("advanced", False)
    st.session_state.setdefault("analysis", None)
    existing_locked = st.session_state.get("locked")
    st.session_state.setdefault("snapshot", None)
    st.session_state.setdefault("round_number", 1)
    entries = service.entries()
    st.session_state.setdefault("setup_complete", bool(entries))
    if entries: st.session_state.setdefault("season", entries[0].season)
    if entries and existing_locked is None:
        from .weekly import WeeklyStore
        st.session_state.locked = any(x.locked and x.season == st.session_state.get("season") and x.round_number == st.session_state.get("round_number", 1) for x in WeeklyStore().versions())
    else:
        st.session_state.setdefault("locked", False)


def _current(st, service):
    season = st.session_state.get("season")
    number = int(st.session_state.get("round_number", 1))
    fixtures = [f for f in service.fixtures() if f.season == season and f.round_number == number]
    entries = [e for e in service.entries() if e.season == season and e.active]
    return season, number, fixtures, entries


def _setup(st, service):
    st.title("Welcome to Last Man Standing")
    st.write("Let's get your competition ready. This only takes a minute.")
    with st.form("setup_form"):
        competition = st.text_input("Competition name", "Premier League Last Man Standing")
        season = st.text_input("Season", "2026/27")
        count = int(st.number_input("Number of players", min_value=1, max_value=20, value=1))
        names = [st.text_input(f"Player {i + 1}", f"Player {i + 1}") for i in range(count)]
        entries = [int(st.number_input(f"Entries for {name or f'Player {i + 1}'}", min_value=1, max_value=10, value=1, key=f"entry_count_{i}")) for i, name in enumerate(names)]
        if st.form_submit_button("Finish setup"):
            try:
                from .models import Entry, Player, Season
                service.create_season(Season(season=season, name=competition))
                from .models import Round
                service.create_round(Round(season=season, round_number=1, selection_deadline=datetime.now(timezone.utc)))
                for name, amount in zip(names, entries):
                    pid = str(uuid.uuid4()); service.add_player(Player(player_id=pid, name=name.strip() or "Player"))
                    for _ in range(amount): service.add_entry(Entry(entry_id=str(uuid.uuid4()), player=pid, season=season))
                st.session_state.update(season=season, setup_complete=True, page="Home")
                st.success("Setup complete. Now get this week's matches."); st.rerun()
            except Exception as exc: _error(st, exc)


def _save_draft(service, analysis, season, number, st):
    from .weekly import RecommendationSnapshot, WeeklyStore
    now = datetime.now(timezone.utc); version = now.strftime("%Y%m%dT%H%M%S%fZ")
    snap = RecommendationSnapshot(version=version, created_at=now, season=season, round_number=number, odds_snapshot_version=f"odds-{len(service.odds())}", forecast_snapshot_version="not-required", active_entries=list(analysis["allocation"]), used_teams={e: service.used_teams(e) for e in analysis["allocation"]}, objective_weights=analysis["objective_weights"], exposure_limits={}, simulation_settings={}, seed=7, optimiser_version="weekly-service", allocation=analysis["allocation"], backups=analysis["backups"], odds_snapshot={q.fixture_id: q.model_dump() for q in service.odds()}, probabilities={r["team"]: r["proportional"] for r in analysis["probabilities"]}, exact_risk={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in analysis["risk"].items()}, risk_estimates={"expected_survivors": float(analysis["risk"]["expected_survivors"])})
    WeeklyStore().save(snap); st.session_state.snapshot = snap


def _home(st, service):
    season, number, fixtures, entries = _current(st, service)
    if not season or not entries: return _setup(st, service)
    st.title("Your LMS week"); st.subheader(f"Round {number}")
    rounds = [r for r in service.repo.list_payloads("rounds") if r.get("season") == season and r.get("round_number") == number]
    deadline = "Not set"
    if rounds: deadline = datetime.fromisoformat(rounds[-1]["selection_deadline"]).astimezone().strftime("%a %d %b, %H:%M")
    c1, c2, c3 = st.columns(3); c1.metric("Deadline", deadline); c2.metric("Active entries", len(entries)); c3.metric("Odds freshness", "Ready" if fixtures else "Not loaded")
    if st.session_state.get("locked"):
        message, action, page = ("Selections are locked. Share them with your group.", "Copy WhatsApp message", "Share") if not any(f.status.value == "played" for f in fixtures) else ("Matches have finished. Record the results to update entries.", "Enter results", "Results")
    elif st.session_state.get("analysis"):
        message, action, page = "Picks are ready for review.", "Review and confirm", "Confirm"
    elif fixtures: message, action, page = "Fixtures are ready.", "Show our best picks", "Choose"
    else: message, action, page = "This round has no matches yet.", "Get this week's matches", "Get"
    st.info(message)
    if st.button(action, type="primary", key="home_primary"): st.session_state.page = page; st.rerun()


def _get(st, service):
    from .providers import OddsApiProvider, ProviderError
    st.title("Get matches"); st.write("We'll find this week's Premier League matches and the latest available odds.")
    provider = OddsApiProvider()
    if not provider.api_key:
        st.warning("Automatic matches are not configured yet. You can use cached matches or enable Advanced mode for manual entry.")
        if st.button("Use cached odds", type="primary"): st.info("No cached provider snapshot is available yet.")
        if st.button("Enter matches manually"): st.session_state.update(advanced=True, page="Settings"); st.rerun()
    if st.session_state.get("locked"):
        st.info("This round is locked. Start the next round before refreshing matches.")
    elif st.button("Get this week's matches", type="primary"):
        try:
            season, number, _, _ = _current(st, service); result = service.refresh_provider_odds(provider, season, number); st.session_state.provider_refresh = result
            gate = service.validate_round(season, number)
            if gate["valid"]:
                st.session_state.analysis = service.analyse_round(season, number); _save_draft(service, st.session_state.analysis, season, number, st)
            st.success(f"{result['events']} matches found")
        except (ProviderError, Exception) as exc: _error(st, exc)
    season, number, fixtures, _ = _current(st, service)
    if fixtures:
        st.write("Matchweek: " + " – ".join(sorted({f.kickoff.astimezone().strftime("%d %b") for f in fixtures})))
        st.write("Odds freshness: latest successful refresh")
        st.write("Bookmaker coverage: imported prices combined into fair probabilities")
        for fixture in sorted(fixtures, key=lambda f: f.kickoff): st.container(border=True).write(f"⚽  {fixture.home_team}  v  {fixture.away_team}")
        if st.button("Show our best picks", type="primary"):
            try:
                if not st.session_state.get("analysis"):
                    gate = service.validate_round(season, number)
                    if gate["valid"]:
                        st.session_state.analysis = service.analyse_round(season, number); _save_draft(service, st.session_state.analysis, season, number, st)
                st.session_state.page = "Choose"; st.rerun()
            except Exception as exc: _error(st, exc)


def _choose(st, service):
    st.title(f"Our recommended selections for Round {st.session_state.round_number}")
    analysis = st.session_state.get("analysis")
    if not analysis: st.warning("Get this week's matches first."); return
    players = {x["player_id"]: x["name"] for x in service.repo.list_payloads("players")}; probs = {r["team"]: r["proportional"] for r in analysis["probabilities"]}
    for index, entry in enumerate(service.entries(), 1):
        if entry.season != st.session_state.season or not entry.active: continue
        team, backup = analysis["allocation"].get(entry.entry_id), analysis["backups"].get(entry.entry_id)
        with st.container(border=True):
            st.subheader(f"{players.get(entry.player, 'Player')} · Entry {index}")
            st.write(f"**{team}** — {probs.get(team, 0):.0%} chance of winning"); st.write(f"Backup: {backup or 'None'}")
            st.caption("This is the strongest available market favourite and has not previously been used by this entry.")
    risk = analysis["risk"]; st.subheader("This round's risk")
    st.write(f"Chance at least one entry survives: **{risk['probability_at_least_one']:.0%}**"); st.write(f"Chance all entries lose: **{risk['wipeout_probability']:.0%}**")
    st.caption("Worst-case round risk is the number of entries that could be lost in this round. This is exact for the supplied probabilities.")
    if st.button("Review these picks", type="primary"): st.session_state.page = "Confirm"; st.rerun()
    with st.expander("Explore other strategies"): st.caption("Experimental alternatives are available in Advanced mode. The recommended picks use the validated default.")


def _confirm(st, service):
    st.title("Confirm and share"); analysis = st.session_state.get("analysis")
    if not analysis: st.warning("There are no picks to review yet."); return
    players = {x["player_id"]: x["name"] for x in service.repo.list_payloads("players")}
    for index, entry in enumerate(service.entries(), 1):
        if entry.season == st.session_state.season and entry.active:
            st.write(f"✅ **{players.get(entry.player, 'Player')} · Entry {index}** — {analysis['allocation'].get(entry.entry_id)} (backup: {analysis['backups'].get(entry.entry_id) or 'none'})")
            st.caption("Previously used: " + (", ".join(service.used_teams(entry.entry_id)) or "None"))
    st.checkbox("I have checked every pick and backup", key="confirmed_review")
    if st.button("Lock our selections", type="primary", disabled=not st.session_state.get("confirmed_review")):
        try:
            from .weekly import RecommendationSnapshot, WeeklyStore
            _save_draft(service, analysis, st.session_state.season, st.session_state.round_number, st); path = WeeklyStore().lock(st.session_state.snapshot.version); st.session_state.snapshot = RecommendationSnapshot.model_validate_json(path.read_text())
            service.save_recommendation_selections(analysis["allocation"], analysis["backups"], st.session_state.round_number); st.session_state.locked = True; st.success("Selections locked. They cannot be changed silently."); st.rerun()
        except Exception as exc: _error(st, exc)
    if st.session_state.get("locked"):
        from .weekly import WeeklyStore
        st.success(f"Selections locked for Round {st.session_state.round_number}")
        st.code(WeeklyStore.whatsapp_message(st.session_state.snapshot), language=None)
        if st.button("Copy WhatsApp message"): st.info("The message above is ready to copy.")


def _results(st, service):
    from .providers import OddsApiProvider, ProviderError
    st.title("Enter results")
    if st.button("Get results automatically", type="primary"):
        try: st.session_state.result_proposals = service.propose_provider_results(OddsApiProvider()); st.success("Results are ready to check. Nothing has changed yet.")
        except (ProviderError, Exception) as exc: _error(st, exc)
    proposals = st.session_state.get("result_proposals", {})
    for proposal in proposals.get("proposals", []):
        fixture = next((f for f in service.fixtures() if f.fixture_id == proposal["fixture_id"]), None)
        if fixture: st.write(f"{fixture.home_team}  **{proposal['home_goals']} – {proposal['away_goals']}**  {fixture.away_team}  · Confirmed")
    if proposals.get("proposals") and st.button("Confirm results and update entries", type="primary"):
        try: st.session_state.survival = service.confirm_provider_results(proposals["proposals"]); st.success("Results confirmed and entries updated.")
        except Exception as exc: _error(st, exc)
    if st.session_state.get("survival"):
        st.subheader("Round complete")
        for entry, status in st.session_state.survival.items(): st.success(f"{entry}: survived") if status == "surviving" else st.error(f"{entry}: eliminated")
        st.write("Teams used this round are now recorded for each entry.")
        if st.button("Continue to next round", type="primary"):
            st.session_state.update(round_number=st.session_state.round_number + 1, locked=False, analysis=None, snapshot=None, survival=None, result_proposals=None, page="Home"); st.rerun()


def _settings(st, service):
    st.title("Settings")
    st.caption("Your local competition settings")
    with st.form("round_settings"):
        from .models import Round
        number = int(st.number_input("Round number", min_value=1, value=int(st.session_state.round_number)))
        deadline = st.datetime_input("Selection deadline", value=datetime.now(timezone.utc))
        if st.form_submit_button("Save round settings"):
            try:
                service.create_round(Round(season=st.session_state.season, round_number=number, selection_deadline=deadline))
                st.session_state.round_number = number; st.success("Round settings saved.")
            except Exception as exc: _error(st, exc)
    st.toggle("Advanced mode", key="advanced")
    st.caption("Advanced mode is for troubleshooting and manual data entry. You normally do not need it.")
    if st.session_state.advanced:
        st.warning("Advanced mode exposes technical data and manual controls.")
        st.subheader("Manual fixture management")
        with st.form("manual_fixture"):
            from .models import Fixture
            home, away = st.text_input("Home team"), st.text_input("Away team"); kickoff = st.datetime_input("Kickoff", value=datetime.now(timezone.utc))
            if st.form_submit_button("Add manual fixture"):
                service.add_fixtures([Fixture(fixture_id=str(uuid.uuid4()), season=st.session_state.season, round_number=st.session_state.round_number, home_team=home, away_team=away, kickoff=kickoff, data_source="manual", collected_at=datetime.now(timezone.utc))]); st.success("Match added.")
        if st.button("Load deterministic historical dry-run"):
            from .models import Fixture, OddsQuote, Player, Entry, Season, Round
            day = datetime(2020, 8, 1, tzinfo=timezone.utc); season = "2020/21"
            service.create_season(Season(season=season, name="Historical dry run", is_sample=True))
            service.create_round(Round(season=season, round_number=1, selection_deadline=day, is_sample=True)); service.create_round(Round(season=season, round_number=2, selection_deadline=day.replace(day=15), is_sample=True))
            fixtures = [Fixture(fixture_id=f"dry-{round_no}-{i}", season=season, round_number=round_no, home_team=f"Home {round_no}-{i}", away_team=f"Away {round_no}-{i}", kickoff=day.replace(day=3 + (round_no - 1) * 10 + i), collected_at=day, data_source="deterministic-dry-run", is_sample=True) for round_no in (1, 2) for i in range(6)]
            service.add_fixtures(fixtures); service.add_odds([OddsQuote(fixture_id=f.fixture_id, bookmaker="dry-run", home=1.5, draw=4.0, away=6.0, collected_at=day, market_timestamp=day, data_source="deterministic-dry-run", is_sample=True) for f in fixtures])
            pid = "dry-player"; service.add_player(Player(player_id=pid, name="Dry Run Player", is_sample=True)); service.add_entry(Entry(entry_id="dry-entry", player=pid, season=season, is_sample=True)); st.session_state.update(season=season, round_number=1, setup_complete=True, page="Get"); st.success("Historical dry-run data loaded. It is demonstration data and does not predict future performance."); st.rerun()
        st.write("CSV import, provider metadata, raw tables, forecasts, simulation settings, diagnostics and audit logs are available here.")


def run() -> None:
    try: import streamlit as st
    except ImportError: print("Install dependencies with: pip install -r requirements.txt"); return
    from .storage import Repository
    from .workflow import LMSWorkflow
    st.set_page_config(page_title="LMS Weekly Manager", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")
    st.markdown("<style>.block-container{max-width:980px;padding-top:2rem;padding-left:1.2rem;padding-right:1.2rem}@media(max-width:700px){.block-container{padding:1rem .7rem}h1{font-size:1.8rem}}</style>", unsafe_allow_html=True)
    service = LMSWorkflow(Repository(Path("data/lms.sqlite3"))); _state(st, service)
    if not st.session_state.setup_complete: _setup(st, service); return
    with st.sidebar:
        st.title("LMS")
        for page in ("Home", "Entries", "History", "Settings"):
            if st.button(page, use_container_width=True, key=f"nav_{page}"): st.session_state.page = page; st.rerun()
    page = st.session_state.get("page", "Home")
    if page == "Home": _home(st, service)
    elif page == "Get": _get(st, service)
    elif page == "Choose": _choose(st, service)
    elif page in ("Confirm", "Share"): _confirm(st, service)
    elif page == "Results": _results(st, service)
    elif page == "Settings": _settings(st, service)
    elif page == "Entries":
        st.title("Entries")
        players = {x["player_id"]: x["name"] for x in service.repo.list_payloads("players")}
        for index, e in enumerate(service.entries(), 1): st.write(f"{players.get(e.player, 'Player')} · Entry {index} · {'active' if e.active else 'eliminated'}")
        st.subheader("Add a player")
        with st.form("add_player"):
            name = st.text_input("Player name")
            if st.form_submit_button("Add player"):
                from .models import Player
                try: service.add_player(Player(player_id=str(uuid.uuid4()), name=name.strip())); st.success("Player added."); st.rerun()
                except Exception as exc: _error(st, exc)
        if players:
            st.subheader("Add an entry")
            with st.form("add_entry"):
                player_id = st.selectbox("Player", list(players), format_func=lambda value: players[value])
                if st.form_submit_button("Add entry"):
                    from .models import Entry
                    try: service.add_entry(Entry(entry_id=str(uuid.uuid4()), player=player_id, season=st.session_state.season)); st.success("Entry added."); st.rerun()
                    except Exception as exc: _error(st, exc)
    elif page == "History": st.title("History"); st.write("Your saved rounds and results will appear here.")


if __name__ == "__main__": run()

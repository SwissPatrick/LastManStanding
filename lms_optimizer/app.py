"""Simple weekly Streamlit journey; business operations stay in LMSWorkflow."""
from datetime import datetime, timezone
from pathlib import Path
import os
import uuid

APP_ROOT = Path(__file__).resolve().parents[1]

def _database_path() -> Path:
    configured = os.environ.get("LMS_DATABASE_PATH")
    return Path(configured).expanduser().resolve() if configured else APP_ROOT / "data" / "lms.sqlite3"

def _recommendation_dir() -> Path:
    configured = os.environ.get("LMS_RECOMMENDATIONS_PATH")
    return Path(configured).expanduser().resolve() if configured else APP_ROOT / "data" / "recommendations"


def _error(st, exc):
    text = str(exc).lower()
    st.error("We couldn't retrieve or save that information. Please try again or use the manual option." if any(x in text for x in ("api", "provider", "timeout")) else "We couldn't complete that step. Please check the information and try again.")


def _state(st, service):
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("advanced", False)
    st.session_state.setdefault("analysis", None)
    existing_locked = st.session_state.get("locked")
    st.session_state.setdefault("snapshot", None)
    entries = service.entries()
    st.session_state.setdefault("setup_complete", bool(entries))
    persisted_season, persisted_round = service.current_context()
    if persisted_season: st.session_state.setdefault("season", persisted_season)
    st.session_state.setdefault("round_number", persisted_round)
    if st.session_state.get("round_number") == 1 and persisted_round != 1: st.session_state.round_number = persisted_round
    if st.session_state.get("analysis") is None and st.session_state.get("season"):
        try:
            gate = service.validate_round(st.session_state.season, st.session_state.round_number)
            if gate["valid"]: st.session_state.analysis = service.analyse_round(st.session_state.season, st.session_state.round_number)
        except Exception: pass
    if entries and existing_locked is None:
        from .weekly import WeeklyStore
        st.session_state.locked = any(x.locked and x.season == st.session_state.get("season") and x.round_number == st.session_state.get("round_number", 1) for x in WeeklyStore(_recommendation_dir()).versions())
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
        st.subheader("Our cartel")
        st.caption("There are five permanent family members. Enter their actual names and each person's number of entries.")
        names = [st.text_input(label, key=f"member_name_{i}") for i, label in enumerate(("Your name", "Your brother's name", "Your dad's name", "Your uncle's name", "Your cousin's name"))]
        entries = [int(st.number_input(f"Entries for {name or label}", min_value=1, max_value=10, value=1, key=f"entry_count_{i}")) for i, (name, label) in enumerate(zip(names, ("You", "Your brother", "Your dad", "Your uncle", "Your cousin")))]
        if st.form_submit_button("Finish setup"):
            try:
                from .models import Entry, FamilyMember, Player, Season
                service.create_season(Season(season=season, name=competition))
                from .models import Round
                service.create_round(Round(season=season, round_number=1, selection_deadline=datetime.now(timezone.utc)))
                for position, (name, amount) in enumerate(zip(names, entries), 1):
                    member_id = str(uuid.uuid4()); friendly_name = name.strip() or ("You" if position == 1 else ["", "Brother", "Dad", "Uncle", "Cousin"][position - 1])
                    service.add_family_member(FamilyMember(member_id=member_id, name=friendly_name, position=position))
                    for _ in range(amount): service.add_entry(Entry(entry_id=str(uuid.uuid4()), member_id=member_id, season=season))
                st.session_state.update(season=season, setup_complete=True, page="Home")
                st.success("Setup complete. Now get this week's matches."); st.rerun()
            except Exception as exc: _error(st, exc)


def _save_draft(service, analysis, season, number, st):
    from .weekly import RecommendationSnapshot, WeeklyStore
    now = datetime.now(timezone.utc); version = now.strftime("%Y%m%dT%H%M%S%fZ")
    snap = RecommendationSnapshot(version=version, created_at=now, season=season, round_number=number, odds_snapshot_version=f"odds-{len(service.odds())}", forecast_snapshot_version="not-required", active_entries=list(analysis["allocation"]), used_teams={e: service.used_teams(e) for e in analysis["allocation"]}, objective_weights=analysis["objective_weights"], exposure_limits={}, simulation_settings={}, seed=7, optimiser_version="weekly-service", allocation=analysis["allocation"], backups=analysis["backups"], odds_snapshot={q.fixture_id: q.model_dump() for q in service.odds()}, probabilities={r["team"]: r["proportional"] for r in analysis["probabilities"]}, exact_risk={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in analysis["risk"].items()}, risk_estimates={"expected_survivors": float(analysis["risk"]["expected_survivors"])})
    _recommendation_dir().mkdir(parents=True, exist_ok=True); WeeklyStore(_recommendation_dir()).save(snap); st.session_state.snapshot = snap


def _home(st, service):
    season, number, fixtures, entries = _current(st, service)
    if not season or not entries: return _setup(st, service)
    st.title("Your LMS week"); st.subheader(f"Round {number}")
    members = service.family_members(); total = len([e for e in service.entries() if e.season == season]); alive = len([e for e in service.entries() if e.season == season and e.active])
    st.markdown(f"### OUR CARTEL\n{len(members)} family members • {total} total entries • {alive} still alive")
    field = service.wider_field(season, number)
    st.markdown(f"### WIDER FIELD\n{field.surviving_entries} entries remain" if field else "### WIDER FIELD\nNo field size recorded yet")
    rounds = [r for r in service.repo.list_payloads("rounds") if r.get("season") == season and r.get("round_number") == number]
    deadline = "Not set"
    if rounds: deadline = datetime.fromisoformat(rounds[-1]["selection_deadline"]).astimezone().strftime("%a %d %b, %H:%M")
    c1, c2, c3 = st.columns(3); c1.metric("Deadline", deadline); c2.metric("Active entries", len(entries)); c3.metric("Odds freshness", "Ready" if fixtures else "Not loaded")
    st.write(f"### THIS WEEK\n{len(entries)} selections required • Deadline {deadline}")
    with st.form("update_field_size"):
        field_count = st.number_input("Current surviving outside entries", min_value=0, value=field.surviving_entries if field else 0, step=1)
        if st.form_submit_button("Update field size"):
            from .models import WiderFieldSnapshot
            try:
                starting = field.starting_entries if field else int(field_count)
                service.save_wider_field(WiderFieldSnapshot(season=season, round_number=number, starting_entries=starting, surviving_entries=int(field_count), recorded_at=datetime.now(timezone.utc)))
                st.success("Wider field size updated."); st.rerun()
            except Exception as exc: _error(st, exc)
    if st.session_state.get("locked"):
        message, action, page = ("Selections are locked. Share them with your group.", "Copy WhatsApp message", "Share") if not any(f.status.value == "played" for f in fixtures) else ("Matches have finished. Record the results to update entries.", "Enter results", "Results")
    elif st.session_state.get("analysis"):
        message, action, page = "Picks are ready for review.", "Review and confirm", "Confirm"
    elif fixtures and service.odds(): message, action, page = "Fixtures are ready.", "Show our best picks", "Choose"
    elif fixtures: message, action, page = "The matches are ready, but we still need the latest odds.", "Get latest odds", "Get"
    else: message, action, page = "This round has no matches yet.", "Get this week's matches", "Get"
    st.info(message)
    if st.button(action, type="primary", key="home_primary"): st.session_state.page = page; st.rerun()


def _get(st, service):
    from .providers import OddsApiProvider, ProviderError
    st.title("Get matches"); st.write("We'll find this week's Premier League matches and the latest available odds.")
    provider = OddsApiProvider()
    if not provider.api_key:
        st.error("The automatic football-data connection has not been configured.")
        st.code("Add ODDS_API_KEY=your-local-key to .env, then restart Streamlit.")
        if st.button("Try again", type="primary"): st.rerun()
        if st.button("Enter matches manually"): st.session_state.update(advanced=True, page="Settings"); st.rerun()
        return
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
    else:
        st.info("This week's matches haven't been loaded yet.")


def _choose(st, service):
    analysis = st.session_state.get("analysis")
    if not analysis:
        season, number, fixtures, entries = _current(st, service)
        st.title(f"Let's prepare Round {number}")
        gate = service.validate_round(season, number) if season else {"eligible_fixture_count": 0}
        if fixtures and gate.get("eligible_fixture_count", 0) < 6:
            with st.container(border=True):
                st.subheader("No selection is required this week")
                st.write(f"Only {gate.get('eligible_fixture_count', 0)} eligible matches are scheduled, so no selection is required under your LMS rules.")
                if st.button("Return home", type="primary"): st.session_state.page = "Home"; st.rerun()
            return
        with st.container(border=True):
            st.subheader("This week's selections are not ready")
            st.write("We need the current fixtures and odds before we can generate recommendations.")
            st.write("✅ Family entries checked" if entries else "⬜ Add your family's entries")
            st.write("✅ Fixtures loaded" if fixtures else "⬜ Fixtures still need to be loaded")
            st.write("✅ Odds available" if service.odds() else "⬜ Latest odds still need to be retrieved")
            if not entries:
                if st.button("Set up our entries", type="primary"): st.session_state.page = "Home"; st.rerun()
            elif not fixtures or not service.odds():
                action = "Get latest odds" if fixtures else "Get this week's matches"
                if st.button(action, type="primary"): st.session_state.page = "Get"; st.rerun()
            if st.button("Return home"): st.session_state.page = "Home"; st.rerun()
        return
    st.title(f"Our recommended selections for Round {st.session_state.round_number}")
    probs = {r["team"]: r["proportional"] for r in analysis["probabilities"]}
    for member, recommendations in analysis.get("recommendations_by_member", {}).items():
        st.subheader(member)
        for item in recommendations:
            team, backup = item["team"], item["backup"]
            with st.container(border=True):
                st.write(f"**{member} — {item['label']}**: {team}")
                st.caption(f"Backup: {backup or 'None'} · {probs.get(team, 0):.0%} chance of winning")
    risk = analysis["risk"]; st.subheader("This round's risk")
    cartel = analysis.get("cartel_risk", {})
    st.write(f"Chance at least one cartel entry survives: **{cartel.get('probability_at_least_one', risk['probability_at_least_one']):.0%}**")
    st.write(f"Chance every family member retains at least one entry: **{cartel.get('probability_every_member', 0):.0%}**")
    if analysis.get("competition_winning_probability") is not None:
        st.write(f"Estimated competition-winning probability: **{analysis['competition_winning_probability']:.2%}**")
    field = service.wider_field(st.session_state.season, st.session_state.round_number)
    st.caption("Competition-winning probability is estimated because outside selections are unavailable." if not field or not field.known_selections else "Competition-winning probability uses the recorded wider-field selections and remains an estimate.")
    st.caption("Worst-case round risk is the number of entries that could be lost in this round. This is exact for the supplied probabilities.")
    if st.button("Review these picks", type="primary"): st.session_state.page = "Confirm"; st.rerun()
    with st.expander("Explore other strategies"): st.caption("Experimental alternatives are available in Advanced mode. The recommended picks use the validated default.")


def _confirm(st, service):
    st.title("Confirm and share"); analysis = st.session_state.get("analysis")
    if not analysis: st.warning("There are no picks to review yet."); return
    members = {x.member_id: x.name for x in service.family_members()}
    for index, entry in enumerate(service.entries(), 1):
        if entry.season == st.session_state.season and entry.active:
            owner = members.get(service.entry_owner(entry), "Family member")
            st.write(f"✅ **{owner} — Entry {index}** — {analysis['allocation'].get(entry.entry_id)} (backup: {analysis['backups'].get(entry.entry_id) or 'none'})")
            st.caption("Previously used: " + (", ".join(service.used_teams(entry.entry_id)) or "None"))
    st.checkbox("I have checked every pick and backup", key="confirmed_review")
    if st.button("Lock our selections", type="primary", disabled=not st.session_state.get("confirmed_review")):
        try:
            from .weekly import RecommendationSnapshot, WeeklyStore
            _save_draft(service, analysis, st.session_state.season, st.session_state.round_number, st); path = WeeklyStore(_recommendation_dir()).lock(st.session_state.snapshot.version); st.session_state.snapshot = RecommendationSnapshot.model_validate_json(path.read_text())
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
    service = LMSWorkflow(Repository(_database_path())); _state(st, service)
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
        members = {x.member_id: x.name for x in service.family_members()}
        for index, e in enumerate(service.entries(), 1): st.write(f"{members.get(service.entry_owner(e), 'Family member')} — Entry {index} · {'active' if e.active else 'eliminated'}")
        st.subheader("Add a player")
        with st.form("add_player"):
            name = st.text_input("Player name")
            if st.form_submit_button("Add player"):
                from .models import Player
                try: service.add_player(Player(player_id=str(uuid.uuid4()), name=name.strip())); st.success("Player added."); st.rerun()
                except Exception as exc: _error(st, exc)
        if members:
            st.subheader("Add an entry")
            with st.form("add_entry"):
                player_id = st.selectbox("Family member", list(members), format_func=lambda value: members[value])
                if st.form_submit_button("Add entry"):
                    from .models import Entry
                    try: service.add_entry(Entry(entry_id=str(uuid.uuid4()), player=player_id, season=st.session_state.season)); st.success("Entry added."); st.rerun()
                    except Exception as exc: _error(st, exc)
    elif page == "History": st.title("History"); st.write("Your saved rounds and results will appear here.")


if __name__ == "__main__": run()

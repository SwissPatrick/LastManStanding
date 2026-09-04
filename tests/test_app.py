from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / "main.py"


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_first_time_user_sees_simple_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMS_DATABASE_PATH", str(tmp_path / "data" / "lms.sqlite3"))
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert app.title[0].value == "Welcome to Last Man Standing"
    assert button(app, "Finish setup")
    assert not any("fixture_id" in str(item.value).lower() for item in app.markdown)


def test_setup_leads_to_single_home_action(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMS_DATABASE_PATH", str(tmp_path / "data" / "lms.sqlite3"))
    app = AppTest.from_file(APP).run()
    button(app, "Finish setup").click().run()
    assert not app.exception
    assert app.title[0].value == "Your LMS week"
    assert button(app, "Get this week's matches")


def test_advanced_mode_keeps_dry_run_and_manual_tools_available(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMS_DATABASE_PATH", str(tmp_path / "data" / "lms.sqlite3"))
    app = AppTest.from_file(APP).run()
    button(app, "Finish setup").click().run()
    button(app, "Settings").click().run()
    app.toggle[0].set_value(True).run()
    assert button(app, "Load deterministic historical dry-run")
    assert button(app, "Add manual fixture")


def test_normal_mode_does_not_show_technical_controls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMS_DATABASE_PATH", str(tmp_path / "data" / "lms.sqlite3"))
    app = AppTest.from_file(APP).run()
    button(app, "Finish setup").click().run()
    assert not app.exception
    widgets = list(app.text_input) + list(app.number_input)
    assert not any(item.label in {"CSV import", "Process workers", "Performance profile", "Fixture ID"} for item in widgets)


def test_import_competition_csv_is_available_from_home_and_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMS_DATABASE_PATH", str(tmp_path / "data" / "lms.sqlite3"))
    app = AppTest.from_file(APP).run()
    button(app, "Finish setup").click().run()
    button(app, "Import competition CSV").click().run()
    assert not app.exception
    assert app.title[0].value == "Import competition CSV"
    button(app, "Entries").click().run()
    assert button(app, "Import competition CSV")

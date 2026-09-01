from streamlit.testing.v1 import AppTest
from pathlib import Path

APP = Path(__file__).parents[1] / "main.py"

def widget_by_label(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_dashboard_renders_manual_workflow_controls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert app.title[0].value == "Premier League Last Man Standing"
    assert widget_by_label(app.button, "Create / update season")
    assert widget_by_label(app.button, "Add fixture")
    assert any("Selections and backups" in heading.value for heading in app.header)


def test_dashboard_can_submit_season_form(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP).run()
    widget_by_label(app.text_input, "Season").set_value("2026/27")
    widget_by_label(app.button, "Create / update season").click().run()
    assert any("Season saved" in alert.value for alert in app.success)


def test_dashboard_loads_deterministic_dry_run_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP).run()
    widget_by_label(app.button, "Load deterministic historical dry-run").click().run()
    assert not app.exception
    assert any("Historical dry-run data loaded" in alert.value for alert in app.success)

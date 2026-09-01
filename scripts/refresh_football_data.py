"""Download Football-Data Premier League E0.csv archives and write a manifest."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from lms_optimizer.data import download_football_data

seasons = [f"{year}/{(year + 1) % 100:02d}" for year in range(2010, 2026)]
entries = download_football_data(seasons)
print(json.dumps([entry.__dict__ for entry in entries], indent=2))

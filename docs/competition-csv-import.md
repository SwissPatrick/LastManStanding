# Competition CSV import

Use **Import competition CSV** from Home or Entries. Download the fictional template, upload a UTF-8 comma- or semicolon-separated export, select the season and the organiser-list reporting round, link only known family rows, then review and confirm the preview.

The importer preserves the original file locally under `data/competition_imports/` (Git-ignored), stores its checksum and audit record, and never treats a blank pick as elimination. A complete list proposes missing active rows as eliminations and requires a separate confirmation. A partial list only adds compatible history.

Outside entries are stored separately from family entries. The wider-field snapshot therefore uses the outside survivor count; the displayed full-list total is never inflated by adding family entries twice. The first mid-season list is labelled an observed surviving field, not an original paid field.

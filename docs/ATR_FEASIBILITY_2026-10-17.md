# ATR sectional collection: live feasibility sample

Checked 17 October 2026 through the normal in-app browser. This is a small
read-only sample, not a sustained crawl or a test of the existing collector.

## Results

| Race | Result | Sectional runners | Splits per runner | Largest split-sum difference |
| --- | --- | ---: | ---: | ---: |
| Cheltenham, 11 March 2025, 13:20, jumps | Times and tools tabs accessible | 11 of 11 | 7 | 0.02 seconds |
| Wolverhampton, 11 March 2025, 16:30, Flat AW | Times and tools tabs accessible | 9 of 9 | 5 | 0.01 seconds |
| Dundalk, 21 March 2025, 17:15, Irish Flat AW | Full result accessible; no sectional tabs offered | Not available | Not available | Not applicable |

Sources:

- https://www.attheraces.com/racecard/Cheltenham/11-March-2025/1320
- https://www.attheraces.com/racecard/Wolverhampton/11-March-2025/1630
- https://www.attheraces.com/racecard/Dundalk/21-March-2025/1715

The two British samples contain 20 runners and 122 individual split values.
Non-runners were absent from the sectional rows, as expected. The sums were
compared with each runner's displayed finish time, not just the winner's time.
Differences are consistent with independently rounded split values.

## Data observed

Both British races exposed individual split times, finish times, race pace
classification, race finishing percentage, per-section pace classes and
front/rear indicators. The tools tabs exposed sectional speeds, top and final
speed, efficiency grades and horse finishing percentages. Wolverhampton also
displayed 0–30mph acceleration times and offered Stride Data and Distance Ran
tabs; those two additional tabs were not tested.

Cheltenham's labels were Start–12f, 12f–8f, 8f–6f, 6f–4f, 4f–2f, 2f–1f,
1f–Finish. Wolverhampton's labels were Start–4f, 4f–3f, 3f–2f, 2f–1f,
1f–Finish. The opening section includes any distance beyond the named marker;
do not assume equal distances or calculate speeds using a fixed one-furlong
length for every column.

The live HTML groups sectional values in runner rows and includes horse links
and official placing attributes. Extraction can therefore preserve identifiers
and column alignment rather than treating every decimal on the page as a split.

## Access observations

Earlier checks in this chat found that plain HTTP requests to the Cheltenham
page and sectional guide returned a roughly 3KB Client Challenge page with
HTTP 200. A successful HTTP status alone is insufficient evidence of race data.
Normal browser navigation and tab selection worked without login or a CAPTCHA
in this sample. No protection-bypass methods were tested.

The robots file read during the earlier check excluded /ajax/, login/profile
paths and URLs containing showResult=yes. Sectional tabs load dynamically from
/ajax/ links. This is not a supported public bulk API, and this investigation
does not establish permission for bulk extraction or redistribution.

## Assessment

Small-scale browser collection is demonstrated for these British races.
Large backfills and unattended daily operation remain unproven. Irish
sectionals cannot be assumed: this Dundalk race offered none, despite a working
full result. One absent Irish race does not establish absence across Ireland.

A collector should preserve source URL, capture time, runner identifiers,
section labels and raw values; distinguish blocked, not-yet-published,
not-covered and parsing-failure outcomes; and validate runner counts and split
sums. Post-race pace/sectionals are only available after racing and must not be
used as current-race pre-race features.

An existing src/attheraces_scraper.py was discovered in the workspace. It was
read but neither executed nor modified. Its transport and coverage claims were
not independently validated by this sample. Production caches and model inputs
were not changed.

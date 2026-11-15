# Feature audit:meaning,source validity and useful missing information

Audit covers151selected inputs over753,734runners/80,135races. It reads the
corrected-history candidate store, leaving the ongoing production refit intact.
The numeric audit has no constant, entirely missing or infinite selected input.
That does not establish correct meaning:the all-weather defect produces a
varying but incorrect feature, which a missing-value scan would miss.

| Finding | Evidence | Action |
|---|---|---|
| All-weather pedigree history never counted actual synthetic surfaces |179,758runners/18,927races;Allweather/Polytrack/Fibresand all miss the old predicate. The AW win rate is its constant0.1prior. |Corrected causal research implementation;one-window matched native comparison running. |
| Missing race classes used as real ordinal numbers |197,646missing current classes become0for class changes;speed adjustment independently assumesclass5.22,251observed known/unknown transitions. |Missing-aware class prototype preserves unknowns and exposes reliability flags;not active yet. |
| Unsupported class arithmetic |259,964rows have no observed comparable current/prior UK-class transition but retain finite old class-distance interactions. This includes cold starts and unknowns,not259,964confirmed model mistakes. |Research projection available;do not manufacture class changes or silently carry older known classes across an unknown latest start. |
| Impossible source weights |3runners have1–5lb;original Sporting Life ride fields themselves contain0-1/0-5. |Future parser rejects these as unknown. Historical replacements need a verified alternative source;no guessed raw-data changes. |
| Missing values fabricated by parsers |Unrecognised weight/distance formerly became140lb/10f;decimal distances could be misread. |Strict parsing now preserves unknowns,handles decimal/half-unit values and rejects malformed units.120affected checks pass. |
| Duplicate/near-duplicate signals |Margin-Elo double-centred pair exactly equal at model precision;trainer-Elo pair differs by at most2.3e-13;another momentum pair by2.3e-17. |Record aliases;test deletion as a family because column subsampling makes immediate deletion a model change. |
| Excluded recency information |680,477comparisons:99.15%source gaps match local history;26,947first-local rows have positive source gaps. |The old scrape-time dismissal is contradicted by data. Separate recency feature candidate needed. |
| Synthetic official ratings |96,842unrated rows receive field medians. Existing has_official_rating flag is useful,so this is not an automatic error. |Check all rating changes/ranks for fake information from the imputation;expose reliability. |
| Speed proxies omit actual time |Current speed inputs use beaten margins,not measured speed;unknown class is assignedclass5. |Audit class/racing-code comparability and develop prior-only time standards using existing winning-time data. |

The283Sand-surface rows are Laytown beach races and remain separate from
synthetic racing. Official source:[LaytownRaces](https://www.laytownstrandraces.ie/).
The internal region labelUKmeans the collector's UK/Ireland scope,not a reliable
country field. It is constant here;country/jurisdiction context is an information
gap,not evidence that every Irish record has the wrong region contract.

## Implemented research features

src/audited_surface_features.py (renamed src/surface_features.py on 2026-11-07) computes true prior-day sire/damsireAW win/place
rates,specialisation edges and support. The corrected sireAWedge changes707,048
later input values. New current-surface and sire-sample-support inputs give the
model context and a reliability signal. Six statistics match the native stage
on4,686real prefix rows after the predicate is corrected. Natural sand,unknown
parents,DNFs,non-runners,pending cards,current/future dates and row order are covered.

src/audited_class_features.py preserves unavailable class asNaN and uses the
latest actual earlier start,including an unknown class;it never skips backward
to a convenient older observed class. It adds known-current/previous/transition
flags and missing-aware class differences. The two families have8passing tests.
These prototypes do not yet modify the active feature store.

## Faster evaluation

The user requested fewer repeated quarterly fits.017retains its first native
2025Q3screen;the controller stops later queued periods after saving the first
model and predictions. An amendment was recorded before any017score existed.
Future screens use one window and cheaper matched baseline/candidate models;
full precomputed historical features are retained. Broader confirmation is
reserved for shortlisted improvements. No automatic live switch follows a screen.

## Evidence

- data/audit/feature_review_017/selected_input_profiles.csv:all151inputs,ranges,coverage,context and source locations.
- missing_by_year.csv,missing_by_race_type.csv and aliases.json:domain-aware coverage and redundancy.
- semantic_findings.json,implausible_weights.csv and source_cases/:source failures and original public evidence.
- surface_features.parquet and surface_projection_receipt.json:causal corrected projection and source hashes.
- class_features.parquet:missing-aware class prototype.
- data/audit/model_improvement/017_surface_family/:original registered comparison,screening amendment and results.

The preceding016market-teacher comparison completed all9,356races:LL1.872045to
1.857492,Brier.08754346to.08714276,about11.73%of the original matched market gap
closed. Its full-history production refit continues separately. The market
remains ahead;these historical improvements do not establish future parity.


## Completed first-window surface screen

On3,532July–September2025races, corrected153-input AWfamily versus selected016:
raceLL1.904964609 ->1.904705849, delta-.000258760,95%race interval
[-.002420343,+.002100156]. Brier.086597206 ->.086614982 (slightly worse).
This is inconclusive screening evidence,not a qualified improvement. The
controller stopped laterperiods at16:30UTCper the user's compute preference.
The frozen originalmanifest remains intact; its3period gate has not passed.

The preceding016productionfit is finished and installed through25October2026.
No additional training was needed to recover its installer bookkeeping failure.

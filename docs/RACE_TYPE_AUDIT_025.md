# Race-type source audit

The importer searched for the substring `chase` before checking `hurdle`.
Consequently Purchase, Chipchase, Chasemore and Chaser could incorrectly
identify a race as a Chase. A sponsor advertising a Chase day could also
override the actual Hurdle or Flat description. Some major chases omit the
discipline word from their title and fell through to Flat.

An audit of the existing2,549 public horse-profile snapshots contains48,310
unique performances. Of these,26,178 match local results in19,118 races.
Dates agree throughout. The source's published race type disagrees with the
local coarse label in23 rows from16 races. This is a source-consistency
finding; no winners or prices were used to select the audit population.

Examples include:

| Title | Stored label | Published kind |
|---|---|---|
| Purchase By3pm Today To WINABMW.ie Mares Maiden Hurdle | Chase | Hurdle |
| William Hill Chipchase Stakes (Group3) | Chase | Flat |
| Chasemore Farm Brigadier Gerard Stakes (Group3) | Chase | Flat |
| Book For Bobbyjo Chase Day ... Mares Maiden Hurdle | Chase | Hurdle |
| Ladbrokes Punchestown Gold Cup (Grade1) | Flat | Chase |
| Paddy Power Irish Gold Cup (Grade1) | Flat | Chase |

Three original result pages were also fetched with their race IDs checked
and raw responses preserved. Their current race summaries contain no
structured `run_type`. Previous-performance type fields on a horse's record
are not evidence of the current race's type and must not be substituted.
The two named Gold Cups are documented exceptions, not a rule that all
Gold Cups are chases: the Ascot race remains Flat.

The proposed parser uses explicit source codes when supplied, whole words,
the final actual discipline in a title, and excludes an advertised 'Chase
Day' from race-type inference. It agrees with every known coarse source
label in the19,118-race comparison:16 disagreements become0, with no new
disagreements. Its23 focused tests pass. It proposes changes to68 local
race labels overall; unverified changes remain audit leads, rather than
automatically rewritten historical records.

Forty stored jumps races with distances under13f are separately flagged.
This is a triage rule, not an automatic correction or a general minimum
distance claim. Some contain literal sponsor words such as Chase Distillery
that still require authoritative type metadata. The source comparison is
conditional on the fixed profile cache's coverage and does not certify all
local race labels as correct.

The current three coarse labels also intentionally collapse bumpers into
Flat:1,119 matched rows/762 races have the source type NH FLAT. Preserving
this distinction is a promising feature hypothesis, separate from fixing
incorrect published kinds. It needs a measured experiment before adoption.

The identity cross-check finds one horse ID with two display names and103
names shared by multiple IDs. These are not automatically errors: names can
change or be reused, and the main horse-history features already prefer
horse IDs. No name-based merge or historical identity rewrite was applied.

The parser is now active in both result and card ingestion, after the023
installation completed. All158 final integration checks pass, including
source-code priority in both entry points and feature-generation coverage.
The pure parser remains inside the existing importer; no extra runtime
dependency or change to the trained feature generation is required. Its
implementation is covered by the forward studies' existing importer hash.
Historical corrections require rebuilding affected
features and rating histories; changing only an indicator on stored rows
would leave contradictory inputs. No claim of predictive improvement is
attached to this metadata audit.

The recent-results refresh found another source-status defect: WITHDRAWN
entries were retained alongside actual starters, causing complete-field
checks to reject three races. These entries are now excluded in results
and cards while started non-finishers remain. All three source responses
were preserved; the corrected refresh recovered30 valid runners, including
genuine fallers, pulled-up horses and unseated riders. Two DORMANT listings
without published results remain retryable rather than receiving invented
outcomes. A Windows sharing conflict in the optional pedigree cache now
uses bounded retries and preserves verified parents if caching stays locked.

The canonical history advanced by763 runners/77 races to2026-10-27, with
754,497 rows and unchanged feature generation. The old753,734-row prefix
was verified unchanged. This required no model refit. The deep readiness
check passes all four links with no serving blockers. A real eight-runner
Nottingham12:46 card on2026-10-28 also passes the public live-input path:
horse IDs are preserved and probability checks pass. Native single-race
and batch output on five recent fields match exactly;100 recent fields
were checked for valid probabilities. The initial operational name check
was corrected to use IDs because title casing changes apostrophes such as
Sue's -> Sue'S; this was not a runner-population change.

The new source-pinned setup is
`data/forecasts/forward/20261028T092600Z_source_repair_025_20261028_084608_624611`,
first eligible2026-10-29. It preserves the earlier studies and has no
observations or scheduled recorder yet. The operational forecast is not a
paired market comparison. Final source snapshots and23 evidence/source
fingerprints are verified in the audit completion record.

Evidence: `data/audit/cached_race_code_review_025/`,
`scripts/audit_cached_race_codes.py`, `src/data_scraper.py:_infer_race_type`,
`tests/test_race_discipline.py`.

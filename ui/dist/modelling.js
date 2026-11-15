export function createModellingUI({api, esc, number, pretty, banner, badge, empty, metric, toast, state, heading, queue, dateLabel}) {
 const $ = selector => document.querySelector(selector);
 const value = text => esc(text ?? 'Not recorded');
 const score = n => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(5) : '—';
 const date = raw => esc(dateLabel(raw));
 const runLink = (run, label = run.name || run.id) => `<a class="link-button" href="#modelling/runs/${esc(run.id)}">${esc(label)}</a>`;
 const benchmarkLink = (id, label) => `<a class="link-button" href="#modelling/benchmarks/${esc(id)}">${esc(label)}</a>`;
 const runType = run => ({win_stack_candidate: 'Win-only candidate', race_predictor: 'Win/place predictor'})[run.type] || pretty(run.type);
 const panel = (title, body, description = '') => `<section class="panel"><div class="panel-head"><div><h2>${esc(title)}</h2>${description ? `<p>${esc(description)}</p>` : ''}</div></div>${body}</section>`;
 const body = content => `<div class="panel-body">${content}</div>`;
 const props = entries => `<dl class="property-grid modelling-properties">${entries.map(([label, text]) => `<div><dt>${esc(label)}</dt><dd>${value(text)}</dd></div>`).join('')}</dl>`;
 const table = (headers, rows) => `<div class="table-wrap"><table><thead><tr>${headers.map(h => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(cells => `<tr>${cells.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
 const threads = max => `<div class="field"><label for="model-threads">Training threads</label><input id="model-threads" name="threads" type="number" required min="1" max="${max}" value="${Math.min(2, max)}"></div>`;
 const blockers = (title, reasons, action = '') => reasons.length ? banner(title, reasons.join(' '), 'error', action) : '';
 const compatibility = run => run.matches == null ? badge('Unverified', 'neutral')
  : run.matches ? badge('Feature definitions match') : badge('Feature mismatch', 'warning');

 function overview(data) {
  const {store, promoted, latest_candidate: candidate, history, sources} = data;
  const readout = `<div class="metric-grid modelling-metrics">
    ${metric('Stored runners', number(store.rows), store.last_date ? 'Features through ' + dateLabel(store.last_date) : 'Feature store unavailable')}
    ${metric('Latest results date', dateLabel(data.latest_results), 'Canonical results archive')}
    ${metric('Forecast history races', number(history.races), history.through ? 'History through ' + dateLabel(history.through) : 'Correction history unavailable')}
   </div>`;
  const coverage = sources.ready ? badge('Coverage checks current') : badge('Coverage needs review', 'warning');
  const historyStatus = !history.through ? 'Unavailable' : history.recipe_count !== 1 ? 'Recipe history needs review'
   : history.lag_days == null ? 'Lag unknown' : history.lag_days > 7 ? `${history.lag_days} days behind results` : 'Within seven days of results';
  const readiness = panel('Modelling inputs', body(`
   <div class="feature-check-row"><span>Feature store</span>${badge(store.state === 'current' ? 'Matches selected dataset and definitions' : pretty(store.state), store.state === 'current' ? '' : 'warning')}</div>
   <div class="feature-check-row"><span>Source coverage</span>${coverage}</div>
   ${sources.blockers.length ? `<ul class="check-list">${sources.blockers.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}
   <div class="feature-check-row"><span>Correction history</span>${badge(historyStatus, history.recipe_count === 1 && history.lag_days != null && history.lag_days <= 7 ? '' : 'neutral')}</div>
   ${props([['Selected dataset', data.dataset], ['Store dataset', store.dataset], ['Feature definition', store.signature]])}
   <div class="actions"><a class="button small" href="#features/overview">Inspect features</a><a class="button small" href="#sources">Review sources</a></div>`));
  const models = panel('Saved models', body(`
   <div class="model-summary"><h3>Promoted win/place run</h3>
    ${promoted ? `${runLink(promoted)} ${compatibility(promoted)}<p>Fitted through ${date(promoted.trained_through)}. ${esc(promoted.promotion)}</p>`
      : `<p>${data.promoted_id ? 'Promoted run metadata is unavailable.' : 'No run is promoted.'}</p>`}
   </div>
   <div class="model-summary"><h3>Latest win-only candidate</h3>
    ${candidate ? `${runLink(candidate)} ${compatibility(candidate)}<p>Fitted through ${date(candidate.trained_through)}.</p>` : '<p>No win-only candidate saved.</p>'}
   </div>
   <div class="actions"><a class="button primary" href="#modelling/benchmarks">Open benchmarks</a><a class="button" href="#modelling/runs">Browse runs</a></div>`));
  return blockers('Feature store needs attention', data.blockers, '<a class="button small" href="#features/builds">Open feature builds</a>') +
   readout + `<div class="two-column">${readiness}${models}</div>`;
 }

 function benchmarkTitle(row) {
  if (row.label.startsWith('modelling/workbench/')) return 'Generalist benchmark';
  return row.label.split('/').slice(-2).join(' / ').replaceAll('_', ' ');
 }
 function benchmarks({rows, inputs}) {
  const usable = inputs.recipes.filter(r => !r.missing_columns?.length);
  const disabled = inputs.blockers.length || !usable.length || !inputs.development_through;
  const form = !disabled ? `<form class="panel-body" data-model-form="benchmark" data-dataset="${esc(inputs.dataset)}" data-build="${esc(inputs.store.build_id)}">
   <div class="form-grid">
    <div class="field"><label for="model-recipe">Win recipe inputs</label><select id="model-recipe" name="recipe">${inputs.recipes.map(r =>
       `<option value="${esc(r.id)}" data-signature="${esc(r.signature)}" ${r.missing_columns.length ? 'disabled' : ''}
        ${r.configured_for_training && !r.missing_columns.length ? 'selected' : ''}>${esc(r.label)} · ${number(r.count)} inputs${r.missing_columns.length ? ' · missing inputs' : ''}</option>`).join('')}</select></div>
    <div class="field"><label for="model-capacity">Fit capacity</label><select id="model-capacity" name="capacity"><option value="native">Native</option><option value="screen">Screen</option></select></div>
    <div class="field"><label for="model-fold-start">First scored date <span class="muted">(optional)</span></label><input id="model-fold-start" name="fold_start" type="date" max="${esc(inputs.development_through)}"></div>
    ${threads(inputs.max_threads)}
   </div>
   <p class="hint">Scoring stops by ${date(inputs.development_through)}. A first scored date limits the quarters scored while retaining earlier training history.</p>
   <div class="actions"><button class="button primary" type="submit">Queue benchmark</button><a class="button small" href="#features/catalog">Inspect feature sets</a></div></form>`
   : body(blockers('Benchmark unavailable for this dataset', inputs.blockers) ||
      empty('No usable feature set', 'Check the registered inputs in Features.'));
  const saved = rows.length ? table(['Benchmark', 'Scored through', 'Capacity', 'Quarters', 'Race log loss', 'Status'], rows.map(row => [
   `${benchmarkLink(row.id, benchmarkTitle(row))}<div class="race-meta">${row.report.created_at ? esc(new Date(row.report.created_at).toLocaleString('en-GB')) : 'Report pending'}</div>`,
   date(row.last_scored), value(pretty(row.capacity)), number(row.fold_count),
   Object.entries(row.report.components || {}).map(([id, m]) => `<div>${score(m.race_log_loss)} <small>${esc(id)}</small></div>`).join('') || '—',
   row.completed ? badge('Report saved') : badge('Incomplete', 'warning')
  ])) : empty('No saved benchmarks', 'Queue a walk-forward benchmark to create the first report.');
  return panel('New walk-forward benchmark', form) + panel('Saved benchmarks', saved);
 }

 function recipeDetails(spec) {
  return `<details class="panel"><summary class="panel-head"><h2>${esc(pretty(spec.id))} recipe</h2>${badge(`${spec.features.length} inputs`, 'neutral')}</summary>
   ${body(props([['Model', pretty(spec.kind || 'win')], ['Objective', pretty(spec.objective)], ['Purge gap', `${spec.purge_days} days`],
     ['Market target blend', spec.soft_label_blend], ['Calibration fraction', spec.calibration_fraction]]) +
     `<ol class="model-input-list">${spec.features.map(name => `<li>${esc(name)}</li>`).join('')}</ol>
     <details><summary>View training settings</summary><pre class="model-settings">${esc(JSON.stringify(spec, null, 2))}</pre></details>`)}
  </details>`;
 }

 function benchmark(row, dataset, build) {
  const results = Object.entries(row.report.components || {});
  const nested = row.report.nested_pool;
  const paired = nested?.paired_difference;
  const scores = results.length ? table(['Component', 'Race log loss', 'Brier', 'Calibration error', 'Top pick accuracy'], results.map(([id, m]) => [
   esc(id), score(m.race_log_loss), score(m.brier), score(m.ece), score(m.top1_accuracy)
  ])) : empty('Report pending', 'Results appear after the benchmark finishes.');
  const pool = paired ? panel('Ensemble versus generalist on matched races', body(props([
   ['Ensemble race log loss', score(nested.pool?.race_log_loss)], ['Generalist race log loss', score(nested.baseline?.race_log_loss)],
   ['Paired change in race log loss', score(paired.mean)], ['95% interval', paired.ci95.map(score).join(' to ')]
  ])), nested.evaluation_status) : '';
  const fit = row.fit_blockers.length ? body(blockers('Candidate fitting unavailable', row.fit_blockers))
   : `<form class="panel-body" data-model-form="fit" data-dataset="${esc(dataset)}" data-build="${esc(build)}" data-benchmark="${esc(row.id)}">
    <div class="form-grid">
     <div class="field"><label for="model-cutoff">Fit through</label><input type="date" id="model-cutoff" name="as_of" required min="${esc(row.last_scored)}" max="${esc(row.latest_fit)}" value="${esc(row.latest_fit)}"></div>
     <div class="field"><label for="model-run-name">Run name <span class="muted">(optional)</span></label><input type="text" id="model-run-name" name="name" maxlength="120"></div>
     ${threads(row.max_threads)}
    </div>
    <p class="hint">The candidate uses the benchmark’s recipes and ${esc(row.capacity)} capacity. Its fitted weights have no held-out score.</p>
    <button class="button primary" type="submit">Queue candidate fit</button></form>`;
  return `<a class="back-link" href="#modelling/benchmarks">← All benchmarks</a>` +
   panel(benchmarkTitle(row), scores, row.evaluation_status) + pool + panel('Fit a candidate', fit) +
   panel('Training and scoring quarters', table(['Quarter', 'Fitted through', 'Scored from', 'Scored through'], row.folds.map(f => [
    value(f.label), date(f.training_last_date), date(f.test_first_date), date(f.test_last_date)
   ]))) + panel('Benchmark inputs', body(props([
    ['Accepted dataset', row.dataset], ['Feature definition', row.feature_signature], ['Training data signature', row.data_signature],
    ['Capacity', pretty(row.capacity)], ['Excluded races in loaded history', row.excluded_races]
   ]))) + row.specs.map(recipeDetails).join('');
 }

 function runs({rows, promoted_id: promoted}) {
  return (promoted && !rows.some(run => run.promoted) ? banner('Promoted run metadata unavailable', promoted, 'error') : '') +
   panel('Saved runs', rows.length ? table(['Run', 'Model', 'Fitted through', 'Inputs', 'Installed feature definitions'], rows.map(run => [
    `${runLink(run)}<div class="race-meta">${esc(run.id)}</div>`, `${esc(runType(run))} ${run.promoted ? badge('Promoted') : ''}`,
    date(run.trained_through), number(run.features), compatibility(run)
   ])) : empty('No saved runs', 'Fit a candidate from a completed benchmark.'));
 }

 function runDetail(run) {
  const metrics = Object.entries(run.metrics);
  return `<a class="back-link" href="#modelling/runs">← All saved runs</a>` +
   panel(run.name || run.id, body(`<div class="actions">${compatibility(run)}${run.promoted ? badge('Promoted') : ''}</div>
    <p class="hint">${esc(run.promotion)}</p>
    ${props([['Run', run.id], ['Created', run.created_at], ['Fitted through', dateLabel(run.trained_through)], ['Inputs', run.features],
     ['Training rows', run.runners], ['Accepted dataset', run.dataset], ['Feature definition', run.signature], ['Model weights', run.has_model ? 'Saved' : 'Unavailable']])}
    ${run.benchmark_id ? benchmarkLink(run.benchmark_id, 'Open source benchmark →') : ''}`), runType(run)) +
   (metrics.length ? panel('Saved hold-out scores', table(['Head', 'Brier', 'Binary log loss', 'Calibration error'], metrics.map(([id, m]) => [
     esc(pretty(id)), score(m.brier_score), score(m.log_loss), score(m.ece)
   ])), `${number(run.holdout_races)} hold-out races`) : banner('No held-out score recorded for this fit', run.performance_claim || '')) +
   Object.values(run.components).filter(m => m.recipe).map(m => recipeDetails(m.recipe)).join('');
 }

 async function render(tab = 'overview', id, ticket) {
  const tabs = {overview: 'Overview', benchmarks: 'Benchmarks', runs: 'Runs'};
  if (!Object.hasOwn(tabs, tab)) tab = 'overview';
  const detail = id && tab !== 'overview';
  const params = {dataset: state.dataset};
  const route = `modelling/${tab}`;
  const data = await api(detail ? `${route}/${encodeURIComponent(id)}` : route, params);
  if (ticket !== state.ticket) return;
  const content = tab === 'overview' ? overview(data) : tab === 'runs' ? (detail ? runDetail(data) : runs(data))
   : detail ? benchmark(data, data.selected_dataset, data.build_id) : benchmarks(data);
  $('#main').innerHTML = heading('Modelling', 'Inspect readiness, benchmark win recipes and fit candidates.', '', 'MODEL DEVELOPMENT') +
   `<nav class="tabs feature-tabs" aria-label="Modelling views">${Object.entries(tabs).map(([key, label]) =>
     `<a href="#modelling/${key}" ${tab === key ? 'class="active" aria-current="page"' : ''}>${label}</a>`).join('')}</nav>` + content;
 }

 document.addEventListener('submit', async event => {
  const form = event.target.closest('[data-model-form]');
  if (!form) return;
  event.preventDefault();
  const button = form.querySelector('button[type="submit"]');
  if (button.disabled) return;
  button.disabled = true;
  try {
   const fields = new FormData(form);
   const params = {dataset: form.dataset.dataset, build_id: form.dataset.build, threads: Number(fields.get('threads'))};
   if (form.dataset.modelForm === 'benchmark') {
    const option = form.elements.recipe.selectedOptions[0];
    await queue('model_benchmark', {...params, feature_set: option.value, feature_set_signature: option.dataset.signature,
     capacity: fields.get('capacity'), fold_start: fields.get('fold_start')});
   } else {
    await queue('model_fit', {...params, benchmark: form.dataset.benchmark, as_of: fields.get('as_of'), name: fields.get('name')});
   }
  } catch (error) {
   toast(error.message, true);
  } finally {
   button.disabled = false;
  }
 });
 return {render};
}

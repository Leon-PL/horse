export function createEvaluationUI({api, esc, number, pretty, empty, metric, dateLabel, panel, body, table, score, date}) {
 const endpoint = 'modelling/evaluation';
 let lastSavedReport = `#${endpoint}`;
 const percent = (value, signed = false) => Number.isFinite(value) ? `${signed && value > 0 ? '+' : ''}${value.toFixed(2)}%` : '—';
 const units = value => value == null ? '—' : `${value > 0 ? '+' : ''}${number(value)} u`;
 const ratio = value => Number.isFinite(value) ? value.toFixed(4) : '—';
 const options = (choices, selected) => choices.map(([key, label]) =>
  `<option value="${esc(key)}" ${selected === key ? 'selected' : ''}>${esc(label)}</option>`).join('');
 const field = (label, name, choices, selected) =>
  `<label class="evaluation-field">${esc(label)}<select name="${name}">${options(choices, selected)}</select></label>`;
 const reportLocation = (id, params) => `#${endpoint}/${encodeURIComponent(id)}${params?.size ? '?' + params : ''}`;
 const href = (id, params) => esc(reportLocation(id, params));

 async function load(segment = '') {
  const [requested, query] = segment.split('?');
  const params = new URLSearchParams(query);
  const menu = await api(endpoint);
  const id = requested || menu.runs[0]?.id || (menu.timing ? 'price-timing' : '');
  const report = id ? await api(`${endpoint}/${encodeURIComponent(id)}`, Object.fromEntries(params)) : null;
  return {menu, id, report, params};
 }

 function profitChart(curve) {
  if (!curve.length) return '';
  const dates = curve.map(p => Date.parse(p.date));
  const pnl = curve.map(p => p.pnl);
  const start = dates[0], end = dates.at(-1);
  const low = Math.min(0, ...pnl), high = Math.max(0, ...pnl);
  const y = n => (151 - 127*(n-low)/(high-low || 1)).toFixed(2);
  const line = `M72 ${y(0)} ` + curve.map((p, i) =>
   `L${(72+548*(end === start ? (i+1)/curve.length : (dates[i]-start)/(end-start))).toFixed(2)} ${y(p.pnl)}`).join(' ');
  const ticks = [...new Set([low,0,high])];
  return panel('Cumulative profit and loss', body(`<figure class="evaluation-chart">
   <svg viewBox="0 0 650 190" role="img" aria-label="Cumulative profit and loss in stake units by race date">
    ${ticks.map(n => `<line x1="72" x2="620" y1="${y(n)}" y2="${y(n)}" class="${n === 0 ? 'chart-zero' : 'chart-gridline'}"/>
     <text x="64" y="${y(n)}" dy=".33em" text-anchor="end">${esc(units(n))}</text>`).join('')}
    <path class="evaluation-line" d="${line}"/>
    <text x="72" y="176">${date(curve[0].date)}</text>
    ${end === start ? '' : `<text x="620" y="176" text-anchor="end">${date(curve.at(-1).date)}</text>`}
   </svg><figcaption>End-of-day totals for the selected bets.</figcaption></figure>`));
 }

 function calibration(bins) {
  const points = bins.filter(b => Number.isFinite(b.predicted) && Number.isFinite(b.observed)
   && b.predicted >= 0 && b.predicted <= 1 && b.observed >= 0 && b.observed <= 1 && b.runners > 0);
  if (!points.length) return '';
  const max = Math.min(1, Math.max(.1, Math.ceil(10*Math.max(...points.flatMap(p => [p.predicted,p.observed])))/10));
  const x = n => (66 + 420*n/max).toFixed(1);
  const y = n => (172 - 132*n/max).toFixed(1);
  const ticks = [0, max/2, max];
  return body(`<figure class="evaluation-chart calibration-chart"><svg viewBox="0 0 520 218" role="img" aria-label="Win calibration on the full evaluated period">
   ${ticks.map(n => `<line x1="66" x2="486" y1="${y(n)}" y2="${y(n)}" class="chart-gridline"/>
    <text x="58" y="${y(n)}" dy=".33em" text-anchor="end">${Math.round(100*n)}%</text>
    <text x="${x(n)}" y="191" text-anchor="middle">${Math.round(100*n)}%</text>`).join('')}
   <path d="M66 172 L486 40" class="chart-reference"/>
   ${points.map(p => `<circle class="calibration-point" cx="${x(p.predicted)}" cy="${y(p.observed)}" r="5">
    <title>${percent(p.predicted*100)} predicted · ${percent(p.observed*100)} observed · ${number(p.runners)} runners</title></circle>`).join('')}
   <text x="66" y="23">Observed win rate</text><text x="276" y="210" text-anchor="middle">Mean predicted probability</text>
  </svg><figcaption>Each dot is a probability band. The dashed line shows perfect calibration.</figcaption></figure>`) +
   table(['Predicted', 'Observed', 'Runners'], points.map(p => [percent(p.predicted*100), percent(p.observed*100), number(p.runners)]));
 }

 function forecast(report) {
  const heads = Object.entries(report.forecast?.heads || {});
  const bins = report.forecast?.bins || [];
  const raceLoss = heads.some(([, data]) => data.race_log_loss != null);
  if (!heads.length && !bins.length) return '';
  const result = heads.length ? table(['Model output', ...(raceLoss ? ['Race log loss'] : []), 'Runner log loss', 'Brier score', 'Calibration error'],
   heads.map(([label, data]) => [esc(label === 'win_classifier' ? 'Win' : 'Place'),
    ...(raceLoss ? [score(data.race_log_loss)] : []), score(data.log_loss), score(data.brier_score), score(data.ece)])) : '';
  const scope = [report.forecast.scope, report.forecast.races ? `${number(report.forecast.races)} races` : null].filter(Boolean).join(' · ');
  return panel('Forecast quality', result + calibration(bins), scope);
 }

 function history(data, id, params) {
  const {rows, page, page_size: size, summary} = data;
  const lastPage = Math.ceil(summary.bets/size);
  const pageLink = (target, label, description = label) => {
   const query = new URLSearchParams(params);
   query.delete('section');
   query.set('page', target);
   return `<a class="button small" href="${href(id, query)}" aria-label="${description}">${label}</a>`;
  };
  const previous = page > 1 ? pageLink(page-1, 'Previous', 'Previous page') : '';
  const next = page < lastPage ? pageLink(page+1, 'Next', 'Next page') : '';
  const range = `${number((page-1)*size+1)}–${number(Math.min(page*size, summary.bets))} of ${number(summary.bets)} bets`;
  const bandQuery = new URLSearchParams(params);
  bandQuery.set('section', 'odds-bands');
  const bandLink = data.odds_breakdown?.length ? `<a class="link-button" href="${href(id, bandQuery)}">Odds-band breakdown ↑</a>` : '';
  const pager = lastPage > 1 ? `<div class="table-foot bet-history-pager">
   <div>Page ${number(page)} of ${number(lastPage)} · ${range} ${bandLink}</div>
   <nav class="actions" aria-label="Bet history pages">
    ${page > 1 ? pageLink(1, 'First', 'First page') + previous + pageLink(page-1, number(page-1), `Page ${page-1}`) : ''}
    <span class="button small primary" aria-current="page" aria-label="Page ${page}">${number(page)}</span>
    ${page < lastPage ? pageLink(page+1, number(page+1), `Page ${page+1}`) + next + pageLink(lastPage, 'Last', 'Last page') : ''}
   </nav>
   <form class="actions bet-history-jump" data-bet-page-form data-source="${esc(id)}" data-query="${esc(params.toString())}">
    <label for="bet-history-page">Go to page</label>
    <input id="bet-history-page" name="page" type="number" inputmode="numeric" value="${page}" min="1" max="${lastPage}" required>
    <button class="button small" type="submit">Go</button>
   </form>
  </div>` : `<div class="table-foot"><span>${range}</span>${bandLink}</div>`;
  const raceName = row => {
   if (id !== 'price-timing' || !row.race_id) return esc(row.race_name || '—');
   const query = new URLSearchParams(params);
   query.delete('section');
   query.set('page', page);
   query.set('race_id', row.race_id);
   return `<a class="link-button" href="${href(id, query)}">${esc(row.race_name || 'View race forecasts')}</a>`;
  };
  const hasBsp = rows.some(r => r.bsp != null);
  const ledger = table(['Race date', 'Horse', 'Course', 'Race name', 'Odds', ...(hasBsp ? ['BSP'] : []), 'Result', 'Stake', 'P&L'], rows.map(row => [
   date(row.race_date),
   row.race_id ? `<a class="link-button" href="#race/${encodeURIComponent(row.race_id)}">${esc(row.horse_name || (row.horse_id ? `Runner ${row.horse_id}` : 'View race'))}</a>` : esc(row.horse_name || '—'),
   esc(row.track || '—'), raceName(row),
   number(row.odds), ...(hasBsp ? [number(row.bsp)] : []),
   row.won === 1 ? 'Won' : row.placed === 1 ? 'Placed' : row.won === 0 ? 'Lost' : '—',
   number(row.stake), units(row.pnl)
  ]));
  const footer = lastPage > 1 ? `<div class="table-foot"><span>${range}</span><nav class="actions" aria-label="Bet history page shortcuts">${previous}${next}</nav></div>` : '';
  const hint = id === 'price-timing' ? 'Open a race name to see the market and both model forecasts for the full field.' : '';
  return `<div id="bet-history" role="region" aria-label="Bet history" tabindex="-1">${panel('Bet history', pager + ledger + footer, hint)}</div>`;
 }

 function raceView(report, params) {
  const {race, cut, cuts} = report;
  const id = 'price-timing';
  const back = new URLSearchParams(params);
  back.delete('race_id');
  const bandQuery = new URLSearchParams(back);
  bandQuery.set('section', 'odds-bands');
  const chance = probability => Number.isFinite(probability) ? percent(probability*100) : '—';
  const forecastPrice = probability => Number.isFinite(probability) && probability > 0 ? `<div class="race-meta">Fair odds ${number(1/probability)}</div>` : '';
  const winner = race.rows.find(row => row.won === 1);
  const controls = body(`<div class="section-toolbar">
   <nav class="actions" aria-label="Race price timing">${Object.entries(cuts).map(([key, label]) => {
    const query = new URLSearchParams(params);
    query.set('cut', key);
    return `<a class="button small ${key === cut ? 'primary' : ''}" href="${href(id, query)}" ${key === cut ? 'aria-current="page"' : ''}>${esc(label)}</a>`;
   }).join('')}</nav>
   <a class="button small" href="#race/${encodeURIComponent(race.id)}">Open race inspector</a>
  </div>
  <p class="hint">${esc([winner?.horse_name ? `Winner ${winner.horse_name}` : null, `${race.rows.length} runners`,
    `${race.bets} simulated ${race.bets === 1 ? 'bet' : 'bets'}`, race.bets ? `P&L ${units(race.pnl)}` : null].filter(Boolean).join(' · '))}</p>`);
  const field = table(['Horse', 'Market price', 'Base model', 'Corrected model', 'Edge vs market', 'Betfair SP', 'Simulated bet'],
   race.rows.map(row => {
    const edge = Number.isFinite(row.corrected_probability) && Number.isFinite(row.market_probability)
     ? (row.corrected_probability-row.market_probability)*100 : null;
    return [
     `<strong>${esc(row.horse_name || `Runner ${row.horse_id}`)}</strong>${row.won === 1 ? '<div class="race-meta">Winner</div>' : ''}`,
     `${number(row.market_price)}${Number.isFinite(row.market_probability) ? `<div class="race-meta">${chance(row.market_probability)} implied</div>` : ''}`,
     chance(row.base_probability) + forecastPrice(row.base_probability),
     chance(row.corrected_probability) + forecastPrice(row.corrected_probability),
     edge === null ? '—' : `${edge > 0 ? '+' : ''}${edge.toFixed(2)} pp`,
     number(row.bsp),
     row.pnl == null ? '—' : `@${number(row.simulated_price)}<div class="race-meta">${units(row.pnl)} P&L</div>`
    ];
   }));
  const details = `The market price is the last traded price at ${cuts[cut]}. The corrected forecast uses that snapshot. Edge is corrected win probability minus the market’s share of the field.`;
  const rule = `Simulated bets use a ${Number(report.edge)*100} pp base edge, ${report.ticks === '1' ? 'one tick below ' : ''}the price adjusted for withdrawals${params.get('band') ? `, and the ${params.get('band')} odds band` : ''}.`;
  return `<div class="actions"><a class="back-link" href="${href(id, back)}">← Back to bet history</a>
   <a class="back-link" href="${href(id, bandQuery)}">Odds-band breakdown</a></div>` +
   panel(race.name || `Race ${race.id}`, controls + field + body(`<p class="hint">${esc(details)} ${esc(rule)}</p>`),
    [dateLabel(race.date), race.track, 'Archived price-timing study'].filter(Boolean).join(' · '));
 }

 function oddsBreakdown(rows, id, params) {
  if (!rows?.length) return '';
  const hasClv = rows.some(row => row.clv?.n);
  const bands = table(['Decimal odds', 'Bets', 'Staked', 'Strike rate', 'Profit / loss', 'ROI', ...(hasClv ? ['Median CLV'] : [])],
   rows.map(row => {
    const query = new URLSearchParams(params);
    query.delete('section');
    query.set('band', row.band);
    query.set('page', 1);
    const label = row.bets ? `<a class="link-button" href="${href(id, query)}" aria-label="View bets at odds ${esc(row.band)}"
     ${params.get('band') === row.band ? 'aria-current="true"' : ''}>${esc(row.band)}</a>` : esc(row.band);
    const close = row.clv?.n ? `${ratio(row.clv.median_clv)}<div class="race-meta">${number(row.clv.n)} matched</div>` : '—';
    return [label, number(row.bets), number(row.staked), percent(row.strike_rate),
     row.bets ? units(row.pnl) : '—', percent(row.roi, true), ...(hasClv ? [close] : [])];
   }));
  return `<div id="odds-band-breakdown" role="region" aria-label="Odds-band breakdown" tabindex="-1">${panel(
   'Odds-band breakdown', bands, 'Bands use the recorded bet odds and include the upper bound. Open a band to see its bets.'
  )}</div>`;
 }

 function betting(report, id, params, assumption) {
  const data = report.bets;
  const summary = data?.summary || report.summary;
  const clv = data?.clv || report.clv;
  if (!summary) return panel('Betting results', empty('No betting results saved', 'Open an evaluated run or the price-timing study.'));
  const hasRows = !!data;
  const show = `<div class="evaluation-metrics">
   ${metric('Bets', number(summary.bets), summary.strike_rate == null ? '' : percent(summary.strike_rate)+' win strike rate')}
   ${metric('Net profit / loss', units(summary.pnl), number(summary.staked ?? summary.total_staked)+' units staked')}
   ${metric('ROI', percent(summary.roi, true), 'Profit as a share of stakes')}
   ${metric('Maximum drawdown', number(summary.drawdown ?? summary.max_drawdown)+' u', hasRows ? 'Measured at end of day' : 'Saved settlement order')}
  </div>`;
  const close = clv?.n ? `<div class="evaluation-clv">
   <div><h3>Closing-line value</h3><p>${number(clv.n)} bets matched to Betfair SP${data?.clv ? ' in this selection' : ' across the full strategy report'}.</p></div>
   <dl><div><dt>Median</dt><dd>${ratio(clv.median_clv)}</dd></div><div><dt>Geometric mean</dt><dd>${ratio(clv.geometric_mean_clv)}</dd></div>
   <div><dt>Beat the close</dt><dd>${percent(clv.pct_beat_close*100)}</dd></div></dl></div>` :
   '<p class="hint">Closing-line value is unavailable for this selection.</p>';
  const periods = data?.quarterly.length ? panel('Quarterly results', table(['Quarter', 'Bets', 'Staked', 'Profit / loss', 'ROI'],
   data.quarterly.map(q => [esc(q.quarter), number(q.bets), number(q.staked), units(q.pnl), percent(q.roi, true)]))) : '';
  return panel('Betting results', body(show + close), assumption) + oddsBreakdown(data?.odds_breakdown, id, params) + (data?.rows.length
   ? profitChart(data.curve) + periods + history(data, id, params) : body(empty('Bet history unavailable',
    hasRows ? 'No bets match the selected quarter and odds band.' : 'This run only retained the betting summary.')));
 }

 function content({menu, id, report, params}) {
  if (!report) return empty('No saved evaluation', 'Evaluation appears here when a saved run or price-timing study has results.');
  const timing = id === 'price-timing';
  if (!timing) lastSavedReport = href(id, params);
  const navigation = menu.timing ? `<nav class="evaluation-modes" aria-label="Evaluation views">
   <a class="button ${timing ? '' : 'primary'}" href="${lastSavedReport}" ${timing ? '' : 'aria-current="page"'}>Saved run reports</a>
   <a class="button ${timing ? 'primary' : ''}" href="#${endpoint}/price-timing" ${timing ? 'aria-current="page"' : ''}>Price timing</a>
  </nav>` : '';
  if (report.race) return navigation + raceView(report, params);
  const sources = menu.runs.map(run => [run.id,run.name]);
  if (!timing && !sources.some(([key]) => key === id)) sources.unshift([id, report.title]);
  const data = report.bets;
  const controls = `<form class="panel-body evaluation-filters ${timing ? 'timing-filters' : ''}" data-evaluation-form data-source="${esc(id)}">
   ${timing ? '' : field('Evaluation', 'source', sources, id)}
   ${timing ? field('Price time', 'cut', Object.entries(report.cuts), report.cut) +
     field('Base probability edge', 'edge', report.edges.map(edge => [edge, `${Number(edge)*100} pp`]), report.edge) +
     field('Price assumption', 'ticks', [['0','Recorded last traded price'],['1','One tick worse']], report.ticks)
    : report.strategies.length ? field('Betting strategy', 'strategy', report.strategies.map(k => [k,pretty(k)]), report.strategy) : ''}
   ${data ? field('Quarter', 'quarter', [['','All quarters'], ...data.quarters.map(q => [q,q])], params.get('quarter') || '') +
    field('Decimal odds', 'band', [['','All odds'], ...data.bands.map(b => [b,b])], params.get('band') || '') : ''}
   ${data?.race_types.length ? field('Race type', 'race_type', [['','All race types'], ...data.race_types.map(type => [type,type])], params.get('race_type') || '') : ''}
   <button class="button primary" type="submit">Apply</button></form>`;
  const assumption = timing ? `${report.cuts[report.cut]} snapshot · ${report.ticks === '1' ? 'one tick below the adjusted last traded price' : 'last traded price adjusted for withdrawals'} · 1 unit per bet · 2% exchange commission. A traded quote is a simulated fill assumption.` :
   ['Odds saved in run · quote time unrecorded · settlement as saved',
    report.strategy === 'top_pick' ? '1 unit per bet' : report.strategy === 'each_way' ? '1 unit per leg' : report.stake_mode === 'flat' ? '1 unit per bet' :
    report.stake_mode === 'kelly' ? 'Saved fractional Kelly stakes' : null,
    report.value_threshold != null ? `base edge ${Math.round(report.value_threshold*10000)/100} pp` : null].filter(Boolean).join(' · ');
  const dates = report.dates?.length === 2 ? `${dateLabel(report.dates[0])} to ${dateLabel(report.dates[1])}` : 'Evaluation dates unrecorded';
  const method = timing ? 'Retrospective walk-forward study' : 'Saved historical evaluation';
  return navigation + panel(report.title, controls, `${method} · ${dates}`) + betting(report, id, params, assumption) + forecast(report);
 }

 document.addEventListener('submit', event => {
  const jump = event.target.closest('[data-bet-page-form]');
  if (jump) {
   event.preventDefault();
   const params = new URLSearchParams(jump.dataset.query);
   params.delete('section');
   params.set('page', new FormData(jump).get('page'));
   location.hash = reportLocation(jump.dataset.source, params);
   return;
  }
  const form = event.target.closest('[data-evaluation-form]');
  if (!form) return;
  event.preventDefault();
  const fields = new FormData(form);
  const id = fields.get('source') || form.dataset.source;
  if (id !== form.dataset.source) {
   location.hash = reportLocation(id);
   return;
  }
  const params = new URLSearchParams();
  for (const [key, value] of fields) if (key !== 'source' && value) params.set(key, value);
  location.hash = reportLocation(id, params);
 });
 return {load, content};
}

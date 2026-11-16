// Matchbook paper record: the 11:00 rule followed forward on live prices. Read-only.
export function createPaperUI({api, esc, number, banner, badge, empty, metric, heading}) {
 const $ = selector => document.querySelector(selector);
 const pct = x => typeof x === 'number' && Number.isFinite(x) ? (x >= 0 ? '+' : '') + x.toFixed(1) + '%' : '—';
 const ratio = x => typeof x === 'number' && Number.isFinite(x) ? x.toFixed(3) : '—';
 const odds = x => typeof x === 'number' && Number.isFinite(x) ? x.toFixed(2) : '—';
 const money = x => typeof x === 'number' && Number.isFinite(x) ? (x < 0 ? '−£' : '£') + Math.abs(x).toFixed(2) : '—';
 const panel = (title, body, description = '') => `<section class="panel"><div class="panel-head"><div><h2>${esc(title)}</h2>${description ? `<p>${esc(description)}</p>` : ''}</div></div>${body}</section>`;
 const table = (headers, rows) => `<div class="table-wrap"><table><thead><tr>${headers.map(h => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(cells => `<tr>${cells.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;

 function curve(points) {
  if (points.length < 2) return '';
  const w = 640, h = 160, pad = 28, ys = points.map(p => p.cumulative), lo = Math.min(0, ...ys), hi = Math.max(0, ...ys);
  const x = i => pad + i * (w - 2 * pad) / (points.length - 1), y = v => h - pad - (v - lo) * (h - 2 * pad) / ((hi - lo) || 1);
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.cumulative).toFixed(1)}`).join('');
  return `<svg class="paper-curve" viewBox="0 0 ${w} ${h}" role="img" aria-label="Cumulative profit in level stakes">
   <line x1="${pad}" x2="${w - pad}" y1="${y(0)}" y2="${y(0)}" stroke="currentColor" stroke-opacity=".25"/>
   <path d="${path}" fill="none" stroke="var(--accent, #2563eb)" stroke-width="2"/>
   <text x="${pad}" y="14" font-size="11" fill="currentColor">Cumulative level-stake profit (units)</text></svg>`;
 }

 function betRows(rows, settled) {
  return rows.map(b => [esc(b.race_date || ''), esc(b.off_time || ''), esc(b.track || ''), `<strong>${esc(b.horse_name || '')}</strong>`,
   ratio(b.p), ratio(b.share), odds(b.back1), money(b.back1_size),
   ...(settled ? [odds(b.bsp), ratio(b.clv), b.status !== 'settled' ? badge('Void', 'neutral') : b.won ? badge('Won') : badge('Lost', 'neutral'), money(b.pnl_fill)]
              : [money(b.fill_stake)])]);
 }

 const share = x => typeof x === 'number' && Number.isFinite(x) ? (100 * x).toFixed(1) + '%' : '—';
 const fair = x => typeof x === 'number' && x > 0 ? (1 / x).toFixed(2) : '—';
 const quote = (price, size) => price > 1 ? `${odds(price)}<span class="pp-size">${money(size)}</span>` : '—';
 const styles = `<style>
  .pp-toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .pp-races{display:grid;gap:10px;padding:0 22px 22px}
  .pp-race{border:1px solid var(--line);border-radius:10px;background:#fff}
  .pp-race>summary{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:11px 14px;cursor:pointer;list-style:none}
  .pp-race>summary::-webkit-details-marker{display:none}
  .pp-race>summary::before{content:'▸';color:var(--muted);transition:transform .15s}
  .pp-race[open]>summary::before{transform:rotate(90deg)}
  .pp-race>summary strong{font-size:13px}
  .pp-race .pp-meta{color:var(--muted);font-size:11px}
  .pp-race .table-wrap{border-top:1px solid var(--line)}
  .pp-race td,.pp-race th{text-align:right;padding:7px 10px;font-size:11px;white-space:nowrap}
  .pp-race td:first-child,.pp-race th:first-child{text-align:left}
  .pp-race tr.pp-bet td{background:var(--mint)}
  .pp-race tr.pp-out td{color:var(--muted);text-decoration:line-through}
  .pp-size{display:block;color:var(--muted);font-size:9px}
  .pp-pos{color:var(--green);font-weight:600}.pp-neg{color:var(--muted)}
  .pp-up{color:var(--green)}.pp-down{color:var(--red)}
  .pp-bar{display:block;height:3px;border-radius:2px;background:var(--green);opacity:.55;margin-top:3px;margin-left:auto}
  .pp-bar.mkt{background:var(--amber)}
 </style>`;
 let chosenDay = null;

 function runnerRow(r, forecast) {
  const move = r.price_0800 > 1 && r.back1 > 1 ? (r.back1 < r.price_0800 ? 'pp-up' : r.back1 > r.price_0800 ? 'pp-down' : '') : '';
  const latestMove = r.latest_back > 1 && r.back1 > 1 ? (r.latest_back < r.back1 ? 'pp-up' : r.latest_back > r.back1 ? 'pp-down' : '') : '';
  const result = r.won === 1 ? badge('Won') : r.won === 0 ? badge('Lost', 'neutral') : '';
  const bet = r.bet ? badge(`Bet £${(r.fill_stake ?? 0).toFixed(0)} @ ${odds(r.fill_price)}`, 'blue') + (result ? ' ' + result : '') : result;
  const bar = (x, cls = '') => typeof x === 'number' ? `<span class="pp-bar ${cls}" style="width:${Math.min(100, 100 * x).toFixed(0)}%"></span>` : '';
  const name = `<strong>${esc(r.horse_name || '')}</strong>`;
  if (!forecast) return {cls: '', cells: [name, share(r.p_model) + bar(r.p_model), fair(r.p_model)]};
  return {cls: r.withdrawn_now ? 'pp-out' : r.bet ? 'pp-bet' : '', cells: [name,
   share(r.p_model), share(r.p) + bar(r.p), fair(r.p), share(r.share) + bar(r.share, 'mkt'),
   `<span class="${r.edge > 0 ? 'pp-pos' : 'pp-neg'}">${typeof r.edge === 'number' ? (r.edge >= 0 ? '+' : '') + (100 * r.edge).toFixed(1) : '—'}</span>`,
   odds(r.price_0800), `<span class="${move}">${quote(r.back1, r.back1_size)}</span>`, quote(r.lay1, r.lay1_size),
   r.latest_back > 1 ? `<span class="${latestMove}">${odds(r.latest_back)}</span><span class="pp-size">${esc(r.latest_at || '')}${r.latest_lay > 1 ? ' · lay ' + odds(r.latest_lay) : ''}</span>` : '—',
   ...(r.bsp != null || r.won != null ? [odds(r.bsp)] : []), bet || '']};
 }

 function racesPanel(day) {
  const forecast = day.stage === 'forecast';
  const settled = day.races.some(race => race.field.some(r => r.bsp != null));
  const headers = forecast ? ['Horse', 'Win model', 'Forecast', 'Fair odds', '11:00 share', 'Edge (pts)', '08:00', '11:00 back', '11:00 lay', 'Latest back', ...(settled ? ['BSP'] : []), 'Bet']
   : ['Horse', 'Win model', 'Fair odds'];
  const pick = day.dates.length ? `<label class="pp-toolbar"><span class="muted">Day</span><select id="paper-day">${day.dates.map(x => `<option${x === day.date ? ' selected' : ''}>${esc(x)}</option>`).join('')}</select></label>` : '';
  const note = !day.stage ? '' : forecast
   ? `Win model is run ${esc(day.win_run || '')}, normalised in the race; Forecast is after the correction with the 11:00 price. 11:00 share is the race-normalised Matchbook price; Edge = Forecast − share. Green prices shortened, red drifted (11:00 against 08:00, latest against 11:00). A bet needs Edge above the rule's value threshold and a back price ≤ 51; struck-through runners have since been withdrawn.`
   : 'Prepared win-model forecasts only; the corrected forecast, prices and bets appear at 11:00.';
  const races = day.races.map(race => {
   const rows = race.field.map(r => runnerRow(r, forecast));
   const flags = [race.bets ? badge(`${race.bets} bet${race.bets > 1 ? 's' : ''}`, 'blue') : '',
    race.eligible === false ? badge('Not eligible', 'neutral') : '',
    race.tight != null && race.tight < 0.8 ? badge('Thin book', 'warning') : ''].join(' ');
   return `<details class="pp-race"${race.bets ? ' open' : ''}><summary><strong>${esc(race.off_time)} ${esc(race.track)}</strong>
    <span class="pp-meta">${race.runners} runners${race.book_pct ? ` · back book ${race.book_pct.toFixed(1)}%` : ''}${race.tight != null ? ` · ${Math.round(100 * race.tight)}% quoted tight` : ''}</span>${flags}</summary>
    <div class="table-wrap"><table><thead><tr>${headers.map(h => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(r => `<tr class="${r.cls}">${r.cells.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div></details>`;
  }).join('');
  const body = day.races.length ? `<p class="hint" style="padding:0 22px">${note}</p><div class="pp-races">${races}</div>` : empty('No paper day yet', 'Races appear once the card is prepared.');
  return `<section class="panel" id="paper-races"><div class="panel-head"><div><h2>Races</h2><p>Whole field for each race: model forecasts against Matchbook prices. Races with bets start open.</p></div>${pick}</div>${body}</section>`;
 }

 function bindDay() {
  const select = $('#paper-day');
  if (!select) return;
  select.addEventListener('change', async () => {
   chosenDay = select.value;
   $('#paper-races').outerHTML = racesPanel(await api('paper/day', {date: chosenDay}));
   bindDay();
  });
 }

 async function render() {
  const [d, day] = await Promise.all([api('paper'), api('paper/day', chosenDay ? {date: chosenDay} : {})]);
  const s = d.summary || {};
  const roi = s.roi_unit || {};
  const ready = d.config ? '' : banner('Paper record not configured', 'Run python -m src.matchbook configure --win-run RUN_ID after fitting the 11:00 correction.', 'warning');
  const metrics = `<div class="metric-grid">
   ${metric('Settled bets', number(s.settled_bets || 0), `${s.settled_days || 0} settled of ${s.days_recorded || 0} recorded days`)}
   ${metric('Geometric CLV vs BSP', ratio(s.geometric_mean_clv), s.geo_clv_ci95 ? `95% ${ratio(s.geo_clv_ci95[0])}–${ratio(s.geo_clv_ci95[1])} · median ${ratio(s.median_clv)}` : 'Taken price / Betfair SP')}
   ${metric('ROI, level stakes', pct(roi.mean), roi.ci_low != null ? `95% ${pct(roi.ci_low)} to ${pct(roi.ci_high)} · 2% commission` : 'After 2% commission')}
   ${metric('Tight-book races', s.tight_races ? ratio(s.tight_races.geometric_mean_clv) : '—', s.tight_races ? `${s.tight_races.bets} bets where ≥${Math.round(100 * s.tight_races.threshold)}% of the field is quoted within 25%` : 'CLV where the 11:00 book was tight')}
   ${metric(`Fill at £${d.rule.stake}`, s.fill_rate != null ? (100 * s.fill_rate).toFixed(0) + '%' : '—', s.pnl_filled != null ? `Profit on filled stakes ${money(s.pnl_filled)}` : 'Stake available at 11:00')}
  </div>`;
  const rule = `<p class="hint">Rule (research 107): back where the corrected forecast beats the race-normalised 11:00 Matchbook share by the value threshold (base ${d.rule.threshold}), best back price ≤ ${d.rule.max_odds}, races off after ${esc(d.rule.clock)}. Settled at Betfair SP results the next day. No bet is placed. Last price snapshot: ${esc(d.last_snapshot || 'none')}.</p>`;
  const fs = s.field || {}, ll = fs.log_loss || {}, edge = (fs.clv_by_edge || {}).forecast_above_share || {};
  const loss = x => typeof x === 'number' ? x.toFixed(4) : '—';
  const field = fs.races ? `<div class="metric-grid">
    ${metric('Forecast log loss', loss(ll.forecast), `Betfair SP ${loss(ll.betfair_sp)} · 11:00 Matchbook ${loss(ll.matchbook_1100)} · win model ${loss(ll.win_model)}`)}
    ${metric('Forecast − SP', typeof fs.forecast_minus_sp === 'number' ? (fs.forecast_minus_sp >= 0 ? '+' : '') + fs.forecast_minus_sp.toFixed(4) : '—', `Per race, ${number(fs.races)} races over ${fs.days} days; below zero beats the close`)}
    ${metric('Edge vs result', typeof fs.edge_result_corr === 'number' ? fs.edge_result_corr.toFixed(3) : '—', 'Correlation of (forecast − 11:00 share) with (won − SP share); above zero is information the close lacks')}
    ${metric('CLV where forecast > share', ratio(edge.geometric_mean_clv), edge.n ? `${number(edge.n)} runners · median ${ratio(edge.median_clv)}` : 'Taken / BSP for every runner the forecast rated above the market')}
   </div>`
   : empty('No whole-field results yet', 'Settlement keeps SP and result for every runner from the first settled day.');
  const days = d.days.length ? table(['Date', 'Status', 'Card races', 'Matchbook races', 'Linked', 'Eligible', 'Bets', 'Settled'],
   d.days.map(r => [esc(r.date), r.error ? badge('Error', 'error') + ` <span class="muted">${esc(r.error)}</span>` : badge(r.status || '', r.status === 'settled' ? '' : 'neutral'),
    number(r.card_races ?? 0), number(r.matchbook_races ?? 0), number(r.races_linked ?? 0), number(r.eligible_races ?? 0), number(r.bets ?? 0), number(r.settled_bets ?? 0)]))
   : empty('No paper days yet', 'The first day appears after the 11:00 forecast runs.');
  const open = d.open_bets.length ? table(['Date', 'Off', 'Track', 'Horse', 'Forecast', '11:00 share', 'Back', 'On offer', `Filled of £${d.rule.stake}`], betRows(d.open_bets, false))
   : empty('No unsettled bets', '');
  const settled = d.settled_bets.length ? table(['Date', 'Off', 'Track', 'Horse', 'Forecast', '11:00 share', 'Back', 'On offer', 'BSP', 'CLV', 'Result', 'Profit (filled)'], betRows(d.settled_bets, true))
   : empty('Nothing settled yet', 'Betfair publishes SP the morning after racing.');
  $('#main').innerHTML = styles + heading('Matchbook paper record', 'The 11:00 rule followed forward on live Matchbook prices, before any money is staked.', '', 'BETTING VALIDATION') +
   ready + metrics + rule + panel('Whole field', `<div class="panel-body">${field}</div>`, 'Every runner of every eligible, settled race: the forecast against the market, not just the bets.') +
   panel('Unsettled paper bets', open) + racesPanel(day) +
   panel('Profit curve', `<div class="panel-body">${curve(d.curve) || empty('Curve appears after two settled days', '')}</div>`) +
   panel('Days', days) + panel('Settled bets', settled);
  bindDay();
 }
 return {render};
}

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

 async function render() {
  const d = await api('paper');
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
  const days = d.days.length ? table(['Date', 'Status', 'Card races', 'Matchbook races', 'Linked', 'Eligible', 'Bets', 'Settled'],
   d.days.map(r => [esc(r.date), r.error ? badge('Error', 'error') + ` <span class="muted">${esc(r.error)}</span>` : badge(r.status || '', r.status === 'settled' ? '' : 'neutral'),
    number(r.card_races ?? 0), number(r.matchbook_races ?? 0), number(r.races_linked ?? 0), number(r.eligible_races ?? 0), number(r.bets ?? 0), number(r.settled_bets ?? 0)]))
   : empty('No paper days yet', 'The first day appears after the 11:00 forecast runs.');
  const open = d.open_bets.length ? table(['Date', 'Off', 'Track', 'Horse', 'Forecast', '11:00 share', 'Back', 'On offer', `Filled of £${d.rule.stake}`], betRows(d.open_bets, false))
   : empty('No unsettled bets', '');
  const settled = d.settled_bets.length ? table(['Date', 'Off', 'Track', 'Horse', 'Forecast', '11:00 share', 'Back', 'On offer', 'BSP', 'CLV', 'Result', 'Profit (filled)'], betRows(d.settled_bets, true))
   : empty('Nothing settled yet', 'Betfair publishes SP the morning after racing.');
  $('#main').innerHTML = heading('Matchbook paper record', 'The 11:00 rule followed forward on live Matchbook prices, before any money is staked.', '', 'BETTING VALIDATION') +
   ready + metrics + rule + panel('Profit curve', `<div class="panel-body">${curve(d.curve) || empty('Curve appears after two settled days', '')}</div>`) +
   panel('Unsettled paper bets', open) + panel('Days', days) + panel('Settled bets', settled);
 }
 return {render};
}

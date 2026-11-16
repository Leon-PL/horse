import {createFeaturesUI} from './features.js';
import {createModellingUI} from './modelling.js';
import {createPaperUI} from './paper.js';
const $ = (q, root = document) => root.querySelector(q);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => value == null ? '—' : Number(value).toLocaleString('en-GB', {maximumFractionDigits: 2});
const pretty = value => String(value ?? '').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const valueText = value => value == null ? 'Unknown' : typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
const sourceLabels = {sl:'Sporting Life',rtv:'Racing TV',atr:'At The Races',betfair:'Betfair',bha:'BHA stewards',pedigree:'Pedigree',fx:'Currency references',weather:'Weather'};
const sourceShort = {sl:'SL',rtv:'RTV',atr:'ATR',betfair:'BF',bha:'BHA',pedigree:'PD',fx:'FX',weather:'WX'};
const state = {page:'overview',dataset:'',boundDataset:'',overview:null,source:'atr',layer:'accepted',race:null,raceTab:'runners',query:'',offset:0,records:new Map(),recordCount:0,changes:[],auto:true,ticket:0};
let token = '', toastTimer;
const icons = {
 overview:'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
 races:'<path d="M4 5h16M4 12h16M4 19h16"/><circle cx="8" cy="5" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="10" cy="19" r="2"/>',
 sources:'<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
 coverage:'<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v4m10-4v4M3 10h18M8 10v11m8-11v11m-13-6h18"/>',
 features:'<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 3v18M15 9v12M3 15h18"/>',
 modelling:'<path d="M4 4v16h16M8 15l4-5 4 2 4-6"/>',
 paper:'<path d="M4 19h16M6 15l4-4 3 3 5-6"/><circle cx="18" cy="8" r="1.5"/>',
 jobs:'<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>',
 review:'<path d="M5 21V4m0 0c6-4 8 4 14 0v10c-6 4-8-4-14 0"/>'
};
const svgIcon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name]}</svg>`;
$('#navigation').innerHTML = Object.entries({overview:'Overview',races:'Races',sources:'Sources',coverage:'Coverage',features:'Features',modelling:'Modelling',paper:'Paper record',jobs:'Jobs',review:'Review queue'}).map(([key,label]) => `<a href="#${key}" data-page="${key}">${svgIcon(key)}${label}</a>`).join('');

function badge(label, kind = '') { return `<span class="badge ${kind}">${esc(label)}</span>`; }
function sourceIcon(id) { return `<span class="source-icon ${id}">${sourceShort[id]}</span>`; }
function when(value) {
 if (!value) return 'Not recorded';
 const d = new Date(value); if (Number.isNaN(d.valueOf())) return value;
 const seconds = Math.max(0, (Date.now() - d)/1000);
 return seconds < 60 ? 'Just now' : seconds < 3600 ? `${Math.floor(seconds/60)}m ago` : seconds < 86400 ? `${Math.floor(seconds/3600)}h ago` : `${Math.floor(seconds/86400)}d ago`;
}
function dateLabel(value) { return value ? new Date(`${value.slice(0,10)}T12:00:00`).toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'}) : 'Not recorded'; }
function toast(message, error = false) {
 const box = $('#toast'); box.textContent = message; box.className = `toast ${error ? 'error' : ''}`; box.hidden = false;
 clearTimeout(toastTimer); toastTimer = setTimeout(() => box.hidden = true, error ? 10000 : 6000);
}
function url(path, params = {}) {
 const search = new URLSearchParams({start:$('#start-date').value,end:$('#end-date').value});
 const dataset = Object.hasOwn(params,'dataset') ? params.dataset : state.boundDataset;
 if (dataset) search.set('dataset',dataset);
 Object.entries(params).forEach(([k,v]) => {if (v != null && v !== '') search.set(k,v);});
 return `/api/${path}?${search}`;
}
async function api(path, params = {}, body) {
 const response = await fetch(url(path,params), body ? {method:'POST',headers:{'Content-Type':'application/json','X-Workbench-Token':token},body:JSON.stringify(body)} : {signal:state.abort?.signal});
 const result = await response.json();
 if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
 return result;
}
function heading(title, description, action = '', eyebrow = 'COLLECTION & DATA QUALITY') {
 return `<div class="page-heading"><div><div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(title)}</h1><p>${esc(description)}</p></div>${action ? `<div class="actions">${action}</div>` : ''}</div>`;
}
function banner(title, description = '', kind = '', action = '') {
 return `<div class="banner ${kind}"><span class="banner-icon">${kind === 'good' ? '✓' : kind === 'error' ? '!' : '◷'}</span><div><strong>${esc(title)}</strong>${description ? `<p>${esc(description)}</p>` : ''}</div>${action ? `<div class="actions">${action}</div>` : ''}</div>`;
}
function metric(label, value, foot, icon = '↗') { return `<div class="metric"><span class="metric-icon">${icon}</span><div class="metric-label">${esc(label)}</div><div class="metric-number">${esc(value)}</div><div class="metric-foot">${esc(foot)}</div></div>`; }
function empty(title, message = '') { return `<div class="empty"><strong>${esc(title)}</strong>${esc(message)}</div>`; }
function storeRecord(record, provider = '') { const key = `record-${++state.recordCount}`; state.records.set(key,{record,provider}); return key; }
function inspectButton(record, provider, label = 'Inspect') { return `<button class="button small" data-action="inspect" data-record="${storeRecord(record,provider)}">${esc(label)}</button>`; }
function primary(source) {
 const preferred = {sl:'results',rtv:'runs',atr:'sectionals',betfair:'bsp',bha:'reports',pedigree:'horses',weather:'races',fx:'rate_values'}[source.id];
 return {entry:source.tables?.[preferred],coverage:source.coverage?.[preferred] || {},table:preferred};
}
function sourceStatus(source) {
 if (!source.accepted_version) return badge('Not published','warning');
 if (source.required && source.check_stale) return badge('Check overdue','warning');
 if (source.adoption?.state === 'error') return badge('Prior version retained','warning');
 return badge('Accepted');
}
function healthTable(sources) {
 return `<div class="table-wrap"><table><thead><tr><th>Source</th><th>Last check</th><th>Accepted observations</th><th>Publication</th><th>Activity</th></tr></thead><tbody>${sources.map(s => {
 const p = primary(s), grain = p.entry?.contract?.grain || 'reference';
 const activity = s.worker?.stale ? badge('Heartbeat stale','error') : s.worker?.active ? badge(s.worker.status.state,'blue') : badge(pretty(s.attempt_state),'neutral');
 return `<tr><td><div class="source-cell">${sourceIcon(s.id)}<div><button class="link-button" data-nav="sources/${s.id}"><strong>${esc(s.label)}</strong></button><small>${s.required ? 'Required input' : 'Optional source'}</small></div></div></td><td><span title="${esc(s.checked_at)}">${esc(when(s.checked_at))}</span></td><td>${number(p.entry?.rows)}<div class="race-meta">${esc(grain)}</div></td><td>${sourceStatus(s)}</td><td>${activity}</td></tr>`;
 }).join('')}</tbody></table></div>`;
}
function workerCard(worker, controls = false) {
 const s = worker.status, done = (s.races?.data || 0)+(s.races?.data_partial || 0)+(s.races?.no_data || 0);
 const known = Math.max(0,(s.total_races || 0)-(s.native_ids?.missing || 0)), percent = known ? Math.min(100,done/known*100) : 0;
 const genericStats=worker.source==='bha'?[['Fixture reports retained',s.reports],['Documents pending',s.pending_documents]]:s.files?[['Published market files',s.files.published],['Pending market files',s.files.pending],['Files processed this batch',s.processed]]:[['Days collected',s.days],['Latest date',s.latest_date || s.end]];
 return `<div class="worker-card"><div class="worker-title"><div class="source-cell">${sourceIcon(worker.source)}<strong>${esc(worker.label)}</strong></div>${badge(worker.stale ? 'Stale' : pretty(s.state),worker.stale ? 'error' : worker.active ? 'blue' : 'neutral')}</div>${s.races ? `<div class="stat-pair"><span>Known-ID races resolved</span><strong>${number(done)} / ${number(known)}</strong></div><div class="progress" aria-label="Known-ID progress"><span style="width:${percent}%"></span></div><div class="stat-pair"><span>Waiting for native IDs</span><strong>${number(s.native_ids?.missing)}</strong></div><div class="stat-pair"><span>Runner rows saved</span><strong>${number(s.saved_rows)}</strong></div>` : genericStats.map(([label,value])=>`<div class="stat-pair"><span>${esc(label)}</span><strong>${typeof value==='string'?esc(value):number(value)}</strong></div>`).join('')}
 ${s.current ? `<hr class="subtle-divider"><div class="eyebrow">CURRENT RACE</div><strong>${esc(s.current.track)} · ${esc(s.current.off_time)}</strong><div class="race-meta">${dateLabel(s.current.race_date)}</div>` : ''}
 <div class="provenance">Checked ${esc(when(s.checked_at))}${s.pid ? ` · PID ${s.pid}` : ''}${worker.request_delay ? ` · ${worker.request_delay}s base request delay` : ''}</div>
 ${controls && worker.source === 'atr' ? `<div class="actions" style="margin-top:15px"><button class="button small" data-action="atr-log">View log</button>${worker.active ? '<button class="button small danger" data-action="stop-atr">Stop gracefully</button>' : '<button class="button small primary" data-action="resume-atr">Resume worker</button>'}</div>` : ''}</div>`;
}
async function overviewPage() {
 const o = state.overview, published = o.sources.filter(s=>s.accepted_version).length, active = o.workers.filter(w=>w.active);
 const readiness = o.readiness;
 $('#main').innerHTML = heading('Ingestion overview','A clear view of collected races, source evidence and the work still to do.','<button class="button primary" data-nav="jobs">Manage collection ↗</button>')+
 banner(readiness.ready ? 'Required source checks are ready' : 'Source checks need attention', readiness.ready ? `${readiness.limitations.length} optional source limitations are recorded. Review them before building features.` : readiness.blockers.join(' · '),' '+(readiness.ready ? 'good' : ''),'<button class="button small" data-nav="sources">Review sources →</button>')+
 `<div class="metric-grid">${metric('Canonical races',number(o.database.races),`${dateLabel(o.database.first_date)} — ${dateLabel(o.database.latest_date)}`,'≋')}${metric('Runner records',number(o.database.runners),'Stored canonical runner appearances','↗')}${metric('Sources published',`${published} / ${o.sources.length}`,'In the selected complete dataset','▱')}${metric('Active collectors',number(active.length),active.length ? active.map(w=>w.label).join(' · ') : 'No active worker heartbeat','◷')}</div>`+
 `<div class="two-column"><section class="panel"><div class="panel-head"><div><h2>Source health</h2><p>Collection and accepted data are tracked separately.</p></div><button class="link-button" data-nav="sources">View all →</button></div>${healthTable(o.sources)}<div class="table-foot"><span>${o.dataset.id ? `Accepted dataset ${o.dataset.id.slice(0,12)}…` : 'No complete dataset has been published'}</span><span>${dateLabel(o.dataset.created_at)}</span></div></section><div><section class="panel"><div class="panel-head"><div><h2>Collection activity</h2><p>Progress from the source workers.</p></div></div>${active.length ? active.map(w=>workerCard(w)).join('<hr class="subtle-divider">') : empty('No active collectors','Check Jobs for the last recorded worker state.')}<div class="table-foot"><button class="link-button" data-nav="jobs">Open jobs &amp; logs →</button></div></section><section class="panel"><div class="panel-head"><h2>Start an inspection</h2></div><div class="panel-body"><p class="hint">Follow a race from its original source through identity checks and accepted observations.</p><div class="actions"><button class="button" data-nav="races">Browse races →</button><button class="button" data-nav="coverage">Explore coverage</button></div></div></section></div></div>`;
}
function raceTable(rows) {
 if (!rows.length) return empty('No races in this selection','Change the date range or search terms. Missing dates are not filled with generated data.');
 return `<div class="table-wrap"><table><thead><tr><th>Date / off</th><th>Race</th><th>Conditions</th><th>Runners</th><th></th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.off_time || '—')}<div class="race-meta">${dateLabel(r.race_date)} · UTC</div></td><td><button class="link-button race-name" data-nav="race/${esc(r.race_id)}">${esc(r.track)}</button><div class="race-meta">${esc(r.race_name || 'Race '+r.race_id)}</div></td><td>${number(r.distance_furlongs)}f · ${esc(r.race_type || 'Type unknown')}<div class="race-meta">${esc(r.going || 'Going unknown')} · ${esc(r.surface || 'Surface unknown')}</div></td><td>${badge(`${r.stored_runners} / ${r.expected_runners ?? '?'}`,r.stored_runners === r.expected_runners ? '' : 'warning')}</td><td><button class="button small" data-nav="race/${esc(r.race_id)}">Inspect →</button></td></tr>`).join('')}</tbody></table></div>`;
}
async function racesPage() {
 const result = await api('races',{query:state.query,offset:state.offset});
 $('#main').innerHTML = heading('Race explorer','Find a race, inspect its field and compare the evidence behind each value.')+
 `<section class="panel"><form class="search-form" data-form="search"><div class="field grow"><label for="race-search">Search races &amp; runners</label><input id="race-search" type="text" placeholder="Course, horse, race name or canonical race ID" value="${esc(state.query)}" maxlength="120"></div><button class="button primary">Search races</button><button type="button" class="button" data-action="clear-search">Clear</button></form>${raceTable(result.rows)}<div class="table-foot"><span>${number(result.total)} matching races · showing ${result.rows.length ? state.offset+1 : 0}–${state.offset+result.rows.length}</span><div class="actions"><button class="button small" data-action="previous" ${state.offset ? '' : 'disabled'}>← Previous</button><button class="button small" data-action="next" ${state.offset+result.rows.length < result.total ? '' : 'disabled'}>Next →</button></div></div></section>`;
}
function sourcePicker(action = 'race-source') { return `<div class="source-picker">${Object.entries(sourceLabels).map(([id,label])=>`<button data-action="${action}" data-source="${id}" class="${state.source===id ? 'active' : ''}">${esc(label)}</button>`).join('')}</div>`; }
function layerPicker() { return `<div class="segmented" aria-label="Observation layer"><button data-action="layer" data-layer="accepted" class="${state.layer==='accepted'?'active':''}">Accepted dataset</button><button data-action="layer" data-layer="native" class="${state.layer==='native'?'active':''}">Latest native</button></div>`; }
function dataTable(rows, provider, name) {
 if (!rows.length) return empty('No observations','This source/table has no attached rows in the selected layer.');
 const layouts={
  comparison:['horse_name','field','canonical','source_value','comparison','source_table','observed_at'],
  changes:['horse_name','field','original','prepared'],
  reports:['race_date','course','report_key','presence_status','enquiry_text','runner_notes_text','observed_at'],
  race_notes:['race_number','off_time_local','text','report_key'],
  unassigned_race_notes:['race_number','off_time_local','text','report_key'],
  documents:['report_key','state','pages','text','observed_at'],
  revisions:['report_key','observed_at','record_hash','source_kind'],
  runs:['horse_id','race_date','fsp','top_speed','avg_length','avg_frequency','match_status','available_at'],
  cards:['horse_id','race_date','rtv_card_tf_rating','rtv_card_tf_stars','rtv_card_tf_tipped','match_status'],
  pre:['horse_key','observed_at','rtv_pre_timeform_rating','rtv_pre_pace_position','match_status','availability_basis'],
  text:['horse_id','race_date','tf_comment','tf_verdict','match_status'],
  scores:['horse_id','race_date','rtv_text_logit','rtv_verdict_rank','match_status'],
  horses:['horse_name','foaled','foaling_date_source','sire','dam','damsire','rtv_country_of_foaling','rtv_coat_colour'],
 };
 const preferred = layouts[name] || ['horse_name','horse_id','sl_horse_id','race_date','finish_position','official_rating','atr_horse_id','atr_finishing_speed_pct','bsp','place_bsp','match_status','match_method','observed_at','available_at','availability_basis'];
 let columns = preferred.filter(c => rows.some(r => Object.hasOwn(r,c)));
 if (!columns.length) columns = Object.keys(rows[0]).filter(c=>!c.startsWith('_')).slice(0,7);
 columns = columns.slice(0,8);
 const headings = {foaled:'Foaling date',foaling_date_source:'Date source',rtv_country_of_foaling:'Country of foaling',rtv_coat_colour:'Coat colour'};
 const dateSources = {racing_tv:'Racing TV',sporting_life:'Sporting Life',conflict:'Conflicting dates'};
 const cellText = (row, column) => valueText(column==='foaling_date_source' ? dateSources[row[column]] || row[column] : row[column]);
 return `<div class="table-wrap"><table class="record-table"><thead><tr>${columns.map(c=>`<th>${esc(headings[c] || pretty(c))}</th>`).join('')}<th>Full record</th></tr></thead><tbody>${rows.map(r=>`<tr>${columns.map(c=>`<td class="data-cell" title="${esc(cellText(r,c).slice(0,1000))}">${r[c] == null ? '<span class="muted">Unknown</span>' : esc(cellText(r,c).slice(0,100))}</td>`).join('')}<td class="inspect-cell">${inspectButton(r,provider)}</td></tr>`).join('')}</tbody></table></div>`;
}
function sourceTables(result) {
 const entries = Object.entries(result.tables || {});
 return (result.message ? banner(result.message,'','') : '')+(entries.length ? entries.map(([name, table],index)=>`<details class="panel" ${index===0 || result.source==='bha' ? 'open' : ''}><summary class="panel-head"><div class="source-table-title"><h2 class="no-margin">${esc(pretty(name))}</h2><small>${number(table.count)} observations${table.truncated ? ' · preview limited to 500' : ''}</small></div>${badge(result.layer==='native' ? 'Native' : 'Accepted',result.layer==='native'?'blue':'')}</summary>${dataTable(table.rows,result.source,name)}</details>`).join('') : empty('No observations in this layer','Check source health and the selected publication. The app does not substitute a different version.'));
}
async function racePage(id) {
 const [race,review] = await Promise.all([api(`races/${encodeURIComponent(id)}`),api('review')]);
 state.race = race; state.review = review; state.changes = [];
 const r = race.header, note = review.notes.find(n=>n.race_id===id);
 $('#main').innerHTML = `<a href="#races" class="back-link">← Back to race explorer</a><section class="race-header"><div class="section-toolbar"><div class="chip-stack">${badge(r.race_type || 'Type unknown','neutral')}${badge(`Race ${r.race_id}`,'neutral')}${race.findings.some(f=>f.severity==='error')?badge('Check field','warning'):badge('Field structure checked')}</div><div class="actions"><button class="button bookmark" data-action="bookmark" title="Bookmark race" aria-label="Bookmark race">${note?.bookmarked ? '★' : '☆'}</button><a class="button small" href="${url(`races/${id}/export`,{format:'csv'})}">Export CSV</a><a class="button small" href="${url(`races/${id}/export`)}">Evidence pack</a></div></div><h1>${esc(r.track)} · ${esc(r.off_time_local)}</h1><div class="race-subheading">${dateLabel(r.race_date)} · ${esc(r.race_name || '')}</div><dl class="race-facts">${[['Distance',`${r.distance_furlongs ?? '?'} furlongs`],['Going',r.going],['Surface',r.surface],['Field',`${race.rows.length} / ${r.num_runners ?? '?'} runners`],['Prize',r.prize_money==null?'Unknown':number(r.prize_money)+' · source units'],['Stored off-time',(r.off_time || 'Unknown')+' UTC']].map(([k,v])=>`<div><dt>${esc(k)}</dt><dd>${esc(v || 'Unknown')}</dd></div>`).join('')}</dl></section>
 <div class="tabs" aria-label="Race inspection tabs">${Object.entries({runners:'Runners',evidence:'Source evidence',prepared:'Prepared inputs',features:'Features',changes:'Publication changes',review:'Review & notes'}).map(([key,label])=>`<button data-action="race-tab" data-tab="${key}" class="${state.raceTab===key?'active':''}">${label}</button>`).join('')}</div><div id="race-content"></div>`;
 await raceContent();
}
async function raceContent() {
 const race=state.race, target=$('#race-content'); if (!target) return;
 $$('.tabs button').forEach(b=>b.classList.toggle('active',b.dataset.tab===state.raceTab));
 if (state.raceTab==='runners') {
  target.innerHTML = race.findings.map(f=>banner(f.message,'',f.severity==='error'?'error':'')).join('')+`<section class="panel"><div class="panel-head"><div><h2>Canonical runner field</h2><p>Stored race records. Open a runner to inspect every available field.</p></div>${badge(`${race.rows.length} runners`,'neutral')}</div><div class="table-wrap"><table><thead><tr><th>Place</th><th>Runner</th><th>Jockey / trainer</th><th>Rating</th><th>Weight</th><th>Source odds</th><th></th></tr></thead><tbody>${race.rows.map(r=>`<tr><td>${esc(r.finish_pos_label || r.finish_position || 'Unknown')}</td><td><strong>${esc(r.horse_name)}</strong><div class="race-meta">ID ${esc(r.horse_id ?? 'Unknown')} · ${r.age ?? '?'}yo · ${esc(r.sex || 'Sex unknown')}</div></td><td>${esc(r.jockey || 'Unknown')}<div class="race-meta">${esc(r.trainer || 'Unknown')}</div></td><td>${number(r.official_rating)}</td><td>${number(r.weight_lbs)} lb</td><td>${number(r.odds)}</td><td>${inspectButton(r,'sl','View runner')}</td></tr>`).join('')}</tbody></table></div></section>`;
 } else if (state.raceTab==='features') {
  await featureUI.raceInTab(race.header.race_id,target);
 } else if (state.raceTab==='evidence') {
  target.innerHTML=sourcePicker()+`<div class="section-toolbar"><div><h2>${esc(sourceLabels[state.source])}</h2><div class="hint no-margin">Native evidence and immutable accepted observations remain separate.</div></div>${layerPicker()}</div><div id="source-content" class="loading"><span class="spinner"></span>Reading source observations…</div>`;
  await loadRaceSource();
 } else if(state.raceTab==='prepared') {
  target.innerHTML=banner('Preview the input to feature building','Runs the existing cleaning and preparation functions for this accepted race. It does not build features or write data.')+'<button class="button primary bottom-space" data-action="prepare">Prepare this race</button><div id="prepared-result"></div>';
 } else if(state.raceTab==='changes') {
  const sets=(await api('datasets')).rows;
  target.innerHTML=`<section class="panel"><div class="panel-head"><div><h2>Compare accepted publications</h2><p>See changes in this race’s source values, identities and timing receipts.</p></div></div><form class="panel-body" data-form="compare"><div class="form-grid"><div class="field"><label for="compare-before">Earlier dataset</label><select id="compare-before">${sets.map((s,i)=>`<option value="${s.id}" ${i===1?'selected':''}>${dateLabel(s.created_at)} · ${s.id.slice(0,12)}</option>`).join('')}</select></div><div class="field"><label for="compare-source">Source</label><select id="compare-source">${Object.entries(sourceLabels).map(([id,label])=>`<option value="${id}" ${id===state.source?'selected':''}>${label}</option>`).join('')}</select></div></div><p class="hint">Compare against the dataset currently selected at the top of the app.</p><button class="button primary" ${sets.length?'':'disabled'}>Compare publications</button></form></section><div id="compare-result"></div>`;
 } else if(state.raceTab==='review') renderRaceReview();
}
const $$ = (q, root=document) => [...root.querySelectorAll(q)];
async function loadRaceSource() {
 const ticket=state.sourceTicket=(state.sourceTicket || 0)+1, source=state.source;
 try {
  const result=await api(`races/${state.race.header.race_id}/source`,{source,layer:state.layer});
  if(ticket===state.sourceTicket && $('#source-content')) {
   state.sourceResult=result;
   $('#source-content').innerHTML=(['sl','atr','rtv','betfair'].includes(source)?'<button class="button bottom-space" data-action="compare-canonical">Compare matching fields with canonical records</button>':'')+sourceTables(result);
  }
 } catch(e) {if(ticket===state.sourceTicket && $('#source-content') && e.name!=='AbortError') $('#source-content').innerHTML=banner(e.message,'','error');}
}
function renderRaceReview() {
 const note=state.review.notes.find(n=>n.race_id===state.race.header.race_id);
 $('#race-content').innerHTML=`<div class="two-column"><div><section class="panel"><div class="panel-head"><h2>Inspection notes</h2></div><form class="panel-body" data-form="note"><div class="field"><label for="race-note">What did you find?</label><textarea id="race-note" maxlength="10000" placeholder="Record the evidence, uncertainty or follow-up needed…">${esc(note?.note || '')}</textarea></div><label class="hint"><input type="checkbox" id="race-bookmarked" ${note?.bookmarked?'checked':''}> Bookmark this race</label><div style="margin-top:16px"><button class="button primary">Save review note</button></div></form></section><section class="panel"><div class="panel-head"><div><h2>Propose a correction</h2><p>A saved, validated proposal for review by the existing repair owner.</p></div></div><div class="panel-body"><div class="form-grid"><div class="field"><label for="proposal-field">Field</label><select id="proposal-field">${['going','surface','distance_furlongs','prize_money','race_class','race_type','jockey','trainer','weight_lbs','draw','official_rating','age','headgear','finish_position','won','finish_pos_label','lengths_behind'].map(f=>`<option value="${f}">${pretty(f)}</option>`).join('')}</select></div><div class="field"><label for="proposal-runner">Runner (runner fields only)</label><select id="proposal-runner">${state.race.rows.map(r=>`<option value="${esc(r.horse_id)}">${esc(r.horse_name)} · ${esc(r.horse_id)}</option>`).join('')}</select></div><div class="field"><label for="proposal-value">Proposed value</label><input id="proposal-value" type="text" placeholder="New value; use null for unknown"></div><div class="field"><label>&nbsp;</label><button class="button" data-action="add-change">Add field change</button></div></div><div id="proposal-changes"></div><form data-form="proposal"><div class="field" style="margin-top:18px"><label for="proposal-reason">Reason</label><textarea id="proposal-reason" required placeholder="Explain why the stored value is incorrect."></textarea></div><div class="field" style="margin-top:14px"><label for="proposal-evidence">Source evidence</label><input id="proposal-evidence" type="text" required placeholder="Source URL, retained-response hash or report reference"></div><p class="hint">Race-level fields apply to the whole field. Result corrections must preserve consistent positions and winner labels. This creates a proposal; it does not edit canonical data.</p><button class="button primary">Validate &amp; save proposal</button></form><div id="proposal-result"></div></div></section></div><section class="panel"><div class="panel-head"><h2>Review principles</h2></div><div class="panel-body"><ul class="check-list"><li>Retain the original source evidence.</li><li>Keep provider and canonical identities separate.</li><li>Distinguish unknown from confirmed absence.</li><li>Review the complete field when changing results.</li><li>Apply repairs through their owner, then publish a new dataset.</li></ul></div></section></div>`;
}
async function sourcesPage(selected) {
 if(selected && sourceLabels[selected]) state.source=selected;
 const o=state.overview,s=o.sources.find(s=>s.id===state.source),p=primary(s),c=p.coverage;
 $('#main').innerHTML=heading('Source health','Inspect collection outcomes, publication, identity quality and source-specific evidence.')+sourcePicker('source-profile')+
 `<div class="section-toolbar"><div class="source-cell">${sourceIcon(s.id)}<div><h2 class="no-margin">${esc(s.label)}</h2><div class="hint no-margin">${s.required?'Required input':'Optional source'} · ${s.publication_lag_days} day expected publication lag</div></div></div>${sourceStatus(s)}</div>`+
 `<div class="source-profile">${metric('Accepted observations',number(p.entry?.rows),p.entry?.contract?.grain || 'Source-native grain')}${['runner','runner snapshot','canonical runner'].includes(p.entry?.contract?.grain)?metric('Linked runners',number(c.linked_runners),c.eligible_runners?`${number(c.eligible_runners)} eligible canonical runners`:'No canonical runner denominator'):metric('Identity grain',s.id==='bha'?'Fixture':s.id==='pedigree'?'Horse':'Reference','No canonical runner attribution is asserted')}${metric('Usable observations',number(c.usable_rows),c.latest_usable_event?`Latest event ${dateLabel(c.latest_usable_event)}`:'Availability not established')}</div>`+
 (s.adoption?.state==='error'?banner('Latest candidate was not accepted',s.adoption.error || 'The previous accepted version is retained.'):s.check_stale?banner('The collection check is overdue','Stored observations remain inspectable. A refresh is needed to establish current coverage.'):banner('Collection and publication are separate',`Last collection check: ${when(s.checked_at)}. Accepted publication: ${dateLabel(s.published_at)}.`,s.accepted_version?'good':''))+
 `<div class="two-column"><section class="panel"><div class="panel-head"><h2>Accepted tables &amp; contracts</h2></div><div class="table-wrap"><table><thead><tr><th>Table</th><th>Rows</th><th>Grain / role</th><th>Declared identity</th><th></th></tr></thead><tbody>${Object.entries(s.tables || {}).filter(([name])=>!name.startsWith('links_')&&!name.startsWith('observations_')).map(([name,e])=>`<tr><td>${esc(name)}</td><td>${number(e.rows)}</td><td>${esc(e.contract?.grain || 'Reference')}<div class="race-meta">${esc(e.contract?.role || 'history')}</div></td><td class="mono">${esc((e.contract?.keys || []).join(', ') || 'See source receipt')}</td><td>${inspectButton({table:name,...e},s.id,'Schema')}</td></tr>`).join('') || '<tr><td colspan="5">No accepted version in this dataset.</td></tr>'}</tbody></table></div><div class="table-foot mono">${esc(s.accepted_version || 'Not published')}</div></section><div><section class="panel"><div class="panel-head"><h2>Identity &amp; missingness</h2></div><div class="panel-body">${Object.keys(c.link_reasons || {}).length?Object.entries(c.link_reasons).map(([reason,count])=>`<div class="stat-pair"><span>${esc(pretty(reason))}</span><strong>${number(count)}</strong></div>`).join(''):'<p class="hint">This table has no runner crosswalk. Its source-native grain is preserved.</p>'}<hr class="subtle-divider">${Object.entries(c.missing_values_by_column || {}).slice(0,10).map(([name,count])=>`<div class="stat-pair"><span>${esc(pretty(name))}</span><strong>${number(count)}</strong></div>`).join('') || '<p class="hint">No missing-field summary is recorded for this table.</p>'}</div></section>${s.worker?`<section class="panel"><div class="panel-head"><h2>Source worker</h2></div>${workerCard(s.worker,true)}</section>`:''}</div></div><details class="panel"><summary class="panel-head"><h2>Source provenance</h2></summary><div class="panel-body"><pre class="mono">${esc(JSON.stringify(s.provenance,null,2))}</pre></div></details>${s.id==='bha'?`<div class="section-toolbar"><div><h2>Stewards’ fixture reports</h2><p class="hint no-margin">Enquiries, runner notes, fixture comments, incidents and PDF race notes.</p></div>${layerPicker()}</div><div id="bha-fixtures" class="loading">Reading retained BHA reports…</div>`:''}`;
 const foaling=s.provenance?.foaling_coverage;
 if(s.id==='pedigree' && foaling) $('.source-profile').insertAdjacentHTML('afterend',
  `<div class="source-profile">${metric('Racing TV foaling dates',number(foaling.racing_tv),`${number(foaling.recovered_from_january_1)} recovered from 1 January`)}${metric('Sporting Life dates',number(foaling.sporting_life),'Full dates from Sporting Life')}${metric('Unresolved dates',number(foaling.unresolved+foaling.conflict),`${number(foaling.conflict)} conflicting source dates`)}</div>`);
 if(s.id==='bha') {try {const r=await api('bha/fixtures',{layer:state.layer});if($('#bha-fixtures')) $('#bha-fixtures').innerHTML=sourceTables(r);}catch(e){if($('#bha-fixtures'))$('#bha-fixtures').innerHTML=banner(e.message,'','error');}}
}
async function coveragePage() {
 const result=await api('coverage');
 $('#main').innerHTML=heading('Coverage explorer','Follow the gaps from a source and date straight into the affected races.')+banner('Every source keeps its own grain',result.message)+`<section class="panel"><div class="panel-head"><div><h2>Accepted observations by day</h2><p>Click a cell to inspect that date’s canonical races.</p></div>${badge(`${result.days.length} days`,'neutral')}</div><div class="table-wrap"><table class="coverage-table"><thead><tr><th>Source / grain</th>${result.days.map(d=>`<th title="${d}">${d.slice(8)}<br>${new Date(d+'T12:00:00').toLocaleDateString('en-GB',{month:'short'})}</th>`).join('')}</tr></thead><tbody>${result.sources.map(s=>`<tr><td><button class="link-button" data-nav="sources/${s.source}">${esc(s.label)}</button><div class="race-meta">${esc(s.grain)}</div></td>${s.cells.map(c=>`<td><button class="heat-cell ${c.state}" data-action="coverage-day" data-day="${c.day}" data-source="${s.source}" aria-label="${esc(s.label)} ${c.day}: ${c.count} observations; ${pretty(c.state)}" title="${esc(`${s.label} · ${c.day}\n${c.count} ${s.grain} observations\n${pretty(c.state)}\nCanonical: ${c.canonical.races} races / ${c.canonical.runners} runners`)}">${c.state==='observed' ? number(c.count) : c.state==='awaiting_publication'?'◷':c.state==='reference'?'·':'—'}</button></td>`).join('')}</tr>`).join('')}</tbody></table></div><div class="legend"><span><i style="background:#b7d7a4"></i>Observations present</span><span><i style="background:#f8e8be"></i>Unverified gap</span><span><i style="background:#e1edf6"></i>Publication window</span><span><i style="background:#eef2ea"></i>No canonical races / reference</span></div></section>`;
 for(const source of result.sources.filter(s=>s.error))$('#main').insertAdjacentHTML('beforeend',banner(`${source.label} coverage could not be read`,source.error,'error'));
}
function jobProgress(job) {
 const coverage=job.result?.foaling_coverage;
 if(job.kind==='foaling_backfill' && coverage){
  return `${number(coverage.recovered_from_january_1)} dates recovered · ${number(coverage.unresolved+coverage.conflict)} unresolved`;
 }
 if(job.progress) return [job.progress.message || job.progress.source,job.progress.phase || job.progress.state].filter(Boolean).join(' · ');
 return '';
}
function modellingJobLink(job) {
 if(job.result?.benchmark_id) return `<a class="button small" href="#modelling/benchmarks/${esc(job.result.benchmark_id)}">Open benchmark</a>`;
 if(job.result?.run_id) return `<a class="button small" href="#modelling/runs/${esc(job.result.run_id)}">Open fitted run</a>`;
 return '';
}
async function jobsPage() {
 const result=await api('jobs');
 const busySources=new Set(result.workers.filter(w=>w.active || w.process_alive && ['running','waiting'].includes(w.status.state)).map(w=>w.source));
 $('#main').innerHTML=heading('Project jobs','Follow source collection, feature builds and modelling jobs.')+
 `<div class="two-column"><div><section class="panel"><div class="panel-head"><div><h2>Refresh a date range</h2><p>Uses the dates selected above and the existing source coordinator.</p></div></div><form class="panel-body" data-form="refresh"><div class="check-grid">${['sl','rtv','atr','betfair','fx','bha'].map(id=>`<label><input name="source" type="checkbox" value="${id}" ${busySources.has(id)?'disabled':'checked'}>${esc(sourceLabels[id])}</label>`).join('')}</div>${busySources.size?`<p class="hint">Active collectors are reserved: ${[...busySources].map(id=>esc(sourceLabels[id])).join(', ')}. Their existing jobs continue independently.</p>`:''}<label class="hint"><input type="checkbox" id="recheck"> Recheck already collected records in this range</label><p class="hint">BHA stewards is included when its collector is free. Historical ranges are limited to 31 settled days per job; provider publication delays and cooldowns are retained.</p><button class="button primary">Queue source refresh</button></form></section><section class="panel"><div class="panel-head"><div><h2>Local data operations</h2><p>Long operations continue independently of this page.</p></div></div><div class="panel-body"><div class="actions"><button class="button" data-action="queue" data-kind="publish">Publish accepted sources</button><button class="button" data-action="queue" data-kind="verify">Verify publication hashes</button><button class="button" data-action="queue" data-kind="validate">Check race fields</button><button class="button" data-action="queue" data-kind="replay">Replay retained SL responses</button></div><p class="hint">Publication adopts local stores and promotes qualified ATR observations. It does not download data. Replay reparses retained Sporting Life responses through the existing ingestion path.</p></div></section><section class="panel"><div class="panel-head"><h2>Workbench job history</h2>${badge(result.rows.length,'neutral')}</div>${result.rows.length?result.rows.map(j=>`<div class="job-row"><header><strong>${esc(pretty(j.kind))}</strong>${badge(pretty(j.state),j.state==='failed'?'error':j.state==='partial'?'warning':j.state==='running'?'blue':'')}</header><div class="race-meta">${esc(when(j.created_at))} · ${esc(j.params?.start || '')}${j.params?.end?' → '+esc(j.params.end):''}</div>${j.error?`<p>${esc(j.error)}</p>`:''}${jobProgress(j)?`<p>${esc(jobProgress(j))}</p>`:''}<div class="actions"><button class="button small" data-action="job-log" data-job="${j.id}">View log</button>${modellingJobLink(j)}${inspectButton(j,'','Receipt')}</div></div>`).join(''):empty('No workbench jobs yet','Existing source workers are shown separately. Page loads never queue operations.')}</section></div><div>${result.workers.map(w=>`<section class="panel"><div class="panel-head"><h2>${esc(w.label)} worker</h2></div>${workerCard(w,true)}</section>`).join('')}</div></div>`;
 $('#recheck').closest('label').insertAdjacentHTML('beforebegin','<label class="hint checkbox-line"><input type="checkbox" id="enrich"> Enrich pedigree and foaling dates for collected runners</label><label class="hint checkbox-line"><input type="checkbox" id="weather"> Collect weather observations (research)</label>');
 const profileButton=$('[data-kind="publish"]');
 const profileRunning=result.rows.some(j=>['refresh','replay','publish','foaling_backfill'].includes(j.kind) && ['queued','running'].includes(j.state));
 profileButton.insertAdjacentHTML('afterend',`<button class="button" data-action="queue" data-kind="foaling_backfill" ${profileRunning?'disabled':''}>Backfill foaling dates</button>`);
 profileButton.closest('.panel-body').insertAdjacentHTML('beforeend','<p class="hint">The foaling-date backfill looks up Racing TV profiles for unresolved horses across the archive. Completed profile lookups are reused when a job is rerun.</p>');
}
async function reviewPage() {
 const [issues,review]=await Promise.all([api('issues'),api('review')]);
 const decisions=Object.fromEntries(review.decisions.map(d=>[d.issue_id,d]));
 $('#main').innerHTML=heading('Review queue','Investigate data problems, keep useful notes and retain a clear correction trail.')+
 `<div class="metric-grid">${metric('Findings in this range',number(issues.rows.length),'Canonical checks and ATR queue issues')}${metric('Fields checked',number(issues.scanned_races),'Bounded to 1,000 races per inspection')}${metric('Bookmarked races',number(review.notes.filter(n=>n.bookmarked).length),'Saved across workbench sessions')}${metric('Correction proposals',number(review.proposals.length),'Saved for review; source data unchanged')}</div>`+
 `<section class="panel"><div class="panel-head"><div><h2>Findings to review</h2><p>${esc(issues.message)}</p></div></div>${issues.rows.length?issues.rows.map(issue=>`<div class="review-row"><div><div class="chip-stack">${badge(issue.source.toUpperCase(),'neutral')}${badge(pretty(issue.code),issue.severity==='error'?'error':'warning')}</div><h3 style="margin-top:8px"><button class="link-button" data-nav="race/${esc(issue.race_id)}">${esc(issue.track)} · ${dateLabel(issue.race_date)} · Race ${esc(issue.race_id)}</button></h3><p>${esc(issue.message)}</p></div><select class="review-status" data-decision="${issue.id}" aria-label="Review state for ${esc(issue.race_id)}">${['open','acknowledged','investigating'].map(s=>`<option value="${s}" ${(decisions[issue.id]?.state || 'open')===s?'selected':''}>${pretty(s)}</option>`).join('')}</select></div>`).join(''):empty('No findings in this bounded check','This is a targeted structural check, not a guarantee that every source value is correct.')}</section><div class="two-column"><section class="panel"><div class="panel-head"><h2>Bookmarked races &amp; notes</h2></div>${review.notes.length?review.notes.map(n=>`<div class="review-row"><div><h3><button class="link-button" data-nav="race/${esc(n.race_id)}">${n.bookmarked?'★ ':''}Race ${esc(n.race_id)}</button></h3><p>${esc(n.note || 'Bookmarked for inspection')}</p><div class="race-meta">${esc(when(n.updated_at))}</div></div></div>`).join(''):empty('Keep a race close','Bookmark a race or save a note from its inspection screen.')}</section><section class="panel"><div class="panel-head"><h2>Correction proposals</h2></div>${review.proposals.length?review.proposals.map(p=>`<div class="job-row"><header><strong>Race ${esc(p.race_id)}</strong>${badge(p.status,'neutral')}</header><div class="race-meta">${esc(when(p.created_at))}</div><a class="button small" style="margin-top:10px" href="/api/review/proposals/${p.id}">Download reviewed plan</a></div>`).join(''):empty('No correction proposals','Create a proposal from a race’s Review & notes tab.')}</section></div>`;
}
function safeURL(value) {try {const u=new URL(value);return u.protocol==='https:'?u.href:null;}catch{return null;}}
function parseArray(value) {if(typeof value==='string'){try{return JSON.parse(value);}catch{return null;}}return value;}
function chart(values, label, unit, markers) {
 if(values.some(v=>v==null || typeof v==='boolean' || typeof v==='object' || typeof v==='string' && !v.trim()))return '';
 const points=values.map(Number);if(points.length<2||points.some(n=>!Number.isFinite(n)))return '';
 if(!Array.isArray(markers)||markers.length!==points.length)markers=null;
 const lo=Math.min(...points),hi=Math.max(...points),span=hi-lo||1;
 const coords=points.map((n,i)=>[18+i/(points.length-1)*614,140-(n-lo)/span*110]);
 const d=coords.map(([x,y],i)=>`${i?'L':'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
 return `<section class="chart"><h3>${esc(label)} <span class="muted">· ${esc(unit)}</span></h3><svg viewBox="0 0 650 170" role="img" aria-label="${esc(label)} in ${esc(unit)}"><path d="M18 30H632M18 85H632M18 140H632" stroke="#e1e8d9" fill="none"/><path d="${d} L632,150 L18,150 Z" fill="#d9e9ca" opacity=".65"/><path d="${d}" stroke="#648a4b" stroke-width="2.5" fill="none"/>${coords.map(([x,y],i)=>`<circle cx="${x}" cy="${y}" r="3" fill="#527c3c"><title>${esc(markers?.[i] || `Observation ${i+1}`)}: ${points[i]} ${esc(unit)}</title></circle>`).join('')}</svg><div class="chart-axis"><span>${esc(markers?.[0] || 'Observation 1')}</span><span>${number(lo)}–${number(hi)} ${esc(unit)}</span><span>${esc(markers?.at(-1) || `Observation ${points.length}`)}</span></div></section>`;
}
function inspectRecord(record,provider='') {
 $('#inspector-title').textContent=record.horse_name || record.course || record.report_key || 'Source record';
 $('#inspector-eyebrow').textContent=provider ? `${sourceLabels[provider]} · full record` : 'RECORDED EVIDENCE';
 const metrics={atr_split_times:['Sectional times','seconds'],atr_running_totals:['Cumulative time','seconds'],atr_sectional_speeds:['Sectional speed','mph'],atr_stride_length:['Stride length','feet'],atr_stride_frequency:['Stride frequency','strides/second']};
 const markerFields={atr_split_times:'atr_section_markers',atr_running_totals:'atr_section_markers',atr_sectional_speeds:'atr_sectional_speed_labels',atr_stride_length:'atr_stride_markers',atr_stride_frequency:'atr_stride_frequency_markers'};
 let charts='';for(const [field,[label,unit]] of Object.entries(metrics)){const values=parseArray(record[field]);if(Array.isArray(values))charts+=chart(values,label,unit,parseArray(record[markerFields[field]]));}
 const payloads=parseArray(record.atr_tab_manifest || record.atr_source_payloads) || [];
 const evidence=[];
 if(record.source_sha256)evidence.push({sha256:record.source_sha256,kind:'Retained source'});
 if(record.sha256)evidence.push({sha256:record.sha256,kind:'Retained document'});
 if(Array.isArray(payloads))evidence.push(...payloads);
 const links=evidence.filter(e=>/^[a-f0-9]{64}$/.test(e.sha256 || '')).map(e=>`<a class="button small" href="/api/evidence?source=${provider}&amp;sha=${e.sha256}">${esc(pretty(e.kind || 'Source'))} evidence ↓</a>`).join('');
 const external=safeURL(record.source_url || record.rtv_profile_url || record.pdf_report_url || record.url);
 const tabCoverage=Array.isArray(payloads)&&payloads.length?`<div class="chip-stack bottom-space">${payloads.map(p=>badge(`${pretty(p.kind)}: ${p.state==='empty'?'confirmed empty':pretty(p.state || 'unknown')}`,p.state==='data'?'':p.state==='empty'?'neutral':'warning')).join('')}</div>`:'';
 $('#inspector-body').innerHTML=tabCoverage+charts+`<div class="actions bottom-space">${links}${external?`<a class="button small" href="${esc(external)}" target="_blank" rel="noopener noreferrer">Open original source ↗</a>`:''}<button class="button small" data-action="copy-record" data-record="${storeRecord(record,provider)}">Copy JSON</button></div><dl class="property-grid">${Object.entries(record).map(([key,value])=>`<dt>${esc(pretty(key))}</dt><dd>${esc(valueText(value))}</dd>`).join('')}</dl>`;
 if(!$('#inspector').open)$('#inspector').showModal();
}
async function queue(kind,params={}) {
 const result=await api('jobs',{}, {kind,params:{start:$('#start-date').value,end:$('#end-date').value,dataset:state.boundDataset,...params}});
 const label=({model_benchmark:'Benchmark',model_fit:'Candidate fit'})[kind] || pretty(kind);
 toast(`${label} queued. Follow its progress in Jobs.`);
 if(state.page==='jobs')await render(false);else location.hash='jobs';
 return result;
}
document.addEventListener('click',async event=>{
 const nav=event.target.closest('[data-nav]');if(nav){location.hash=nav.dataset.nav;return;}
 const button=event.target.closest('[data-action]');if(!button)return;
 const action=button.dataset.action;
 button.disabled=true;
 try {
  if(action==='inspect'){const r=state.records.get(button.dataset.record);inspectRecord(r.record,r.provider);}
  else if(action==='copy-record'){await navigator.clipboard.writeText(JSON.stringify(state.records.get(button.dataset.record).record,null,2));toast('Record JSON copied.');}
  else if(action==='clear-search'){state.query='';state.offset=0;await racesPage();}
  else if(action==='previous'||action==='next'){state.offset=Math.max(0,state.offset+(action==='next'?50:-50));await racesPage();}
  else if(action==='race-tab'){state.raceTab=button.dataset.tab;await raceContent();}
  else if(action==='race-source'){state.source=button.dataset.source;await raceContent();}
  else if(action==='source-profile'){state.source=button.dataset.source;location.hash=`sources/${state.source}`;}
  else if(action==='layer'){state.layer=button.dataset.layer;if(state.page==='race')await raceContent();else await sourcesPage(state.source);}
  else if(action==='prepare'){$('#prepared-result').innerHTML='<div class="loading"><span class="spinner"></span>Preparing the accepted race…</div>';const r=await api(`races/${state.race.header.race_id}/prepared`);$('#prepared-result').innerHTML=banner(`${r.prepared_count} of ${r.original_count} runners prepared`,r.message,'good')+`<section class="panel"><div class="panel-head"><h2>Value changes during preparation</h2></div>${dataTable(r.changes,'sl','changes')}</section><details class="panel"><summary class="panel-head"><h2>Prepared runner records</h2></summary>${dataTable(r.rows,'sl','prepared')}</details>`;}
  else if(action==='bookmark'){const id=state.race.header.race_id,n=state.review.notes.find(n=>n.race_id===id);await api('review/note',{}, {race_id:id,note:n?.note || '',bookmarked:!n?.bookmarked});state.review=await api('review');button.textContent=!n?.bookmarked?'★':'☆';toast(!n?.bookmarked?'Race bookmarked.':'Bookmark removed.');}
  else if(action==='coverage-day'){$('#start-date').value=button.dataset.day;$('#end-date').value=button.dataset.day;state.source=button.dataset.source;state.query='';state.offset=0;location.hash='races';}
  else if(action==='queue')await queue(button.dataset.kind);
  else if(action==='stop-atr'){const r=await api('workers/atr/stop',{},{});toast(r.message);await render(false);}
  else if(action==='resume-atr')await queue('atr_resume');
  else if(action==='atr-log'){const r=await api('workers/atr/log');inspectRecord({log:r.text},'atr');}
  else if(action==='job-log'){const r=await api(`jobs/${button.dataset.job}/log`);inspectRecord({log:r.text});}
  else if(action==='compare-canonical'){
   const result=state.sourceResult, rows=[];
   const canonical=new Map(state.race.rows.map(r=>[String(r.horse_id).replace(/\.0$/,''),r]));
   for(const [table,entries] of Object.entries(result.tables))for(const record of entries.rows){
    const id=record.sl_horse_id || record.link_sl_horse_id || (['sl','rtv'].includes(result.source)?record.horse_id:null);
    if(id==null || (record.match_status && record.match_status!=='matched') || (record.link_match_status && record.link_match_status!=='matched'))continue;
    const target=canonical.get(String(id).replace(/\.0$/,''));if(!target)continue;
    const fields=['horse_name','race_date','track','off_time','finish_position','num_runners','distance_furlongs','surface','going','official_rating','prize_money'];
    for(const field of fields){const sourceField=field==='finish_position'&&Object.hasOwn(record,'atr_official_position')?'atr_official_position':field;if(!Object.hasOwn(record,sourceField)||target[field]==null||record[sourceField]==null)continue;
     rows.push({horse_name:target.horse_name,field:pretty(field),canonical:target[field],source_value:record[sourceField],comparison:String(target[field])===String(record[sourceField])?'Same':'Different; review evidence',source_table:table,source_field:sourceField,observed_at:record.observed_at || record.collected_at || null});
    }
   }
   $('#inspector-title').textContent='Canonical & source comparison';$('#inspector-eyebrow').textContent=`${sourceLabels[result.source]} · ${result.layer}`;
   $('#inspector-body').innerHTML=banner('Only comparable fields and established runner identities are aligned','Different does not by itself mean incorrect. Timeform ratings, BSP and other distinct measurements are kept separate.')+dataTable(rows,result.source,'comparison');$('#inspector').showModal();
  }
  else if(action==='add-change'){const field=$('#proposal-field').value,raw=$('#proposal-value').value;state.changes.push({field,horse_id:$('#proposal-runner').value,value:raw==='null'?null:raw});renderProposalChanges();}
  else if(action==='remove-change'){state.changes.splice(Number(button.dataset.index),1);renderProposalChanges();}
 }catch(e){toast(e.message,true);if(action==='prepare'&&$('#prepared-result'))$('#prepared-result').innerHTML=banner(e.message,'','error');}
 finally{button.disabled=false;}
});
function renderProposalChanges(){if($('#proposal-changes'))$('#proposal-changes').innerHTML=state.changes.map((c,i)=>`<div class="proposal-change"><span>${esc(pretty(c.field))} → ${esc(valueText(c.value))} <span class="muted">(runner ${esc(c.horse_id)})</span></span><button class="link-button" data-action="remove-change" data-index="${i}" aria-label="Remove field change">×</button></div>`).join('');}
document.addEventListener('submit',async event=>{
 const form=event.target.closest('[data-form]');if(!form)return;event.preventDefault();
 const submit=$('button:not([type=button])',form);if(submit)submit.disabled=true;
 try{
  if(form.dataset.form==='search'){state.query=$('#race-search').value.trim();state.offset=0;await racesPage();}
  else if(form.dataset.form==='refresh'){const sources=$$('input[name=source]:checked',form).map(n=>n.value);await queue('refresh',{sources,recheck:$('#recheck').checked,enrich:$('#enrich').checked,weather:$('#weather').checked});}
  else if(form.dataset.form==='note'){await api('review/note',{}, {race_id:state.race.header.race_id,note:$('#race-note').value,bookmarked:$('#race-bookmarked').checked});state.review=await api('review');toast('Inspection note saved.');}
  else if(form.dataset.form==='proposal'){const r=await api('review/proposals',{}, {race_id:state.race.header.race_id,changes:state.changes,reason:$('#proposal-reason').value,evidence:$('#proposal-evidence').value,fingerprint:state.race.fingerprint});$('#proposal-result').innerHTML=banner(r.validation_passed?'Proposal saved; field validation passed':'Proposal saved with validation findings',r.findings.map(f=>f.message).join(' · ') || r.application,r.validation_passed?'good':'')+`<a class="button small" href="/api/review/proposals/${r.id}">Download correction proposal</a>`;toast('Correction proposal saved. Canonical data is unchanged.');}
  else if(form.dataset.form==='compare'){const r=await api(`races/${state.race.header.race_id}/compare`,{source:$('#compare-source').value,before:$('#compare-before').value,after:state.boundDataset});$('#compare-result').innerHTML=r.changes.length?`<section class="panel"><div class="panel-head"><h2>${r.changes.length} changed observations</h2></div>${r.changes.map(c=>`<div class="review-row"><div><h3>${esc(c.table)} · ${esc(c.kind)}</h3><p>${esc(c.fields.join(', '))}</p><div class="mono">${esc(c.identity)}</div></div>${inspectButton(c,r.source,'View change')}</div>`).join('')}</section>`:banner('No changes found','The selected race’s accepted source rows are identical across these datasets.','good');}
 }catch(e){toast(e.message,true);}finally{if(submit)submit.disabled=false;}
});
document.addEventListener('change',async event=>{
 if(event.target.matches('[data-decision]')){try{await api('review/decision',{}, {issue_id:event.target.dataset.decision,state:event.target.value,note:''});toast('Review state saved.');}catch(e){toast(e.message,true);}}
});
async function render(showLoading=true) {
 state.abort?.abort();state.abort=new AbortController();
 const ticket=++state.ticket,segments=(location.hash.slice(1)||'overview').split('/');
 state.page=segments[0];
 if(!['overview','races','race','sources','coverage','features','modelling','paper','jobs','review'].includes(state.page)){location.hash='overview';return;}
 $('.contextbar').classList.toggle('modelling-context',state.page==='modelling');
 $('.contextbar').classList.toggle('evaluation-context',state.page==='modelling' && segments[1]==='evaluation');
 $$('#navigation a').forEach(a=>a.classList.toggle('active',a.dataset.page===(state.page==='race'?'races':state.page)));
 if(showLoading)$('#main').innerHTML='<div class="loading"><span class="spinner"></span>Reading your local data…</div>';
 try{
  if(state.page==='features') {
   const summary=await api('features/overview',{dataset:state.dataset});if(ticket!==state.ticket)return;
   state.boundDataset=state.dataset || summary.selected_dataset || '';
   await featureUI.render(segments[1],decodeURIComponent(segments[2]||''),summary);
  } else if(state.page==='paper') {
   await paperUI.render();
  } else if(state.page==='modelling') {
   await modellingUI.render(segments[1],segments[2],ticket);
   if(ticket!==state.ticket)return;
  } else {
   const o=await api('overview',{dataset:state.dataset});if(ticket!==state.ticket)return;
   state.overview=o;state.boundDataset=state.dataset || o.dataset.id || '';
   if(state.page==='overview')await overviewPage();else if(state.page==='races')await racesPage();else if(state.page==='race')await racePage(decodeURIComponent(segments[1]||''));else if(state.page==='sources')await sourcesPage(segments[1]);else if(state.page==='coverage')await coveragePage();else if(state.page==='jobs')await jobsPage();else if(state.page==='review')await reviewPage();
  }
  $('.breadcrumb').innerHTML=`Project <span>/</span> ${({features:'Features',modelling:'Modelling',paper:'Paper record'})[state.page] || 'Ingestion'}`;
  $('#last-checked').textContent=`Checked ${new Date().toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'})} · Local workspace`;
  document.title=`${state.page==='race'?'Race inspection':pretty(state.page)} · Racing Workbench`;
 }catch(e){if(ticket===state.ticket)$('#main').innerHTML=heading('Unable to load this view','Your source data has not been changed.')+banner(e.message,'Check the selected date range, dataset or local source availability.','error')+'<button class="button" data-nav="overview">Back to overview</button>';}
}
$('#close-inspector').addEventListener('click',()=>$('#inspector').close());
$('.skip').addEventListener('click',e=>{e.preventDefault();$('#main').focus();});
$('#inspector').addEventListener('click',e=>{if(e.target===$('#inspector')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
$('#refresh').addEventListener('click',()=>render(false));
$('#auto-refresh').addEventListener('change',e=>state.auto=e.target.checked);
$('#apply-dates').addEventListener('click',()=>{state.offset=0;render();});
$('#dataset').addEventListener('change',e=>{state.dataset=e.target.value;render();});
window.addEventListener('hashchange',()=>{render();window.scrollTo(0,0);});
setInterval(()=>{if(state.auto && (['overview','jobs'].includes(state.page) || state.page==='features' && ['overview','builds'].includes(location.hash.split('/')[1] || 'overview')) && !['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName) && !$('#inspector').open)render(false);},30000);
const featureUI=createFeaturesUI({api,url,esc,number,pretty,banner,badge,empty,metric,toast,state,heading,queue,render,raceContent});
const paperUI=createPaperUI({api,esc,number,banner,badge,empty,metric,heading});
const modellingUI=createModellingUI({api,esc,number,pretty,banner,badge,empty,metric,toast,state,heading,queue,dateLabel});
async function boot(){try{const session=await api('session');token=session.token;const yesterday=new Date();yesterday.setDate(yesterday.getDate()-1);const end=yesterday.toISOString().slice(0,10);yesterday.setDate(yesterday.getDate()-6);$('#start-date').value=yesterday.toISOString().slice(0,10);$('#end-date').value=end;const sets=await api('datasets');$('#dataset').insertAdjacentHTML('beforeend',sets.rows.map(s=>`<option value="${s.id}">${dateLabel(s.created_at)} · ${s.id.slice(0,10)}</option>`).join(''));await render();}catch(e){$('#main').innerHTML=heading('Workspace connection unavailable','Start the local workbench server and reload this page.')+banner(e.message,'','error');}}
boot();

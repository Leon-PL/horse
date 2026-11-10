export function createFeaturesUI(ctx) {
 const {api,url,esc,number,pretty,banner,badge,empty,metric,toast,state} = ctx;
 const $ = q=>document.querySelector(q);
 const f={set:'',family:'',query:'',offset:0,race:'',tab:'overview',summary:null,quality:null,qualityKey:''};
 const setParams=()=>({feature_set:f.set,family:f.family,query:f.query,offset:f.offset});
 const nameOf=id=>f.summary.feature_sets.find(s=>s.id===id)?.label || id;
 const panel=(title,description,body,action='')=>`<section class="panel"><div class="panel-head"><div><h2>${esc(title)}</h2><p>${esc(description)}</p></div>${action}</div>${body}</section>`;
 const short=v=>v?String(v).slice(0,12)+'…':'Not recorded';
 const scope=()=>`${$('#start-date').value}/${$('#end-date').value}/${f.set}/${f.family}/${f.summary?.store.build_id}/${f.summary?.feature_sets.find(x=>x.id===f.set)?.signature}`;
 function controls(search=true) {
  return `<form class="feature-controls" data-feature-form="filter"><label>Feature set<select data-feature-set>${f.summary.feature_sets.map(s=>`<option value="${esc(s.id)}" ${f.set===s.id?'selected':''}>${esc(s.label)} · ${s.count} inputs</option>`).join('')}<option value="all" ${f.set==='all'?'selected':''}>All stored candidates & available definitions</option></select></label><label>Family<select data-feature-family><option value="">All families</option>${f.summary.families.map(x=>`<option value="${x.id}" ${f.family===x.id?'selected':''}>${esc(x.label)}</option>`).join('')}</select></label>${search?`<label class="feature-search">Find a feature<input name="query" value="${esc(f.query)}" placeholder="Name or definition" maxlength="120"></label><button class="button small">Filter</button>`:''}</form>`;
 }
 function provenance() {
  const s=f.summary.store;
  return `${s.reasons.length?banner(s.state==='missing'?'Feature store is not available':'Stored features need an update',s.reasons.join(' '),'',`<a class="button small" href="#features/builds">Build controls →</a>`):''}<div class="feature-provenance"><span>${badge('Stored values','neutral')} Through <strong>${esc(s.last_date || 'Not built')}</strong></span><span>Stored dataset <span class="mono">${esc(short(s.dataset))}</span></span><span>Selected sources <span class="mono">${esc(short(f.summary.selected_dataset))}</span></span></div>`;
 }
 async function render(tab='overview',race='',summary=null) {
  const ticket=state.ticket;
  f.tab=['overview','inspector','catalog','quality','builds'].includes(tab)?tab:'overview';
  if(race) f.race=race;
  f.summary=summary || await api('features/overview');
  if(ticket!==state.ticket || state.page!=='features')return;
  if(!f.set || f.set!=='all'&&!f.summary.feature_sets.some(s=>s.id===f.set))f.set=f.summary.feature_sets.find(s=>s.configured_for_training)?.id || f.summary.feature_sets[0]?.id || 'all';
  $('#main').innerHTML=ctx.heading('Feature engineering','Inspect stored values, understand inputs and prepare shared features for each model role.','', 'FEATURE ENGINEERING')+
   `<div class="tabs feature-tabs" aria-label="Feature engineering views">${Object.entries({overview:'Overview',inspector:'Race inspector',catalog:'Feature catalog',quality:'Quality checks',builds:'Builds & validation'}).map(([key,label])=>`<a href="#features/${key}" class="${f.tab===key?'active':''}" ${f.tab===key?'aria-current="page"':''}>${label}</a>`).join('')}</div>`+provenance()+controls(f.tab==='catalog'||f.tab==='inspector')+'<div id="feature-content"></div>';
  await content();
 }
 async function content() {
  const target=$('#feature-content');if(!target)return;
  if(f.tab==='overview') {
   const o=f.summary,s=o.store;
   target.innerHTML=`<div class="metrics-grid feature-metrics">${metric('Stored runners',number(s.rows),'Shared computed history')}${metric('Stored inputs',number(s.columns),'Numeric candidates in this generation')}${metric('Named feature sets',number(o.feature_sets.length),'Independent model input selections')}${metric('Model roles',number(o.model_roles.length),'Configured roles, not fitted models')}</div>`+
    `<div class="two-column">${panel('Model roles & their inputs','One shared store can serve several general and specialist selections.',o.model_roles.length?`<div class="panel-body feature-role-list">${o.model_roles.map(m=>`<article class="feature-role"><div class="feature-role-icon">${esc(m.label[0])}</div><div><h3>${esc(m.label)}</h3><p>${esc(m.scope)}</p><div class="actions">${m.feature_sets.map(id=>`<button class="button small" data-feature-select="${esc(id)}">${esc(nameOf(id))}</button>`).join('')}</div></div>${badge('Configured','neutral')}</article>`).join('')}</div>`:empty('No model roles configured','Add model roles and named selections in the feature-set registry.'))}
    ${panel('Readiness has three parts','Source coverage, feature freshness and model compatibility are independent.',`<div class="panel-body"><div class="feature-check-row"><span>Feature generation</span>${badge(pretty(s.state),s.state==='current'?'':'warning')}</div><div class="feature-check-row"><span>Saved model / definition check</span>${badge(pretty(o.saved_model.state),o.saved_model.state==='definition_match'?'':'neutral')}</div><p class="hint">${esc(o.saved_model.note)}</p><a class="link-button" href="#sources">Review source coverage →</a></div>`)}</div>`+
    panel('Explore feature families','Descriptions and coverage are inspected separately from predictive value.',`<div class="feature-family-grid">${o.families.map(x=>`<button class="feature-family" data-feature-family-link="${x.id}"><strong>${esc(x.label)}</strong><span>${esc(x.description)}</span><span class="link-button">Explore →</span></button>`).join('')}</div>`)+
    `<details class="panel"><summary class="panel-head"><h2>Generation details</h2></summary><div class="panel-body"><dl class="property-grid">${[['Current code',o.code_signature],['Stored definition',s.signature],['Feature build',s.build_id],['Stored source dataset',s.dataset],['Saved run reference',o.saved_model.run_id],['Last store operation',s.built_at]].map(([k,v])=>`<div><dt>${esc(k)}</dt><dd class="mono">${esc(v || 'Not recorded')}</dd></div>`).join('')}</dl><p class="hint">${esc(o.architecture)}</p></div></details>`;
  } else if(f.tab==='catalog')await catalog(target);
  else if(f.tab==='inspector')await inspector(target);
  else if(f.tab==='quality') quality(target);
  else await builds(target);
 }
 async function catalog(target) {
  const r=await api('features/catalog',setParams());
  target.innerHTML=panel('Feature catalog',`${r.total} matching features. Model-role membership is configured; saved-model membership is not inferred.`,r.rows.length?`<div class="table-wrap"><table class="feature-catalog"><thead><tr><th>Feature</th><th>Family / source</th><th>Stored</th><th>Configured sets</th><th>Definition</th></tr></thead><tbody>${r.rows.map(x=>`<tr><td><button class="link-button" data-feature-definition="${esc(x.name)}"><strong>${esc(x.label)}</strong></button><div class="mono race-meta">${esc(x.name)}</div></td><td>${esc(x.family_label)}<small>${esc(x.source)}</small></td><td>${badge(x.stored?'Available':'Not built',x.stored?'':'warning')}<small>${esc(x.dtype || '')}</small></td><td>${x.feature_sets.map(id=>badge(nameOf(id),'neutral')).join(' ') || '<span class="muted">Candidate</span>'}</td><td>${badge(x.documentation==='documented'?'Documented':'Family only','neutral')}</td></tr>`).join('')}</tbody></table></div>`:empty('No matching features','Try another set, family or search term.'))+pager(r.offset,r.limit,r.total);
 }
 function pager(offset,limit,total) {
  return `<div class="table-foot"><span>${number(Math.min(offset+limit,total))} of ${number(total)} features</span><div class="actions"><button class="button small" data-feature-page="${Math.max(0,offset-limit)}" ${offset?'':'disabled'}>Previous</button><button class="button small" data-feature-page="${offset+limit}" ${offset+limit<total?'':'disabled'}>Next</button></div></div>`;
 }
 async function inspector(target) {
  let choices=[];try{choices=(await api('features/races')).rows;}catch(e){if(f.summary.store.state!=='missing')toast(e.message,true);}
  target.innerHTML=panel('Choose a stored race','Stored values keep their original feature generation and dataset.',`<form class="panel-body feature-race-picker" data-feature-form="race"><label>Stored races in this date range<select id="feature-race-choice"><option value="">Choose a race</option>${choices.map(x=>`<option value="${esc(x.race_id)}" ${String(x.race_id)===f.race?'selected':''}>${esc(String(x.race_date).slice(0,10))} · ${esc(x.track || '')} ${esc(x.off_time || '')}</option>`).join('')}</select></label><label>Or race ID<input id="feature-race-id" value="${esc(f.race)}" placeholder="Canonical race ID" maxlength="80"></label><button class="button primary">Inspect values</button></form>`)+`<div id="feature-race-values"></div>`;
  if(f.race) await raceValues($('#feature-race-values'),f.race);
  else $('#feature-race-values').innerHTML=empty('Choose a race to inspect','You can also open Features from any race in the race explorer. Narrow the dates to find another stored field.');
 }
 async function raceValues(target,race) {
  if(!target)return;
  const request=f.raceTicket=(f.raceTicket || 0)+1;
  try {
   const r=await api(`features/races/${encodeURIComponent(race)}`,setParams());
   if(request!==f.raceTicket || !target.isConnected)return;
   if(f.summary.store.build_id && r.store.build_id!==f.summary.store.build_id)throw new Error('The feature store changed. Refresh this view before comparing values.');
   const attrs=`feature_set=${encodeURIComponent(f.set)}&family=${encodeURIComponent(f.family)}&query=${encodeURIComponent(f.query)}&offset=${f.offset}`;
   target.innerHTML=panel(`Stored values · race ${race}`,r.message,r.rows.length&&r.columns.length?`<div class="table-wrap feature-matrix"><table><thead><tr><th class="sticky-runner">Runner</th>${r.columns.map(c=>`<th title="${esc(c.name)}">${esc(c.label)}<small>${esc(c.name)}</small></th>`).join('')}</tr></thead><tbody>${r.rows.map(row=>`<tr><td class="sticky-runner"><strong>${esc(row.horse_name || row.horse_id)}</strong><small>ID ${esc(row.horse_id)}</small></td>${r.columns.map(c=>`<td>${c.stored?`<button class="feature-value ${row[c.name]==null?'missing':''}" data-feature-value="${esc(c.name)}" data-race="${esc(race)}" data-horse="${esc(row.horse_id)}" aria-label="Inspect ${esc(c.label)} for ${esc(row.horse_name || row.horse_id)}">${r.invalid_values.some(v=>v.horse_id===String(row.horse_id)&&v.column===c.name)?'Invalid':row[c.name]==null?'Missing':number(row[c.name])}</button>`:badge('Not built','warning')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`:empty(r.rows.length?'No columns in this selection':'This race has no stored feature rows',r.rows.length?'Choose another family or update the store to build newer definitions.':'It may be newer than the store. Use Builds & validation to update the shared history.'),`<a class="button small" href="/api/features/races/${encodeURIComponent(race)}/export?${attrs}">Export shown columns</a>`)+pager(r.offset,r.limit,r.total_columns)+`<p class="hint">Build ${esc(short(r.store.build_id))} · definition ${esc(short(r.store.signature))} · source dataset ${esc(short(r.store.dataset))}. Missing and zero are different values. Click a cell for its recorded context.</p>`;
  }catch(e){if(target.isConnected)target.innerHTML=banner('Stored feature inspection unavailable',e.message,'');}
 }
 function quality(target) {
  if(f.qualityKey!==scope())f.quality=null;
  target.innerHTML=panel('Bounded quality check','Use the date range above. Check up to 5,000 stored runners and 256 features over at most 31 days.',`<div class="panel-body"><div class="actions"><button class="button primary" data-feature-action="quality">Run quality check</button><button class="button" data-feature-action="check-job">Save check as a job</button></div><p class="hint">Nulls and constants can be expected for debutants, restricted race types or uncovered sources. These checks report observations; they do not certify every feature as leakage-free.</p></div>`)+`<div id="feature-quality-results">${f.quality?qualityResults(f.quality):empty('No check run for this selection','Choose a feature set, family and date range, then run a bounded check.')}</div>`;
 }
 function qualityResults(r) {
  return `${$('#end-date').value>String(f.summary.store.last_date || '')?banner('The requested dates extend beyond stored history',`The store is built through ${f.summary.store.last_date || 'an unknown date'}. These diagnostics cover available stored rows only.`):''}${r.truncated?banner('The cohort was capped at 5,000 runners','Narrow the dates for a smaller complete cohort.'):''}${r.missing_columns.length?banner('Required columns are not stored',r.missing_columns.join(', '),'error'):''}`+
   `<div class="metrics-grid feature-metrics">${metric('Runners checked',number(r.rows_checked),'Stored cohort')}${metric('Columns checked',number(r.columns_checked),'Selected set / family')}${metric('Missing columns',number(r.missing_columns.length),'Schema requirements')}${metric('Duplicate runner IDs',number(r.duplicate_keys),'Within this checked cohort')}</div>`+
   panel('Observed values',r.scope,`<div class="table-wrap"><table><thead><tr><th>Feature</th><th>Observation</th><th>Missing</th><th>Distinct</th><th>Zeros</th><th>Range</th><th>Infinite</th></tr></thead><tbody>${r.rows.map(x=>`<tr><td><button class="link-button mono" data-feature-definition="${esc(x.name)}">${esc(x.name)}</button></td><td>${badge(pretty(x.state),x.state==='invalid'?'error':x.state==='observed'?'neutral':'warning')}</td><td>${number(x.missing_pct)}%</td><td>${number(x.distinct)}</td><td>${number(x.zeros)}</td><td>${number(x.minimum)} → ${number(x.maximum)}</td><td>${number(x.infinite)}</td></tr>`).join('')}</tbody></table></div>`);
 }
 async function builds(target) {
  const jobs=(await api('jobs')).rows.filter(j=>j.kind.startsWith('feature_'));
  target.innerHTML=`<div class="two-column">${panel('Update shared feature history','The selected source dataset is fixed when the job is queued.',`<form class="panel-body" data-feature-form="build"><p>One shared store supplies every configured feature set. Updating does not change feature selections or fit a model.</p><p class="hint">The existing builder chooses append or rebuild from its compatibility checks. It keeps the previous published store intact until a successful commit.</p><label class="checkbox-line"><input type="checkbox" id="feature-force-rebuild"> Force a full historical rebuild</label><p class="hint">Accepted dataset: <span class="mono">${esc(short(f.summary.selected_dataset))}</span></p><button class="button primary" ${f.summary.selected_dataset?'':'disabled'}>Queue feature update</button></form>`)}${panel('Build comparison','Compare definitions and values within a fixed cohort.',`<div class="panel-body"><p>Stored value exports include their generation and dataset. Keep these exports when reviewing changes.</p><p class="hint">Automatic comparisons between complete feature generations need retained comparison artifacts and are not yet available. This view never compares old values against a new dataset silently.</p><a class="button" href="#features/inspector">Inspect / export a race →</a></div>`)}</div>`+
   panel('Feature jobs','Jobs continue independently of this page; stage messages are available in their logs.',jobs.length?jobs.slice(0,12).map(j=>`<div class="job-row"><header><strong>${esc(pretty(j.kind))}</strong>${badge(pretty(j.state),j.state==='failed'?'error':'neutral')}</header><p>${esc(j.progress?.message || j.error || 'Recorded workbench operation')}</p><div class="actions"><button class="button small" data-action="job-log" data-job="${esc(j.id)}">View log</button><a class="button small" href="#jobs">All jobs</a></div></div>`).join(''):empty('No feature jobs queued','Browsing never builds features.'));
 }
 async function showDefinition(name) {
  const r=await api('features/catalog',{feature_set:'all',query:name,limit:100});
  const x=r.rows.find(row=>row.name===name);if(!x)throw new Error('Feature definition is not available');
  $('#inspector-eyebrow').textContent='FEATURE DEFINITION';$('#inspector-title').textContent=x.label;
  $('#inspector-body').innerHTML=`<p>${esc(x.description)}</p>${x.definition_note?banner('Individual definition pending',x.definition_note):''}${x.availability_rule?banner('Current availability rule',x.availability_rule):''}<dl class="property-grid">${[['Column',x.name],['Family',x.family_label],['Source family',x.source],['Units',x.units || 'Not documented'],['Formula',x.formula || 'Not recorded'],['Stored dtype',x.dtype || 'Not built'],['Configured sets',x.feature_sets.map(nameOf).join(', ') || 'Candidate'],['Configured model roles',x.model_roles.join(', ') || 'None']].map(([k,v])=>`<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl>`;$('#inspector').showModal();
 }
 async function showValue(button) {
  const request=f.valueTicket=(f.valueTicket || 0)+1;
  const r=await api(`features/races/${encodeURIComponent(button.dataset.race)}/value`,{column:button.dataset.featureValue,horse_id:button.dataset.horse});
  if(request!==f.valueTicket)return;
  $('#inspector-eyebrow').textContent='STORED FEATURE VALUE';$('#inspector-title').textContent=r.feature.label;
  $('#inspector-body').innerHTML=`<div class="feature-value-hero"><div><span class="hint">${esc(r.runner.horse_name || r.runner.horse_id)}</span><strong>${r.invalid?'Invalid (infinite)':r.missing?'Missing':number(r.runner[r.feature.name])}</strong><span class="mono">${esc(r.feature.name)}</span></div>${badge(r.invalid?'Invalid value':r.missing?'Missing value':'Stored measurement',r.invalid?'error':r.missing?'warning':'neutral')}</div><p>${esc(r.explanation)}</p><p>${esc(r.feature.description)}</p>${r.feature.definition_note?banner('Detailed calculation trace unavailable',r.feature.definition_note):''}${r.support?`<p class="hint">Support: ${esc(r.support.column)} = ${number(r.support.value)}. ${esc(r.support.note)}</p>`:''}${r.availability?banner('Availability rule',r.availability):''}<p class="hint">Build ${esc(short(r.build_id))} · definition ${esc(short(r.generation))} · source dataset ${esc(short(r.dataset))}</p><h3>Prior starts for context</h3><p class="hint">${esc(r.history_message)}</p>${r.history.length?`<div class="table-wrap"><table><thead><tr><th>Date</th><th>Course</th><th>Prior result</th><th>Evidence</th></tr></thead><tbody>${r.history.map(h=>`<tr><td>${esc(String(h.race_date).slice(0,10))}</td><td>${esc(h.track || '')}</td><td>${esc(h.finish_position ?? 'Unknown')}</td><td>${r.dataset?`<button class="button small" data-feature-evidence="${esc(h.race_id)}" data-dataset="${esc(r.dataset)}" data-source="${['atr','rtv','bha'].includes(r.feature.family)?r.feature.family:'sl'}">Open source</button>`:'Dataset not recorded'}</td></tr>`).join('')}</tbody></table></div>`:empty('No earlier local starts in this context query','This does not establish that the horse has never raced elsewhere.')}`;
  $('#inspector').showModal();
 }
 async function refresh() { f.offset=Math.max(0,f.offset); if(state.page==='race')await ctx.raceContent();else await render(f.tab,f.race); }
 document.addEventListener('click',async e=>{
  const b=e.target.closest('[data-feature-action],[data-feature-select],[data-feature-family-link],[data-feature-page],[data-feature-definition],[data-feature-value],[data-feature-evidence]');if(!b)return;
  b.disabled=true;
  try {
   if(b.dataset.featureSelect){f.set=b.dataset.featureSelect;f.offset=0;if(state.page==='features'&&f.tab==='overview')location.hash='features/catalog';else await refresh();}
   else if(b.dataset.featureFamilyLink){f.family=b.dataset.featureFamilyLink;f.offset=0;location.hash='features/catalog';}
   else if(b.dataset.featurePage!=null){f.offset=Number(b.dataset.featurePage);await refresh();}
   else if(b.dataset.featureDefinition)await showDefinition(b.dataset.featureDefinition);
   else if(b.dataset.featureValue)await showValue(b);
   else if(b.dataset.featureEvidence){state.dataset=b.dataset.dataset;state.raceTab='evidence';state.source=b.dataset.source;state.layer='accepted';const select=$('#dataset');if(![...select.options].some(o=>o.value===state.dataset))select.add(new Option(`Stored dataset ${short(state.dataset)}`,state.dataset));select.value=state.dataset;$('#inspector').close();location.hash=`race/${encodeURIComponent(b.dataset.featureEvidence)}`;}
   else if(b.dataset.featureAction==='quality'){const target=$('#feature-quality-results');target.innerHTML='<div class="loading"><span class="spinner"></span>Reading the bounded stored cohort…</div>';const key=scope(),result=await api('features/quality',setParams());if(f.summary.store.build_id&&result.build_id!==f.summary.store.build_id)throw new Error('The feature store changed. Refresh and run this check again.');if(scope()!==key || state.page!=='features' || f.tab!=='quality')return;f.quality=result;f.qualityKey=key;target.innerHTML=qualityResults(result);}
   else if(b.dataset.featureAction==='check-job')await ctx.queue('feature_check',{feature_set:f.set,family:f.family});
  }catch(error){toast(error.message,true);if(b.dataset.featureAction==='quality'&&$('#feature-quality-results'))$('#feature-quality-results').innerHTML=banner('Quality check unavailable',error.message,'');}finally{b.disabled=false;}
 });
 document.addEventListener('submit',async e=>{
  const form=e.target.closest('[data-feature-form]');if(!form)return;e.preventDefault();
  const button=form.querySelector('button');if(button)button.disabled=true;
  try {
   if(form.dataset.featureForm==='filter'){f.query=form.elements.query?.value.trim() || '';f.offset=0;await refresh();}
   else if(form.dataset.featureForm==='race'){f.race=$('#feature-race-id').value.trim() || $('#feature-race-choice').value;f.offset=0;history.replaceState(null,'',`#features/inspector/${encodeURIComponent(f.race)}`);await raceValues($('#feature-race-values'),f.race);}
   else if(form.dataset.featureForm==='build')await ctx.queue('feature_update',{rebuild:$('#feature-force-rebuild').checked});
  }catch(error){toast(error.message,true);}finally{if(button)button.disabled=false;}
 });
 document.addEventListener('change',async e=>{
  try {
  if(e.target.matches('[data-feature-set]')){f.set=e.target.value;f.offset=0;await refresh();}
  else if(e.target.matches('[data-feature-family]')){f.family=e.target.value;f.offset=0;await refresh();}
  else if(e.target.id==='feature-race-choice')$('#feature-race-id').value=e.target.value;
  }catch(error){toast(error.message,true);}
 });
 async function raceInTab(race,target) {
  f.summary=await api('features/overview');f.race=String(race);
  if(!target.isConnected || state.raceTab!=='features')return;
  if(!f.set)f.set=f.summary.feature_sets.find(s=>s.configured_for_training)?.id || f.summary.feature_sets[0]?.id || 'all';
  target.innerHTML=provenance()+controls(true)+'<div id="feature-race-values"></div>';
  await raceValues($('#feature-race-values'),race);
 }
 return {render,raceInTab};
}

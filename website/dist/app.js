import {parseCsv} from './replay-inference.js';
import { percent, ranked, recallAt, departures, predictionDelta, colorClass, cardKind, boardEntries, predictionChanges, predictionHistory, opponentComparison, gameForMethod } from './model-view.js';
const $ = selector => document.querySelector(selector);
const escape = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data;
let images = {};
const state = { game:0, frame:0, reveal:false, limit:10, selected:null, method:'historical' };
let playTimer = null;
let animatedFrame = '';
function stopPlayback() {
 if(playTimer)clearInterval(playTimer);playTimer=null;
 $('#play-toggle').textContent='▶ Play replay';$('#play-toggle').setAttribute('aria-pressed','false');
}
function chooseFrame(index) { stopPlayback();state.frame=index;render(); }
function symbol(card) {
 const image=images[card];
 if(image) return `<img class="card-thumb" src="${escape(image.image)}" alt="${escape(card)} card" loading="lazy" width="48" height="67">`;
 const meta=data.cards[card]||{};
 return `<span class="card-symbol ${colorClass(meta)}" aria-hidden="true">${cardKind(meta).includes('Land')?'◇':'✦'}</span>`;
}
function activeGame() { return gameForMethod(data.cases[state.game],state.method); }
function render() {
 hideCardPreview();
 const game=activeGame(), frame=game.frames[state.frame];
 $('#replay-origin').textContent=game.imported?'IMPORTED PUBLIC REPLAY':'REAL HELD-OUT GAME';
 $('#game-meta').textContent=`${game.imported?'Supported public prefix':game.complete?'Full recorded game':'Early-turn excerpt'} · ${game.on_play?'Opponent plays first':'Opponent draws first'}`;
 $('#frame-label').textContent=`Turn-end state ${state.frame+1} of ${game.frames.length}`;
 const timeline=$('#timeline'); timeline.replaceChildren();
 game.frames.forEach((f,i)=>{
  const button=document.createElement('button');
  button.className=(i===state.frame?'active':i<state.frame?'completed':'');
  button.setAttribute('aria-pressed',String(i===state.frame));
  button.innerHTML=`<small>${f.step.active_player==='target'?'Target':'Observer'}</small><strong>Turn ${f.step.turn}</strong>`;
  button.addEventListener('click',()=>chooseFrame(i));timeline.append(button);
 });
 const selectedTurn=timeline.children[state.frame];
 if(selectedTurn)timeline.scrollLeft=Math.max(0,selectedTurn.offsetLeft-timeline.clientWidth/2+selectedTurn.clientWidth/2);
 $('#frame-slider').max=game.frames.length-1;$('#frame-slider').value=state.frame;
 $('#prev').disabled=state.frame===0;$('#next').disabled=state.frame===game.frames.length-1;
 const step=frame.step;
 $('#action-summary').innerHTML=`<b>${step.active_player==='target'?'Target’s':'Observer’s'} turn ${step.turn}</b>${step.draws?`<span class="draw">+${step.draws} unknown card${step.draws===1?'':'s'} drawn</span>`:''}${step.departed.map(card=>`<span class="action-chip" tabindex="0" data-preview-card="${escape(card)}">${escape(card)} left hand</span>`).join('')}${!step.draws&&!step.departed.length?'<span class="draw">No modeled target-hand movement this half-turn</span>':''}`;
 $('#hand-size').textContent=step.hand_size;
 $('#compatible').textContent=state.method==='historical'?frame.compatible_decks.toLocaleString():'—';
 $('#method-note').textContent=frame.prediction_error||data.methods[state.method].description;
 $('#departed-count').textContent=departures(game.frames,state.frame).length;
 $('#reveal').disabled=frame.labels_available===false;
 if(frame.labels_available===false)state.reveal=false;
 $('#reveal').textContent=state.reveal?'Hide opponent’s hand':'Reveal opponent’s hand';
 $('#reveal').setAttribute('aria-pressed',String(state.reveal));
 $('#hand-status').textContent=state.reveal?'REVEALED':'HIDDEN';
 $('#hand-mode').textContent=frame.labels_available===false?'No private labels provided':state.reveal?'Evaluation labels · hidden from the model':'Identities hidden';
 const predictions=ranked(frame.probabilities,state.limit);
 if(!state.selected) state.selected=predictions[0]?.[0];
 const list=$('#prediction-list');list.replaceChildren();
 if(frame.prediction_error){const note=document.createElement('p');note.className='change-empty';note.textContent=frame.prediction_error+' Choose a smoothed or tree method.';list.append(note);}
 predictions.forEach(([card,p],i)=>{
  const button=document.createElement('button');button.className=`prediction-row ${state.selected===card?'selected':''}`;
  button.dataset.previewCard=card;
  button.setAttribute('aria-pressed',String(state.selected===card));
  const delta=predictionDelta(game.frames,state.frame,card);
  const hit=state.reveal?(frame.actual_hand[card]?'<span class="hit">✓ In hand</span>':'<span class="miss">Not held</span>'):'';
  button.innerHTML=`<span class="rank">${String(i+1).padStart(2,'0')}</span>${symbol(card)}<div><div class="card-name">${escape(card)} ${hit}</div><div class="card-kind">${escape(cardKind(data.cards[card]))}</div></div><div class="probability">${percent(p)}<div class="prob-bar"><span style="width:${p*100}%"></span></div><div class="prob-delta">${delta===null?'Awaiting next opponent turn':Math.abs(delta)<0.0005?'Unchanged':`${delta>=0?'+':''}${(delta*100).toFixed(1)} pp`}</div></div>`;
  button.addEventListener('click',()=>{state.selected=card;render();});list.append(button);
 });
 const hand=$('#actual-hand');hand.replaceChildren();
 if(state.reveal) {
  for(const [card,copies] of Object.entries(frame.actual_hand)) {
   const tile=document.createElement('div');tile.className='hand-card illustrated';
   tile.innerHTML=`${images[card]?`<button class="hand-image-button" aria-label="Enlarge ${escape(card)}"><img class="full-card-image" src="${escape(images[card].image)}" alt="${escape(card)}" loading="lazy" width="488" height="680"></button>`:symbol(card)}<span class="card-title">${escape(card)}</span><span class="copies">${copies} cop${copies===1?'y':'ies'} · ${percent(frame.probabilities[card]||0)} predicted</span>`;
   if(tile.querySelector('.hand-image-button'))tile.querySelector('.hand-image-button').dataset.previewCard=card;
   tile.querySelector('.hand-image-button')?.addEventListener('click',()=>showCard(card));hand.append(tile);
  }
 } else {
  for(let i=0;i<step.hand_size;i++){
   const tile=document.createElement('div');tile.className='hand-card back';tile.setAttribute('aria-label','Unknown card');tile.innerHTML='<span aria-hidden="true">◈</span><small>HIDDEN HAND</small>';hand.append(tile);
  }
 }
 if(!step.hand_size)hand.innerHTML='<p class="empty-hand">The target’s hand is empty.</p>';
 $('#snapshot-score').textContent=frame.labels_available===false?'Imported public replay: actual hand labels are unavailable. Predictions have not been audited against private identities.':state.reveal?`This snapshot: ${percent(recallAt(frame.probabilities,frame.actual_hand,state.limit))} of actual card types recovered in the top ${state.limit}.`:'Reveal the labels to check which guesses were right. They were never used to make these predictions.';
 renderBoard(frame,game);
 renderMethods(frame);
 renderChanges(game);
 renderDetail(frame,game);
 wireImageFallbacks();
 $('#selection-note').textContent=game.imported?`${game.coverage_note} ${game.stop_reason||'All supplied states processed; this does not establish complete game coverage.'} Imported games are not held-out benchmark examples.`:`${game.complete?`${game.selection}. Every one of its ${game.recorded_snapshot_count} available turn-end states passed the hand audit and prior support checks; this is a selected demonstration, not a new benchmark.`:data.selection+'.'} ${data.methods[state.method].label}: ${data.methods[state.method].description} ${data.method_comparison_note} ${game.stop_reason?'This excerpt stops when movement accounting becomes unsupported.':''}`;
 const ending=$('#game-result');ending.hidden=!(game.complete&&state.frame===game.frames.length-1);
 if(!ending.hidden)ending.textContent=`End of recorded game · ${game.recorded_winner==='target'?'Opponent':'You / observer'} won. The dataset records the result but not the ending action. This is the last available board state; a concession or other ending may not appear on it.`;
}
function renderBoard(frame,game) {
 const previous=state.frame ? game.frames[state.frame-1].board : null;
 const frameKey=`${game.id}:${state.frame}`;
 const shouldAnimate=frameKey!==animatedFrame;
 animatedFrame=frameKey;
 $('#board-turn-label').textContent=`Turn ${frame.step.turn} · ${frame.step.active_player==='target'?'Opponent':'Observer'}`;
 for(const player of ['target','observer']) {
  const board=frame.board[player], old=previous?.[player];
  $(`#${player}-life`).textContent=board.life;
  $(`#${player}-active`).hidden=frame.step.active_player!==player;
  $(`.${player}-half`).classList.toggle('active-half',frame.step.active_player===player);
  $(`#${player}-land-count`).textContent=`${board.lands.length}`;
  $(`#${player}-permanent-count`).textContent=`${board.creatures.length+board.other.length}`;
  renderZone($(`#${player}-lands`),board.lands,old?.lands||[],shouldAnimate);
  renderZone($(`#${player}-permanents`),[...board.creatures,...board.other],old?[...old.creatures,...old.other]:[],shouldAnimate);
 }
 const observerHand=$('#observer-hand');observerHand.replaceChildren();
 for(let i=0;i<frame.board.observer.hand_size;i++){
  const card=document.createElement('div');card.className='observer-card-back';card.textContent='◈';card.setAttribute('aria-hidden','true');observerHand.append(card);
 }
 const caption=document.createElement('small');caption.textContent=`${frame.board.observer.hand_size} cards · identities unavailable`;observerHand.append(caption);
}
function renderZone(container,cards,previous,animate) {
 container.replaceChildren();
 if(!cards.length){container.innerHTML='<span class="empty-zone">No cards in this zone</span>';return;}
 boardEntries(cards,previous).forEach(({card,isNew},index)=>{
  const button=document.createElement('button');button.className=`board-card ${isNew?'new-card':''} ${isNew&&animate?'entering-card':''}`;
  button.style.setProperty('--order',index);
  button.setAttribute('aria-label',`${card}${isNew?', appeared this snapshot':''}. Open card preview`);
  button.dataset.previewCard=card;
  button.innerHTML=images[card]?`<img src="${escape(images[card].image)}" alt="${escape(card)}" loading="lazy" width="488" height="680">`:`<span class="board-card-placeholder">${escape(card)}</span>`;
  if(isNew)button.insertAdjacentHTML('beforeend','<span class="new-card-badge">NEW</span>');
  button.addEventListener('click',()=>showCard(card));container.append(button);
 });
}
function renderMethods(frame) {
 const container=$('#method-comparison');container.replaceChildren();
 for(const [method,info] of Object.entries(data.methods)){
  const section=document.createElement('div');section.className='method-comparison-row';
  const heading=document.createElement('button');heading.className='method-choice';heading.textContent=info.label;
  heading.setAttribute('aria-pressed',String(method===state.method));
  heading.addEventListener('click',()=>{stopPlayback();state.method=method;$('#method-select').value=method;render();});section.append(heading);
  if(frame.method_errors?.[method]){const note=document.createElement('p');note.className='change-empty';note.textContent=frame.method_errors[method];section.append(note);}
  for(const [card,p] of ranked(frame.methods[method]||{},3)){
   const button=document.createElement('button');button.className='method-guess';button.dataset.previewCard=card;
   button.innerHTML=`<span>${escape(card)}</span><b>${percent(p)}</b>`;
   button.addEventListener('click',()=>{stopPlayback();state.method=method;state.selected=card;$('#method-select').value=method;render();});section.append(button);
  }
  container.append(section);
 }
}
function renderChanges(game) {
 const changes=predictionChanges(game.frames,state.frame), container=$('#prediction-changes');
 container.replaceChildren();
 if(!changes&&game.frames[state.frame].prediction_error){container.textContent='Selected method is unavailable at this state.';return;}
 if(!changes){container.innerHTML='<p class="change-empty">Changes appear after two recorded opponent turns and stay visible during your turn.</p>';return;}
 const comparison=opponentComparison(game.frames,state.frame);
 const previous=game.frames[comparison.previous], current=game.frames[comparison.current];
 const context=document.createElement('p');context.className='change-empty';
 context.textContent=`Opponent turn ${previous.step.turn} → ${current.step.turn} · Hand: ${previous.step.hand_size} → ${current.step.hand_size} cards${state.method==='historical'?` · Compatible decks: ${previous.compatible_decks.toLocaleString()} → ${current.compatible_decks.toLocaleString()}`:''}`;
 container.append(context);
 if(!changes.rises.length&&!changes.falls.length){
  const empty=document.createElement('p');empty.className='change-empty';empty.textContent='Predictions unchanged at displayed precision. Board activity alone does not update this baseline.';container.append(empty);return;
 }
 for(const [label,items,kind] of [['More likely',changes.rises,'rise'],['Less likely',changes.falls,'fall']]){
  if(!items.length)continue;
  const heading=document.createElement('h3');heading.className='change-label';heading.textContent=label;container.append(heading);
  for(const item of items){
   const button=document.createElement('button');button.className=`change-row ${kind}`;button.dataset.previewCard=item.card;
   button.setAttribute('aria-label',`${item.card}: ${percent(item.before)} to ${percent(item.after)}. Inspect prediction history`);
   button.innerHTML=`<span>${escape(item.card)}<small>${percent(item.before)} → ${percent(item.after)}</small></span><b>${item.delta>0?'+':''}${(item.delta*100).toFixed(1)} pp</b>`;
   button.addEventListener('click',()=>{state.selected=item.card;render();$('#card-detail').closest('details').open=true;});container.append(button);
  }
 }
}
function renderTrend(game,card) {
 const points=predictionHistory(game.frames,state.frame,card);
 const container=document.createElement('div');container.className='prediction-trend';
 const coordinates=points.flatMap((p,i)=>p.probability===null?[]:[`${10+i*280/Math.max(1,points.length-1)},${100-p.probability*90}`]);
 const description=points.map(p=>`State ${p.index+1}, ${p.player==='target'?'opponent':'observer'} turn ${p.turn}: ${p.probability===null?'Unavailable':percent(p.probability)}`).join('; ');
 container.innerHTML=`<h3>Probability over the replay so far</h3><svg viewBox="0 0 300 120" role="img" aria-label="${escape(card)} prediction history"><title>${escape(description)}</title><path d="M10 10 V100 H290" fill="none" stroke="currentColor" opacity=".25"/><text x="12" y="20">100%</text><text x="12" y="116">0%</text><polyline points="${coordinates.join(' ')}" fill="none" stroke="#d5b777" stroke-width="2.5"/>${coordinates.map(point=>{const [x,y]=point.split(',');return `<circle cx="${x}" cy="${y}" r="3" fill="#d5b777"/>`;}).join('')}</svg><small>State 1 → ${points.length} · Current: ${points.at(-1).probability===null?'Unavailable':percent(points.at(-1).probability)}. Future states are hidden.</small>`;
 const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Read values';details.append(summary);
 const values=document.createElement('p');values.textContent=description;details.append(values);container.append(details);
 return container;
}
function renderDetail(frame,game) {
 const card=state.selected;
 if(!card){$('#card-detail').textContent='No card predictions available.';return;}
 const p=frame.probabilities[card]||0, baseline=frame.baseline[card]||0;
 const seen=departures(game.frames,state.frame).filter(c=>c===card).length;
 const delta=predictionDelta(game.frames,state.frame,card);
 $('#card-detail').innerHTML=`<div class="detail-content"><div class="detail-title">${escape(card)}</div><div class="detail-prob">${frame.prediction_error?'Unavailable':percent(p)}</div><p>Chance the target holds <b>at least one copy</b> at this snapshot. It is not a prediction of when they will play it.</p><div class="detail-comparison"><div><small>Hand-size-only baseline</small><b>${percent(baseline)}</b></div><div><small>Change vs previous opponent turn</small><b>${delta===null?'—':`${delta>=0?'+':''}${(delta*100).toFixed(1)} pp`}</b></div></div><p>${escape(data.methods[state.method].description)}</p><p>${seen} copies of this card have publicly left hand. The detailed method descriptions above explain how each model uses this evidence.</p></div>`;
 const comparison=document.createElement('div');comparison.className='card-method-comparison';
 const heading=document.createElement('h3');heading.textContent='This card across methods';comparison.append(heading);
 for(const [method,info] of Object.entries(data.methods)){
  const row=document.createElement('p');row.textContent=`${info.label}: ${(frame.methods[method]===null?'Unavailable':percent(frame.methods[method][card]||0))}`;comparison.append(row);
 }
 $('#card-detail .detail-content').append(comparison);
 $('#card-detail .detail-content').append(renderTrend(game,card));
 if(images[card]) {
  const button=document.createElement('button');button.className='detail-image-button';button.dataset.previewCard=card;button.setAttribute('aria-label',`Enlarge ${card}`);
  button.innerHTML=`<img class="full-card-image" src="${escape(images[card].image)}" alt="${escape(card)}" loading="lazy" width="488" height="680"><span>Click to view full card ↗</span>`;
  button.addEventListener('click',()=>showCard(card));$('#card-detail .detail-content').prepend(button);
 }
}
function wireImageFallbacks() {
 for(const img of document.querySelectorAll('img:not(#hover-card-image)'))img.addEventListener('error',()=>{
  const fallback=document.createElement('span');fallback.className='image-fallback';fallback.textContent=img.alt;img.replaceWith(fallback);
 },{once:true});
}
let previewAnchor=null;
function hideCardPreview() {
 previewAnchor=null;
 const preview=$('#card-hover-preview');if(preview)preview.hidden=true;
}
function showCardPreview(anchor) {
 const card=anchor?.dataset.previewCard, art=images[card];
 if(!art||$('#card-dialog').open)return;
 previewAnchor=anchor;
 const preview=$('#card-hover-preview'), image=$('#hover-card-image');
 image.src=art.image;image.alt=card;
 $('#hover-card-name').textContent=card;
 preview.hidden=false;
 const rect=anchor.getBoundingClientRect(), gap=16, edge=10;
 const {width,height}=preview.getBoundingClientRect();
 const right=rect.right+gap;
 const left=right+width<=window.innerWidth-edge?right:rect.left-width-gap;
 preview.style.left=`${Math.max(edge,Math.min(left,window.innerWidth-width-edge))}px`;
 preview.style.top=`${Math.max(edge,Math.min(rect.top+(rect.height-height)/2,window.innerHeight-height-edge))}px`;
}
document.addEventListener('pointerover',event=>{
 if(event.pointerType==='touch')return;
 const anchor=event.target.closest('[data-preview-card]');
 if(anchor&&anchor!==previewAnchor)showCardPreview(anchor);
});
document.addEventListener('pointerout',event=>{
 if(previewAnchor&&previewAnchor.contains(event.target)&&!previewAnchor.contains(event.relatedTarget))hideCardPreview();
});
document.addEventListener('focusin',event=>{
 const anchor=event.target.closest('[data-preview-card]');if(anchor)showCardPreview(anchor);
});
document.addEventListener('focusout',hideCardPreview);
document.addEventListener('keydown',event=>{if(event.key==='Escape')hideCardPreview();});
window.addEventListener('scroll',hideCardPreview,true);
window.addEventListener('resize',hideCardPreview);
window.addEventListener('blur',hideCardPreview);
document.addEventListener('visibilitychange',hideCardPreview);
$('#hover-card-image').addEventListener('error',hideCardPreview);
function showCard(card) {
 hideCardPreview();
 stopPlayback();
 const image=images[card];if(!image)return;
 const dialog=$('#card-dialog');$('#dialog-title').textContent=card;
 $('#dialog-image').src=image.image;$('#dialog-image').alt=card;
 $('#dialog-credit').textContent=`Illustrated by ${image.artist || 'the credited card artist'} · Magic: The Gathering © Wizards of the Coast`;
 $('#dialog-source').href=image.source_url;dialog.showModal();
}
function renderBenchmark() {
 const report=data.experiments[Number($('#experiment-select').value)];
 const publicMetrics=report.metrics.public_movements,baseline=report.metrics.no_history;
 $('#experiment-stats').innerHTML=`<span><b>${report.evaluated_games.toLocaleString()}</b> evaluated games</span><span><b>${report.evaluated_snapshots.toLocaleString()}</b> snapshots</span><span><b>${report.training_decks.toLocaleString()}</b> training decks</span><span>Rounds 1–${report.settings.max_turn} · Basic lands included</span>`;
 const bars=['Recall@5','Recall@10','Recall@20'];
 $('#benchmark-chart').innerHTML=`<div class="chart-legend"><span><i class="legend-square"></i>Public card movements</span><span><i class="legend-square grey"></i>Hand size only</span></div>${bars.map(metric=>`<div class="chart-group"><div class="chart-label">${metric}</div><div class="chart-bars"><div class="chart-bar" style="--value:${publicMetrics[metric]*100}%"><span style="width:${publicMetrics[metric]*100}%"></span><b>${percent(publicMetrics[metric])}</b></div><div class="chart-bar baseline" style="--value:${baseline[metric]*100}%"><span style="width:${baseline[metric]*100}%"></span><b>${percent(baseline[metric])}</b></div></div></div>`).join('')}<div class="chart-legend" style="margin-top:25px;margin-bottom:4px"><span>MRR: <b>${publicMetrics.MRR.toFixed(3)}</b> vs ${baseline.MRR.toFixed(3)}</span><span>Brier ↓: <b>${publicMetrics.Brier.toFixed(5)}</b> vs ${baseline.Brier.toFixed(5)}</span></div>`;
 const ci=report.paired_differences?.['Recall@20']?.ci95;
 $('#benchmark-interpretation').innerHTML=`<b>Public movements improve top-20 hand recall by ${((publicMetrics['Recall@20']-baseline['Recall@20'])*100).toFixed(1)} percentage points.</b><small>${ci?`95% paired bootstrap interval: ${(ci[0]*100).toFixed(1)} to ${(ci[1]*100).toFixed(1)} percentage points, resampling whole draft events. `:''}Snapshots are averaged within each game, then games are averaged equally. File-prefix experiments; these are not full-set benchmarks.</small>`;
}
function showView(view) {
 if(!['explorer','progress'].includes(view))view='explorer';
 if(view!=='explorer')stopPlayback();
 for(const section of document.querySelectorAll('.view'))section.hidden=section.id!==view;
 for(const button of document.querySelectorAll('[data-view]')){const active=button.dataset.view===view;button.classList.toggle('active',active);button.setAttribute('aria-current',active?'page':'false');}
}
for(const button of document.querySelectorAll('[data-view]'))button.addEventListener('click',()=>{location.hash=button.dataset.view;});
window.addEventListener('hashchange',()=>showView(location.hash.slice(1)));
showView(location.hash.slice(1));
let importWorker=null, importRows=null, importPayload=null, importSequence=0;
function setupReplayImport(){
 const status=$('#import-status'), button=$('#run-import');
 function finish(){button.disabled=false;$('#cancel-import').hidden=true;importWorker?.terminate();importWorker=null;}
 $('#replay-file').addEventListener('change',async event=>{
  importWorker?.terminate();importWorker=null;$('#cancel-import').hidden=true;
  button.disabled=true;importRows=null;importPayload=null;$('#csv-game-label').hidden=true;
  const sequence=++importSequence;
  const file=event.target.files[0];if(!file)return;
  try{
   if(file.size>8*1024*1024)throw Error('File exceeds 8 MB. Export a smaller plain CSV or JSON replay.');
   const text=await file.text();if(sequence!==importSequence)return;
   if(file.name.toLowerCase().endsWith('.csv')){
    importRows=parseCsv(text);if(importRows.length>1000)throw Error('Choose a CSV containing at most 1,000 game rows.');
    const picker=$('#csv-game');picker.replaceChildren();
    importRows.forEach((row,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${row.draft_id||'Game'} · match ${row.match_number||'?'} · game ${row.game_number||'?'}`;picker.append(option);});
    $('#csv-game-label').hidden=false;status.textContent=`${importRows.length} game rows found. Choose one, then run predictions.`;
   }else if(file.name.toLowerCase().endsWith('.json')){
    importPayload=JSON.parse(text);status.textContent='Public replay JSON loaded. Ready to run predictions.';
   }else throw Error('Choose a plain .csv or public replay .json file. Replay URLs and Arena log files are not supported yet.');
   button.disabled=false;
  }catch(error){status.textContent=error.message;}
 });
 button.addEventListener('click',()=>{
  if(!importRows&&!importPayload)return;
  stopPlayback();hideCardPreview();button.disabled=true;$('#cancel-import').hidden=false;
  status.textContent='Loading saved models and running predictions locally in your browser…';
  try{
   importWorker=new Worker('replay-worker.js',{type:'module'});
   importWorker.onmessage=event=>{
    const result=event.data;
    if(result.type==='progress'){status.textContent=`Scored ${result.done} of ${result.total} supplied states…`;return;}
    if(result.type==='error'){status.textContent=result.message;finish();return;}
    if(result.type==='result'){
     const old=data.cases.findIndex(g=>g.imported);
     if(old>=0)data.cases[old]=result.game;else data.cases.push(result.game);
     const index=old>=0?old:data.cases.length-1,picker=$('#game-select');
     if(old<0){const option=document.createElement('option');option.value=index;picker.append(option);}
     picker.options[index].textContent=`Imported replay · ${result.game.frames.length} states`;
     state.game=index;state.frame=0;state.reveal=false;state.selected=null;state.method='smoothed_prior';
     picker.value=index;$('#method-select').value=state.method;
     status.textContent=`Loaded ${result.game.frames.length} states. ${result.game.stop_reason||'All supplied states processed.'} Public inputs only; actual hand labels are unavailable.`;
     finish();render();
    }
   };
   importWorker.onerror=()=>{status.textContent='Replay processing failed. Check your file and try again.';finish();};
   importWorker.postMessage(importRows?{row:importRows[Number($('#csv-game').value)]}:{payload:importPayload});
  }catch(error){status.textContent=error.message;finish();}
 });
 $('#cancel-import').addEventListener('click',()=>{finish();status.textContent='Import cancelled. Your previous replay is unchanged.';});
}
async function init(){
 try {
  const [response,imageResponse]=await Promise.all([fetch('data.json'),fetch('card-images.json')]);
  if(!response.ok||!imageResponse.ok)throw new Error('Data request failed');data=await response.json();images=await imageResponse.json();
  if(!data.cases.length)throw new Error('No replay cases available');
  for(const [method,info] of Object.entries(data.methods)){const option=document.createElement('option');option.value=method;option.textContent=info.label;$('#method-select').append(option);}
  $('#method-select').addEventListener('change',()=>{stopPlayback();state.method=$('#method-select').value;render();});
  data.cases.forEach((game,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${game.label} · ${game.complete?game.recorded_turns+' turns': 'excerpt'} · ${game.frames.length} states`;$('#game-select').append(option);});
  $('#game-select').addEventListener('change',()=>{stopPlayback();state.game=Number($('#game-select').value);state.frame=0;state.selected=null;state.reveal=false;render();});
  $('#frame-slider').addEventListener('input',()=>chooseFrame(Number($('#frame-slider').value)));
  $('#prev').addEventListener('click',()=>chooseFrame(Math.max(0,state.frame-1)));
  $('#next').addEventListener('click',()=>chooseFrame(Math.min(activeGame().frames.length-1,state.frame+1)));
  $('#reveal').addEventListener('click',()=>{state.reveal=!state.reveal;render();});
  $('#top-select').addEventListener('change',()=>{state.limit=Number($('#top-select').value);render();});
  $('#experiment-select').addEventListener('change',renderBenchmark);
  $('#dialog-close').addEventListener('click',()=>$('#card-dialog').close());
 $('#card-dialog').addEventListener('click',event=>{if(event.target===$('#card-dialog'))$('#card-dialog').close();});
 $('#card-dialog').addEventListener('close',()=>wireImageFallbacks());
 $('#play-toggle').addEventListener('click',()=>{
  if(playTimer){stopPlayback();return;}
  if(state.frame===activeGame().frames.length-1){state.frame=0;render();}
  $('#play-toggle').textContent='Ⅱ Pause';$('#play-toggle').setAttribute('aria-pressed','true');
  playTimer=setInterval(()=>{state.frame++;render();if(state.frame>=activeGame().frames.length-1)stopPlayback();},2200);
 });
 window.addEventListener('keydown',event=>{
  if(event.target!==document.body||$('#card-dialog').open||$('#explorer').hidden)return;
  if(event.key==='ArrowRight'){event.preventDefault();chooseFrame(Math.min(activeGame().frames.length-1,state.frame+1));}
  if(event.key==='ArrowLeft'){event.preventDefault();chooseFrame(Math.max(0,state.frame-1));}
 });
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stopPlayback();});
  setupReplayImport();
  render();renderBenchmark();$('#loading').hidden=true;$('#explorer-content').hidden=false;
 } catch(error) { $('#loading').textContent='The replay data could not be loaded. Please refresh the page.';console.error(error); }
}
init();

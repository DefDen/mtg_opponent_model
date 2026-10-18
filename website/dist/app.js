import { percent, ranked, recallAt, departures, predictionDelta, colorClass, cardKind, boardEntries, predictionChanges, predictionHistory } from './model-view.js';
const $ = selector => document.querySelector(selector);
const escape = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data;
let images = {};
const state = { game:0, frame:0, reveal:false, limit:10, selected:null };
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
function activeGame() { return data.cases[state.game]; }
function render() {
 hideCardPreview();
 const game=activeGame(), frame=game.frames[state.frame];
 $('#game-meta').textContent=`${game.complete?'Full recorded game':'Early-turn excerpt'} · ${game.on_play?'Opponent plays first':'Opponent draws first'}`;
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
 $('#compatible').textContent=frame.compatible_decks.toLocaleString();
 $('#departed-count').textContent=departures(game.frames,state.frame).length;
 $('#reveal').textContent=state.reveal?'Hide opponent’s hand':'Reveal opponent’s hand';
 $('#reveal').setAttribute('aria-pressed',String(state.reveal));
 $('#hand-status').textContent=state.reveal?'REVEALED':'HIDDEN';
 $('#hand-mode').textContent=state.reveal?'Evaluation labels · hidden from the model':'Identities hidden';
 const predictions=ranked(frame.probabilities,state.limit);
 if(!state.selected) state.selected=predictions[0]?.[0];
 const list=$('#prediction-list');list.replaceChildren();
 predictions.forEach(([card,p],i)=>{
  const button=document.createElement('button');button.className=`prediction-row ${state.selected===card?'selected':''}`;
  button.dataset.previewCard=card;
  button.setAttribute('aria-pressed',String(state.selected===card));
  const delta=predictionDelta(game.frames,state.frame,card);
  const hit=state.reveal?(frame.actual_hand[card]?'<span class="hit">✓ In hand</span>':'<span class="miss">Not held</span>'):'';
  button.innerHTML=`<span class="rank">${String(i+1).padStart(2,'0')}</span>${symbol(card)}<div><div class="card-name">${escape(card)} ${hit}</div><div class="card-kind">${escape(cardKind(data.cards[card]))}</div></div><div class="probability">${percent(p)}<div class="prob-bar"><span style="width:${p*100}%"></span></div><div class="prob-delta">${delta===null?'First snapshot':Math.abs(delta)<0.0005?'Unchanged':`${delta>=0?'+':''}${(delta*100).toFixed(1)} pp`}</div></div>`;
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
 $('#snapshot-score').textContent=state.reveal?`This snapshot: ${percent(recallAt(frame.probabilities,frame.actual_hand,state.limit))} of actual card types recovered in the top ${state.limit}.`:'Reveal the labels to check which guesses were right. They were never used to make these predictions.';
 renderBoard(frame,game);
 renderChanges(game);
 renderDetail(frame,game);
 wireImageFallbacks();
 $('#selection-note').textContent=`${game.complete?`${game.selection}. Every one of its ${game.recorded_snapshot_count} available turn-end states passed the hand audit and prior support checks; this is a selected demonstration, not a new benchmark.`:data.selection+'.'} Predictions use the no-mulligan 5,000-game experiment’s ${data.training_decks.toLocaleString()} training decks. ${game.stop_reason?'This excerpt stops when movement accounting becomes unsupported.':''}`;
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
function renderChanges(game) {
 const changes=predictionChanges(game.frames,state.frame), container=$('#prediction-changes');
 container.replaceChildren();
 if(!changes){container.innerHTML='<p class="change-empty">First recorded state: use Next or Play to see how predictions change.</p>';return;}
 const previous=game.frames[state.frame-1], current=game.frames[state.frame];
 const context=document.createElement('p');context.className='change-empty';
 context.textContent=`Hand: ${previous.step.hand_size} → ${current.step.hand_size} cards · Compatible decks: ${previous.compatible_decks.toLocaleString()} → ${current.compatible_decks.toLocaleString()}`;
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
 const coordinates=points.map((p,i)=>`${10+i*280/Math.max(1,points.length-1)},${100-p.probability*90}`);
 const description=points.map(p=>`State ${p.index+1}, ${p.player==='target'?'opponent':'observer'} turn ${p.turn}: ${percent(p.probability)}`).join('; ');
 container.innerHTML=`<h3>Probability over the replay so far</h3><svg viewBox="0 0 300 120" role="img" aria-label="${escape(card)} prediction history"><title>${escape(description)}</title><path d="M10 10 V100 H290" fill="none" stroke="currentColor" opacity=".25"/><text x="12" y="20">100%</text><text x="12" y="116">0%</text><polyline points="${coordinates.join(' ')}" fill="none" stroke="#d5b777" stroke-width="2.5"/>${coordinates.map(point=>{const [x,y]=point.split(',');return `<circle cx="${x}" cy="${y}" r="3" fill="#d5b777"/>`;}).join('')}</svg><small>State 1 → ${points.length} · Current: ${percent(points.at(-1).probability)}. Future states are hidden.</small>`;
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
 $('#card-detail').innerHTML=`<div class="detail-content"><div class="detail-title">${escape(card)}</div><div class="detail-prob">${percent(p)}</div><p>Chance the target holds <b>at least one copy</b> at this snapshot. It is not a prediction of when they will play it.</p><div class="detail-comparison"><div><small>Hand-size-only baseline</small><b>${percent(baseline)}</b></div><div><small>Change since last snapshot</small><b>${delta===null?'—':`${delta>=0?'+':''}${(delta*100).toFixed(1)} pp`}</b></div></div><p>${seen?`${seen} cop${seen===1?'y has':'ies have'} already left the hand publicly. The model removes those copies; additional copies can still be held.`:'This card has not publicly left the hand yet. Its score comes from the plausible decks and the target’s current hand size.'}</p><p style="margin-top:10px">${frame.compatible_decks.toLocaleString()} historical decks contain enough copies to explain all departures so far. The model weights them by the observed card frequencies, then estimates what remains in hand.</p></div>`;
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
async function init(){
 try {
  const [response,imageResponse]=await Promise.all([fetch('data.json'),fetch('card-images.json')]);
  if(!response.ok||!imageResponse.ok)throw new Error('Data request failed');data=await response.json();images=await imageResponse.json();
  if(!data.cases.length)throw new Error('No replay cases available');
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
  render();renderBenchmark();$('#loading').hidden=true;$('#explorer-content').hidden=false;
 } catch(error) { $('#loading').textContent='The replay data could not be loaded. Please refresh the page.';console.error(error); }
}
init();

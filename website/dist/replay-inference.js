// Portable inference for the exported fitted BLB models. No hidden labels are inputs.
const sum = xs => xs.reduce((a,b)=>a+b,0);
const count = xs => {const r={};for(const x of xs)r[x]=(r[x]||0)+1;return r;};
const rising = (a,n) => sum(Array.from({length:n},(_,i)=>Math.log(a+i)));
const falling = (a,n) => n>a?-Infinity:sum(Array.from({length:n},(_,i)=>Math.log(a-i)));
function choose(n,k){if(k<0||k>n)return 0;let r=1;for(let i=1;i<=Math.min(k,n-k);i++)r*= (n-i+1)/i;return r;}
function evidence(history){
 let size=history.opening_hand_size, required=size, unknown=0, removed=0;const seen={};
 for(const s of history.steps){
  if(size+s.draws-s.departed.length-(s.unknown_departures||0)!==s.hand_size)throw Error('Hand size does not match public movements.');
  required=Math.max(required,removed+unknown+size+s.draws);
  for(const c of s.departed)seen[c]=(seen[c]||0)+1;
  removed+=s.departed.length;unknown+=s.unknown_departures||0;size=s.hand_size;
 }
 return {seen,size,required};
}
function smoothed(model,history){
 const {seen,size:n,required}=evidence(history), k=sum(Object.values(seen));
 const values=model.cards.map(c=>seen[c]||0), kl=sum(values.filter((_,i)=>model.land[i])),ks=k-kl;
 const groups={};
 for(const [g,alpha] of Object.entries(model.alpha)){
  let term=0;
  for(const isLand of [true,false]){
   const indices=model.cards.map((_,i)=>i).filter(i=>model.land[i]===isLand);
   const observed=isLand?kl:ks;
   if(observed)term+=sum(indices.map(i=>rising(alpha[i],values[i])))-rising(sum(indices.map(i=>alpha[i])),observed);
  }
  groups[g]=term;
 }
 const components=model.components.filter(([,s,l])=>s>=required&&kl<=l&&ks<=s-l)
  .map(([g,s,l,p])=>({g,lands:l-kl,spells:s-l-ks,log:Math.log(p)+falling(l,kl)+falling(s-l,ks)-falling(s,k)+groups[g]}));
 if(!components.length)throw Error('No finite BLB deck shape can explain these public movements.');
 const max=Math.max(...components.map(c=>c.log)),weights=components.map(c=>Math.exp(c.log-max)),denom=sum(weights);
 const result=model.cards.map(()=>0);
 for(let j=0;j<components.length&&n;j++){
  const {g,lands,spells}=components[j],weight=weights[j]/denom;
  for(const isLand of [true,false]){
   const available=isLand?lands:spells,other=isLand?spells:lands;
   if(!available)continue;
   const indices=model.cards.map((_,i)=>i).filter(i=>model.land[i]===isLand);
   const aTotal=sum(indices.map(i=>model.alpha[g][i]+values[i]));
   for(const i of indices){
    const alpha=model.alpha[g][i]+values[i];let absent=1,presence=0;
    for(let m=0;m<=Math.min(n,available);m++){
     if(m)absent*=(aTotal-alpha+m-1)/(aTotal+m-1);
     if(n-m<=other)presence+=choose(available,m)*choose(other,n-m)/choose(available+other,n)*(1-absent);
    }
    result[i]+=weight*presence;
   }
  }
 }
 return Object.fromEntries(model.cards.map((c,i)=>[c,Math.max(0,Math.min(1,result[i]))]));
}
function historical(model,history){
 const {seen,size,required}=evidence(history),index=new Map(model.cards.map((c,i)=>[c,i]));
 const observations=Object.entries(seen).map(([c,n])=>[index.get(c),n]);const supported=[];
 for(const sparse of model.historical_decks){
  const cards=new Map(sparse),total=sum(sparse.map(([,n])=>n));
  if(total<required||observations.some(([i,n])=>(cards.get(i)||0)<n))continue;
  let removed=0,log=0;
  for(const [i,n] of observations){for(let j=0;j<n;j++)log+=Math.log(cards.get(i)-j)-Math.log(total-removed++);cards.set(i,cards.get(i)-n);}
  supported.push({cards,total:total-removed,log});
 }
 if(!supported.length)return {probabilities:null,count:0};
 const max=Math.max(...supported.map(d=>d.log)),denom=sum(supported.map(d=>Math.exp(d.log-max)));
 const p=model.cards.map(()=>0);
 for(const d of supported){const weight=Math.exp(d.log-max)/denom;
  for(const [i,n] of d.cards){let absent=1;for(let j=0;j<size;j++)absent*=Math.max(d.total-n-j,0)/(d.total-j);p[i]+=weight*(1-absent);}
 }
 return {probabilities:Object.fromEntries(model.cards.map((c,i)=>[c,p[i]])),count:supported.length};
}
function features(model,history,prior,board,includeBoard){
 const step=history.steps.at(-1),seen=count(history.steps.flatMap(s=>s.departed)),last={};
 for(const s of history.steps)for(const c of s.departed)last[c]=s.turn;
 const shared=[step.hand_size,history.opening_hand_size,step.turn,+(step.active_player==='target'),
  sum(history.steps.map(s=>s.draws)),sum(history.steps.map(s=>s.departed.length)),sum(history.steps.map(s=>s.unknown_departures||0))];
 const target=board.target,observer=board.observer;
 const onBoard=p=>count([...p.lands,...p.creatures,...p.other]);const tc=onBoard(target),oc=onBoard(observer);
 const colors=[...'WUBRG'];
 return model.cards.map(card=>{
  const m=model.metadata[card]||{colors:'',types:'',mana_value:0};
  const row=[prior[card]||0,seen[card]||0,step.turn-(last[card]||0),...shared,m.mana_value,
   +['Plains','Island','Swamp','Mountain','Forest'].includes(card),
   ...['Land','Creature','Instant','Sorcery','Artifact','Enchantment','Planeswalker'].map(t=>+m.types.includes(t)),...colors.map(c=>+m.colors.includes(c))];
  if(includeBoard)row.push(target.life,observer.life,target.lands.length,observer.lands.length,target.creatures.length,observer.creatures.length,
   target.other.length,observer.other.length,tc[card]||0,oc[card]||0,...colors.map(c=>target.lands.filter(l=>(model.metadata[l]?.colors||'').includes(c)).length));
  return row.map(Math.fround); // Match the float32 Python feature matrix.
 });
}
function boosted(estimator,rows){return rows.map(row=>{
 if(row[0]<=0)return 0;
 let score=estimator.base;
 for(const tree of estimator.trees){let node=tree[0];while(!node[5])node=tree[Number.isNaN(row[node[1]])?(node[6]?node[3]:node[4]):row[node[1]]<=node[2]?node[3]:node[4]];score+=node[0];}
 return 1/(1+Math.exp(-score));
});}
export function scoreHistory(model,history,board){
 const known=new Set(model.cards);
 for(const s of history.steps)for(const c of s.departed)if(!known.has(c))throw Error(`Card outside trained BLB pool: ${c}`);
 const smooth=smoothed(model,history),finite=historical(model,history);
 const baseline=historical(model,{opening_hand_size:history.steps.at(-1).hand_size,steps:[]}).probabilities;
 const methods={historical:finite.probabilities,hand_size:baseline,smoothed_prior:smooth};
 for(const method of ['boosted_history','boosted_board']){
  const values=boosted(model.trees[method],features(model,history,smooth,board,method==='boosted_board'));
  methods[method]=Object.fromEntries(model.cards.map((c,i)=>[c,values[i]]));
 }
 return {methods,compatible_decks:finite.count,baseline};
}
export function validatePublicReplay(payload,model){
 if(!payload||payload.schema_version!==1||payload.expansion!=='BLB')throw Error('Expected public replay JSON schema_version 1, expansion BLB.');
 if(!Number.isInteger(payload.opening_hand_size)||payload.opening_hand_size<0||payload.opening_hand_size>7)throw Error('Opening hand size must be 0–7.');
 if(typeof payload.on_play!=='boolean')throw Error('on_play must be true or false.');
 if(!Array.isArray(payload.frames)||!payload.frames.length||payload.frames.length>60)throw Error('Provide 1–60 turn-end frames.');
 const known=new Set(model.cards);let previousOrdinal=-1;const steps=[];
 const frames=payload.frames.map(frame=>{
  const s=frame?.step;
  if(!s||!Number.isInteger(s.turn)||s.turn<1||s.turn>30||!['target','observer'].includes(s.active_player))throw Error('Invalid turn or active_player.');
  for(const key of ['draws','hand_size','unknown_departures'])if(!Number.isInteger(key==='unknown_departures'?(s[key]??0):s[key])||(s[key]??0)<0||(s[key]??0)>100)throw Error(`Invalid ${key}.`);
  if(!Array.isArray(s.departed)||s.departed.length>100||s.departed.some(c=>typeof c!=='string'||!known.has(c)))throw Error('Departures must be card names in the trained BLB pool.');
  const ordinal=s.turn*2+(s.active_player===(payload.on_play?'target':'observer')?0:1);
  if(previousOrdinal>=0&&ordinal!==previousOrdinal+1)throw Error('Frames must be consecutive half-turns in play/draw order.');
  if(previousOrdinal<0&&ordinal!==2)throw Error('Replay must begin at the first half-turn.');previousOrdinal=ordinal;
  const board={};
  for(const player of ['target','observer']){
   const b=frame.board?.[player];if(!b||!Number.isFinite(b.life)||Math.abs(b.life)>10000)throw Error('Both public boards need finite life totals.');
   board[player]={life:b.life};
   for(const zone of ['lands','creatures','other']){
    if(!Array.isArray(b[zone])||b[zone].length>200||b[zone].some(c=>typeof c!=='string'||!c.length||c.length>150))throw Error('Invalid public board zone.');
    board[player][zone]=[...b[zone]];
   }
  }
  if(!Number.isInteger(frame.board.observer.hand_size)||frame.board.observer.hand_size<0||frame.board.observer.hand_size>100)throw Error('Observer public hand_size is required.');
  board.observer.hand_size=frame.board.observer.hand_size;
  // Whitelist public fields. Never propagate actual hands, draw identities, or decklists.
  const step={turn:s.turn,active_player:s.active_player,draws:s.draws,departed:[...s.departed],hand_size:s.hand_size,unknown_departures:s.unknown_departures||0};
  steps.push(step);evidence({opening_hand_size:payload.opening_hand_size,steps});
  return {step,board};
 });
 return {opening_hand_size:payload.opening_hand_size,on_play:payload.on_play,frames};
}
export function parseCsv(text){
 text=text.replace(/^\uFEFF/,'');
 const rows=[];let row=[],field='',quoted=false,closed=false;
 for(let i=0;i<text.length;i++){
  const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++;}else if(quoted){quoted=false;closed=true;}else if(!field&&!closed)quoted=true;else throw Error('Invalid CSV quoting.');}
  else if(!quoted&&(c===','||c==='\n'||c==='\r')){row.push(field);field='';closed=false;if(c!==','){if(c==='\r'&&text[i+1]==='\n')i++;if(row.some(v=>v))rows.push(row);row=[];}}
  else {if(closed)throw Error('Unexpected characters after CSV closing quote.');field+=c;}
 }
 if(quoted)throw Error('Unterminated CSV quote.');if(field||row.length){row.push(field);rows.push(row);}
 if(rows.length<2)throw Error('CSV needs a header and at least one game row.');
 const header=rows.shift();header[0]=header[0].replace(/^\uFEFF/,'');
 if(new Set(header).size!==header.length)throw Error('Duplicate CSV columns.');
 return rows.map(values=>{if(values.length!==header.length)throw Error('Incomplete CSV row.');return Object.fromEntries(header.map((h,i)=>[h,values[i]]));});
}
export function wideRowToPublic(row,names){
 if(!('user_turn_1_eot_user_cards_in_hand' in row))throw Error('Expected wide 17Lands replay CSV with user_turn_1_ columns, not game-data or long-form CSV.');
 if(row.expansion!=='BLB')throw Error('Only BLB replay rows are supported by these trained models.');
 if(!['True','False'].includes(row.on_play))throw Error('Invalid CSV on_play.');
 const mulligans=Number(row.num_mulligans);if(!Number.isInteger(mulligans)||mulligans<0||mulligans>7)throw Error('Invalid mulligan count.');
 const ids=s=>(s||'').split('|').filter(Boolean),named=s=>ids(s).map(id=>{if(!names[id])throw Error(`Unknown public Arena card ID: ${id}`);return names[id];});
 const opening=ids(row.opening_hand).length;if(opening!==7-mulligans)throw Error('Unsupported opening hand size.');
 const onPlay=row.on_play==='True',frames=[];let reason=null,size=opening;
 const sides=onPlay?['user','oppo']:['oppo','user'];
 outer:for(let turn=1;turn<=30;turn++)for(const side of sides){
  const p=`${side}_turn_${turn}_`,a=row[p+'eot_user_life'],b=row[p+'eot_oppo_life'];
  if(a===undefined||b===undefined||(+a===0&&+b===0))break outer;
  if(side==='user'&&row[p+'cards_tutored']){reason='Tutor: target draw identities or selection cannot be modeled.';break outer;}
  const departed=(side==='user'?['lands_played','creatures_cast','non_creatures_cast','cards_discarded','user_instants_sorceries_cast']:['user_instants_sorceries_cast']).flatMap(key=>named(row[p+key]));
  const draws=side==='user'?ids(row[p+'cards_drawn']).length:0,hand=ids(row[p+'eot_user_cards_in_hand']).length;
  if(size+draws-departed.length!==hand){reason='Unaccounted hand movement: stopped before this state.';break outer;}
  const board={};for(const [player,key] of [['target','user'],['observer','oppo']])board[player]={life:Number(row[p+`eot_${key}_life`]),
   lands:named(row[p+`eot_${key}_lands_in_play`]),creatures:named(row[p+`eot_${key}_creatures_in_play`]),other:named(row[p+`eot_${key}_non_creatures_in_play`])};
  board.observer.hand_size=Number(row[p+'eot_oppo_cards_in_hand']||0);
  frames.push({step:{turn,active_player:side==='user'?'target':'observer',draws,departed,hand_size:hand},board});size=hand;
 }
 if(!frames.length)throw Error(reason||'No supported half-turn records found. Expected wide 17Lands replay CSV.');
 return {schema_version:1,expansion:'BLB',opening_hand_size:opening,on_play:onPlay,frames,stop_reason:reason,
  source_game:[row.draft_id,row.match_number,row.game_number].join(':')};
}
export function scoreReplay(model,payload,onProgress=()=>{}){
 const publicReplay=validatePublicReplay(payload,model),steps=[],frames=[];let reason=payload.stop_reason||null;
 for(const f of publicReplay.frames){
  steps.push(f.step);let scores;
  try{scores=scoreHistory(model,{opening_hand_size:publicReplay.opening_hand_size,steps},f.board);}
  catch(error){reason=`Prediction stopped: ${error.message}`;break;}
  frames.push({...f,...scores,probabilities:scores.methods.historical||{},actual_hand:{},metrics:{},labels_available:false,
   method_errors:scores.methods.historical?{}:{historical:'No historical training deck matches this public history.'}});
  onProgress(frames.length,publicReplay.frames.length);
 }
 if(!frames.length)throw Error(reason||'No supported predictions.');
 return {id:'imported-replay',label:'Imported replay',on_play:publicReplay.on_play,opening_hand_size:publicReplay.opening_hand_size,frames,
  complete:false,imported:true,source_game:payload.source_game||'uploaded public replay',stop_reason:reason,
  coverage_note:'Public count-consistent prefix only; no private-hand identity audit. Cast source zones remain unverified. Mulligans use a uniform kept-hand approximation.'};
}

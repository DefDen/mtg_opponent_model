export const percent = value => `${(value * 100).toFixed(1)}%`;
export function ranked(probabilities, limit = 10) {
 return Object.entries(probabilities).filter(([, p]) => p > 0).sort((a,b) => b[1] - a[1] || a[0].localeCompare(b[0], 'en')).slice(0,limit);
}
export function recallAt(probabilities, actualHand, k) {
 const targets = Object.keys(actualHand).filter(card=>actualHand[card]>0);
 if (!targets.length) return 0;
 const predictions = new Set(ranked(probabilities,k).map(([card])=>card));
 return targets.filter(card=>predictions.has(card)).length/targets.length;
}
export function departures(frames, index) {
 return frames.slice(0,index+1).flatMap(frame=>frame.step.departed);
}
// Hold the latest opponent-to-opponent comparison through observer turns.
export function opponentComparison(frames, index) {
 let current=index;
 while(current>=0&&frames[current].step.active_player!=='target')current--;
 let previous=current-1;
 while(previous>=0&&frames[previous].step.active_player!=='target')previous--;
 return previous<0?null:{previous,current};
}
export function predictionDelta(frames, index, card) {
 const comparison=opponentComparison(frames,index);
 if(!comparison||frames[comparison.current].prediction_error||frames[comparison.previous].prediction_error)return null;
 return (frames[comparison.current].probabilities[card]||0)-(frames[comparison.previous].probabilities[card]||0);
}
export function colorClass(metadata = {}) {
 const colors = metadata.colors || '';
 const present = 'WUBRG'.split('').filter(c=>colors.includes(c));
 return present.length===1 ? `color-${present[0]}` : '';
}
export function cardKind(metadata = {}) {
 return (metadata.types || 'Card').split(/ — | - /)[0];
}

export function boardEntries(cards, previousCards = []) {
 const old = new Map();
 for(const card of previousCards)old.set(card,(old.get(card)||0)+1);
 return cards.map((card,index)=>{
  const count=old.get(card)||0;
  if(count)old.set(card,count-1);
  return {card,key:`${card}:${index}`,isNew:!count};
 });
}

// Compare the full candidate pool, including cards whose probability fell to zero.
export function predictionChanges(frames, index, limit = 3) {
 const comparison=opponentComparison(frames,index);
 if(!comparison||frames[comparison.current].prediction_error||frames[comparison.previous].prediction_error)return null;
 const before = frames[comparison.previous].probabilities, after = frames[comparison.current].probabilities;
 const cards = new Set([...Object.keys(before), ...Object.keys(after)]);
 const changes = [...cards].map(card => ({card, before:before[card]||0, after:after[card]||0,
  delta:(after[card]||0)-(before[card]||0)}));
 const order = (a,b) => Math.abs(b.delta)-Math.abs(a.delta) || a.card.localeCompare(b.card, 'en');
 return {rises:changes.filter(c=>c.delta>=0.0005).sort(order).slice(0,limit),
  falls:changes.filter(c=>c.delta<=-0.0005).sort(order).slice(0,limit)};
}
export function predictionHistory(frames, index, card) {
 return frames.slice(0,index+1).map((frame,i)=>({index:i, turn:frame.step.turn,
  player:frame.step.active_player, probability:frame.prediction_error?null:(frame.probabilities[card]||0)}));
}

export function gameForMethod(game, method) {
 return {...game,frames:game.frames.map(frame=>({...frame,
  prediction_error:frame.method_errors?.[method],
  probabilities:frame.methods&&method in frame.methods?(frame.methods[method]||{}):(method==='hand_size'?frame.baseline:frame.probabilities)}))};
}

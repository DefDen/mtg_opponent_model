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
export function predictionDelta(frames, index, card) {
 if (!index) return null;
 return (frames[index].probabilities[card]||0)-(frames[index-1].probabilities[card]||0);
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
 if (!index) return null;
 const before = frames[index-1].probabilities, after = frames[index].probabilities;
 const cards = new Set([...Object.keys(before), ...Object.keys(after)]);
 const changes = [...cards].map(card => ({card, before:before[card]||0, after:after[card]||0,
  delta:(after[card]||0)-(before[card]||0)}));
 const order = (a,b) => Math.abs(b.delta)-Math.abs(a.delta) || a.card.localeCompare(b.card, 'en');
 return {rises:changes.filter(c=>c.delta>=0.0005).sort(order).slice(0,limit),
  falls:changes.filter(c=>c.delta<=-0.0005).sort(order).slice(0,limit)};
}
export function predictionHistory(frames, index, card) {
 return frames.slice(0,index+1).map((frame,i)=>({index:i, turn:frame.step.turn,
  player:frame.step.active_player, probability:frame.probabilities[card]||0}));
}

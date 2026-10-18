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

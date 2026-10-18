import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ranked,recallAt,predictionDelta,departures,colorClass,boardEntries} from '../dist/model-view.js';
import {readFileSync} from 'node:fs';
test('ranking and recall use card types, omit zero-probability cards',()=>{
 assert.deepEqual(ranked({A:.8,B:0,C:.2},5),[['A',.8],['C',.2]]);
 assert.equal(recallAt({A:.8,B:0},{A:2,B:1},5),.5);
});
test('departure counts and prediction deltas stop at the selected half-turn',()=>{
 const frames=[{step:{departed:['A']},probabilities:{A:.5}},{step:{departed:['A','B']},probabilities:{A:.25}}];
 assert.deepEqual(departures(frames,0),['A']);
 assert.deepEqual(departures(frames,1),['A','A','B']);
 assert.equal(predictionDelta(frames,0,'A'),null);
 assert.equal(predictionDelta(frames,1,'A'),-.25);
});
test('colorless and multicolor cards are not labeled a single color',()=>{
 assert.equal(colorClass({colors:'RG'}),'');assert.equal(colorClass({colors:'G'}),'color-G');
});
test('exported replay counts are consistent; probabilities and labels stay distinct',()=>{
 const data=JSON.parse(readFileSync(new URL('../dist/data.json',import.meta.url)));
 assert.equal(data.training_decks,data.experiments[1].training_decks);
 assert.equal(data.cases.filter(game=>!game.complete).length,8);
 assert.ok(data.cases.some(game=>game.complete));
 for(const game of data.cases){
  let size=game.opening_hand_size;
  for(const frame of game.frames){
   assert.equal(size+frame.step.draws-frame.step.departed.length,frame.step.hand_size);
   assert.equal(Object.values(frame.actual_hand).reduce((a,b)=>a+b,0),frame.step.hand_size);
   assert.equal('actual_hand' in frame.step,false);
   assert.ok(Object.values(frame.probabilities).every(p=>p>=0&&p<=1));
   assert.ok(Math.abs(recallAt(frame.probabilities,frame.actual_hand,20)-frame.metrics['Recall@20'])<1e-12);
   size=frame.step.hand_size;
  }
 }
});
test('default replay covers every recorded state through its final turn',()=>{
 const data=JSON.parse(readFileSync(new URL('../dist/data.json',import.meta.url)));
 const game=data.cases[0];
 assert.equal(game.complete,true);
 assert.equal(game.frames.length,game.recorded_snapshot_count);
 assert.equal(game.frames.length,17);
 assert.equal(game.frames.at(-1).step.turn,game.recorded_turns);
 assert.equal(game.recorded_turns,9);
 assert.equal(game.recorded_winner,'observer');
 assert.equal(game.stop_reason,null);
 assert.ok(game.frames.every(frame=>frame.compatible_decks>0));
});
test('every displayed card has a bundled JPEG and attribution',()=>{
 const data=JSON.parse(readFileSync(new URL('../dist/data.json',import.meta.url)));
 const images=JSON.parse(readFileSync(new URL('../dist/card-images.json',import.meta.url)));
 const visible=new Set();
 for(const game of data.cases)for(const frame of game.frames){
  for(const card of Object.keys(frame.actual_hand))visible.add(card);
  for(const [card] of ranked(frame.probabilities,20))visible.add(card);
  for(const card of frame.step.departed)visible.add(card);
  for(const player of Object.values(frame.board))for(const zone of ['lands','creatures','other'])for(const card of player[zone])visible.add(card);
 }
 for(const card of visible){
  const art=images[card];assert.ok(art,`Missing artwork for ${card}`);
  assert.match(art.image,/^assets\/cards\/[a-f0-9-]+\.jpg$/);
  assert.match(art.source_url,/^https:\/\/scryfall.com\//);
  assert.ok(art.artist);
  const jpeg=readFileSync(new URL(`../dist/${art.image}`,import.meta.url));
  assert.equal(jpeg[0],255);assert.equal(jpeg[1],216);
  assert.equal(jpeg[jpeg.length-2],255);assert.equal(jpeg[jpeg.length-1],217);
 }
});
test('board transitions preserve duplicates and omit cards that left play',()=>{
 assert.deepEqual(boardEntries(['A','A','B'],['A','C']).map(({card,isNew})=>({card,isNew})),[
  {card:'A',isNew:false},{card:'A',isNew:true},{card:'B',isNew:true}]);
 assert.deepEqual(boardEntries([],['C']),[]);
});
test('all replay frames contain public board snapshots for both sides',()=>{
 const data=JSON.parse(readFileSync(new URL('../dist/data.json',import.meta.url)));
 for(const game of data.cases)for(const frame of game.frames){
  assert.deepEqual(Object.keys(frame.board).sort(),['observer','target']);
  for(const player of Object.values(frame.board)){
   assert.ok(Number.isFinite(player.life));
   for(const zone of ['lands','creatures','other'])assert.ok(Array.isArray(player[zone]));
   assert.equal('actual_hand' in player,false);
  }
  assert.ok(Number.isInteger(frame.board.observer.hand_size));
 }
});

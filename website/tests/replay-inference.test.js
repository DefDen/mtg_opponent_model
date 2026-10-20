import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parseCsv,wideRowToPublic,scoreHistory,scoreReplay,validatePublicReplay} from '../dist/replay-inference.js';
import {gameForMethod,predictionDelta} from '../dist/model-view.js';
const read=name=>JSON.parse(readFileSync(new URL('../dist/'+name,import.meta.url)));
const model=read('browser-models.json'),data=read('data.json'),example=read('public-replay-example.json');
const clone=x=>JSON.parse(JSON.stringify(x));

test('browser inference matches all five Python-exported methods on every saved state',()=>{
 for(const game of data.cases){const steps=[];
  for(const frame of game.frames){steps.push(frame.step);
   const result=scoreHistory(model,{opening_hand_size:game.opening_hand_size,steps},frame.board);
   assert.equal(result.compatible_decks,frame.compatible_decks);
   for(const [method,probs] of Object.entries(result.methods))for(const card of model.cards)
    assert.ok(Math.abs(probs[card]-(frame.methods[method][card]||0))<1e-9,`${game.id}: ${method}: ${card}`);
  }
 }
});
test('import validates ordering, format, accounting, pool, and board data',()=>{
 for(const mutate of [
  x=>x.expansion='OTHER', x=>x.on_play='True', x=>x.frames[0].step.draws=-1,
  x=>x.frames[0].step.hand_size++,x=>x.frames[0].step.departed=['Not a BLB card'],
  x=>x.frames[1].step.turn=5,x=>delete x.frames[0].board.target,
  x=>x.frames[0].board.observer.hand_size='7',x=>x.frames=[],
 ]){const payload=clone(example);mutate(payload);assert.throws(()=>validatePublicReplay(payload,model));}
});
test('public import strips all private labels and decklists; they cannot change scores',()=>{
 const a=clone(example);a.frames=a.frames.slice(0,2);
 const b=clone(a);b.decklist={'Forest':40};b.actual_hand={'Mountain':7};
 b.frames[0].actual_hand={'Mountain':7};b.frames[0].step.draw_identities=['Mountain'];
 assert.deepEqual(validatePublicReplay(a,model),validatePublicReplay(b,model));
 assert.deepEqual(scoreReplay(model,a),scoreReplay(model,b));
 const game=scoreReplay(model,b);
 assert.ok(game.frames.every(f=>f.labels_available===false&&Object.keys(f.actual_hand).length===0));
});
test('historical prior exhaustion is explicit; smoothed and tree methods remain available',()=>{
 const p=clone(example);p.frames=p.frames.slice(0,1);p.opening_hand_size=7;
 const absent=model.cards.find(c=>!model.historical_decks.some(deck=>deck.some(([i])=>model.cards[i]===c)));
 assert.ok(absent);
 p.frames[0].step.draws=0;p.frames[0].step.departed=[absent];p.frames[0].step.hand_size=6;
 const game=scoreReplay(model,p),frame=game.frames[0];
 assert.equal(frame.methods.historical,null);assert.match(frame.method_errors.historical,/No historical/);
 assert.ok(frame.methods.boosted_history);assert.ok(frame.methods.smoothed_prior);
 assert.deepEqual(gameForMethod(game,'historical').frames[0].probabilities,{});
 assert.equal(predictionDelta(gameForMethod(game,'historical').frames,0,'Forest'),null);
});
test('CSV parser handles quoted commas, escaped quotes, BOM, CRLF, and rejects incomplete rows',()=>{
 assert.deepEqual(parseCsv('\uFEFFa,b\r\n"A, B","a""b"\r\n'),[{a:'A, B',b:'a"b'}]);
 assert.throws(()=>parseCsv('a,b\n1'));
 assert.throws(()=>parseCsv('a,b\n"unfinished'));
 assert.throws(()=>parseCsv('a,a\n1,2'));
 assert.throws(()=>parseCsv('a,b\n"quoted"junk,2'));
 assert.deepEqual(parseCsv('\uFEFF"a",b\n1,2'),[{a:'1',b:'2'}]);
});
test('wide CSV adapter uses private-list lengths only and stops before unexplained movement',()=>{
 const row={expansion:'BLB',on_play:'True',num_mulligans:'0',opening_hand:'x|x|x|x|x|x|x',
  draft_id:'x',match_number:'1',game_number:'1'};
 for(const side of ['user','oppo']){
  const p=side+'_turn_1_';Object.assign(row,{[p+'eot_user_life']:'20',[p+'eot_oppo_life']:'20',
   [p+'eot_user_cards_in_hand']:'x|x|x|x|x|x|x',[p+'eot_oppo_cards_in_hand']:'7'});
 }
 const a=wideRowToPublic(row,{}),other=Object.fromEntries(Object.entries(row).map(([k,v])=>[k,v.replaceAll('x','secret')]));
 other.draft_id=row.draft_id;
 assert.deepEqual(wideRowToPublic(other,{}),a);
 row.oppo_turn_1_eot_user_cards_in_hand+='|y';
 const stopped=wideRowToPublic(row,{});assert.equal(stopped.frames.length,1);assert.match(stopped.stop_reason,/Unaccounted/);
 row.user_turn_1_cards_tutored='secret';assert.throws(()=>wideRowToPublic(row,{}),/Tutor/);
});
test('all trained-pool prediction cards have bundled art for imported games',()=>{
 const images=read('card-images.json');for(const card of model.cards)assert.ok(images[card],card);
});
test('worker loads model tables once, reports progress, and returns validation errors',async()=>{
 const previousSelf=globalThis.self,previousFetch=globalThis.fetch,messages=[];let loads=0;
 globalThis.self={postMessage:message=>messages.push(message)};
 globalThis.fetch=async url=>{assert.equal(url,'browser-models.json');loads++;return {ok:true,json:async()=>model};};
 try{
  await import('../dist/replay-worker.js');
  const payload=clone(example);payload.frames=payload.frames.slice(0,2);
  await self.onmessage({data:{payload}});
  assert.equal(messages.filter(m=>m.type==='progress').length,2);
  assert.equal(messages.at(-1).type,'result');assert.equal(messages.at(-1).game.frames.length,2);
  await self.onmessage({data:{payload:{expansion:'OTHER'}}});
  assert.equal(messages.at(-1).type,'error');assert.equal(loads,1);
 }finally{globalThis.self=previousSelf;globalThis.fetch=previousFetch;}
});

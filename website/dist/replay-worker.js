import {wideRowToPublic,scoreReplay} from './replay-inference.js';
let tables;
self.onmessage=async event=>{
 try{
  if(!tables){const response=await fetch('browser-models.json');if(!response.ok)throw Error('Could not load browser models.');tables=await response.json();}
  const payload=event.data.row?wideRowToPublic(event.data.row,tables.arena_names):event.data.payload;
  const game=scoreReplay(tables,payload,(done,total)=>self.postMessage({type:'progress',done,total}));
  self.postMessage({type:'result',game});
 }catch(error){self.postMessage({type:'error',message:error.message});}
};

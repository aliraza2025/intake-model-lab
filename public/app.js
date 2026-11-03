import { CONFIG, runDemo } from './core.js';
import { el, download, showError } from './shared.js';
const $=id=>document.getElementById(id);
let current=null;
let lastInput=null;
function values(){const data={};for(const field of CONFIG.fields){const node=$(field.id);data[field.id]=field.type==='number'?Number(node.value):node.value;}return data;}
function draw(result){
 current=result;
 const sections=(result.sections||[]).map(section=>el('section',{className:'result-section'},el('h3',{},section.title),section.body?el('p',{},section.body):null,section.items?el('ul',{},...section.items.map(item=>el('li',{},item))):null,section.rows?el('div',{className:'table-scroll'},el('table',{},section.headers?el('thead',{},el('tr',{},...section.headers.map(cell=>el('th',{},cell)))):null,el('tbody',{},...section.rows.map(row=>el('tr',{},...row.map(cell=>el('td',{},String(cell??'—')))))))):null));
 $('result').replaceChildren(...[el('div',{className:'eyebrow'},result.status||'REVIEWABLE OUTPUT'),el('h2',{},result.title||CONFIG.title),result.message?el('p',{},result.message):null,
 result.metrics?el('div',{className:'metrics'},...result.metrics.map(([label,value,note])=>el('div',{className:'metric'},el('span',{},label),el('strong',{},String(value)),el('small',{},note||'')))):null,...sections,
 result.actions?el('div',{className:'button-row'},...result.actions.map(action=>el('button',{className:action.id==='approve'?'primary':'quiet',onclick:()=>execute(action.id)},action.label))):null,
 el('details',{},el('summary',{},'Inspect the complete result'),el('pre',{},JSON.stringify(result.data||result,null,2)))].filter(Boolean));
 $('result').setAttribute('aria-busy','false');$('error').hidden=true;
}
async function execute(action){const input=action&&lastInput?lastInput:values();lastInput=input;const button=$('run');button.disabled=true;button.textContent='Working…';$('result').setAttribute('aria-busy','true');try{
 let result;
 if($('runtime').value==='browser-ai') { const {runBrowser}=await import('./inference.js');result=await runBrowser(input,message=>$('progress').textContent=message); }
 else if($('runtime').value==='local') { const response=await fetch('./api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input,action,previous:current?.data}),signal:AbortSignal.timeout(300000)});if(!response.headers.get('content-type')?.includes('application/json'))throw new Error('Run npm start locally to use the installed AI integration.');const body=await response.json();if(!response.ok)throw new Error(body.error||'Local processing failed.');result=body; }
 else result=await runDemo(input,current?.data,action);
 draw(result);
 }catch(error){showError(error);}finally{button.disabled=false;button.textContent=CONFIG.action||'Run example';$('result').setAttribute('aria-busy','false');}}
$('inputs').replaceChildren(...CONFIG.fields.map(field=>el('label',{},field.label,field.type==='select'?el('select',{id:field.id},...field.options.map(([value,label])=>el('option',{value},label))):el(field.type==='textarea'?'textarea':'input',{id:field.id,type:field.type==='textarea'?undefined:field.type||'text',value:field.value,rows:field.rows||5,min:field.min,max:field.max,maxLength:field.maxLength||100000,required:''}))));
for(const field of CONFIG.fields)if(field.type==='select')$(field.id).value=field.value;
$('runtime').replaceChildren(el('option',{value:'example'},CONFIG.baselineLabel||'Browser example / no model'),el('option',{value:'local'},CONFIG.localLabel||'Local integration / requires setup'),...(CONFIG.browserAI?[el('option',{value:'browser-ai'},'AI on this device / download models')]:[]));
$('form').addEventListener('submit',event=>{event.preventDefault();execute();});
$('example').onclick=()=>{for(const field of CONFIG.fields)$(field.id).value=field.value;current=null;execute();};
$('export').onclick=()=>{if(current)download(`${CONFIG.slug}-result.json`,JSON.stringify(current,null,2),'application/json');};
if(CONFIG.audio){const label=el('label',{},'Local recording (WAV, MP3, or WebM)',el('input',{type:'file',accept:'audio/*',id:'audio-file'}));$('form').append(label,el('button',{type:'button',className:'quiet',onclick:async()=>{try{const file=$('audio-file').files[0];if(!file)throw new Error('Choose an audio file first.');if(file.size>8_000_000)throw new Error('Choose an audio file below 8 MB.');const reader=new FileReader();const encoded=await new Promise((resolve,reject)=>{reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(file);});$('progress').textContent='Transcribing with the locally installed model…';const response=await fetch('./api/audio',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({audio:encoded,filename:file.name}),signal:AbortSignal.timeout(300000)});if(!response.headers.get('content-type')?.includes('application/json'))throw new Error('Audio transcription requires the local server.');const result=await response.json();if(!response.ok)throw new Error(result.error||'Transcription failed.');draw(result);$('progress').textContent='Local transcription complete.';}catch(error){showError(error);}}},'Transcribe locally'));
}
execute();

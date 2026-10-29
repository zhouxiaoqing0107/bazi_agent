import asyncio
import logging
import os
import secrets
import threading
import time
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from agent_service import Bazi, DepType, bazi_agent, get_bazi, langfuse

app = FastAPI()
logger = logging.getLogger(__name__)
# Single-process sessions: history stays on the server, isolated by a random token.
sessions = {}
SESSION_TTL = 24 * 60 * 60
MAX_SESSIONS = 1000
IDLE_LIMIT = 120
last_active = datetime.now().timestamp()


def note_activity():
    global last_active
    last_active = datetime.now().timestamp()


def watch_idle():
    while True:
        time.sleep(5)
        busy = any(session['lock'].locked() for session in list(sessions.values()))
        if not busy and datetime.now().timestamp() - last_active >= IDLE_LIMIT:
            logger.warning('两分钟无操作，进程退出')
            os._exit(0)


threading.Thread(target=watch_idle, daemon=True).start()


class BirthRequest(Bazi):
    target: Literal['整体命局', '事业财运', '婚姻感情', '流年运势', '大运分析']


class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(default='', max_length=5000)


def prune_sessions():
    now = datetime.now().timestamp()
    for key, session in list(sessions.items()):
        if now - session['updated'] > SESSION_TTL and not session['lock'].locked():
            del sessions[key]


@app.post('/chart')
async def chart(body: BirthRequest):
    prune_sessions()
    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(503, '当前会话较多，请稍后重试')
    note_activity()
    result = await run_in_threadpool(get_bazi, body)
    token = secrets.token_urlsafe(32)
    sessions[token] = dict(
        chart=result, target=body.target, gender=body.gender,
        history=[], turn=0, lock=asyncio.Lock(), updated=datetime.now().timestamp(),
    )
    note_activity()
    return {'session_id': token, 'chart': result, 'target': body.target}


@app.post('/chat')
async def chat(body: ChatRequest):
    prune_sessions()
    session = sessions.get(body.session_id)
    if session is None:
        raise HTTPException(404, '会话已过期或服务已重启，请重新提交出生资料')
    if session['lock'].locked():
        raise HTTPException(409, '上一条问题仍在分析，请稍候')
    note_activity()
    async with session['lock']:
        turn = session['turn'] + 1
        if turn > 5:
            raise HTTPException(422, '已完成 5 轮分析')
        question = body.question.strip()
        if turn > 1 and not question:
            raise HTTPException(422, '请输入追问内容')
        if turn == 1:
            question = (
                f"请根据已经生成的排盘，分析{session['target']}。"
                '页面已展示完整排盘，请重点给出主题解读。不做真太阳时校正。'
            )
        try:
            result = await bazi_agent.run(
                user_prompt=question,
                deps=DepType(bazi=session['chart'], gender=session['gender'],
                             target=session['target'], turn=turn),
                message_history=session['history'],
            )
            session['history'] = result.all_messages()
            session['turn'] = turn
            return {'answer': str(result.output), 'turn': turn}
        except Exception:
            logger.exception('Agent analysis failed')
            raise HTTPException(502, '分析暂时失败，已保留排盘和历史记录，请重试')
        finally:
            note_activity()
            session['updated'] = datetime.now().timestamp()
            try:
                await run_in_threadpool(langfuse.flush)
            except Exception:
                logger.warning('Telemetry flush failed', exc_info=True)


@app.get('/', response_class=HTMLResponse)
async def home():
    return PAGE


@app.post('/exit')
def exit_process():
    os._exit(0)


PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>八字排盘与解读</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fc;color:#23324a;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:760px;margin:auto;padding:30px 18px}h1{margin:0;font-size:30px}h2{font-size:19px;margin:0 0 15px}p{margin:8px 0}.muted{color:#748096;font-size:13px}.card{background:white;border:1px solid #e3e8f2;border-radius:20px;padding:22px;margin-top:20px}label,legend{font-weight:600;font-size:14px}fieldset{border:0;padding:0;margin:18px 0}select,textarea,button{font:inherit}select,textarea{width:100%;padding:12px;border:1px solid #d8deeb;border-radius:10px;background:#fafbff;color:inherit}textarea{min-height:100px;resize:vertical}.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:9px}.chip{position:relative}.chip input{position:absolute;opacity:0}.chip span{display:block;border:1px solid #dce2ef;border-radius:10px;padding:8px 14px;font-weight:400;cursor:pointer}.chip input:checked+span{color:#385eda;border-color:#7996f2;background:#eef2ff}.chip input:focus-visible+span{outline:3px solid #9db3ff}.primary{background:#4269e8;color:white;border:0;border-radius:12px;padding:13px 20px;cursor:pointer;width:100%;margin-top:14px}button:disabled{opacity:.55;cursor:wait}.wheels{display:flex;gap:6px;margin-top:10px}.wheel-wrap{flex:1;min-width:0;text-align:center}.wheel-wrap small{color:#748096}.wheel{height:150px;overflow-y:auto;scroll-snap-type:y mandatory;overscroll-behavior:contain;padding:54px 0;scrollbar-width:thin;border:1px solid #dce2ef;border-radius:12px;background:linear-gradient(#fafbff 35%,#eaf0ff 36%,#eaf0ff 64%,#fafbff 65%)}.wheel button{display:block;height:42px;width:100%;flex-shrink:0;border:0;background:transparent;scroll-snap-align:center;color:#748096;cursor:pointer}.wheel button.selected{color:#3157ce;font-weight:700}.entry{white-space:pre-wrap;overflow-wrap:anywhere;margin:12px 0;padding:16px;border-radius:12px;background:#f6f8fd}.entry.user{background:#eef2ff}.entry.error{color:#b73146;background:#fff3f4}.entry strong{display:block;margin-bottom:6px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #e7ebf4}.pillars{font-size:25px;letter-spacing:3px;color:#365ac8}#status{color:#4269e8} [hidden]{display:none!important}@media(max-width:420px){.card{padding:16px}.wheels{gap:3px}.wheel button{font-size:14px}}
</style></head><body><main>
<h1>八字排盘与解读</h1><p class="muted">填写出生资料，先查看排盘，再阅读主题分析。</p>
<form id="birth-form" class="card">
<h2>出生资料</h2>
<label>公历生日与出生时间（滚动或点击选择）</label>
<div class="wheels" id="wheels"></div><p id="birthday" class="muted" aria-live="polite"></p>
<fieldset><legend>性别</legend><div class="chips" id="genders"></div></fieldset>
<fieldset><legend>解读主题 · Target</legend><div class="chips" id="targets"></div></fieldset>
<p class="muted">按填写的公历时间排盘，不做真太阳时校正。解读供传统文化参考。</p>
<button class="primary" id="start" type="submit">生成排盘并开始分析</button>
</form>
<p id="status" role="status" aria-live="polite"></p>
<section id="chart" class="card" hidden><h2>你的排盘 · lunar_python</h2><div id="chart-content"></div></section>
<section id="conversation" class="card" hidden><h2>分析与追问记录</h2><div id="messages" aria-live="polite"></div><button id="retry" class="primary" hidden type="button">重试本次分析</button></section>
<form id="follow-form" class="card" hidden><label for="question">继续追问</label><textarea id="question" maxlength="5000" required placeholder="结合以上排盘与分析，继续提问……"></textarea><button id="send" class="primary">发送追问</button></form>
<button id="exit" type="button" class="primary" style="background:#eef1f8;color:#5c6b84">退出</button>
</main><script>
const $=id=>document.getElementById(id);
let sessionId=null,busy=false,turn=0,pendingQuestion='';
const values={year:1994,month:1,day:7,hour:12,minute:0}, wheels={};
function days(){return new Date(values.year,values.month,0).getDate()}
function updateBirthday(){$('birthday').textContent=`已选：${values.year}年${values.month}月${values.day}日 ${String(values.hour).padStart(2,'0')}:${String(values.minute).padStart(2,'0')}`}
function fillWheel(key,min,max){
 const el=wheels[key];el.replaceChildren();
 for(let n=min;n<=max;n++){const b=document.createElement('button');b.type='button';b.textContent=key==='year'?n:String(n).padStart(2,'0');b.dataset.value=n;b.classList.toggle('selected',n===values[key]);b.onclick=()=>{el.scrollTo({top:(n-min)*42,behavior:'smooth'})};el.append(b)}
 el.scrollTop=(values[key]-min)*42;
}
[['year','年',1900,new Date().getFullYear()],['month','月',1,12],['day','日',1,31],['hour','时',0,23],['minute','分',0,59]].forEach(([key,label,min,max])=>{
 const wrap=document.createElement('div');wrap.className='wheel-wrap';const caption=document.createElement('small');caption.textContent=label;
 const el=document.createElement('div');el.className='wheel';el.setAttribute('aria-label',label+'，可滚动或点击选择');wrap.append(caption,el);$('wheels').append(wrap);wheels[key]=el;fillWheel(key,min,key==='day'?days():max);
 el.addEventListener('scroll',()=>{const index=Math.max(0,Math.min(el.children.length-1,Math.round(el.scrollTop/42)));const next=Number(el.children[index].dataset.value);if(values[key]===next)return;values[key]=next;Array.from(el.children).forEach((b,i)=>b.classList.toggle('selected',i===index));if(key==='year'||key==='month'){values.day=Math.min(values.day,days());fillWheel('day',1,days())}updateBirthday()});
});updateBirthday();
function radios(id,name,options){options.forEach((text,i)=>{const label=document.createElement('label');label.className='chip';const input=document.createElement('input');input.type='radio';input.name=name;input.value=text;input.checked=i===0;const span=document.createElement('span');span.textContent=text;label.append(input,span);$(id).append(label)})}
radios('genders','gender',['男','女']);radios('targets','target',['整体命局','事业财运','婚姻感情','流年运势','大运分析']);
async function request(url,body){const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'资料格式有误，请检查后重试');return data}
function plain(text){return text.replace(/^\s{0,3}#{1,6}\s+/gm,'').replace(/^\s*([-*_])\1{2,}\s*$/gm,'').replace(/^[ \t]*[*+-][ \t]+/gm,'· ').replace(/\*\*|__|~~|`+/g,'').replace(/\*/g,'')}
function entry(title,text,kind=''){const el=document.createElement('div');el.className='entry '+kind;const head=document.createElement('strong');head.textContent=title;el.append(head,document.createTextNode(text));$('messages').append(el);return el}
function setBusy(value){busy=value;$('start').disabled=value||!!sessionId;$('send').disabled=value;$('retry').disabled=value;$('question').disabled=value}
function showChart(data){const box=$('chart-content');box.replaceChildren();const summary=document.createElement('p');summary.textContent=`${data.chart['性别']} · ${data.target}`;const birth=document.createElement('p');birth.textContent='出生时间：'+data.chart['出生时间'];const pillars=document.createElement('p');pillars.className='pillars';pillars.textContent=data.chart['四柱'];const start=document.createElement('p');start.textContent='起运日期：'+data.chart['起运日期'];box.append(summary,birth,pillars,start);const wrap=document.createElement('div');wrap.className='table-wrap';const table=document.createElement('table');const row=document.createElement('tr');['大运干支','开始年份','结束年份'].forEach(t=>{const th=document.createElement('th');th.textContent=t;row.append(th)});const thead=document.createElement('thead');thead.append(row);table.append(thead);const tbody=document.createElement('tbody');data.chart['大运'].forEach(item=>{const tr=document.createElement('tr');['干支','开始年份','结束年份'].forEach(k=>{const td=document.createElement('td');td.textContent=item[k];tr.append(td)});tbody.append(tr)});table.append(tbody);wrap.append(table);box.append(wrap);$('chart').hidden=false}
async function analyze(){
 setBusy(true);$('retry').hidden=true;$('conversation').hidden=false;$('status').textContent=turn?'正在思考你的追问……':'排盘已生成，正在分析……';
 const waiting=entry('分析中','正在思考，请稍候……');
 try{const data=await request('/chat',{session_id:sessionId,question:pendingQuestion});waiting.remove();entry('第 '+data.turn+'/5 轮',plain(data.answer));turn=data.turn;$('question').value='';pendingQuestion='';if(turn>=5){$('follow-form').hidden=true;$('status').textContent='5 轮分析已完成';}else{$('follow-form').hidden=false;$('status').textContent='分析完成，可继续追问';}}
 catch(error){waiting.remove();entry('本次请求未完成',error.message,'error');$('retry').hidden=false;$('status').textContent='排盘与已有记录已保留，可重试';}
 finally{setBusy(false)}
}
$('birth-form').addEventListener('submit',async event=>{event.preventDefault();if(busy||sessionId)return;setBusy(true);$('status').textContent='正在生成排盘……';try{const data=await request('/chart',{date:`${values.year}-${String(values.month).padStart(2,'0')}-${String(values.day).padStart(2,'0')}`,hour:values.hour,minute:values.minute,gender:document.querySelector('[name=gender]:checked').value,target:document.querySelector('[name=target]:checked').value});sessionId=data.session_id;showChart(data);Array.from($('birth-form').elements).forEach(el=>el.disabled=true);await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));await analyze()}catch(error){$('status').textContent=error.message}finally{setBusy(false)}});
$('follow-form').addEventListener('submit',async event=>{event.preventDefault();if(busy||!$('retry').hidden)return;const text=$('question').value.trim();if(!text)return;pendingQuestion=text;entry('你',text,'user');await analyze()});
$('retry').onclick=()=>{if(!busy)analyze()};
$('exit').onclick=()=>{$('status').textContent='正在退出……';fetch('/exit',{method:'POST'})};
</script></body></html>'''

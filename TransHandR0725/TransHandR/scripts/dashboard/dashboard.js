const $=id=>document.getElementById(id);
let liveState={},recordConfig=null,systemConfig=null,episodes=[],configDirty=false,systemDirty=false,replay=null,replayTimer=null,liveTickBusy=false;
let robotReplayEpisode=null,robotReplayPreviewReady=false,robotReplayStatusTimer=null,robotReplayLocal=false;
const fmt=(v,d=1)=>Number.isFinite(Number(v))?Number(v).toFixed(d):'—';
const bytes=n=>{const u=['B','KiB','MiB','GiB','TiB'];let i=0,v=Number(n)||0;while(v>=1024&&i<u.length-1){v/=1024;i++}return `${v.toFixed(i<2?0:1)} ${u[i]}`};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function toast(message,error=false){const e=$('toast');e.textContent=message;e.style.borderColor=error?'#77343a':'#315c40';e.style.color=error?'#fecaca':'#b7f7cb';e.classList.remove('hidden');setTimeout(()=>e.classList.add('hidden'),2800)}

document.querySelectorAll('.nav').forEach(button=>button.onclick=()=>showView(button.dataset.view));
function showView(name){document.querySelectorAll('.nav').forEach(x=>x.classList.toggle('active',x.dataset.view===name));document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));$(`${name}View`).classList.add('active');history.replaceState(null,'',name==='data'?'/data':'/overview');if(name==='data')loadEpisodes()}
if(location.pathname==='/data')showView('data');

function refreshImages(){const t=Date.now();$('rgb').src=`/api/rgb.jpg?t=${t}`;$('depth').src=`/api/depth.jpg?t=${t}`;$('headRgb').src=`/api/d435i_rgb.jpg?t=${t}`;$('headDepth').src=`/api/d435i_depth.jpg?t=${t}`}
function canvas(id,draw){const c=$(id),d=devicePixelRatio||1,r=c.getBoundingClientRect();if(!r.width)return;c.width=r.width*d;c.height=r.height*d;const x=c.getContext('2d');x.scale(d,d);draw(x,r.width,r.height)}
function handBars(state,action,handType){canvas('joints',(x,w,h)=>{x.fillStyle='#0a0d0e';x.fillRect(0,0,w,h);const inspire=handType==='inspire',names=inspire?['little','ring','middle','index','thumb bend','thumb rotate']:Array.from({length:Math.min(10,Math.max(state?.length||0,action?.length||0))},(_,i)=>`joint ${i}`),top=10,row=(h-29)/Math.max(1,names.length),max=inspire?1000:255;x.font='10px sans-serif';names.forEach((name,i)=>{const y=top+i*row,sv=Number(state?.[i]),av=Number(action?.[i]),left=82,bw=Math.max(20,w-left-56);x.fillStyle='#89938f';x.fillText(name,8,y+10);x.fillStyle='#222829';x.fillRect(left,y+2,bw,9);if(Number.isFinite(sv)){x.fillStyle='#4ade80';x.fillRect(left,y+2,Math.max(0,Math.min(bw,bw*sv/max)),9)}if(Number.isFinite(av)){const ax=left+Math.max(0,Math.min(bw,bw*av/max));x.strokeStyle='#f6bd60';x.lineWidth=2;x.beginPath();x.moveTo(ax,y);x.lineTo(ax,y+14);x.stroke()}x.fillStyle='#aeb6b2';x.fillText(`${Number.isFinite(sv)?Math.round(sv):'—'} / ${Number.isFinite(av)?Math.round(av):'—'}`,left+bw+5,y+10)});x.fillStyle='#4ade80';x.fillText('■ actual',8,h-4);x.fillStyle='#f6bd60';x.fillText('│ target',72,h-4)})}
function tactileColor(value){
  if(!Number.isFinite(value)||value<0)return '#151a1a';
  const t=Math.sqrt(Math.max(0,Math.min(1,value/255))),stops=[[18,28,27],[34,197,94],[245,158,11],[239,68,68]],scaled=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(scaled)),f=scaled-i;
  return `rgb(${stops[i].map((v,j)=>Math.round(v+(stops[i+1][j]-v)*f)).join(',')})`;
}
function tactileHeatmaps(t){palmHeatmap(liveState.palm);$('palmGap').textContent=`14×15 · ${fmt(liveState.streams?.palm?.hz)} Hz · age ${fmt(liveState.streams?.palm?.age_ms)} ms`;canvas('tactile',(x,w,h)=>{
  x.fillStyle='#0a0d0e';x.fillRect(0,0,w,h);
  const matrices=t?.matrix;
  if(!Array.isArray(matrices)||matrices.length!==5){x.fillStyle='#7f8985';x.fillText('等待有效的触觉阵列',14,24);return}
  const names=['THUMB','INDEX','MIDDLE','RING','PINKY'],margin=10,gap=7,groupWidth=(w-margin*2-gap*4)/5;
  x.textAlign='center';x.font='9px sans-serif';
  matrices.forEach((matrix,finger)=>{
    const validRows=matrix.map(row=>row.map(Number)).filter(row=>row.some(value=>Number.isFinite(value)&&value>=0)),rows=validRows.length,cols=validRows.reduce((count,row)=>Math.max(count,row.reduce((last,value,index)=>Number.isFinite(value)&&value>=0?index+1:last,0)),0),availableHeight=h-61,cell=Math.max(3,Math.min((groupWidth-7)/Math.max(1,cols),availableHeight/Math.max(1,rows))),gridWidth=cell*cols,gridHeight=cell*rows,left=margin+finger*(groupWidth+gap)+(groupWidth-gridWidth)/2,top=26;
    x.fillStyle='#aeb6b2';x.fillText(names[finger],margin+finger*(groupWidth+gap)+groupWidth/2,14);
    for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
      const value=Number(validRows[row][col]);x.fillStyle=tactileColor(value);x.fillRect(left+col*cell+.6,top+row*cell+.6,Math.max(1,cell-1.2),Math.max(1,cell-1.2));
    }
    const maxValue=Number(t.max?.[finger]),contacts=Number(t.contact_area?.[finger]||0),summaryY=Math.min(h-9,top+gridHeight+16);
    x.fillStyle='#cbd1ce';x.fillText(`${Number.isFinite(maxValue)?Math.round(maxValue):'—'} · ${contacts}`,margin+finger*(groupWidth+gap)+groupWidth/2,summaryY);
  });
  x.textAlign='start';
})}
function palmHeatmap(p){canvas('palm',(x,w,h)=>{
  x.fillStyle='#0a0d0e';x.fillRect(0,0,w,h);
  const matrix=p?.matrix;
  if(!Array.isArray(matrix)||matrix.length!==14||matrix.some(row=>!Array.isArray(row)||row.length!==15)){
    x.fillStyle='#7f8985';x.fillText('等待掌心触觉阵列',14,24);return;
  }
  const cell=Math.min((w-38)/15,(h-35)/14),left=(w-cell*15)/2,top=(h-cell*14)/2;
  for(let row=0;row<14;row++)for(let col=0;col<15;col++){
    x.fillStyle=tactileColor(Number(matrix[row][col]));
    x.fillRect(left+col*cell+.5,top+row*cell+.5,Math.max(1,cell-1),Math.max(1,cell-1));
  }
})}
function tracker(t){canvas('tracker',(x,w,h)=>{x.fillStyle='#0a0d0e';x.fillRect(0,0,w,h);x.strokeStyle='#27302d';x.beginPath();x.moveTo(w/2,15);x.lineTo(w/2,h-26);x.moveTo(14,h/2);x.lineTo(w-14,h/2);x.stroke();if(!t?.valid||!t.quat){x.fillStyle='#fb7185';x.fillText(`TRACKER INVALID · ${t?.state||'OFFLINE'}`,14,25);x.fillStyle='#87918d';x.fillText(`${t?.map_state||'MAP UNKNOWN'} · packets ${t?.packet_count||0} · poses ${t?.pose_count||0}`,14,44);return}let px=w/2+(t.pos?.[0]||0)*95,py=h/2-(t.pos?.[2]||0)*95,q=t.quat,[qw,qx,qy,qz]=q;const R=[[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],[2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],[2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]],colors=['#fb7185','#4ade80','#60a5fa'];for(let i=0;i<3;i++){x.strokeStyle=colors[i];x.lineWidth=3;x.beginPath();x.moveTo(px,py);x.lineTo(px+R[0][i]*42,py-R[2][i]*42-R[1][i]*15);x.stroke()}x.fillStyle='#fff';x.beginPath();x.arc(px,py,5,0,7);x.fill();x.fillStyle='#87918d';x.fillText(`quat [${q.map(v=>fmt(v,3)).join(', ')}]`,12,h-9)})}

function profileIdentity(fromRecord=false){if(fromRecord&&recordConfig)return{control_source:recordConfig.control_source,target_hand_model:recordConfig.hand_type,hand_side:recordConfig.hand_side};return{control_source:$('controlSource')?.value||systemConfig?.config?.control_source||'vision_pro',target_hand_model:$('targetHand')?.value||systemConfig?.config?.target_hand_model||'inspire',hand_side:$('handSideSelect')?.value||systemConfig?.config?.hand_side||'right'}}
async function loadConfig(){const query=new URLSearchParams(profileIdentity()).toString(),r=await fetch(`/api/recording/config?${query}`,{cache:'no-store'});recordConfig=await r.json();configDirty=false;$('configState').textContent='已保存';$('configState').className='saved';$('taskName').value=`${recordConfig.task_name} / ${recordConfig.hand_type}`;$('nextFile').value=recordConfig.next_file;$('maxDuration').textContent=`${recordConfig.max_duration_s}s`;$('maxSize').textContent=`${recordConfig.max_size_gb} GiB`;$('dataRoot').textContent=recordConfig.output_dir;renderChannels();updateTelemetryVisibility()}
async function loadSystemConfig(){try{const r=await fetch('/api/system/config',{cache:'no-store'});if(!r.ok){$('systemPanel').classList.add('hidden');return}$('systemPanel').classList.remove('hidden');const incoming=await r.json(),shouldApply=!systemConfig||!systemDirty||incoming.state!=='STOPPED';systemConfig=incoming;if(shouldApply){const c=systemConfig.config;$('controlSource').value=c.control_source;systemSelectionChanged(c.target_hand_model,c.hand_side);if(incoming.state!=='STOPPED')systemDirty=false}renderSystem();if(recordConfig)updateChannelLock();updateTelemetryVisibility()}catch{}}
function markSystemDirty(){systemDirty=true}
function openDeviceSettings(){
  if(!systemConfig||systemConfig.state!=='STOPPED'){toast('停止系统后才能修改设备设置',true);return}
  const c=systemConfig.config,source=$('controlSource').value,model=$('targetHand').value,side=$('handSideSelect').value;
  $('deviceAvpIp').value=c.avp_ip||'';$('deviceLeftGloveSerial').value=c.left_glove_serial||'';$('deviceRightGloveSerial').value=c.right_glove_serial||'';
  $('deviceHandSerial').value=c.hand_serial||'';$('deviceCanInterface').value=c.can_interface||'';
  $('deviceD405Serial').value=c.d405_serial||'';$('deviceD435iSerial').value=c.d435i_serial||'';
  $('deviceRecordingDir').value=c.recording_dir||'';
  $('deviceVisionFields').classList.toggle('hidden',source!=='vision_pro');
  $('deviceLinkerforceFields').classList.toggle('hidden',source!=='linkerforce');
  const profile=systemConfig.profiles?.find(p=>p.control_source===source&&p.robot_model===model);
  const needsCalibration=source==='linkerforce'&&(!!profile?.requires_calibration||['o6','inspire'].includes(model));
  $('calibrationPanel').classList.toggle('hidden',!needsCalibration);
  $('calibrationTitle').textContent=`${model.toUpperCase()} · ${side.toUpperCase()} 标定`;
  const showFeedback=source==='linkerforce'&&model==='inspire';
  $('inspireFeedbackSetting').classList.toggle('hidden',!showFeedback);
  $('enableForceFeedback').checked=!!c.feedback_enabled_profiles?.[`linkerforce:inspire:${side}`];
  renderCalibration();$('deviceSettingsError').classList.add('hidden');$('deviceSettingsModal').classList.remove('hidden');
}
function closeDeviceSettings(){$('deviceSettingsModal').classList.add('hidden')}
function modalBackdropClose(event,id){if(event.target.id===id&&id==='deviceSettingsModal')closeDeviceSettings()}
async function saveDeviceSettings(keepOpen=false){
  const button=$('saveDeviceSettings'),payload={avp_ip:$('deviceAvpIp').value.trim(),left_glove_serial:$('deviceLeftGloveSerial').value.trim(),right_glove_serial:$('deviceRightGloveSerial').value.trim(),hand_serial:$('deviceHandSerial').value.trim(),can_interface:$('deviceCanInterface').value.trim(),d405_serial:$('deviceD405Serial').value.trim(),d435i_serial:$('deviceD435iSerial').value.trim(),recording_dir:$('deviceRecordingDir').value.trim()};
  const identity=profileIdentity();
  Object.assign(payload,identity);
  if(identity.control_source==='linkerforce'&&identity.target_hand_model==='inspire'){
    payload.feedback_enabled_profiles={[`linkerforce:inspire:${identity.hand_side}`]:$('enableForceFeedback').checked};
  }
  button.disabled=true;$('deviceSettingsError').classList.add('hidden');
  try{
    const r=await fetch('/api/system/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),body=await r.json();
    if(!r.ok)throw Error(body.error||'设备设置保存失败');
    systemConfig=body.config;systemDirty=false;renderSystem();await Promise.allSettled([loadConfig(),loadDisk(),loadEpisodes()]);if(!keepOpen)closeDeviceSettings();toast('设备设置已保存');return true;
  }catch(e){$('deviceSettingsError').textContent=e.message;$('deviceSettingsError').classList.remove('hidden');toast(e.message,true);return false}
  finally{button.disabled=false}
}
function systemSelectionChanged(preferredModel,preferredSide){if(!systemConfig)return;const source=$('controlSource').value,profiles=systemConfig.profiles.filter(p=>p.control_source===source),current=preferredModel||$('targetHand').value;$('targetHand').innerHTML=profiles.map(p=>`<option value="${esc(p.robot_model)}">${esc(p.robot_model.toUpperCase())}</option>`).join('');if(profiles.some(p=>p.robot_model===current))$('targetHand').value=current;const profile=profiles.find(p=>p.robot_model===$('targetHand').value)||profiles[0],side=preferredSide||$('handSideSelect').value;$('handSideSelect').innerHTML=(profile?.hand_sides||[]).map(v=>`<option value="${v}">${v==='left'?'左手':'右手'}</option>`).join('');if(profile?.hand_sides.includes(side))$('handSideSelect').value=side;updateTelemetryVisibility()}
async function changeProfile(kind){if(configDirty&&!(await saveConfig()))return;if(kind==='source')systemSelectionChanged();else if(kind==='model')systemSelectionChanged($('targetHand').value);markSystemDirty();await loadConfig()}
function updateTelemetryVisibility(){const source=$('controlSource')?.value||liveState.control_source||systemConfig?.config?.control_source,model=$('targetHand')?.value||liveState.hand_type||systemConfig?.config?.target_hand_model,side=$('handSideSelect')?.value||liveState.hand_side||liveState.vision_hand_side||systemConfig?.config?.hand_side||'right',showInspire=model==='inspire',showO6=source==='linkerforce'&&model==='o6';$('inspireModelCard').classList.toggle('hidden',!showInspire);$('o6TactileCard').classList.toggle('hidden',!showO6);$('o6PalmCard').classList.toggle('hidden',!showO6);$('o6ModelCard').classList.toggle('hidden',!showO6);window.inspireHandViewer?.configure({visible:showInspire,model,side});window.o6HandViewer?.setVisible(showO6)}
window.addEventListener('o6-viewer-ready',updateTelemetryVisibility);
window.addEventListener('inspire-viewer-ready',updateTelemetryVisibility);
function renderCalibration(){if(!systemConfig)return;const c=systemConfig.calibration||{},identity=profileIdentity(),expectedKey=`${identity.target_hand_model}:${identity.hand_side}`,matches=c.key===expectedKey,running=systemConfig.state==='CALIBRATING'&&matches,valid=matches&&(!!c.valid||c.state==='COMPLETE'),summary=running?`${c.instruction||'正在标定'} · ${c.progress||0}%`:valid?`已标定${c.timestamp?' · '+new Date(c.timestamp).toLocaleDateString():''}`:matches&&c.state==='FAILED'?`失败 · ${c.error||'请重试'}`:'尚未标定',modalOpen=!$('calibrationModal').classList.contains('hidden');if($('calibrationSummary'))$('calibrationSummary').textContent=summary;if($('startCalibration'))$('startCalibration').disabled=systemConfig.state!=='STOPPED';if(running||modalOpen){if(running)$('calibrationModal').classList.remove('hidden');$('calibrationProfile').textContent=`${expectedKey} · ${identity.hand_side.toUpperCase()}`;$('calibrationStep').textContent=(c.step||'connecting').toUpperCase();$('calibrationInstruction').textContent=c.instruction||'请按提示保持手势';$('calibrationProgress').style.width=`${Math.max(0,Math.min(100,Number(c.progress)||0))}%`;$('calibrationPercent').textContent=`${Number(c.progress)||0}%`;$('calibrationError').textContent=c.error||'';$('calibrationError').classList.toggle('hidden',!c.error);$('cancelCalibration').classList.toggle('hidden',!running);$('finishCalibration').classList.toggle('hidden',running)}}
async function startCalibration(){if(!(await saveDeviceSettings(true)))return;try{let r=await fetch('/api/system/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(profileIdentity())}),body=await r.json();if(!r.ok)throw Error(body.error||'配置保存失败');closeDeviceSettings();$('calibrationModal').classList.remove('hidden');r=await fetch('/api/calibration/start',{method:'POST'});body=await r.json();if(!r.ok)throw Error(body.error||'标定启动失败');systemDirty=false;await loadSystemConfig()}catch(e){toast(e.message,true);await loadSystemConfig()}}
async function cancelCalibration(){try{const r=await fetch('/api/calibration/cancel',{method:'POST'}),body=await r.json();if(!r.ok)throw Error(body.error||'取消失败');closeCalibration();await loadSystemConfig()}catch(e){toast(e.message,true)}}
function closeCalibration(){$('calibrationModal').classList.add('hidden')}
function renderSystem(){
  if(!systemConfig)return;
  const state=systemConfig.state,locked=state!=='STOPPED',labels={STOPPED:'已停止',STARTING:'启动中',RUNNING:'运行中',STOPPING:'停止中',CALIBRATING:'标定中',ERROR:'异常'};
  $('systemBadge').textContent=labels[state]||state;$('systemBadge').classList.toggle('error',state==='ERROR');
  $('systemStart').textContent=state==='STOPPED'?'启动系统':state==='ERROR'?'复位系统':'停止系统';
  $('systemStart').classList.toggle('stop',state!=='STOPPED');$('deviceSettingsButton').disabled=locked;
  ['controlSource','targetHand','handSideSelect'].forEach(id=>$(id).disabled=locked);
  $('processes').innerHTML=Object.entries(systemConfig.processes||{}).map(([name,p])=>`<span class="${p.running?'running':''}">${esc(name)} · ${p.running?'RUN':'STOP'}</span>`).join('');
  const inspire=systemConfig.config?.control_source==='linkerforce'&&systemConfig.config?.target_hand_model==='inspire';
  const feedbackRunning=inspire&&state==='RUNNING'&&!!systemConfig.processes?.handretarget?.running;
  $('feedbackStatus').classList.toggle('hidden',!feedbackRunning);
  const feedbackEnabled=feedbackRunning&&!!systemConfig.config?.feedback_enabled_profiles?.[`linkerforce:inspire:${systemConfig.config?.hand_side}`];
  $('feedbackStatus').classList.toggle('enabled',feedbackEnabled);
  $('feedbackStatus').textContent=`力反馈 · ${feedbackEnabled?'开启':'关闭'}`;
  $('systemError').textContent=systemConfig.error||'';$('systemError').classList.toggle('hidden',!systemConfig.error);renderCalibration();
}
async function toggleSystem(){if(!systemConfig)return;const stopped=systemConfig.state==='STOPPED',button=$('systemStart');button.disabled=true;try{if(stopped){if(configDirty&&!(await saveConfig()))return;const payload=profileIdentity();let r=await fetch('/api/system/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),body=await r.json();if(!r.ok)throw Error(body.error||'配置保存失败');r=await fetch('/api/system/start',{method:'POST'});body=await r.json();if(!r.ok)throw Error(body.error||'启动失败');systemDirty=false;toast('系统已启动')}else{const r=await fetch('/api/system/stop',{method:'POST'}),body=await r.json();if(!r.ok)throw Error(body.error||'停止失败');systemDirty=false;toast('系统已停止')}await loadSystemConfig();await loadConfig()}catch(e){toast(e.message,true);await loadSystemConfig()}finally{button.disabled=false}}
function renderChannels(){if(!recordConfig)return;$('channelCount').textContent=recordConfig.channels.length;$('channels').innerHTML=recordConfig.channels.map(c=>`<label class="channel ${c.available?'':'disabled'}"><input type="checkbox" data-channel="${esc(c.id)}" ${c.selected?'checked':''} onchange="markConfigDirty()"><span class="channel-content"><span class="channel-title"><b>${esc(c.label)}</b>${c.required?'<i class="tag required">必选</i>':c.recommended?'<i class="tag">推荐</i>':''}</span><code>${esc(c.path)}</code><p>${esc(c.available?c.description:'当前控制源不支持该数据')}</p></span></label>`).join('');updateChannelLock();updateSelectedCount()}
function updateChannelLock(){if(!recordConfig)return;const locked=!!systemConfig&&systemConfig.state!=='STOPPED';document.querySelectorAll('#channels [data-channel]').forEach(input=>{const channel=recordConfig.channels.find(c=>c.id===input.dataset.channel);input.disabled=!!(channel?.required||!channel?.available||locked||liveState.recording)});$('saveConfig').disabled=locked||!!liveState.recording}
function updateSelectedCount(){$('selectedCount').textContent=document.querySelectorAll('[data-channel]:checked').length}
function markConfigDirty(){configDirty=true;$('configState').textContent='未保存';$('configState').className='unsaved';updateSelectedCount()}
function selectRecommended(){document.querySelectorAll('[data-channel]').forEach(x=>{const c=recordConfig.channels.find(v=>v.id===x.dataset.channel);if(!x.disabled)x.checked=!!c.recommended});markConfigDirty()}
async function saveConfig(){const channels=[...document.querySelectorAll('[data-channel]:checked')].map(x=>x.dataset.channel),identity=profileIdentity(true),r=await fetch('/api/recording/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({...identity,channels})}),body=await r.json();if(!r.ok){toast(body.error||'保存失败',true);return false}recordConfig=body.config;configDirty=false;$('configState').textContent='已保存';$('configState').className='saved';renderChannels();toast('通道配置已保存');return true}
async function toggleRecording(){if(configDirty&&!(await saveConfig()))return;const action=liveState.recording?'stop':'start',button=$('recordButton');button.disabled=true;$('recordError').classList.add('hidden');try{const r=await fetch(`/api/${action}`,{method:'POST'}),body=await r.json();if(!r.ok)throw Error(body.error||'操作失败');toast(action==='start'?'录制已开始':'Episode 已保存');setTimeout(()=>{loadConfig();loadEpisodes()},500)}catch(e){$('recordError').textContent=e.message;$('recordError').classList.remove('hidden');toast(e.message,true)}finally{button.disabled=false}}

function renderHeadCameraStatus(running, stream) {
  const preview = liveState.head_preview || {};
  const enabled = running && !!preview.enabled;
  const error = enabled ? preview.error || '' : '';
  const state = !running ? '未启动' : !enabled ? '未启用' : error ? '采集异常'
    : !preview.valid ? '无图像' : liveState.valid?.d435i ? '有效' : '预览有效 · 未对齐';
  $('headState').textContent = state;
  $('headState').title = error || (preview.valid && !liveState.valid?.d435i
    ? 'D435i 有原始图像，但当前 D405 帧未匹配到有效 D435i 帧' : '');
  $('headError').textContent = error;
  $('headError').title = error;
  $('headError').classList.toggle('hidden', !error);
  $('headGap').textContent = `${fmt(stream.hz)} Hz · age ${fmt(stream.age_ms)} ms`;
  $('headGap').title = `D405 对齐差值: ${fmt(liveState.sync?.d435i_dt_ms)} ms`;
}
async function liveTick(){if(liveTickBusy)return;liveTickBusy=true;try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error();const previousRecording=!!liveState.recording;liveState=await r.json();const managed=!!liveState.system,running=managed?liveState.system.state==='RUNNING':true,stream=key=>liveState.streams?.[key]||{};$('connection').textContent=liveState.recording?`正在录制 · ${fmt(liveState.elapsed_s)}s`:running?'采集服务在线':'平台在线';$('statusDot').classList.toggle('bad',liveState.system?.state==='ERROR');$('recordBadge').textContent=liveState.recording?'录制中':'空闲';$('recordButton').textContent=liveState.recording?'■ 停止并保存':'▷ 开始录制';$('recordButton').classList.toggle('stop',!!liveState.recording);$('recordButton').disabled=!running;$('masterFps').textContent=`${fmt(stream('d405').hz||liveState.fps)} FPS`;$('depthState').textContent=liveState.valid?.d405?'有效':'无效';renderHeadCameraStatus(running,stream('head_camera'));$('handGap').textContent=`STATE ${fmt(stream('hand_state').hz)} Hz · ACTION ${fmt(stream('action').hz)} Hz`;$('tactileGap').textContent=`10×4 · ${fmt(stream('tactile').hz)} Hz · age ${fmt(stream('tactile').age_ms)} ms`;$('trackerState').textContent=`${liveState.tracker?.state||'—'} · ${fmt(stream('tracker').hz)} Hz`;const source=liveState.control_source||$('controlSource')?.value,model=liveState.hand_type||$('targetHand')?.value,side=liveState.hand_side||liveState.vision_hand_side||$('handSideSelect')?.value||'right';handBars(liveState.hand_state,liveState.action,model);if(model==='inspire'){window.inspireHandViewer?.configure({visible:true,model,side});window.inspireHandViewer?.update(liveState)}else if(source==='linkerforce'&&model==='o6'){tactileHeatmaps(liveState.tactile);window.o6HandViewer?.update(liveState)}tracker(liveState.tracker);renderValidity();if(recordConfig){$('nextFile').value=running?`episode_${String(liveState.episode_index).padStart(6,'0')}.hdf5`:'系统启动后确定';if(previousRecording!==!!liveState.recording)updateChannelLock()}}catch{$('connection').textContent='平台服务断开';$('statusDot').classList.add('bad')}finally{liveTickBusy=false}}
function renderValidity(){
  const channel=id=>recordConfig?.channels?.find(c=>c.id===id);
  const values=[['D405','d405','d405'],['D435i','d435i','d435i'],['VISION','vision_hand','right_hand'],['EXOSKELETON','operator_exoskeleton','operator'],['HAND STATE','hand_state','hand_state'],['ACTION','action','action'],['TACTILE','tactile','tactile'],['TRACKER','vive_tracker','tracker']];
  if(liveState.control_source==='linkerforce'&&liveState.hand_type==='inspire')values.splice(7,0,['INSPIRE FORCE','inspire_force','force']);
  const enabled=values.filter(([,id])=>channel(id)?.selected),systemRunning=!liveState.system||liveState.system.state==='RUNNING';
  const ok=systemRunning?enabled.filter(([,id,key])=>liveState.valid?.[key]).length:0;
  $('healthScore').textContent=systemRunning?`${ok}/${enabled.length}`:'—';
  $('health').innerHTML=values.map(([name,id,key])=>{
    const item=channel(id),selected=!!item?.selected,available=!!item?.available,valid=!!liveState.valid?.[key];
    let state=!available?'不适用':!selected?'未选择':!systemRunning?'系统未运行':valid?'有效':'无有效样本';
    if(key==='force'&&available&&systemRunning)state+=` · ${fmt(liveState.streams?.force?.hz)} Hz · age ${fmt(liveState.streams?.force?.age_ms)} ms`;
    const klass=!available||!selected||!systemRunning?'off':valid?'ok':'bad';
    return `<div class="health-item ${klass}"><b>${name}</b><span>${state}</span></div>`;
  }).join('');
}
async function loadDisk(){try{const d=await(await fetch('/api/disk')).json(),used=d.used/d.total*100;$('diskFree').textContent=`${bytes(d.free)} 可用`;$('diskBar').style.width=`${used}%`;$('diskBar').style.background=used>90?'#ef4444':used>75?'#f59e0b':'#22c55e';$('diskPath').textContent=d.path}catch{}}

async function loadEpisodes(){try{const data=await(await fetch('/api/episodes',{cache:'no-store'})).json();episodes=data.episodes||[];$('episodeCount').textContent=episodes.length;renderEpisodes()}catch{$('episodeList').innerHTML='<div class="empty">数据列表加载失败</div>'}}
function renderEpisodes(){const q=$('search').value.trim().toLowerCase(),sort=$('sort').value;let list=episodes.filter(x=>x.id.toLowerCase().includes(q));list.sort((a,b)=>sort==='size'?b.size_bytes-a.size_bytes:sort==='name'?a.name.localeCompare(b.name):b.modified.localeCompare(a.modified));if(!list.length){$('episodeList').innerHTML='<div class="empty">没有匹配的 Episode</div>';return}$('episodeList').innerHTML=list.map(e=>{const id=encodeURIComponent(e.id).replace(/'/g,'%27');return `<div class="episode"><div class="episode-main"><div class="episode-name">${esc(e.name)}<span class="quality ${e.quality}">${e.quality==='ok'?'合格':e.quality==='warn'?'警告':'异常'}</span></div><div class="episode-path">${esc(e.id)}</div><div class="episode-meta"><span>${new Date(e.modified).toLocaleString()}</span><span>${bytes(e.size_bytes)}</span><span>${e.frames} 帧</span><span>${fmt(e.fps)} Hz</span><span>${fmt(e.duration_s)} s</span></div></div><div class="episode-actions"><button onclick="openReplay('${id}')">▷ 可视化</button><button onclick="openRobotReplay('${id}')">↗ 真机回放</button><a href="/api/episode/download?id=${id}">↓ 下载</a></div></div>`}).join('')}
function robotReplayChanged(){$('robotO6Can').disabled=!$('robotWithO6').checked;robotReplayPreviewReady=false;$('robotPrepare').disabled=true;$('robotReplayState').textContent='参数已修改，请重新离线预检'}
function robotReplayNumber(id,label){const field=$(id);if(!field.checkValidity())throw Error(`${label}：${field.validationMessage}`);return Number(field.value)}
function robotReplayPayload(){if(!robotReplayEpisode)throw Error('未选择 Episode');const hand=$('robotWithO6').checked;if(hand&&!$('robotO6Can').checkValidity())throw Error(`O6 CAN 接口：${$('robotO6Can').validationMessage}`);return{id:robotReplayEpisode.id,scale:robotReplayNumber('robotScale','平移倍率'),speed:robotReplayNumber('robotSpeed','速度倍率'),axis_map:$('robotAxisMap').value,max_displacement:robotReplayNumber('robotMaxDisplacement','最大相对位移'),countdown:robotReplayNumber('robotCountdown','倒计时'),with_o6_hand:hand,o6_can:hand?$('robotO6Can').value.trim():''}}
async function robotReplayPost(action,payload){const response=await fetch(`/api/robot-replay/${action}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),body=await response.json();if(!response.ok)throw Error(body.error||'真机回放请求失败');return body}
function openRobotReplay(encoded){const id=decodeURIComponent(encoded),episode=episodes.find(x=>x.id===id);if(!episode)return;robotReplayEpisode=episode;robotReplayPreviewReady=false;$('robotReplayEpisode').textContent=id;$('robotWithO6').checked=episode.hand_type==='o6';$('robotO6Can').value=systemConfig?.config?.can_interface||'can1';$('robotO6Can').disabled=!$('robotWithO6').checked;$('robotReplayState').textContent='请先离线预检；打开窗口不会连接机器人';$('robotReplayOutput').textContent='';$('robotPrepareWord').value='';$('robotConfirmWord').value='';$('robotReplayModal').classList.remove('hidden');clearInterval(robotReplayStatusTimer);pollRobotReplay();robotReplayStatusTimer=setInterval(pollRobotReplay,700)}
function closeRobotReplay(){$('robotReplayModal').classList.add('hidden');clearInterval(robotReplayStatusTimer);robotReplayStatusTimer=null;robotReplayEpisode=null}
async function previewRobotReplay(){try{const result=await robotReplayPost('preview',robotReplayPayload());robotReplayPreviewReady=true;$('robotReplayOutput').textContent=result.summary;$('robotReplayState').textContent='离线预检通过；真机准备前确认运动空间与实体急停';$('robotPrepare').disabled=!robotReplayLocal}catch(e){robotReplayPreviewReady=false;$('robotPrepare').disabled=true;$('robotReplayState').textContent=e.message;toast(e.message,true)}}
async function startRobotReplay(){try{const result=await robotReplayPost('start',{...robotReplayPayload(),confirmation:$('robotPrepareWord').value.trim()});robotReplayPreviewReady=false;renderRobotReplayStatus(result)}catch(e){toast(e.message,true);$('robotReplayState').textContent=e.message}}
async function restoreRobotStart(){try{const countdown=robotReplayNumber('robotCountdown','倒计时');const result=await robotReplayPost('restore',{confirmation:$('robotPrepareWord').value.trim(),countdown});robotReplayPreviewReady=false;renderRobotReplayStatus(result)}catch(e){toast(e.message,true);$('robotReplayState').textContent=e.message}}
async function confirmRobotReplay(){try{const result=await robotReplayPost('confirm',{confirmation:$('robotConfirmWord').value.trim()});renderRobotReplayStatus(result)}catch(e){toast(e.message,true);$('robotReplayState').textContent=e.message}}
async function cancelRobotReplay(){if(!window.confirm('仅停止后续软件指令，不是实体急停；MoveJ 可能继续。确定请求停止吗？'))return;try{renderRobotReplayStatus(await robotReplayPost('cancel',{}))}catch(e){toast(e.message,true)}}
function renderRobotReplayStatus(s){const phases={idle:'未运行',preparing:s.kind==='restore'?'正在恢复固定起点（不回放）':'正在恢复固定起点',awaiting_replay:'固定起点已验证，等待输入 REPLAY',armed:'已确认，倒计时/检查中',replaying:'正在回放',cancelling:'正在请求停止后续指令',completed:'回放进程结束',restored:'固定起点已恢复（未回放）',failed:'真机任务报错',unavailable:'网页真机回放未启用'};if(s.local!==undefined)robotReplayLocal=!!s.local;$('robotReplayState').textContent=`${phases[s.phase]||s.phase}${s.trace_file?' · 日志 '+s.trace_file:''}${!robotReplayLocal?' · 真机操作只允许本机网页':''}`;if(s.lines?.length)$('robotReplayOutput').textContent=s.lines.join('\n');const active=['preparing','awaiting_replay','armed','replaying','cancelling'].includes(s.phase);$('robotPrepare').disabled=!robotReplayPreviewReady||!robotReplayLocal||active;$('robotRestore').disabled=!robotReplayLocal||active;$('robotConfirm').disabled=!robotReplayLocal||s.phase!=='awaiting_replay';$('robotCancel').disabled=!robotReplayLocal||!active}
async function pollRobotReplay(){if($('robotReplayModal').classList.contains('hidden'))return;try{const response=await fetch('/api/robot-replay/status',{cache:'no-store'}),status=await response.json();if(robotReplayPreviewReady&&!['preparing','awaiting_replay','armed','replaying','cancelling'].includes(status.phase)){robotReplayLocal=!!status.local;$('robotPrepare').disabled=!robotReplayLocal;$('robotRestore').disabled=!robotReplayLocal;return}renderRobotReplayStatus(status)}catch(e){$('robotReplayState').textContent='回放状态服务不可用';$('robotRestore').disabled=true}}
function openReplay(encoded){const id=decodeURIComponent(encoded),e=episodes.find(x=>x.id===id);if(!e)return;replay={episode:e,index:0,playing:false};$('replayTitle').textContent=e.name;$('replayMeta').textContent=`${e.schema||'unknown schema'} · ${e.frames} frames · ${fmt(e.fps)} Hz · ${e.hand_type||'—'} / ${e.vision_hand_side||'—'} hand`;$('frameSlider').max=Math.max(0,e.frames-1);$('frameSlider').value=0;$('qualityDetail').innerHTML=Object.entries(e.valid_ratios||{}).map(([k,v])=>`<div>${esc(k)} valid<b>${fmt(Number(v)*100)}%</b></div>`).join('')||'<div>质量信息<b>旧格式未缓存</b></div>';[['replayD435Rgb','d435i_rgb'],['replayD435Depth','d435i_depth'],['replayD405Rgb','d405_rgb'],['replayD405Depth','d405_depth']].forEach(([id,stream])=>$(id).parentElement.classList.toggle('hidden',!(e.streams||[]).includes(stream)));$('replayModal').classList.remove('hidden');setReplayFrame(0)}
function closeReplay(){$('replayModal').classList.add('hidden');clearInterval(replayTimer);replayTimer=null;replay=null}
function setReplayFrame(index){if(!replay)return;replay.index=Number(index);$('frameSlider').value=replay.index;$('frameLabel').textContent=`${replay.index+1} / ${replay.episode.frames}`;const base=`/api/episode/frame.jpg?id=${encodeURIComponent(replay.episode.id)}&index=${replay.index}&stream=`,streams={replayD435Rgb:'d435i_rgb',replayD435Depth:'d435i_depth',replayD405Rgb:'d405_rgb',replayD405Depth:'d405_depth'};Object.entries(streams).forEach(([id,stream])=>{if((replay.episode.streams||[]).includes(stream))$(id).src=base+stream})}
function togglePlay(){if(!replay)return;replay.playing=!replay.playing;$('playButton').textContent=replay.playing?'Ⅱ':'▷';clearInterval(replayTimer);if(replay.playing)replayTimer=setInterval(()=>{let next=replay.index+1;if(next>=replay.episode.frames)next=0;setReplayFrame(next)},Math.max(40,1000/(replay.episode.fps||30)))}
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('deviceSettingsModal').classList.contains('hidden')){closeDeviceSettings();return}if(e.target.matches('input,select'))return;if(e.key.toLowerCase()==='r'&&!liveState.recording)toggleRecording();if(e.key.toLowerCase()==='s'&&liveState.recording)toggleRecording();if(e.key==='Escape'&&!$('replayModal').classList.contains('hidden'))closeReplay()});

(async()=>{await loadSystemConfig();await Promise.all([loadConfig(),loadDisk(),loadEpisodes()]);updateTelemetryVisibility()})().catch(()=>{});liveTick();refreshImages();setInterval(liveTick,100);setInterval(refreshImages,1000/30);setInterval(loadSystemConfig,1200);setInterval(loadDisk,10000);

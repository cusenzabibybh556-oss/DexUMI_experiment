// 腕部标定页提供只读采样与离线求解；与机器人回放按钮相互独立。
let wristRequestBusy=false,wristPollBusy=false;
const wristPhases={idle:'尚未连接',connecting:'正在连接设备',ready:'可采样',countdown:'倒计时，请保持姿态',sampling:'正在采样，请保持静止',stopping:'正在断开',stopped:'已结束，样本已保留',failed:'连接或采集失败',unavailable:'请通过数采平台启动此页面'};

function renderWristCapture(s){
  const active=!!s.active,ready=s.phase==='ready',local=s.local!==false,solution=s.solution||{},solving=solution.phase==='running';
  $('wristMode').textContent=s.mock?'模拟设备 · 不连接硬件':'只读';
  $('wristState').textContent=`${wristPhases[s.phase]||s.phase}${['countdown','sampling'].includes(s.phase)?' · '+fmt(s.remaining_s,1)+' s':''}${!local?' · 请在本机网页连接与采样':''}`;
  $('wristError').textContent=s.error||'';$('wristError').classList.toggle('hidden',!s.error);
  $('wristConnect').disabled=wristRequestBusy||active||solving||!local||s.enabled===false;
  $('wristStop').disabled=wristRequestBusy||!active||!local;
  $('wristSample').disabled=wristRequestBusy||!active||!ready||!local;
  $('wristDownload').disabled=!s.file;
  $('wristSolve').disabled=wristRequestBusy||active||solving||!s.file||!local;
  $('wristSolveStatus20').disabled=solving;
  $('wristSolutionDownload').disabled=!solution.report||solving;
  const solveNames={idle:'结束采集后可求解',running:'正在离线求解',passed:'数值与独立验证检查通过（未应用至回放）',review_required:'已计算，结果需复核',failed:'未得到可用标定，请查看原因'};
  $('wristSolutionState').textContent=solveNames[solution.phase]||solveNames.idle;
  $('wristSolutionOutput').textContent=solution.text||'';
  ['wristRobotIp','wristTrackerIndex','wristMapLabel','wristMountLabel'].forEach(id=>$(id).disabled=active);
  const poseText=p=>p?`位置 m：[${p.position_m.map(v=>fmt(v,6)).join(', ')}]\n四元数 WXYZ：[${p.quaternion_wxyz.map(v=>fmt(v,6)).join(', ')}]\n接收帧龄：${fmt(p.age_ms,0)} ms`:'尚未收到位姿';
  const t=s.tracker;
  $('wristTrackerLive').textContent=poseText(t)+(t?`\n设备：${t.mac}\nstatus=${t.tracking_status} · ${t.tracking_status===2?'OK':'状态未确认或跟踪无效；原样保存'}`:'');
  $('wristTrackerLive').classList.toggle('wrist-pending',!!t&&(t.tracking_status!==2||t.age_ms>500));
  $('wristRobotLive').textContent=poseText(s.robot)+(s.robot?`\n查询耗时：${fmt(s.robot.request_duration_ms,1)} ms\n机器人：${s.robot_id}\n来源：${s.robot.pose_source==='response_get_joint_state+official_model_fk'?'实测关节角 → 模型腕位姿（支持拖动示教）':'旧版末端接口；拖动模式可能返回旧姿态，请重连采集'}`:'');
  const groups=s.groups||[];
  $('wristCount').textContent=`已保存 ${groups.length} 组 · 已排除 ${groups.filter(g=>g.excluded).length} 组`;
  $('wristFile').textContent=s.file?`保存位置：${s.file}`:'';
  $('wristSamples').innerHTML=groups.slice().reverse().map(g=>{
    const reasons=g.reasons||[],quality=g.excluded?'已排除':reasons.length?reasons.join('；'):'采样质量检查通过，尚未完成标定';
    const spread=g.robot_statistics?`左腕波动半径 ${fmt(g.robot_statistics.position_radius_mm,2)} mm / ${fmt(g.robot_statistics.rotation_radius_deg,2)}°`:'';
    return `<tr><td>${Number(g.id)} · ${esc(g.label||'未备注')}</td><td>${Number(g.tracker_frames)} / ${Number(g.robot_frames)}</td><td>${Number(g.pair_count)}</td><td class="${reasons.length?'wrist-pending':''}">${esc(quality)}<small>${esc(spread)}</small></td><td><button onclick="excludeWristSample(${Number(g.id)})" ${g.excluded||!active||!ready||!local?'disabled':''}>排除此组</button></td></tr>`;
  }).join('')||'<tr><td colspan="5">尚无样本。每个姿态保持静止，分别采集。</td></tr>';
}

async function pollWristCapture(){
  if(wristPollBusy||wristRequestBusy||!$('wrist-calibrationView').classList.contains('active'))return;
  wristPollBusy=true;
  try{const r=await fetch('/api/wrist-calibration/status',{cache:'no-store'});const s=await r.json();if(!r.ok)throw Error(s.error||'状态读取失败');renderWristCapture(s)}
  catch(e){$('wristState').textContent='腕部采集服务不可用：'+e.message;$('wristSample').disabled=true}
  finally{wristPollBusy=false}
}

async function wristPost(action,payload={}){
  if(wristRequestBusy)return;
  wristRequestBusy=true;
  ['wristConnect','wristSample','wristStop'].forEach(id=>$(id).disabled=true);
  try{const r=await fetch(`/api/wrist-calibration/${action}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const s=await r.json();if(!r.ok)throw Error(s.error||'采集操作失败');renderWristCapture(s)}
  catch(e){toast(e.message,true);$('wristError').textContent=e.message;$('wristError').classList.remove('hidden')}
  finally{wristRequestBusy=false;await pollWristCapture()}
}
function startWristCapture(){return wristPost('start',{robot_ip:$('wristRobotIp').value.trim(),tracker_index:Number($('wristTrackerIndex').value),map_label:$('wristMapLabel').value,mount_label:$('wristMountLabel').value})}
function sampleWristPose(){return wristPost('sample',{countdown:Number($('wristCountdown').value),duration:Number($('wristDuration').value),label:$('wristLabel').value})}
function stopWristCapture(){return wristPost('stop')}
function excludeWristSample(id){return wristPost('exclude',{id})}
function solveWristCalibration(){return wristPost('solve',{include_status20:$('wristSolveStatus20').checked})}
pollWristCapture();setInterval(pollWristCapture,500);

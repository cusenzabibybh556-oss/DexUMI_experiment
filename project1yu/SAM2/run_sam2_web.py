"""
SAM2 Web GUI —— 在浏览器中交互式选择目标并追踪视频

用法:
    /home/zmy/anaconda3/envs/sam2/bin/python run_sam2_web.py \
        --video /home/zmy/Downloads/one.mp4 \
        --output /home/zmy/Downloads/one_segmented.mp4

然后浏览器打开 http://<你的服务器IP>:8080 进行操作
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import numpy as np
import cv2
from flask import Flask, render_template_string, request, jsonify, send_file

from sam2.build_sam import build_sam2_video_predictor

app = Flask(__name__)

# 全局状态（单用户场景足够了）
STATE = {
    "predictor": None,
    "inference_state": None,
    "video_path": "",
    "output_path": "",
    "mask_output_path": "",
    "video_w": 0,
    "video_h": 0,
    "video_total_frames": 0,
    "ann_frame_idx": 0,
    "ann_obj_id": 1,
    "first_frame_bytes": None,
    "preview_mask_bytes": None,
    "tracking_progress": 0,
    "tracking_total": 0,
    "tracking_running": False,
    "tracking_done": False,
    "tracking_error": None,
    "tracking_result": "",
    "tracking_mask_result": "",
    "lock": threading.Lock(),
}

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>SAM2 - Interactive Object Selection</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #1a1a2e; color: #eee; display: flex; flex-direction: column;
       align-items: center; min-height: 100vh; padding: 20px; }
h1 { margin-bottom: 10px; font-size: 1.4em; color: #00d2ff; }
#container { position: relative; display: inline-block; border: 2px solid #333;
             border-radius: 8px; overflow: hidden; cursor: crosshair; }
#main-img { display: block; max-width: 90vw; max-height: 70vh; }
#overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
           pointer-events: none; }
.info { margin-top: 12px; font-size: 0.9em; color: #aaa; text-align: center; max-width: 600px; }
.buttons { margin-top: 16px; display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }
.btn { padding: 10px 24px; border: none; border-radius: 6px; font-size: 1em;
       cursor: pointer; font-weight: 600; transition: all 0.2s; }
.btn-reset { background: #555; color: #fff; }
.btn-preview { background: #0077b6; color: #fff; }
.btn-track { background: #06d6a0; color: #000; }
.btn:hover { filter: brightness(1.2); transform: translateY(-1px); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; filter: none; }
.status { margin-top: 12px; padding: 8px 16px; border-radius: 6px; font-size: 0.9em; display: none; }
.status.info { display: block; background: #1e3a5f; color: #90e0ef; }
.status.success { display: block; background: #1b4332; color: #95d5b2; }
.status.error { display: block; background: #4a1a1a; color: #f08080; }
.left-click-hint { color: #00ff88; }
.right-click-hint { color: #ff4466; }
.points-list { margin-top: 8px; font-size: 0.85em; color: #ccc; }
</style>
</head>
<body>
<h1>🎯 SAM2 — 在目标上点击标记</h1>
<div id="container">
    <img id="main-img" src="/frame" alt="First frame">
    <canvas id="overlay"></canvas>
</div>
<div class="info">
    <span class="left-click-hint">🟢 左键点击 = 标记目标</span> &nbsp;&nbsp;
    <span class="right-click-hint">🔴 右键点击 = 排除背景</span> &nbsp;&nbsp;
    (可以多点几次增强精度)
</div>
<div class="points-list" id="points-list"></div>
<div class="buttons">
    <button class="btn btn-reset" onclick="resetAll()">🔄 重置</button>
    <button class="btn btn-preview" id="btn-preview" onclick="doPreview()">🔍 预览分割</button>
    <button class="btn btn-track"  id="btn-track"  onclick="doTrack()">✅ 开始追踪全视频</button>
</div>
<div id="progress-bar-wrap" style="display:none; margin-top:14px; width:90%; max-width:500px; background:#333; border-radius:8px; height:28px; overflow:hidden; position:relative;">
    <div id="progress-fill" style="width:0%; height:100%; background:linear-gradient(90deg,#00d2ff,#06d6a0); border-radius:8px; transition: width 0.3s;"></div>
    <div id="progress-text" style="position:absolute; top:0; left:0; width:100%; height:100%; display:flex; align-items:center; justify-content:center; font-size:0.8em; color:#fff; font-weight:600;">0%</div>
</div>
<div id="status" class="status"></div>

<script>
const img = document.getElementById('main-img');
const overlay = document.getElementById('overlay');
const pointsList = document.getElementById('points-list');
const statusDiv = document.getElementById('status');

let points = [];
let labels = [];
let previewMaskUrl = null;
let naturalW = 0, naturalH = 0;

img.onload = () => {
    naturalW = img.naturalWidth;
    naturalH = img.naturalHeight;
    resizeCanvas();
};
window.onresize = resizeCanvas;

function resizeCanvas() {
    const rect = img.getBoundingClientRect();
    overlay.width = rect.width;
    overlay.height = rect.height;
    overlay.style.width = rect.width + 'px';
    overlay.style.height = rect.height + 'px';
    drawAll();
}

function imgToNatural(ex, ey) {
    const rect = img.getBoundingClientRect();
    const sx = naturalW / rect.width;
    const sy = naturalH / rect.height;
    return [Math.round((ex - rect.left) * sx), Math.round((ey - rect.top) * sy)];
}

function drawAll() {
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Draw preview mask
    if (previewMaskUrl) {
        ctx.globalAlpha = 0.4;
        const maskImg = new Image();
        maskImg.src = previewMaskUrl;
        ctx.drawImage(maskImg, 0, 0, overlay.width, overlay.height);
        ctx.globalAlpha = 1.0;
    }

    // Draw points
    const rect = img.getBoundingClientRect();
    points.forEach(([nx, ny], i) => {
        const x = (nx / naturalW) * rect.width;
        const y = (ny / naturalH) * rect.height;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = labels[i] === 1 ? 'rgba(0,255,130,0.8)' : 'rgba(255,60,80,0.8)';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        // Cross
        ctx.beginPath();
        ctx.moveTo(x-14, y); ctx.lineTo(x+14, y);
        ctx.moveTo(x, y-14); ctx.lineTo(x, y+14);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    });
}

img.addEventListener('click', (e) => {
    e.preventDefault();
    const [nx, ny] = imgToNatural(e.clientX, e.clientY);
    points.push([nx, ny]);
    labels.push(1);
    previewMaskUrl = null;
    updateList();
    drawAll();
});

img.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const [nx, ny] = imgToNatural(e.clientX, e.clientY);
    points.push([nx, ny]);
    labels.push(0);
    previewMaskUrl = null;
    updateList();
    drawAll();
});

function updateList() {
    if (points.length === 0) {
        pointsList.innerHTML = '';
        return;
    }
    let html = '标记: ';
    points.forEach(([x,y], i) => {
        const emoji = labels[i] === 1 ? '🟢' : '🔴';
        html += `${emoji}(${x},${y}) `;
    });
    pointsList.innerHTML = html;
}

async function resetAll() {
    points = [];
    labels = [];
    previewMaskUrl = null;
    updateList();
    drawAll();
    setStatus('已重置', 'info');
}

async function doPreview() {
    if (points.length === 0) { setStatus('请先在图片上点击标记目标', 'error'); return; }
    setStatus('正在生成预览...', 'info');
    try {
        const resp = await fetch('/preview', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({points, labels})
        });
        const data = await resp.json();
        if (data.error) { setStatus('预览失败: ' + data.error, 'error'); return; }
        previewMaskUrl = data.mask_url;
        drawAll();
        setStatus('预览就绪 ✅ — 满意后点击"开始追踪"', 'success');
    } catch(e) { setStatus('请求失败: ' + e, 'error'); }
}

async function doTrack() {
    if (points.length === 0) { setStatus('请先在图片上点击标记目标', 'error'); return; }
    document.getElementById('btn-track').disabled = true;
    document.getElementById('btn-preview').disabled = true;
    setStatus('正在追踪全视频，请耐心等待... ⏳', 'info');
    document.getElementById('progress-bar-wrap').style.display = 'block';
    try {
        const resp = await fetch('/track', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({points, labels})
        });
        const data = await resp.json();
        if (data.error) { setStatus('追踪失败: ' + data.error, 'error'); document.getElementById('progress-bar-wrap').style.display = 'none'; return; }
        // 轮询进度
        let poll = setInterval(async () => {
            const r = await fetch('/progress');
            const p = await r.json();
            const pct = p.total > 0 ? Math.round(p.progress / p.total * 100) : 0;
            document.getElementById('progress-fill').style.width = pct + '%';
            document.getElementById('progress-text').textContent = p.progress + ' / ' + p.total + ' 帧 (' + pct + '%)';
            if (!p.running) {
                clearInterval(poll);
                document.getElementById('progress-bar-wrap').style.display = 'none';
                document.getElementById('progress-fill').style.width = '0%';
                if (p.error) {
                    setStatus('追踪失败: ' + p.error, 'error');
                } else {
                    setStatus('✅ 完成！<br>🟢 绿布叠加: ' + p.result + '<br>⚫ 二值掩码: ' + p.mask_result, 'success');
                }
            }
        }, 1000);
    } catch(e) { setStatus('请求失败: ' + e, 'error'); document.getElementById('progress-bar-wrap').style.display = 'none'; }
    document.getElementById('btn-track').disabled = false;
    document.getElementById('btn-preview').disabled = false;
}

function setStatus(msg, cls) {
    statusDiv.textContent = msg;
    statusDiv.className = 'status ' + cls;
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/frame")
def get_frame():
    return send_file(io.BytesIO(STATE["first_frame_bytes"]), mimetype="image/jpeg")


@app.route("/preview", methods=["POST"])
def preview():
    data = request.json
    points = np.array(data["points"], dtype=np.float32)
    labels = np.array(data["labels"], dtype=np.int32)

    try:
        _, _, mask_logits = STATE["predictor"].add_new_points_or_box(
            STATE["inference_state"],
            frame_idx=STATE["ann_frame_idx"],
            obj_id=STATE["ann_obj_id"],
            points=points,
            labels=labels,
            clear_old_points=True,
        )
        mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)
        # 创建蓝色半透明蒙版
        mask_color = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_color[mask > 0] = [0, 100, 255, 120]

        buf = io.BytesIO()
        # 用 PNG 支持透明
        from PIL import Image as PILImage
        img = PILImage.fromarray(mask_color, "RGBA")
        img.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return jsonify({"mask_url": f"data:image/png;base64,{b64}"})
    except Exception as e:
        return jsonify({"error": str(e)})


def _run_tracking(points, labels):
    """在后台线程中执行追踪"""
    try:
        # 需要用新的 inference_state，因为 preview 可能已经污染了
        STATE["predictor"].reset_state(STATE["inference_state"])
        STATE["predictor"].add_new_points_or_box(
            STATE["inference_state"],
            frame_idx=STATE["ann_frame_idx"],
            obj_id=STATE["ann_obj_id"],
            points=points,
            labels=labels,
            clear_old_points=True,
        )

        video_segments = {}
        total = STATE["video_total_frames"]
        for out_frame_idx, out_obj_ids, out_mask_logits in STATE["predictor"].propagate_in_video(
            STATE["inference_state"]
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            with STATE["lock"]:
                STATE["tracking_progress"] = out_frame_idx + 1

        # 写输出
        video_w = STATE["video_w"]
        video_h = STATE["video_h"]
        cap = cv2.VideoCapture(STATE["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(STATE["output_path"], fourcc, fps, (video_w, video_h))
        mask_writer = cv2.VideoWriter(STATE["mask_output_path"], fourcc, fps, (video_w, video_h), isColor=False)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in video_segments:
                for obj_id, mask in video_segments[frame_idx].items():
                    m = mask[0]
                    m = cv2.resize(m.astype(np.uint8), (video_w, video_h), interpolation=cv2.INTER_NEAREST)
                    # 绿布叠加视频
                    overlay = frame.copy()
                    overlay[m > 0] = [0, 255, 0]
                    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                    # 二值掩码视频（白=目标，黑=背景）
                    mask_bin = (m > 0).astype(np.uint8) * 255
                    mask_writer.write(mask_bin)
            else:
                mask_writer.write(np.zeros((video_h, video_w), dtype=np.uint8))
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        mask_writer.release()
        with STATE["lock"]:
            STATE["tracking_running"] = False
            STATE["tracking_done"] = True
            STATE["tracking_result"] = STATE["output_path"]
            STATE["tracking_mask_result"] = STATE["mask_output_path"]
    except Exception as e:
        with STATE["lock"]:
            STATE["tracking_running"] = False
            STATE["tracking_error"] = str(e)


@app.route("/track", methods=["POST"])
def track():
    if STATE["tracking_running"]:
        return jsonify({"error": "追踪已在进行中"})

    data = request.json
    points = np.array(data["points"], dtype=np.float32)
    labels = np.array(data["labels"], dtype=np.int32)

    with STATE["lock"]:
        STATE["tracking_running"] = True
        STATE["tracking_done"] = False
        STATE["tracking_error"] = None
        STATE["tracking_progress"] = 0
        STATE["tracking_total"] = STATE["video_total_frames"]

    t = threading.Thread(target=_run_tracking, args=(points, labels), daemon=True)
    t.start()
    return jsonify({"status": "started", "total": STATE["video_total_frames"]})


@app.route("/progress")
def progress():
    with STATE["lock"]:
        return jsonify({
            "running": STATE["tracking_running"],
            "done": STATE["tracking_done"],
            "progress": STATE["tracking_progress"],
            "total": STATE["tracking_total"],
            "error": STATE["tracking_error"],
            "result": STATE["tracking_result"],
            "mask_result": STATE["tracking_mask_result"],
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="./output_segmented.mp4")
    parser.add_argument("--model", default="base_plus", choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    model_configs = {
        "tiny":       ("configs/sam2.1/sam2.1_hiera_t.yaml",   "sam2.1_hiera_tiny.pt"),
        "small":      ("configs/sam2.1/sam2.1_hiera_s.yaml",   "sam2.1_hiera_small.pt"),
        "base_plus":  ("configs/sam2.1/sam2.1_hiera_b+.yaml",  "sam2.1_hiera_base_plus.pt"),
        "large":      ("configs/sam2.1/sam2.1_hiera_l.yaml",   "sam2.1_hiera_large.pt"),
    }
    config_file, ckpt_name = model_configs[args.model]
    ckpt_dir = os.path.expanduser("~/.cache/sam2/checkpoints")

    print(f"[1/3] 加载模型: {args.model}")
    predictor = build_sam2_video_predictor(
        config_file,
        ckpt_path=os.path.join(ckpt_dir, ckpt_name),
        device=args.device,
    )

    print(f"[2/3] 读取视频: {args.video}")
    inference_state = predictor.init_state(video_path=args.video)
    video_h = inference_state["video_height"]
    video_w = inference_state["video_width"]
    print(f"      视频尺寸: {video_w}x{video_h}")

    # 获取总帧数
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"      总帧数: {total_frames}")

    # 读取第一帧
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("❌ 无法读取视频帧")
        return

    # 压缩为 JPEG
    _, jpg = cv2.imencode(".jpg", first_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 自动生成掩码视频输出路径
    base, ext = os.path.splitext(args.output)
    mask_path = f"{base}_mask{ext}"

    STATE.update({
        "predictor": predictor,
        "inference_state": inference_state,
        "video_path": args.video,
        "output_path": args.output,
        "mask_output_path": mask_path,
        "video_w": video_w,
        "video_h": video_h,
        "video_total_frames": total_frames,
        "first_frame_bytes": jpg.tobytes(),
    })

    print(f"\n{'='*60}")
    print(f"  🌐 浏览器打开: http://<服务器IP>:{args.port}")
    print(f"     本地访问:     http://localhost:{args.port}")
    print(f"     ")
    print(f"  🟢 左键点击目标  |  🔴 右键排除背景")
    print(f"  🔍 点「预览分割」查看效果")
    print(f"  ✅ 满意后点「开始追踪全视频」")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

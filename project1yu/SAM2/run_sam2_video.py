"""
SAM2 视频分割 —— 交互式选择目标，自动追踪全视频

用法（无 GUI 模式，推荐远程 SSH 使用）:
    # 先导出第一帧图片查看
    /home/zmy/anaconda3/envs/sam2/bin/python run_sam2_video.py \
        --video /home/zmy/Downloads/one.mp4 \
        --extract-frame frame0.jpg

    # 看图后指定坐标运行 (格式: x,y 多组空格分隔)
    /home/zmy/anaconda3/envs/sam2/bin/python run_sam2_video.py \
        --video /home/zmy/Downloads/one.mp4 \
        --output /home/zmy/Downloads/one_segmented.mp4 \
        --points 320,240 500,300 \
        --labels 1 0

用法（有 GUI 模式）:
    /home/zmy/anaconda3/envs/sam2/bin/python run_sam2_video.py \
        --video /path/to/video.mp4 --gui

操作说明（GUI 模式）:
    - 左键点击: 标记你要追踪的物体（绿色加号）
    - 右键点击: 标记背景/排除区域（红色减号）
    - 按 空格:  预览当前点击的分割效果
    - 按 r:     清除所有标记，重新开始
    - 按 回车: 确认标记，开始追踪全视频
    - 按 q:    退出
"""
import argparse
import os
import sys
import numpy as np
import cv2

from sam2.build_sam import build_sam2_video_predictor


# ─── GUI 模式相关（仅当 --gui 时使用） ───

class ClickState:
    def __init__(self):
        self.points = []
        self.labels = []
        self.preview_mask = None


def draw_ui(frame, state, video_w, video_h):
    display = frame.copy()
    if state.preview_mask is not None:
        mask = state.preview_mask.astype(np.uint8)
        mask = cv2.resize(mask, (video_w, video_h), interpolation=cv2.INTER_NEAREST)
        overlay = display.copy()
        overlay[mask > 0] = [255, 128, 0]
        display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, (255, 128, 0), 2)
    for (px, py), label in zip(state.points, state.labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.drawMarker(display, (px, py), color, cv2.MARKER_CROSS, 15, 2)
    info_lines = [
        "Left-click: mark object  |  Right-click: mark background",
        "SPACE: preview  |  R: reset  |  ENTER: track  |  Q: quit",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(display, line, (10, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return display


def mouse_callback(event, x, y, flags, param):
    state = param
    if event == cv2.EVENT_LBUTTONDOWN:
        state.points.append([x, y])
        state.labels.append(1)
        state.preview_mask = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.points.append([x, y])
        state.labels.append(0)
        state.preview_mask = None


def run_gui(predictor, inference_state, first_frame, video_w, video_h,
            ann_frame_idx, ann_obj_id):
    """GUI 交互模式"""
    state = ClickState()
    window_name = "SAM2 - Click to select object"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(video_w, 1280), min(video_h, 720))
    cv2.setMouseCallback(window_name, mouse_callback, state)

    while True:
        display = draw_ui(first_frame, state, video_w, video_h)
        cv2.imshow(window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord('r'):
            state.points.clear()
            state.labels.clear()
            state.preview_mask = None
            print("🔄 已清除所有标记")
        elif key == ord(' '):
            if len(state.points) == 0:
                print("⚠️  请先在画面上点击标记目标")
                continue
            print("⏳ 正在预览分割效果...")
            points_np = np.array(state.points, dtype=np.float32)
            labels_np = np.array(state.labels, dtype=np.int32)
            try:
                _, _, mask_logits = predictor.add_new_points_or_box(
                    inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id,
                    points=points_np, labels=labels_np, clear_old_points=True,
                )
                state.preview_mask = (mask_logits[0, 0] > 0.0).cpu().numpy()
                print("✅ 预览就绪")
            except Exception as e:
                print(f"❌ 预览失败: {e}")
        elif key == 13:
            if len(state.points) == 0:
                print("⚠️  请先在画面上点击标记目标")
                continue
            cv2.destroyAllWindows()
            break

    return np.array(state.points, dtype=np.float32), np.array(state.labels, dtype=np.int32)


# ─── 主流程 ───

def run_tracking(predictor, inference_state, video_segments, video_w, video_h,
                 ann_frame_idx, ann_obj_id, points_np, labels_np, output_path):
    """提交标记并追踪全视频"""
    predictor.add_new_points_or_box(
        inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id,
        points=points_np, labels=labels_np, clear_old_points=True,
    )

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    cap = cv2.VideoCapture(inference_state.get("_video_path", ""))
    # Fallback: 重新打开
    if not cap.isOpened():
        cap = cv2.VideoCapture(cv2.CAP_ANY)
    cap = cv2.VideoCapture(cv2.CAP_ANY)
    # 直接用之前读取的 cap 不行，重开
    cap.release()

    print(f"⏳ 正在写入输出视频: {output_path}")
    cap_out = cv2.VideoCapture(inference_state.get("_video_path", ""))
    # Save video_path from args
    # ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output", default="./output_segmented.mp4", help="输出视频路径")
    parser.add_argument("--model", default="base_plus",
                        choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"], help="设备 (GPU 加速，无 GPU 请用 cpu)")
    parser.add_argument("--frame", type=int, default=0,
                        help="在第几帧标记目标 (默认第0帧)")
    # 无 GUI 模式参数
    parser.add_argument("--extract-frame", metavar="PATH",
                        help="导出指定帧为图片后退出 (用于查看画面)")
    parser.add_argument("--points", nargs="+", default=[],
                        help="点击坐标, 如: --points 320,240 500,300")
    parser.add_argument("--labels", nargs="+", type=int, default=[],
                        help="标签: 1=前景 0=背景, 如: --labels 1 0")
    parser.add_argument("--gui", action="store_true",
                        help="启用 GUI 交互模式 (需要图形界面)")
    args = parser.parse_args()

    # ── 1. 加载模型 ──
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

    # ── 2. 初始化视频 ──
    print(f"[2/3] 读取视频: {args.video}")
    inference_state = predictor.init_state(video_path=args.video)
    video_h = inference_state["video_height"]
    video_w = inference_state["video_width"]
    print(f"      视频尺寸: {video_w}x{video_h}")

    # 读取指定帧
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("❌ 无法读取视频帧")
        return
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # ── 2.5 导出帧模式 ──
    if args.extract_frame:
        cv2.imwrite(args.extract_frame, first_frame)
        print(f"✅ 第 {args.frame} 帧已导出: {args.extract_frame}")
        print(f"   图片尺寸: {video_w}x{video_h}")
        print(f"   查看后运行:")
        print(f"   /home/zmy/anaconda3/envs/sam2/bin/python run_sam2_video.py \\")
        print(f"       --video {args.video} \\")
        print(f"       --output {args.output} \\")
        print(f"       --points <x>,<y> [<x>,<y> ...] \\")
        print(f"       --labels 1 [0 ...]")
        return

    ann_frame_idx = args.frame
    ann_obj_id = 1

    # ── 3. 获取标记 ──
    if args.gui:
        print(f"\n{'='*60}")
        print(f"  在画面上操作（第 {args.frame} 帧）:")
        print(f"    左键点击 = 标记物体       右键点击 = 排除区域")
        print(f"    空格     = 预览分割效果   R        = 重新开始")
        print(f"    回车     = 确认并开始追踪  Q        = 退出")
        print(f"{'='*60}\n")
        points_np, labels_np = run_gui(
            predictor, inference_state, first_frame_rgb, video_w, video_h,
            ann_frame_idx, ann_obj_id,
        )
    elif args.points:
        # 解析 --points "x,y" 格式
        pts = []
        for p in args.points:
            x, y = p.split(",")
            pts.append([float(x), float(y)])
        points_np = np.array(pts, dtype=np.float32)
        labels_np = np.array(args.labels, dtype=np.int32)
        if len(points_np) != len(labels_np):
            print(f"❌ --points ({len(points_np)} 个) 和 --labels ({len(labels_np)} 个) 数量不一致")
            return
        print(f"✅ 使用指定坐标: {pts}")
        print(f"   标签: {args.labels}  (1=前景, 0=背景)")
    else:
        print("❌ 请指定标记方式:")
        print("   GUI 模式:    --gui")
        print("   命令行模式:  --points x,y --labels 1")
        print("   或先导出帧:  --extract-frame frame0.jpg")
        return

    # ── 4. 追踪全视频并保存 ──
    print(f"\n[3/3] 追踪全视频 → {args.output}")

    predictor.add_new_points_or_box(
        inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id,
        points=points_np, labels=labels_np, clear_old_points=True,
    )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # 写输出
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (video_w, video_h))

    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                m = mask[0]
                m = cv2.resize(m.astype(np.uint8), (video_w, video_h),
                               interpolation=cv2.INTER_NEAREST)
                overlay = frame.copy()
                overlay[m > 0] = [0, 255, 0]
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        writer.write(frame)
        if frame_idx % 50 == 0:
            print(f"  进度: {frame_idx}/{total} 帧")
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"✅ 完成！输出: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_act.py —— 在 trush-bin/ 下运行 act-main 的 MuJoCo 仿真 / ACT 策略。

本文件会自动把 act-main 目录加入 sys.path，复用其中的环境、脚本策略与训练代码，
无需修改 act-main 内部文件。

四种模式:
    demo    仅运行 MuJoCo 仿真演示（脚本策略开环执行），不需要数据集 / checkpoint
    record  采集脚本演示数据 (HDF5)
    train   训练 ACT 策略
    eval    评估已训练的 ACT 策略

快速开始:
    # 1) 只看仿真演示（无显示环境也能跑，会保存视频到 --video_path）
    python run_act.py demo --task sim_transfer_cube_scripted --save_video

    # 2) 采集 5 条演示数据
    python run_act.py record --task sim_transfer_cube_scripted --dataset_dir ./data --num_episodes 5

    # 3) 训练 ACT
    python run_act.py train --task sim_transfer_cube_scripted --dataset_dir ./data --ckpt_dir ./ckpt

    # 4) 评估（自动加载 policy_best.ckpt）
    python run_act.py eval --task sim_transfer_cube_scripted --ckpt_dir ./ckpt --temporal_agg

依赖（见 act-main/README.md 与 conda_env.yaml）:
    - demo/record:  dm_control 1.0.x + mujoco 2.3.x + pyquaternion + matplotlib + numpy（record 另需 h5py）
    - train/eval:   还需 torch + einops + GPU
可用同目录下的 setup_act_env.sh 一键创建 aloha 环境。
"""

import argparse
import os
import sys

# 把 act-main 加入模块搜索路径（act-main 位于本文件上一级目录）
ACT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'act-main'))
if ACT_DIR not in sys.path:
    sys.path.insert(0, ACT_DIR)

# 默认任务参数（与 act-main/constants.py 保持一致，避免依赖其写死的 DATA_DIR）
TASKS = {
    'sim_transfer_cube_scripted': {'episode_len': 400, 'camera_names': ['top']},
    'sim_transfer_cube_human': {'episode_len': 400, 'camera_names': ['top']},
    'sim_insertion_scripted': {'episode_len': 400, 'camera_names': ['top']},
    'sim_insertion_human': {'episode_len': 500, 'camera_names': ['top']},
}


def _require_task(task):
    if task not in TASKS:
        raise SystemExit(f"未知 task: {task}，可选: {list(TASKS)}")


def _save_video(frames, path, fps):
    """用 OpenCV 把 RGB 帧列表保存为 mp4。"""
    import cv2
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"视频已保存: {path}")


def run_demo(args):
    import numpy as np
    import matplotlib
    if not args.onscreen_render:
        matplotlib.use('Agg')  # 无显示环境也能运行
    import matplotlib.pyplot as plt
    from constants import DT
    from ee_sim_env import make_ee_sim_env
    from scripted_policy import PickAndTransferPolicy, InsertionPolicy

    _require_task(args.task)
    episode_len = args.episode_len or TASKS[args.task]['episode_len']
    if 'sim_transfer_cube' in args.task:
        policy_cls = PickAndTransferPolicy
    elif 'sim_insertion' in args.task:
        policy_cls = InsertionPolicy
    else:
        raise SystemExit(f"demo 仅支持 transfer_cube / insertion 任务，收到: {args.task}")

    env = make_ee_sim_env(args.task)
    ts = env.reset()
    policy = policy_cls(inject_noise=False)

    if args.onscreen_render:
        plt.ion()
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][args.camera])
    else:
        plt_img = None

    frames = []
    rewards = []
    print(f"开始演示: {args.task} (episode_len={episode_len}, camera={args.camera})")
    for step in range(episode_len):
        action = policy(ts)
        ts = env.step(action)
        rewards.append(ts.reward)
        img = ts.observation['images'][args.camera]
        if args.save_video:
            frames.append(img)
        if args.onscreen_render:
            plt_img.set_data(img)
            plt.pause(DT)

    if args.onscreen_render:
        plt.close()

    rewards = np.array(rewards)
    max_reward = env.task.max_reward
    success = bool((rewards == max_reward).any())
    print(f"演示完成: episode_return={rewards.sum():.2f}, "
          f"highest_reward={rewards.max():.2f}/{max_reward}, success={success}")

    if args.save_video and frames:
        os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
        _save_video(frames, args.video_path, fps=int(round(1.0 / DT)))


def run_record(args):
    import record_sim_episodes as rse

    _require_task(args.task)
    rse.main(dict(
        task_name=args.task,
        dataset_dir=args.dataset_dir,
        num_episodes=args.num_episodes,
        onscreen_render=args.onscreen_render,
    ))


def run_train_or_eval(args, is_eval):
    import constants
    import imitate_episodes as ie

    _require_task(args.task)
    # 让 load_data 读取用户指定的数据集目录，而不是 constants.py 里写死的 DATA_DIR
    constants.SIM_TASK_CONFIGS[args.task]['dataset_dir'] = args.dataset_dir

    ie.main(dict(
        eval=is_eval,
        ckpt_dir=args.ckpt_dir,
        policy_class=args.policy_class,
        onscreen_render=args.onscreen_render,
        task_name=args.task,
        batch_size=args.batch_size,
        seed=args.seed,
        num_epochs=args.num_epochs,
        lr=args.lr,
        kl_weight=args.kl_weight,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
        temporal_agg=args.temporal_agg,
    ))


def build_parser():
    p = argparse.ArgumentParser(description='运行 act-main 的 MuJoCo 仿真 / ACT 策略')
    sub = p.add_subparsers(dest='mode', required=True)

    def add_common(sp, task_default='sim_transfer_cube_scripted'):
        sp.add_argument('--task', default=task_default, choices=list(TASKS))
        sp.add_argument('--onscreen_render', action='store_true', help='实时渲染（需要图形界面）')

    # demo
    d = sub.add_parser('demo', help='仅运行 MuJoCo 仿真演示（脚本策略）')
    add_common(d)
    d.add_argument('--episode_len', type=int, default=None, help='默认取任务配置')
    d.add_argument('--camera', default='angle', choices=['top', 'angle', 'vis', 'front_close'])
    d.add_argument('--save_video', action='store_true')
    d.add_argument('--video_path', default='act_demo.mp4')
    d.set_defaults(func=run_demo)

    # record
    r = sub.add_parser('record', help='采集脚本演示数据')
    add_common(r)
    r.add_argument('--dataset_dir', required=True)
    r.add_argument('--num_episodes', type=int, default=5)
    r.set_defaults(func=lambda a: run_record(a))

    # train / eval
    for name, is_eval, help_txt in [('train', False, '训练 ACT'), ('eval', True, '评估 ACT')]:
        t = sub.add_parser(name, help=help_txt)
        add_common(t)
        t.add_argument('--dataset_dir', required=True)
        t.add_argument('--ckpt_dir', required=True)
        t.add_argument('--policy_class', default='ACT', choices=['ACT', 'CNNMLP'])
        t.add_argument('--batch_size', type=int, default=8)
        t.add_argument('--seed', type=int, default=0)
        t.add_argument('--num_epochs', type=int, default=2000)
        t.add_argument('--lr', type=float, default=1e-5)
        t.add_argument('--kl_weight', type=int, default=10)
        t.add_argument('--chunk_size', type=int, default=100)
        t.add_argument('--hidden_dim', type=int, default=512)
        t.add_argument('--dim_feedforward', type=int, default=3200)
        t.add_argument('--temporal_agg', action='store_true')
        t.set_defaults(func=lambda a, e=is_eval: run_train_or_eval(a, e))

    return p


def main():
    args = build_parser().parse_args()
    # 先做个依赖自检，给出清晰报错
    if args.mode in ('demo', 'record'):
        try:
            import dm_control  # noqa: F401
        except ImportError:
            raise SystemExit(
                "缺少 dm_control。请先安装依赖（建议用同目录 setup_act_env.sh 创建 aloha 环境）：\n"
                "  conda env create -f act-main/conda_env.yaml\n  conda activate aloha\n"
                "  pip install -e act-main/detr"
            )
    if args.mode == 'record':
        try:
            import h5py  # noqa: F401
        except ImportError:
            raise SystemExit("缺少 h5py。record 模式需要：pip install h5py")
    args.func(args)


if __name__ == '__main__':
    main()

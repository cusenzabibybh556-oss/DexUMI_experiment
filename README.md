# Dexumi 完整工程

本仓库交付范围为整个 `Dexumi/` 下的项目源码，保留现有相对目录关系，包括历史项目、副本、嵌入式代码、SDK、机器人模型和根目录回放工具。数据采集入口位于 `TransHandR0725/TransHandR`。

## 项目与安装说明

| 目录 | 内容 | 安装/使用文档 |
|---|---|---|
| `TransHandR0725/TransHandR/` | 当前多源采集、网页、Vision Pro 推理、LinkerHand/Inspire SDK | [采集链路安装 README](TransHandR0725/TransHandR/README.md)、[现场使用说明](TransHandR0725/TransHandR/scripts/usage.md) |
| `linkerhand_telop_python-master/` | ROS 1/ROS 2 手套遥操作、O6 bridge 与展示资源 | [上游说明](linkerhand_telop_python-master/README_zh.md)；本地采集接入按上面的采集 README 第 4 节 |
| `project1yu/pyvut/` | VIVE Tracker 当前本地实现 | [安装、HID、配对与建图](project1yu/pyvut/README.md) |
| `project1yu/pyvut-main/` | 保留的 Tracker 原始/历史实现 | [说明](project1yu/pyvut-main/README.md) |
| `project1yu/tele/`、`project1yu/tele copy/` | 重定向、机械臂与灵巧手 SDK、模型 | [tele](project1yu/tele/README.md)、[tele copy](<project1yu/tele copy/README.md>) |
| `project1yu/TeleOperation/` | 另一套遥操作实现 | [项目说明](<project1yu/TeleOperation/TeleOperation 项目简易 README.md>) |
| `project1yu/act-main/` | ACT 训练/推理代码 | [README](project1yu/act-main/README.md)、[Conda 配置](project1yu/act-main/conda_env.yaml) |
| `tron2_act/` | Top＋Tracker＋O6 的 ACT 数据、训练与离线评估，无硬件控制 | [训练使用说明](tron2_act/README.md)、[默认配置](configs/tron2_act.yaml) |
| `tron2_vision/` | Top前景标签编辑、轻量分割训练、自动处理与ACT输入缓存 | [数据处理链路](tron2_vision/README.md) |
| `project1yu/DexUMI-main/` | DexUMI 原始工程，含嵌入式代码 | [README](project1yu/DexUMI-main/README.md)、[环境配置](project1yu/DexUMI-main/environment.yml) |
| 根目录 `replay_*.py` 等 | TRON2/O6 回放与测试 | [固定起点参数](tron2_fixed_start_profiles.md)、[命令记录](tron2_tracker_replay_commands.md) |

项目总览见 [DEXUMI_工程内容整理.md](DEXUMI_工程内容整理.md)。子项目文档中原机器的绝对路径、网络与串口配置需按目标机器调整。历史/训练项目的 README 原样保留，尚未全部重新验证安装。

## 环境覆盖范围

采集 README 已汇总采集 Python 3.10、系统 ROS Humble、Vision Pro Python 3.8 三部分安装命令。TRON2 回放仍需额外交付厂商 `limxsdk`；SAM3/DiffuEraser 位于此目录之外，当前完整源码快照不包含它。其他训练项目按各自环境文件安装，不能将所有依赖混装进一个环境。

## 准备整个工程上传 GitHub

保留原工作区所有 `.git` 与 `.gitignore`。使用导出工具，在相邻新目录生成全部项目源码的统一仓库快照：

```bash
cd /absolute/path/to/Dexumi
python3 tools/prepare_github_export.py --destination ../Dexumi-github
```

目标目录必须不存在；工具不会覆盖现有目录。它会：

- 复制整个工程中的源码、文档、配置、SDK 与模型资源，保留历史项目和 `tele copy` 等副本。
- 跳过嵌套 `.git`、虚拟环境、缓存、ROS 构建目录、采集输出、模型权重与压缩包；原文件保留在原工作区。
- 仅在发布副本中清空已识别的 JSON/YAML 密码配置并替换 Wi-Fi SSID；部署时填写本机值。此规则不是完整的凭据审计。
- 初始化新的 `main` 仓库，按明确文件清单暂存全部交付文件，避免子项目 `.gitignore` 意外漏掉源码；不会创建子模块。
- 生成 `github-export/manifest.json`，记录每个文件的校验和、排除项、脱敏配置和原仓库位置；不会自动提交或推送。

进入导出目录检查后，提交并推送到自己的 GitHub 仓库：

```bash
cd ../Dexumi-github
git diff --cached --stat
git status --short
git commit -m "Import complete Dexumi source snapshot"
# 将下方地址换为目标仓库，且先确认目标为空或适合本次初始导入
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

新快照不携带各子仓库的提交历史；原仓库和历史仍保存在原工作区。源码全量上传不等于环境、数据和权重全部备份，权重路径与 SHA-256 见 [采集迁移报告](TransHandR0725/TransHandR/docs/migration_audit.md)。普通 Git 上传单文件上限为 100 MiB，大制品另用 LFS 或制品下载管理，见 [GitHub 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

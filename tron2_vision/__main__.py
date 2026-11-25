import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Top前景：标签修正 → 分割训练 → 自动处理")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="从原始HDF5和SAM3试验缓存导出待修正标签")
    export.add_argument("--data-root", default="/home/zmy/workplace/insert-data/transhandr_demo/o6")
    export.add_argument("--cache-root", required=True)
    export.add_argument("--episodes", type=int, nargs="+", required=True)
    export.add_argument("--samples", type=int, default=5)
    export.add_argument("--frames", nargs="*", default=[], help="额外原始帧，例如66:100")
    export.add_argument("--output", required=True)
    serve = commands.add_parser("serve", help="打开本地画笔／多边形标注服务")
    serve.add_argument("--project", required=True)
    serve.add_argument("--port", type=int, default=8091)
    train = commands.add_parser("train", help="只使用已检查标签，按episode隔离验证")
    train.add_argument("--project", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--validation-episodes", type=int, nargs="+", required=True)
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--input-size", type=int, nargs=2, default=[240, 320])
    train.add_argument("--device", default="cuda")
    train.add_argument("--no-pretrained", action="store_true")
    process = commands.add_parser("process", help="同一冻结模型逐帧导出无损实验缓存")
    process.add_argument("--checkpoint", required=True)
    process.add_argument("--data-root", default="/home/zmy/workplace/insert-data/transhandr_demo/o6")
    process.add_argument("--episodes", type=int, nargs="+", required=True)
    process.add_argument("--output", required=True)
    process.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "export":
        from .annotations import export_project
        result = export_project(args.data_root, args.cache_root, args.episodes, args.output, args.samples, args.frames)
        print(f"Exported {len(result['frames'])} draft masks")
    elif args.command == "serve":
        from .annotations import serve
        serve(args.project, args.port)
    elif args.command == "train":
        from .pipeline import train
        result = train(args.project, args.output, args.validation_episodes, args.steps, args.batch_size, args.input_size, args.device, not args.no_pretrained)
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    else:
        from .pipeline import process
        process(args.checkpoint, args.data_root, args.episodes, args.output, args.device)


if __name__ == "__main__":
    main()

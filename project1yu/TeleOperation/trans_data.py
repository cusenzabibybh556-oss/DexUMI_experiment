import os
import shutil


SOURCE_DIR = "/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/data/robot_data/Orange_in_bawl/l10_masks"
TARGET_DIR = "/media/ub/My Book/04-Imitation Learning/自采抓取数据/Orange_in_bawl/l10_masks"


def main():
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f"源目录不存在: {SOURCE_DIR}")

    os.makedirs(TARGET_DIR, exist_ok=True)

    file_names = sorted(
        # name for name in os.listdir(SOURCE_DIR) if name.lower().endswith(".hdf5")
        name for name in os.listdir(SOURCE_DIR) if name.lower().endswith(".npy")
    )

    if not file_names:
        print("源目录中没有找到 .hdf5 文件。")
        return

    total_count = len(file_names)
    copied_count = 0
    skipped_count = 0

    for index, file_name in enumerate(file_names, start=1):
        src_path = os.path.join(SOURCE_DIR, file_name)
        dst_path = os.path.join(TARGET_DIR, file_name)
        progress = f"[{index}/{total_count}]"

        if os.path.exists(dst_path):
            skipped_count += 1
            print(f"{progress} 跳过已存在文件: {file_name}")
            continue

        print(f"{progress} 复制: {file_name}")
        shutil.copy2(src_path, dst_path)
        copied_count += 1

    print(
        f"完成。共复制 {copied_count} 个文件，跳过 {skipped_count} 个重名文件。"
    )


if __name__ == "__main__":
    main()

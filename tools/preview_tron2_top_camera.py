#!/usr/bin/env python3
"""只读订阅 TRON2 Top，将其光学视场匹配到 HDF5 头部相机。

source /opt/ros/humble/setup.bash
/usr/bin/python3 tools/preview_tron2_top_camera.py --snapshot

默认实时显示；--snapshot 保存一组真实画面后退出。只订阅图像和查询设备信息，
不设置相机参数或控制机器人。校正只匹配内参，不补偿两个安装位置的平移视差。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import cv2
import h5py
import numpy as np


DEFAULT_REFERENCE = "/home/zmy/workplace/insert-data/transhandr_demo/o6/episode_000065.hdf5"


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    k: list[float]
    d: list[float]
    distortion_model: str

    def __post_init__(self):
        matrix = np.asarray(self.k, dtype=float).reshape(3, 3)
        if (self.width <= 0 or self.height <= 0 or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1])
                or not np.isfinite(self.d).all()):
            raise ValueError("相机尺寸或内参无效")
        # inverse_brown_conrady 非零系数不能按 OpenCV 正向畸变系数使用。
        if self.distortion_model not in {"plumb_bob", "rational_polynomial"}:
            if np.any(np.asarray(self.d) != 0):
                raise ValueError(f"暂不支持非零畸变模型：{self.distortion_model}")

    @property
    def matrix(self):
        return np.asarray(self.k, dtype=np.float64).reshape(3, 3)

    def pinhole_fov_deg(self):
        """无畸变针孔视场近似；宽角原图还受镜头畸变影响。"""
        k = self.matrix
        return [float(np.degrees(2 * np.arctan(self.width / (2 * k[0, 0])))),
                float(np.degrees(2 * np.arctan(self.height / (2 * k[1, 1]))))]


class CameraViewAlignment:
    """像素重映射可复用于推理；apply 保留输入的 RGB/BGR 通道顺序。"""

    def __init__(self, source: CameraModel, target: CameraModel, rotation=None):
        self.source, self.target = source, target
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        if (self.rotation.shape != (3, 3) or not np.isfinite(self.rotation).all()
                or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.rotation), 1)):
            raise ValueError("目标到 Top 的旋转矩阵必须属于 SO(3)")
        # 对目标像素去畸变、转成 Top 光学坐标系光线，再投影回原图；不包含相机平移。
        uu, vv = np.meshgrid(np.arange(target.width), np.arange(target.height))
        pixels = np.stack([uu, vv], axis=-1).astype(np.float64).reshape(-1, 1, 2)
        rays = cv2.undistortPoints(pixels, target.matrix, np.asarray(target.d))
        xyz = np.column_stack([rays.reshape(-1, 2), np.ones(len(pixels))])
        xyz = xyz @ self.rotation.T
        if np.any(xyz[:, 2] <= 0):
            raise ValueError("虚拟视角朝向了 Top 相机背面")
        projected, _ = cv2.projectPoints(
            xyz, np.zeros(3), np.zeros(3), source.matrix, np.asarray(source.d))
        projected = projected.reshape(target.height, target.width, 2).astype(np.float32)
        self.map_x, self.map_y = projected[..., 0].copy(), projected[..., 1].copy()
        valid = cv2.remap(np.ones((source.height, source.width), np.float32),
                          self.map_x, self.map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
        self.valid_fraction = float(np.mean(valid > 0.999))

    def apply(self, image):
        if image.shape[:2] != (self.source.height, self.source.width):
            raise ValueError("图像尺寸与 Top 标定不一致；需要读取当前分辨率的 camera_info")
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(CameraModel(**data["source"]), CameraModel(**data["target"]),
                   data.get("rotation_target_to_source"))

    def footprint(self):
        """返回输出画面四条边在 Top 原图内的曲线，包含畸变影响。"""
        points = np.stack([self.map_x, self.map_y], axis=-1)
        boundary = np.concatenate([points[0], points[1:, -1],
                                   points[-1, -2::-1], points[-2:0:-1, 0]])
        return np.round(boundary).astype(np.int32).reshape(-1, 1, 2)


def read_reference(path, frame):
    with h5py.File(path, "r") as f:
        raw = f.attrs["d435i_rgb_intrinsics"]
        if isinstance(raw, bytes):
            raw = raw.decode()
        intr = json.loads(raw)
        ds = f["observations/images/d435i_rgb"]
        if not 0 <= frame < len(ds):
            raise ValueError(f"参考帧应在 0 到 {len(ds) - 1} 之间")
        rgb = ds[frame]
        if not np.any(rgb):
            raise ValueError("参考 RGB 帧是全零占位数据")
        model = CameraModel(intr["width"], intr["height"],
                            [intr["fx"], 0, intr["ppx"],
                             0, intr["fy"], intr["ppy"], 0, 0, 1],
                            intr["coeffs"], intr["model"])
        if rgb.shape != (model.height, model.width, 3):
            raise ValueError("参考帧尺寸与其内参不一致")
        return model, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def optical_rotation(look_down_deg=0.0, look_right_deg=0.0):
    """只移动取景方向；正值表示向 Top 光学坐标的下方/右方看。"""
    # 光学坐标为 x 向右、y 向下、z 向前；向下看对应绕 x 轴负转。
    pitch = cv2.Rodrigues(np.array([-np.radians(look_down_deg), 0., 0.]))[0]
    yaw = cv2.Rodrigues(np.array([0., np.radians(look_right_deg), 0.]))[0]
    return yaw @ pitch


def query_readonly(node, service_type, service_name, request, timeout=2.0):
    import rclpy
    client = node.create_client(service_type, service_name)
    try:
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        return future.result() if future.done() else None
    finally:
        node.destroy_client(client)


def read_device_metadata(node):
    from rcl_interfaces.srv import GetParameters
    fields = ["serial_no", "rgb_camera.color_profile", "rgb_camera.color_format",
              "rotation_filter.enable", "rotation_filter.rotation"]
    reply = query_readonly(node, GetParameters, "/camera/top/get_parameters",
                           GetParameters.Request(names=fields))
    result = {}
    if reply:
        value_fields = {1: "bool_value", 2: "integer_value", 3: "double_value",
                        4: "string_value"}
        result["parameters"] = {
            name: getattr(value, value_fields[value.type], None)
            for name, value in zip(fields, reply.values) if value.type in value_fields}
    # 标准 ROS 安装不一定带 RealSense 自定义服务；缺失时仍可使用 CameraInfo。
    try:
        from realsense2_camera_msgs.srv import DeviceInfo
    except ImportError:
        return result
    reply = query_readonly(node, DeviceInfo, "/camera/top/device_info", DeviceInfo.Request())
    if reply:
        result["device"] = {key: getattr(reply, key)
                            for key in reply.get_fields_and_field_types()}
    return result


def comparison(image, aligned, alignment):
    native = image.copy()
    cv2.polylines(native, [alignment.footprint()], True, (0, 210, 255), 2)
    panels = []
    for frame, title in [(native, "TOP RAW | yellow = sampled area"),
                         (aligned, "TOP -> training camera intrinsics")]:
        # 仅排版预览；aligned.png 保持目标相机的原始像素尺寸。
        panel = np.zeros((520, 640, 3), np.uint8)
        panel[40:] = cv2.resize(frame, (640, 480))
        cv2.putText(panel, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (245, 245, 245), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-hdf5", type=Path, default=Path(DEFAULT_REFERENCE))
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--image-topic", default="/camera/top/color/image_raw/compressed")
    parser.add_argument("--info-topic", default="/camera/top/color/camera_info")
    parser.add_argument("--look-down-deg", type=float, default=0.0,
                        help="虚拟取景向下转的角度；不会驱动机器人头部")
    parser.add_argument("--look-right-deg", type=float, default=0.0,
                        help="虚拟取景向右转的角度；不会驱动机器人头部")
    parser.add_argument("--snapshot", action="store_true", help="保存真实对比图后退出，不打开窗口")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("camera_alignment_results") / datetime.now(
                            ZoneInfo("Asia/Shanghai")).strftime("top_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    target, reference = read_reference(args.reference_hdf5, args.reference_frame)
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, CompressedImage
    rclpy.init()
    node = rclpy.create_node("top_camera_intrinsics_preview")
    latest = {}

    def on_info(msg):
        latest["info"] = msg

    def on_image(msg):
        decoded = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            latest.update(image=decoded, image_stamp_ns=msg.header.stamp.sec * 10**9
                          + msg.header.stamp.nanosec, received=time.monotonic())

    node.create_subscription(CameraInfo, args.info_topic, on_info, qos_profile_sensor_data)
    node.create_subscription(CompressedImage, args.image_topic, on_image, qos_profile_sensor_data)
    try:
        deadline = time.monotonic() + 15
        while not {"info", "image"}.issubset(latest):
            if time.monotonic() > deadline:
                raise RuntimeError("15 秒内未收到 Top 图像和 CameraInfo；检查网络与 ROS_DOMAIN_ID")
            rclpy.spin_once(node, timeout_sec=0.1)
        msg = latest["info"]
        source = CameraModel(msg.width, msg.height, list(msg.k), list(msg.d),
                             msg.distortion_model)
        alignment = CameraViewAlignment(source, target, optical_rotation(
            args.look_down_deg, args.look_right_deg))
        if alignment.valid_fraction < 1.0:
            raise ValueError(f"虚拟取景超出 Top 原图，只有 {alignment.valid_fraction:.2%} "
                             "输出像素有效；请减小虚拟角度")
        metadata = read_device_metadata(node)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        print("Top K:", source.matrix, "\nTop D:", source.d)
        print("针孔视场（水平/垂直，度）：", source.pinhole_fov_deg())
        print("训练参考 K:", target.matrix)
        print(f"有效输出像素：{alignment.valid_fraction:.2%}")
        print(f"输出目录：{args.output_dir.resolve()}")

        def save_current():
            image = latest["image"]
            aligned = alignment.apply(image)
            panel = comparison(image, aligned, alignment)
            for name, frame in [("top_raw.png", image), ("top_aligned.png", aligned),
                                ("comparison.png", panel), ("training_reference.png", reference)]:
                if not cv2.imwrite(str(args.output_dir / name), frame):
                    raise RuntimeError(f"图片保存失败：{name}")
            config = {"schema": "top-to-training-intrinsics-v1",
                      "captured_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                      "image_topic": args.image_topic, "info_topic": args.info_topic,
                      "optical_frame": msg.header.frame_id,
                      "image_stamp_ns": latest["image_stamp_ns"],
                      "source": asdict(source), "target": asdict(target),
                      "device_metadata": metadata,
                      "reference_hdf5": str(args.reference_hdf5.resolve()),
                      "reference_frame": args.reference_frame,
                      "source_pinhole_fov_deg": source.pinhole_fov_deg(),
                      "target_pinhole_fov_deg": target.pinhole_fov_deg(),
                      "valid_fraction": alignment.valid_fraction,
                      "rotation_target_to_source": alignment.rotation.tolist(),
                      "virtual_look_down_deg": args.look_down_deg,
                      "virtual_look_right_deg": args.look_right_deg,
                      "extrinsics_calibrated": False,
                      "assumption": "same optical center; only the supplied virtual rotation; translation parallax is not corrected",
                      "source_sampling_bounds_xy": [float(alignment.map_x.min()),
                          float(alignment.map_y.min()), float(alignment.map_x.max()),
                          float(alignment.map_y.max())]}
            (args.output_dir / "calibration.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return panel

        save_current()
        if not args.snapshot:
            print("实时预览：S 保存当前帧，Q / Esc 退出；光学校正未包含安装外参。")
            window = "Top: raw and training view | S save | Q exit"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 520)
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.02)
                current = latest["info"]
                if (current.width != source.width or current.height != source.height
                        or not np.allclose(current.k, source.k)
                        or not np.allclose(current.d, source.d)):
                    raise RuntimeError("Top 内参已变化，请重新运行以生成新的校正参数")
                image = latest["image"]
                panel = comparison(image, alignment.apply(image), alignment)
                if time.monotonic() - latest["received"] > 1:
                    cv2.putText(panel, "STALE: no new frames", (15, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(window, panel)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("s"):
                    save_current()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if not args.snapshot:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

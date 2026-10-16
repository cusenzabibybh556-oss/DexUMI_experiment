import sys
import os
# 将上一级目录（tele/，其中含 constant.py）加入模块搜索路径，
# 替代原机器上写死的 /home/user-iip/code/TeleOperation/Robot_control
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pyrealsense2 as rs
import cv2
import numpy as np
from constant import*
import time

class RealsenseCamera:
    def __init__(self):
        # self.pipeline = rs.pipeline()
        self.config_list = [rs.config() for i in range(len(CAMERA_ID))]

        self.pipeline_list = [rs.pipeline() for i in range(len(CAMERA_ID))]
        self.align = rs.align(rs.stream.color)


        # self.config.enable_stream(rs.stream.depth, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.z16, IMAGE_FPS)
        # self.config.enable_stream(rs.stream.color, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.bgr8, IMAGE_FPS)
        if len(CAMERA_ID) == 1: # one realsense
                self.config_list[0].enable_device(CAMERA_ID[0])
                self.config_list[0].enable_stream(rs.stream.depth, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.z16, IMAGE_FPS)
                self.config_list[0].enable_stream(rs.stream.color, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.bgr8, IMAGE_FPS)
                self.pipeline_list[0].start(self.config_list[0])
        elif len(CAMERA_ID) >= 2: # two realsense
            for n, config in enumerate(self.config_list):
                config.enable_device(CAMERA_ID[n])
                config.enable_stream(rs.stream.depth, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.z16, IMAGE_FPS)
                config.enable_stream(rs.stream.color, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.bgr8, IMAGE_FPS)
                # self.config_list[n] = config
            # start pipeline
            for n, pipeline in enumerate(self.pipeline_list):
                pipeline.start(self.config_list[n])

    def wait_frames(self, frame_id=None, Align = True):
        frame_list = [None for i in range(len(CAMERA_ID))]
        if frame_id != None: # give a frame id
            frame_list[n] = self.pipeline_list[frame_id].wait_for_frames()
        else: # get all frame
            if len(CAMERA_ID) == 1:
                frame_list.append(self.pipeline_list[0].wait_for_frames())
            else:
                for n, _ in enumerate(self.pipeline_list):
                    frame_list[n] = self.pipeline_list[n].wait_for_frames()
        if not Align:
            return frame_list
        elif Align:
            aligned_frame_list = [None for i in range(len(CAMERA_ID))]
            for n, _ in enumerate(frame_list):
                aligned_frame_list[n] = self.align.process(frame_list[n])
            return aligned_frame_list

    def rgb_image(self, frame):
        rgbframe = self.rgb_frame(frame)
        rgbimage = np.asanyarray(rgbframe.get_data())
        return rgbimage

    def depth_image(self, frame):
        depthframe = self.depth_frame(frame)
        depthimage =np.asanyarray(depthframe.get_data())
        return depthimage

    def rgb_frame(self, frame):
        rgbframe = frame.get_color_frame()
        return rgbframe
    
    def depth_frame(self, frame):
        depthframe = frame.get_depth_frame()
        return depthframe
    
    def depth_colormap(self,depth_image):
        return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)

    def stop(self):
        for pipeline in self.pipeline_list:
            pipeline.stop()



if __name__ == '__main__':

    cap = RealsenseCamera() # 
    while True:
        start = time.time()
        frames = cap.wait_frames()
        img0 = cap.rgb_image(frames[0])
        img1 = cap.rgb_image(frames[1])
        cv2.imshow("img0", img0)
        cv2.imshow("img1", img1)
        print("FPS:", 1/(time.time()-start), "f/s")
        if cv2.waitKey(1) == ord("q"):
            break

    cap.stop()
    cv2.destroyAllWindows()

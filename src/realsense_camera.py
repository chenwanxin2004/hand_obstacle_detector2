#!/usr/bin/env python3
"""
RealSense相机管理模块
支持RealSense深度相机和普通摄像头
"""
import cv2 as cv
import numpy as np
from typing import Tuple, Optional, Dict, Any
import time
import pyrealsense2 as rs
class RealSenseCamera:
    """
    RealSense深度相机管理类
    """
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        初始化RealSense相机
        
        Args:
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # 初始化RealSense管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置流
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # 深度对齐器
        self.align = rs.align(rs.stream.color)
        
        # 深度可视化
        self.depth_scale = None
        self.depth_visualizer = rs.colorizer()
        
        self.is_running = False
        
    def start(self) -> bool:
        """
        启动相机
        
        Returns:
            bool: 启动是否成功
        """
        try:
            # 启动管道
            profile = self.pipeline.start(self.config)
            
            # 获取深度比例
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            self.is_running = True
            print(f"✅ RealSense相机启动成功")
           
            
            return True
            
        except Exception as e:
            print(f"❌ RealSense相机启动失败: {e}")
            return False
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取深度帧和彩色帧
        
        Returns:
            Tuple[深度帧, 彩色帧]: 深度图(米)和彩色图
        """
        if not self.is_running:
            return None, None
            
        try:
            # 等待帧
            frames = self.pipeline.wait_for_frames()
            
            # 对齐深度帧到彩色帧
            aligned_frames = self.align.process(frames)
            
            # 获取对齐后的深度帧和彩色帧
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 转换深度单位为米
            if self.depth_scale:
                depth_image = depth_image.astype(np.float32) * self.depth_scale
            
            return depth_image, color_image
            
        except Exception as e:
            print(f"❌ 获取帧失败: {e}")
            return None, None
    
    def create_depth_visualization(self, depth_image: np.ndarray) -> np.ndarray:
        """
        创建深度图可视化
        
        Args:
            depth_image: 深度图像(米)
            
        Returns:
            np.ndarray: 彩色深度图
        """
        if depth_image is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 转换为16位深度图用于可视化
        if self.depth_scale and self.depth_scale > 0:
            depth_16bit = (depth_image / self.depth_scale).astype(np.uint16)
        else:
            # 如果没有深度比例，直接使用深度值
            depth_16bit = (depth_image * 1000).astype(np.uint16)  # 转换为毫米
        
        # 使用OpenCV的颜色映射进行可视化
        # 将深度值归一化到0-255范围
        depth_normalized = cv.normalize(depth_16bit, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        # 应用颜色映射
        colorized_image = cv.applyColorMap(depth_normalized, cv.COLORMAP_JET)
        
        return colorized_image
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        获取相机信息
        
        Returns:
            Dict: 相机信息
        """
        return {
            'type': 'RealSense',
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'depth_scale': self.depth_scale,
            'is_running': self.is_running
        }
    
    def is_available(self) -> bool:
        """
        检查相机是否可用
        
        Returns:
            bool: 相机是否可用
        """
        return self.is_running
    
    def cleanup(self):
        """
        清理资源
        """
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
            print("✅ RealSense相机已停止")





def create_camera(**kwargs) -> Any:
    """
    相机工厂函数（仅RealSense）
    
    Args:
        **kwargs: RealSense 初始化参数（如 width, height, fps）
        
    Returns:
        相机对象（RealSenseCamera）
    """
 
    camera = RealSenseCamera(**kwargs)
    if camera.start():
        return camera
    raise RuntimeError("Failed to start RealSense camera")


def main():
    """
    测试相机功能
    """
    print("🚀 测试相机功能...")
    
    # 创建相机（仅RealSense）
    camera = create_camera()
    
    if not camera.is_available():
        print("❌ 相机不可用")
        return
    
    print(f"📷 相机信息: {camera.get_camera_info()}")
    
    try:
        frame_count = 0
        while frame_count < 100:  # 测试100帧
            depth_frame, color_frame = camera.get_frames()
            
            if depth_frame is not None and color_frame is not None:
                # 显示彩色图
                cv.imshow('Color Frame', color_frame)
                
                # 显示深度图
                depth_vis = camera.create_depth_visualization(depth_frame)
                cv.imshow('Depth Frame', depth_vis)
                
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"处理帧: {frame_count}")
                
                # 按'q'退出
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("❌ 无法获取帧")
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    
    finally:
        camera.cleanup()
        cv.destroyAllWindows()
        print("✅ 测试完成")


if __name__ == "__main__":
    main()

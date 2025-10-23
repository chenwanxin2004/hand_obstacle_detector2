#!/usr/bin/env python3
"""
RealSense相机管理模块
支持RealSense深度相机和普通摄像头
"""

import cv2 as cv
import numpy as np
from typing import Tuple, Optional, Dict, Any
import time

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️ pyrealsense2 not available, using mock camera")

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
            print(f"   分辨率: {self.width}x{self.height}")
            print(f"   帧率: {self.fps}")
            print(f"   深度比例: {self.depth_scale}")
            
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


class MockCamera:
    """
    模拟相机类（使用普通摄像头）
    """
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        初始化模拟相机
        
        Args:
            camera_index: 摄像头索引
            width: 图像宽度
            height: 图像高度
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        
        # 初始化摄像头
        self.cap = cv.VideoCapture(camera_index)
        
        if self.cap.isOpened():
            # 设置分辨率
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
            
            # 设置帧率
            self.cap.set(cv.CAP_PROP_FPS, 30)
            
            self.is_running = True
            print(f"✅ 模拟相机启动成功 (摄像头 {camera_index})")
            print(f"   分辨率: {width}x{height}")
        else:
            self.is_running = False
            print(f"❌ 无法打开摄像头 {camera_index}")
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取深度帧和彩色帧（模拟版本）
        
        Returns:
            Tuple[深度帧, 彩色帧]: 模拟深度图和彩色图
        """
        if not self.is_running:
            return None, None
            
        ret, color_image = self.cap.read()
        if not ret:
            return None, None
        
        # 调整图像大小
        color_image = cv.resize(color_image, (self.width, self.height))
        
        # 创建模拟深度图（基于图像亮度）
        gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
        
        # 简单的深度模拟：基于亮度反比
        # 较暗的区域假设更近，较亮的区域假设更远
        depth_image = (255 - gray).astype(np.float32) / 255.0
        
        # 添加一些随机噪声使深度图更真实
        noise = np.random.normal(0, 0.05, depth_image.shape)
        depth_image = np.clip(depth_image + noise, 0.1, 2.0)
        
        return depth_image, color_image
    
    def create_depth_visualization(self, depth_image: np.ndarray) -> np.ndarray:
        """
        创建深度图可视化（模拟版本）
        
        Args:
            depth_image: 深度图像(米)
            
        Returns:
            np.ndarray: 彩色深度图
        """
        if depth_image is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 将深度图转换为0-255范围
        if depth_image.max() > depth_image.min():
            depth_normalized = ((depth_image - depth_image.min()) / 
                               (depth_image.max() - depth_image.min()) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
        
        # 应用颜色映射
        depth_colored = cv.applyColorMap(depth_normalized, cv.COLORMAP_JET)
        
        return depth_colored
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        获取相机信息
        
        Returns:
            Dict: 相机信息
        """
        return {
            'type': 'Mock',
            'camera_index': self.camera_index,
            'width': self.width,
            'height': self.height,
            'is_running': self.is_running
        }
    
    def is_available(self) -> bool:
        """
        检查相机是否可用
        
        Returns:
            bool: 相机是否可用
        """
        return self.is_running and self.cap.isOpened()
    
    def cleanup(self):
        """
        清理资源
        """
        if self.cap.isOpened():
            self.cap.release()
            self.is_running = False
            print("✅ 模拟相机已停止")


def create_camera(camera_type: str = "auto", **kwargs) -> Any:
    """
    相机工厂函数
    
    Args:
        camera_type: 相机类型 ("realsense", "mock", "auto")
        **kwargs: 其他参数
        
    Returns:
        相机对象
    """
    if camera_type == "realsense":
        if not REALSENSE_AVAILABLE:
            print("⚠️ RealSense不可用，使用模拟相机")
            return MockCamera(**kwargs)
        
        camera = RealSenseCamera(**kwargs)
        if camera.start():
            return camera
        else:
            print("⚠️ RealSense启动失败，使用模拟相机")
            return MockCamera(**kwargs)
    
    elif camera_type == "mock":
        return MockCamera(**kwargs)
    
    elif camera_type == "auto":
        # 自动选择：优先尝试RealSense
        if REALSENSE_AVAILABLE:
            camera = RealSenseCamera(**kwargs)
            if camera.start():
                return camera
        
        # RealSense不可用或启动失败，使用模拟相机
        print("🔄 自动切换到模拟相机")
        return MockCamera(**kwargs)
    
    else:
        raise ValueError(f"不支持的相机类型: {camera_type}")


def main():
    """
    测试相机功能
    """
    print("🚀 测试相机功能...")
    
    # 创建相机
    camera = create_camera("auto")
    
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

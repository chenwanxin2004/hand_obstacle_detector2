#!/usr/bin/env python3
"""
YOLOv8分割障碍物检测模块
使用YOLOv8-seg进行语义分割，辅助生成障碍物掩膜
"""

import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
import os
import torch
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)
import onnxruntime as ort
 
# 稳定的检测结果类型别名，避免外部依赖内部结构细节
DetectionResult = Dict[str, Any]

# 轻量级的ONNX结果包装，提供与Ultralytics结果最小一致的接口
class _ONNXResult:
    def __init__(self, outputs, original_shape):
        self.outputs = outputs
        self.original_shape = original_shape
        self.masks = None  # 分割掩膜占位
        self.boxes = None  # 边界框占位
        self.names = {i: f'class_{i}' for i in range(80)}

    def __iter__(self):
        return iter([self])

    def __getitem__(self, index):
        if index == 0:
            return self
        raise IndexError("ONNXResult only supports index 0")

class YOLOObstacleDetector:
    """
    YOLOv8分割障碍物检测器
    使用YOLOv8-seg进行语义分割，识别和分离障碍物
    """
    
    def __init__(self, 
                 model_path: str = "yolov8n-seg.pt",
                 confidence_threshold: float = 0.5,
                 device: str = "auto",
                 use_quantized: bool = True,
                 quantization_type: str = "fp16"):
        """
        初始化YOLOv8障碍物检测器
        
        Args:
            model_path: YOLOv8分割模型路径
            confidence_threshold: 检测置信度阈值
            device: 运行设备 ("cpu", "cuda", "auto")
            use_quantized: 是否使用量化模型
            quantization_type: 量化类型 ("fp16", "int8", "original")
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.use_quantized = use_quantized
        self.quantization_type = quantization_type
        
        # 根据量化类型选择模型路径
        if use_quantized:
            self.model_path = self._get_quantized_model_path(quantization_type)
        
        # 动态类别ID集合（初始化后根据模型类别填充）
        self.obstacle_class_ids = set()
        self.hand_class_ids = set()
        
        self.model = None
        self.is_initialized = False
        
        # 性能统计
        self.inference_times = []
        
        # 初始化模型
        self._initialize_model()
    
    def _get_quantized_model_path(self, quantization_type: str) -> str:
        """
        根据量化类型获取对应的模型路径
        
        Args:
            quantization_type: 量化类型
            
        Returns:
            量化模型路径
        """
        quantized_models_dir = "src/quantized_models"
        
        if quantization_type == "fp16":
            # 优先选择ONNX格式，其次OpenVINO
            onnx_path = os.path.join(quantized_models_dir, "yolov8n-seg_fp16.onnx")
            onnx_fallback = "src/yolov8n-seg.onnx"  # 导出的ONNX文件
            openvino_path = os.path.join(quantized_models_dir, "yolov8n-seg_fp16")
            
            if os.path.exists(onnx_path):
                return onnx_path
            elif os.path.exists(onnx_fallback):
                logger.info(f"Use exported ONNX model: {onnx_fallback}")
                return onnx_fallback
            elif os.path.exists(openvino_path):
                return openvino_path
            else:
                logger.warning("FP16 quantized model not found, using original .pt model")
                return "yolov8n-seg.pt"
                
        elif quantization_type == "int8":
            # 优先选择ONNX格式，其次OpenVINO
            onnx_path = os.path.join(quantized_models_dir, "yolov8n-seg_int8.onnx")
            openvino_path = os.path.join(quantized_models_dir, "yolov8n-seg_int8")
            
            if os.path.exists(onnx_path):
                return onnx_path
            elif os.path.exists(openvino_path):
                return openvino_path
            else:
                logger.warning("INT8 quantized model not found, falling back to FP16")
                return self._get_quantized_model_path("fp16")
                
        else:
            return "yolov8n-seg.pt"
    
    def _initialize_model(self):
        """
        初始化YOLOv8模型（支持PyTorch和ONNX格式）
        """
        try:
            logger.info(f"Loading YOLOv8 segmentation model: {self.model_path}")
            
            # 检查是否为ONNX模型
            if self.model_path.endswith('.onnx'):
                # 加载ONNX模型
                self.model = self._load_onnx_model()
                self.model_type = "onnx"
                
            else:
                # 加载PyTorch模型
                self.model = YOLO(self.model_path)
                self.model_type = "pytorch"
                
                # 设置设备
                if self.device == "auto":
                    self.device = "cuda" if self.model.device.type == "cuda" else "cpu"
            
            logger.info(f"YOLOv8 model loaded successfully ({self.model_type})")
            logger.info(f"Device: {self.device}")
            logger.info(f"Confidence threshold: {self.confidence_threshold}")

            # 动态读取类别并建立障碍物/手部类别ID集合
            try:
                if self.model_type == "onnx":
                    names_map = self.model['names']
                else:
                    names_map = self.model.names

                # 排除人类（手部）类别，其余一律判为障碍物
                self.hand_class_ids = {i for i, n in names_map.items() if n == 'person'}
                self.obstacle_class_ids = set(names_map.keys()) - self.hand_class_ids

                logger.info(f"Num classes: {len(names_map)}")
                if self.hand_class_ids:
                    logger.info(f"Hand-related class IDs: {sorted(list(self.hand_class_ids))}")
                logger.info(f"Obstacle class ID count: {len(self.obstacle_class_ids)}")
            except Exception as _:
                # 回退策略：未知类别名时，将全部视为障碍物
                self.hand_class_ids = set()
                # 若无法获取类别总数，则不打印细节
                logger.warning("Class names unavailable, treating all classes as obstacles by default")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"YOLOv8 model initialization failed: {e}")
            self.is_initialized = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if not self.is_initialized:
            return {"error": "模型未初始化"}
        
        info = {
            "model_path": self.model_path,
            "quantization_type": self.quantization_type,
            "use_quantized": self.use_quantized,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "model_size": self._get_model_size(),
            "average_inference_time": np.mean(self.inference_times) if self.inference_times else 0,
            "fps": 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        }
        
        return info
    
    def _get_model_size(self) -> str:
        """获取模型文件大小"""
        try:
            if os.path.exists(self.model_path):
                size_bytes = os.path.getsize(self.model_path)
                if size_bytes < 1024 * 1024:
                    return f"{size_bytes / 1024:.1f} KB"
                else:
                    return f"{size_bytes / (1024 * 1024):.1f} MB"
            else:
                return "Unknown"
        except:
            return "Unknown"
    
    
    
    def _load_onnx_model(self):
        """
        加载ONNX模型
        """
        try:
            # 设置ONNX Runtime提供者
            providers = ['CPUExecutionProvider']
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            # 创建ONNX Runtime会话
            session = ort.InferenceSession(self.model_path, providers=providers)
            
            # 获取输入输出信息
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()
            
            logger.info(f"ONNX input: {input_info.name}, shape: {input_info.shape}")
            logger.info(f"ONNX outputs: {len(output_info)}")
            
            # 创建模型包装器
            model_wrapper = {
                'session': session,
                'input_name': input_info.name,
                'input_shape': input_info.shape,
                'output_names': [output.name for output in output_info],
                'names': {i: f'class_{i}' for i in range(80)}  # COCO数据集80个类别
            }
            
            return model_wrapper
            
        except Exception as e:
            logger.error(f"ONNX model load failed: {e}")
            raise e
    
    def _run_onnx_inference(self, image: np.ndarray):
        """
        运行ONNX模型推理
        """
        try:
            # 预处理图像
            input_tensor = self._preprocess_image_for_onnx(image)
            
            # 运行推理
            outputs = self.model['session'].run(
                self.model['output_names'], 
                {self.model['input_name']: input_tensor}
            )
            
            # 后处理结果
            results = self._postprocess_onnx_outputs(outputs, image.shape)
            
            return results
            
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return None
    
    def _preprocess_image_for_onnx(self, image: np.ndarray) -> np.ndarray:
        """
        为ONNX模型预处理图像
        """
        # 调整图像大小到模型输入尺寸
        input_size = 640  # YOLOv8默认输入尺寸
        resized = cv.resize(image, (input_size, input_size))
        
        # 转换为RGB
        rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        
        # 归一化到[0,1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # 转换为CHW格式并添加batch维度
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def _postprocess_onnx_outputs(self, outputs, original_shape):
        """
        后处理ONNX模型输出
        """
        # 这里需要根据YOLOv8-seg的ONNX输出格式进行后处理
        # 由于ONNX输出格式复杂，这里先返回一个简单的包装器
        # 实际应用中需要根据具体的ONNX模型输出格式进行解析
        return _ONNXResult(outputs, original_shape)
    
    def detect_obstacles(self, 
                        image: np.ndarray, 
                        hand_landmarks_3d: Optional[List] = None) -> DetectionResult:
        """
        检测图像中的障碍物
        
        Args:
            image: 输入图像 (BGR格式)
            hand_landmarks_3d: 手部关键点3D坐标列表
            
        Returns:
            Dict: 检测结果
        """
        if not self.is_initialized:
            return self._get_empty_result()
        
        start_time = time.time()
        
        try:
            # 根据模型类型进行推理
            if self.model_type == "onnx":
                results = self._run_onnx_inference(image)
            else:
                # PyTorch模型推理
                results = self.model(image, 
                                   conf=self.confidence_threshold,
                                   device=self.device,
                                   verbose=False)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # 保持最近100次的推理时间
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            # 处理检测结果
            detection_result = self._process_detection_results(results[0], image.shape)
            
            # 生成障碍物掩膜
            obstacle_mask = self._generate_obstacle_mask(
                detection_result, image.shape, hand_landmarks_3d
            )
            
            detection_result.update({
                'obstacle_mask': obstacle_mask,
                'inference_time': inference_time,
                'fps': 1.0 / inference_time if inference_time > 0 else 0
            })
            
            return detection_result
            
        except Exception as e:
            logger.error(f"YOLOv8 inference failed: {e}")
            return self._get_empty_result()
    
    def _process_detection_results(self, result, image_shape: Tuple[int, int, int]) -> DetectionResult:
        """
        处理YOLOv8检测结果
        
        Args:
            result: YOLOv8检测结果
            image_shape: 图像形状 (height, width, channels)
            
        Returns:
            Dict: 处理后的检测结果
        """
        height, width = image_shape[:2]
        
        detection_result = {
            'obstacles': [],
            'hand_regions': [],
            'obstacle_count': 0,
            'hand_region_count': 0,
            'total_detections': 0
        }
        
        if result.masks is None:
            return detection_result
        
        # 处理每个检测结果
        for i, (box, mask, conf, cls) in enumerate(zip(
            result.boxes.xyxy.cpu().numpy(),
            result.masks.data.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy()
        )):
            class_id = int(cls)
            class_name = result.names[class_id]
            confidence = float(conf)
            
            # 调整掩膜大小到原图尺寸
            mask_resized = cv.resize(mask, (width, height))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            detection_info = {
                'id': i,
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': box.tolist(),
                'mask': mask_binary,
                'area': np.sum(mask_binary > 0)
            }
            
            # 分类为障碍物或手部区域（基于类别ID集合）
            if class_id in self.obstacle_class_ids:
                detection_result['obstacles'].append(detection_info)
                detection_result['obstacle_count'] += 1
            elif class_id in self.hand_class_ids:
                detection_result['hand_regions'].append(detection_info)
                detection_result['hand_region_count'] += 1
            
            detection_result['total_detections'] += 1
        
        return detection_result
    
    def _generate_obstacle_mask(self, 
                               detection_result: DetectionResult,
                               image_shape: Tuple[int, int, int],
                               hand_landmarks_3d: Optional[List] = None) -> np.ndarray:
        """
        生成障碍物掩膜（改进版本，确保与深度图对齐）
        
        Args:
            detection_result: 检测结果
            image_shape: 图像形状
            hand_landmarks_3d: 手部关键点3D坐标
            
        Returns:
            np.ndarray: 障碍物掩膜
        """
        height, width = image_shape[:2]
        obstacle_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 合并所有障碍物掩膜
        for obstacle in detection_result['obstacles']:
            mask = obstacle['mask']
            
            # 确保掩膜尺寸与图像尺寸一致
            if mask.shape != (height, width):
                mask = cv.resize(mask, (width, height))
            
            # 二值化掩膜
            mask_binary = (mask > 128).astype(np.uint8) * 255
            obstacle_mask = cv.bitwise_or(obstacle_mask, mask_binary)
        
        # 排除手部区域
        obstacle_mask = self._exclude_hand_regions(
            obstacle_mask, detection_result['hand_regions'], hand_landmarks_3d
        )
        
        # 后处理：噪声过滤和形态学操作
        obstacle_mask = self._post_process_mask(obstacle_mask)
        
        return obstacle_mask
    
    def _exclude_hand_regions(self, 
                             obstacle_mask: np.ndarray,
                             hand_regions: List[Dict],
                             hand_landmarks_3d: Optional[List] = None) -> np.ndarray:
        """
        从障碍物掩膜中排除手部区域（调试版本）
        
        Args:
            obstacle_mask: 原始障碍物掩膜
            hand_regions: 手部区域检测结果
            hand_landmarks_3d: 手部关键点3D坐标
            
        Returns:
            np.ndarray: 排除手部区域后的掩膜
        """
        result_mask = obstacle_mask.copy()
        original_pixels = np.sum(obstacle_mask > 0)
        
        # 排除YOLOv8检测到的手部区域
        for hand_region in hand_regions:
            hand_mask = hand_region['mask']
            # 膨胀手部区域以确保完全排除
            kernel = np.ones((15, 15), np.uint8)
            hand_mask_dilated = cv.dilate(hand_mask, kernel, iterations=1)
            result_mask = cv.bitwise_and(result_mask, cv.bitwise_not(hand_mask_dilated))
        
        # 排除MediaPipe检测到的手部关键点区域（减少膨胀，避免过度排除）
        if hand_landmarks_3d:
            hand_landmark_mask = self._create_hand_landmark_mask(
                hand_landmarks_3d, obstacle_mask.shape
            )
            result_mask = cv.bitwise_and(result_mask, cv.bitwise_not(hand_landmark_mask))
        
        final_pixels = np.sum(result_mask > 0)
        excluded_pixels = original_pixels - final_pixels
        
        # 调试信息
        if excluded_pixels > 0:
            logger.debug(f"Hand-region exclusion: {excluded_pixels} pixels removed")
        
        return result_mask
    
    def _create_hand_landmark_mask(self, 
                                  hand_landmarks_3d: List,
                                  mask_shape: Tuple[int, int]) -> np.ndarray:
        """
        基于手部关键点创建手部区域掩膜（减少膨胀，避免过度排除）
        
        Args:
            hand_landmarks_3d: 手部关键点3D坐标列表
            mask_shape: 掩膜形状
            
        Returns:
            np.ndarray: 手部区域掩膜
        """
        height, width = mask_shape
        hand_mask = np.zeros((height, width), dtype=np.uint8)
        
        for landmark in hand_landmarks_3d:
            if len(landmark) >= 2 and landmark[2] > 0:  # 有效深度
                x, y = int(landmark[0]), int(landmark[1])
                if 0 <= x < width and 0 <= y < height:
                    # 为每个关键点创建较小的膨胀区域
                    cv.circle(hand_mask, (x, y), 10, 255, -1)  # 减少半径从20到10
        
        # 减少膨胀以确保不会过度排除障碍物
        kernel = np.ones((10, 10), np.uint8)  # 减少核大小从25到10
        hand_mask = cv.dilate(hand_mask, kernel, iterations=1)
        
        return hand_mask
    
    def _post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        掩膜后处理：噪声过滤和形态学操作
        
        Args:
            mask: 原始掩膜
            
        Returns:
            np.ndarray: 处理后的掩膜
        """
        # 移除小的噪声区域
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_small)
        
        # 填充小的空洞
        kernel_medium = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_medium)
        
        return mask
    
    def _get_empty_result(self) -> DetectionResult:
        """
        获取空的检测结果
        
        Returns:
            Dict: 空结果
        """
        return {
            'obstacles': [],
            'hand_regions': [],
            'obstacle_count': 0,
            'hand_region_count': 0,
            'total_detections': 0,
            'obstacle_mask': np.zeros((480, 640), dtype=np.uint8),
            'inference_time': 0.0,
            'fps': 0.0
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        获取性能统计信息
        
        Returns:
            Dict: 性能统计
        """
        if not self.inference_times:
            return {'avg_inference_time': 0.0, 'avg_fps': 0.0, 'max_inference_time': 0.0}
        
        avg_time = np.mean(self.inference_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0
        max_time = np.max(self.inference_times)
        
        return {
            'avg_inference_time': avg_time,
            'avg_fps': avg_fps,
            'max_inference_time': max_time,
            'total_inferences': len(self.inference_times)
        }
    
    def cleanup(self):
        """
        清理资源
        """
        if self.model is not None:
            del self.model
            self.model = None
        self.is_initialized = False
        logger.info("YOLOv8 obstacle detector cleaned up")

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

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not available, YOLOv8-seg功能不可用")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ onnxruntime not available, ONNX模型功能不可用")

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
        
        # 障碍物类别（使用YOLOv8模型自带的类别名称）
        # 这些是COCO数据集的80个类别，YOLOv8会自动识别
        self.obstacle_class_names = {
            # 家具类
            'chair', 'couch', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush',
            
            # 交通工具类
            'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'bicycle',
            
            # 其他物体
            'bottle', 'wine glass', 'cup', 'fork', 'knife',
            'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake',
            
            # 运动用品
            'sports ball', 'tennis racket',
        }
        
        # 手部相关类别（需要排除）
        self.hand_related_class_names = {
            'person',  # 人体，包含手部
        }
        
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
                print(f"📦 使用导出的ONNX模型: {onnx_fallback}")
                return onnx_fallback
            elif os.path.exists(openvino_path):
                return openvino_path
            else:
                print(f"⚠️  FP16量化模型不存在，使用原始模型")
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
                print(f"⚠️  INT8量化模型不存在，使用FP16模型")
                return self._get_quantized_model_path("fp16")
                
        else:
            return "yolov8n-seg.pt"
    
    def _initialize_model(self):
        """初始化YOLOv8模型"""
        try:
            if not YOLO_AVAILABLE:
                print("❌ ultralytics不可用，无法初始化YOLOv8模型")
                return
            
            print(f"🔄 加载YOLOv8模型: {self.model_path}")
            print(f"   量化类型: {self.quantization_type}")
            print(f"   设备: {self.device}")
            
            # 加载模型（修复PyTorch 2.6的weights_only问题）
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                if "weights_only" in str(e):
                    # 对于.pt文件，使用weights_only=False
                    import torch
                    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.SegmentationModel'])
                    self.model = YOLO(self.model_path)
                else:
                    raise e
            
            # 设置设备
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.is_initialized = True
            print(f"✅ YOLOv8模型加载成功")
            
        except Exception as e:
            print(f"❌ YOLOv8模型加载失败: {e}")
            self.is_initialized = False
            print("❌ YOLOv8不可用，请安装ultralytics: pip install ultralytics")
    
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
    
    def benchmark_performance(self, test_image: np.ndarray, num_runs: int = 20) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            test_image: 测试图像
            num_runs: 测试次数
            
        Returns:
            性能测试结果
        """
        if not self.is_initialized:
            return {"error": "模型未初始化"}
        
        print(f"🔄 开始性能基准测试 ({num_runs}次运行)...")
        
        # 预热
        for _ in range(5):
            _ = self.model(test_image, verbose=False)
        
        # 性能测试
        times = []
        for i in range(num_runs):
            start_time = time.time()
            _ = self.model(test_image, verbose=False)
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            if (i + 1) % 5 == 0:
                print(f"   完成 {i + 1}/{num_runs} 次测试")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        
        results = {
            "average_time": avg_time,
            "std_time": std_time,
            "fps": fps,
            "min_time": np.min(times),
            "max_time": np.max(times)
        }
        
        print(f"✅ 性能测试完成:")
        print(f"   平均推理时间: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"   FPS: {fps:.2f}")
        print(f"   最小/最大时间: {np.min(times):.4f}s / {np.max(times):.4f}s")
        
        return results
    
    def _initialize_model(self):
        """
        初始化YOLOv8模型（支持PyTorch和ONNX格式）
        """
        try:
            print(f"🔄 加载YOLOv8分割模型: {self.model_path}")
            
            # 检查是否为ONNX模型
            if self.model_path.endswith('.onnx'):
                if not ONNX_AVAILABLE:
                    print("❌ ONNX Runtime不可用，无法加载ONNX模型")
                    self.is_initialized = False
                    return
                
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
            
            print(f"✅ YOLOv8模型加载成功 ({self.model_type})")
            print(f"   设备: {self.device}")
            print(f"   置信度阈值: {self.confidence_threshold}")
            print(f"   障碍物类别数: {len(self.obstacle_class_names)}")
            
            if self.model_type == "pytorch":
                print(f"   模型自带类别: {len(self.model.names)} 个")
                print(f"   模型支持的类别: {list(self.model.names.values())}")
            
            self.is_initialized = True
            
        except Exception as e:
            print(f"❌ YOLOv8模型初始化失败: {e}")
            self.is_initialized = False
    
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
            
            print(f"   ONNX模型输入: {input_info.name}, 形状: {input_info.shape}")
            print(f"   ONNX模型输出数量: {len(output_info)}")
            
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
            print(f"❌ ONNX模型加载失败: {e}")
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
            print(f"❌ ONNX推理失败: {e}")
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
        
        class ONNXResult:
            def __init__(self, outputs, original_shape):
                self.outputs = outputs
                self.original_shape = original_shape
                self.masks = None  # 分割掩膜
                self.boxes = None  # 边界框
                self.names = {i: f'class_{i}' for i in range(80)}
            
            def __iter__(self):
                return iter([self])
        
        return ONNXResult(outputs, original_shape)
    
    def detect_obstacles(self, 
                        image: np.ndarray, 
                        hand_landmarks_3d: Optional[List] = None) -> Dict[str, Any]:
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
            print(f"❌ YOLOv8推理失败: {e}")
            return self._get_empty_result()
    
    def _process_detection_results(self, result, image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
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
            
            # 分类为障碍物或手部区域
            if class_name in self.obstacle_class_names:
                detection_result['obstacles'].append(detection_info)
                detection_result['obstacle_count'] += 1
            elif class_name in self.hand_related_class_names:
                detection_result['hand_regions'].append(detection_info)
                detection_result['hand_region_count'] += 1
            
            detection_result['total_detections'] += 1
        
        return detection_result
    
    def _generate_obstacle_mask(self, 
                               detection_result: Dict[str, Any],
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
            print(f"🔍 手部区域排除: {excluded_pixels} 像素被排除")
        
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
    
    def _get_empty_result(self) -> Dict[str, Any]:
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
    
    def visualize_detection(self, 
                           image: np.ndarray, 
                           detection_result: Dict[str, Any]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detection_result: 检测结果
            
        Returns:
            np.ndarray: 可视化图像
        """
        vis_image = image.copy()
        
        # 绘制障碍物
        for obstacle in detection_result['obstacles']:
            bbox = obstacle['bbox']
            class_name = obstacle['class_name']
            confidence = obstacle['confidence']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            cv.putText(vis_image, label, (x1, y1 - 10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制手部区域
        for hand_region in detection_result['hand_regions']:
            bbox = hand_region['bbox']
            class_name = hand_region['class_name']
            confidence = hand_region['confidence']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            cv.putText(vis_image, label, (x1, y1 - 10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 添加统计信息
        stats_text = [
            f"Obstacles: {detection_result['obstacle_count']}",
            f"Hand Regions: {detection_result['hand_region_count']}",
            f"FPS: {detection_result['fps']:.1f}"
        ]
        
        for i, text in enumerate(stats_text):
            cv.putText(vis_image, text, (10, 30 + i * 25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def cleanup(self):
        """
        清理资源
        """
        if self.model is not None:
            del self.model
            self.model = None
        self.is_initialized = False
        print("✅ YOLOv8障碍物检测器已清理")


def main():
    """
    测试YOLOv8障碍物检测器
    """
    print("🚀 测试YOLOv8障碍物检测器...")
    
    # 创建检测器
    detector = YOLOObstacleDetector(
        model_path="yolov8n-seg.pt",
        confidence_threshold=0.5
    )
    
    if not detector.is_initialized:
        print("❌ 检测器初始化失败")
        return
    
    # 测试图像（使用摄像头或测试图像）
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    try:
        frame_count = 0
        while frame_count < 100:  # 测试100帧
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测障碍物
            detection_result = detector.detect_obstacles(frame)
            
            # 可视化结果
            vis_frame = detector.visualize_detection(frame, detection_result)
            
            # 显示障碍物掩膜
            obstacle_mask = detection_result['obstacle_mask']
            mask_colored = cv.applyColorMap(obstacle_mask, cv.COLORMAP_JET)
            
            # 显示结果
            cv.imshow('YOLOv8 Obstacle Detection', vis_frame)
            cv.imshow('Obstacle Mask', mask_colored)
            
            frame_count += 1
            if frame_count % 10 == 0:
                stats = detector.get_performance_stats()
                print(f"帧 {frame_count}: {stats['avg_fps']:.1f} FPS")
            
            # 按'q'退出
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    
    finally:
        cap.release()
        cv.destroyAllWindows()
        detector.cleanup()
        print("✅ 测试完成")


if __name__ == "__main__":
    main()

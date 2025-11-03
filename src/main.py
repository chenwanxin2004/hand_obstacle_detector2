import cv2 as cv
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
import math
import time
from collections import deque

# Import camera modules
from realsense_camera import create_camera

# Import YOLOv8 obstacle detector
try:
    from yolo_obstacle_detector import YOLOObstacleDetector
    YOLO_OBSTACLE_AVAILABLE = True
except ImportError:
    YOLO_OBSTACLE_AVAILABLE = False
    print("⚠️ YOLOv8障碍物检测器不可用")

class HandObstacleContactDetector:
    """
    手部触碰障碍物检测器
    功能：
    1. 实时检测手部关键点
    2. 结合深度图计算手部与障碍物距离
    3. 判断手部是否触碰障碍物
    4. 提供触碰反馈和可视化
    """
    
    def __init__(self, 
                 contact_threshold=0.03,  # 触碰阈值（米）- 3cm，更合理的阈值
                 warning_threshold=0.06,  # 警告阈值（米）- 6cm，更合理的阈值
                 max_num_hands=2,
                 min_detection_confidence=0.3,  # 降低检测置信度阈值
                 min_tracking_confidence=0.3,  # 降低跟踪置信度阈值
                 use_yolo_obstacle=True,  # 是否使用YOLOv8障碍物检测
                 yolo_model_path="yolov8n-seg.pt"):  # YOLOv8模型路径
        """
        初始化手部触碰检测器
        
        Args:
            contact_threshold: 触碰检测阈值（米）
            warning_threshold: 接近警告阈值（米）
            max_num_hands: 最大检测手数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 触碰检测参数
        self.contact_threshold = contact_threshold
        self.warning_threshold = warning_threshold
        
        # YOLOv8障碍物检测器
        self.use_yolo_obstacle = use_yolo_obstacle and YOLO_OBSTACLE_AVAILABLE
        self.yolo_obstacle_detector = None
        
        if self.use_yolo_obstacle:
            try:
                self.yolo_obstacle_detector = YOLOObstacleDetector(
                    model_path=yolo_model_path,
                    confidence_threshold=0.5,
                    use_quantized=True,  # 使用量化模型
                    quantization_type="fp16"  # FP16量化
                )
                print("✅ YOLOv8障碍物检测器初始化成功")
            except Exception as e:
                print(f"❌ YOLOv8障碍物检测器初始化失败: {e}")
                self.use_yolo_obstacle = False
        else:
            print("🔄 使用基础障碍物检测算法")
        
        # 手部关键点索引（用于触碰检测的关键点）
        self.contact_points = [
            'THUMB_TIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_TIP', 'PINKY_TIP'
        ]
        
        self.LANDMARK_INDICES = {
            'WRIST': 0,
            'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
            'INDEX_FINGER_MCP': 5, 'INDEX_FINGER_PIP': 6, 'INDEX_FINGER_DIP': 7, 'INDEX_FINGER_TIP': 8,
            'MIDDLE_FINGER_MCP': 9, 'MIDDLE_FINGER_PIP': 10, 'MIDDLE_FINGER_DIP': 11, 'MIDDLE_FINGER_TIP': 12,
            'RING_FINGER_MCP': 13, 'RING_FINGER_PIP': 14, 'RING_FINGER_DIP': 15, 'RING_FINGER_TIP': 16,
            'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20
        }
        
        # 状态历史（用于平滑检测）
        self.contact_history = deque(maxlen=5)
        self.warning_history = deque(maxlen=5)
        
        # 统计信息
        self.total_contacts = 0
        self.total_warnings = 0
        
        # 性能优化：缓存机制
        self._obstacle_mask_cache = None
        self._last_hand_depths = None
        self._cache_valid = False
        
        # 自适应检测参数
        self._adaptive_mode = True  # 启用自适应模式
        self._recent_distances = deque(maxlen=30)  # 记录最近30帧的距离
        self._environment_analyzed = False
        
    def detect_hand_contact(self, color_image: np.ndarray, depth_image: np.ndarray) -> Dict:
        """
        检测手部触碰障碍物（改进版本，解决时间戳对齐问题）
        
        Args:
            color_image: RGB彩色图像
            depth_image: 深度图像（米为单位）
            
        Returns:
            检测结果字典
        """
        # 记录开始时间，确保时间戳对齐
        start_time = time.time()
        
        results = self.hands.process(cv.cvtColor(color_image, cv.COLOR_BGR2RGB))
        
        detection_result = {
            'hands_detected': False,
            'contact_detected': False,
            'warning_detected': False,
            'hands_info': [],
            'contact_points': [],
            'warning_points': [],
            'min_distance': float('inf'),
            'contact_count': 0,
            'warning_count': 0,
            'processing_time': 0.0
        }
        
        if results.multi_hand_landmarks:
            detection_result['hands_detected'] = True
            
            # 获取所有手部的3D坐标（用于生成障碍物掩膜）
            all_hand_landmarks_3d = []
            height, width = depth_image.shape
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    
                    if 0 <= x < width and 0 <= y < height:
                        depth = depth_image[y, x]
                        all_hand_landmarks_3d.append([x, y, depth if depth > 0 else 0])
                    else:
                        all_hand_landmarks_3d.append([x, y, 0])
            
            # 生成障碍物掩膜（使用YOLOv8或基础算法）
            if self.use_yolo_obstacle and self.yolo_obstacle_detector is not None:
                try:
                    # 使用YOLOv8检测障碍物
                    yolo_result = self.yolo_obstacle_detector.detect_obstacles(color_image, all_hand_landmarks_3d)
                    obstacle_mask = yolo_result.get('obstacle_mask', np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8))
                    
                    # 检查YOLOv8是否检测到足够的障碍物
                    obstacle_count = yolo_result.get('obstacle_count', 0)
                    mask_pixels = np.sum(obstacle_mask > 0)
                    
                    # 如果YOLOv8检测到的障碍物太少，使用深度备用检测
                    if obstacle_count < 2 or mask_pixels < 1000:
                        print(f"🔧 YOLOv8检测不足，启用深度备用检测")
                        if depth_image is not None:
                            depth_backup_mask = self._generate_depth_backup_mask(color_image, all_hand_landmarks_3d, depth_image)
                            obstacle_mask = cv.bitwise_or(obstacle_mask, depth_backup_mask)
                    
                    # 存储YOLOv8检测结果用于可视化
                    self._last_yolo_result = yolo_result
                except Exception as e:
                    print(f"❌ YOLOv8障碍物检测失败: {e}")
                    obstacle_mask = self._generate_depth_backup_mask(color_image, all_hand_landmarks_3d, depth_image)
            else:
                # 未启用YOLO时，使用深度备用掩膜作为兜底
                obstacle_mask = self._generate_depth_backup_mask(color_image, all_hand_landmarks_3d, depth_image)
            detection_result['obstacle_mask'] = obstacle_mask
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 获取手部标签
                hand_label = results.multi_handedness[idx].classification[0].label
                hand_confidence = results.multi_handedness[idx].classification[0].score
                
                # 检测手部触碰（使用障碍物掩膜）
                hand_contact_info = self._detect_single_hand_contact_with_mask(
                    hand_landmarks, depth_image, color_image.shape, obstacle_mask
                )
                
                hand_info = {
                    'hand_id': idx,
                    'label': hand_label,
                    'confidence': hand_confidence,
                    'landmarks': hand_landmarks,  # 添加landmarks信息用于绘制
                    'contact_detected': hand_contact_info['contact_detected'],
                    'warning_detected': hand_contact_info['warning_detected'],
                    'contact_points': hand_contact_info['contact_points'],
                    'warning_points': hand_contact_info['warning_points'],
                    'min_distance': hand_contact_info['min_distance']
                }
                
                detection_result['hands_info'].append(hand_info)
                
                # 更新总体检测结果
                if hand_contact_info['contact_detected']:
                    detection_result['contact_detected'] = True
                    detection_result['contact_count'] += len(hand_contact_info['contact_points'])
                    detection_result['contact_points'].extend(hand_contact_info['contact_points'])
                
                if hand_contact_info['warning_detected']:
                    detection_result['warning_detected'] = True
                    detection_result['warning_count'] += len(hand_contact_info['warning_points'])
                    detection_result['warning_points'].extend(hand_contact_info['warning_points'])
                
                # 更新最小距离
                if hand_contact_info['min_distance'] < detection_result['min_distance']:
                    detection_result['min_distance'] = hand_contact_info['min_distance']
        
        # 更新历史状态
        self._update_detection_history(detection_result)
        
        # 自适应阈值调整
        if self._adaptive_mode:
            self._update_adaptive_thresholds(detection_result)
        
        # 记录处理时间
        detection_result['processing_time'] = time.time() - start_time
        
        return detection_result

    def _detect_single_hand_contact_with_mask(self, hand_landmarks, depth_image: np.ndarray, image_shape: Tuple[int, int, int], obstacle_mask: np.ndarray) -> Dict:
        """
        使用障碍物掩膜检测单只手的触碰情况
        """
        height, width = image_shape[:2]
        
        contact_info = {
            'contact_detected': False,
            'warning_detected': False,
            'contact_points': [],
            'warning_points': [],
            'min_distance': float('inf'),
            'debug_info': []  # 添加调试信息
        }
        
        # 检测每个关键点
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            
            if 0 <= x < width and 0 <= y < height:
                hand_depth = depth_image[y, x]
                
                if hand_depth > 0:
                    # 首先检查手部是否悬空
                    is_suspended = self._is_hand_suspended(x, y, hand_depth, depth_image)
                    
                    if is_suspended:
                        # 手部悬空，跳过此关键点
                        debug_info = {
                            'landmark_id': idx,
                            'pixel_coords': (x, y),
                            'hand_depth': hand_depth,
                            'distance': float('inf'),
                            'contact_threshold': self.contact_threshold,
                            'warning_threshold': self.warning_threshold,
                            'suspended': True,
                            'suspension_type': 'suspended'
                        }
                        contact_info['debug_info'].append(debug_info)
                        continue
                    
                    # 使用障碍物掩膜计算距离
                    distance = self._calculate_obstacle_distance(x, y, hand_depth, depth_image, obstacle_mask)
                    
                    # 记录调试信息
                    debug_info = {
                        'landmark_id': idx,
                        'pixel_coords': (x, y),
                        'hand_depth': hand_depth,
                        'distance': distance,
                        'contact_threshold': self.contact_threshold,
                        'warning_threshold': self.warning_threshold,
                        'suspended': False
                    }
                    contact_info['debug_info'].append(debug_info)
                    
                    if distance != float('inf'):
                        # 额外验证：确保距离计算是可靠的
                        if self._validate_distance_measurement(x, y, hand_depth, distance, depth_image):
                            if distance < self.contact_threshold:
                                # 触碰检测
                                contact_info['contact_detected'] = True
                                contact_info['contact_points'].append({
                                    'landmark_id': idx,
                                    'pixel_coords': (x, y),
                                    'depth': hand_depth,
                                    'distance': distance
                                })
                            elif distance < self.warning_threshold:
                                # 警告检测
                                contact_info['warning_detected'] = True
                                contact_info['warning_points'].append({
                                    'landmark_id': idx,
                                    'pixel_coords': (x, y),
                                    'depth': hand_depth,
                                    'distance': distance
                                })
                            
                            # 更新最小距离
                            if distance < contact_info['min_distance']:
                                contact_info['min_distance'] = distance
        
        return contact_info
    
    def _create_hand_region_mask(self, depth_image: np.ndarray, hand_landmarks_3d: list) -> np.ndarray:
        """
        创建手部区域掩膜（精确版本，避免过度膨胀）
        """
        height, width = depth_image.shape
        hand_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 为每个手部关键点创建精确区域
        for landmark in hand_landmarks_3d:
            if landmark[2] > 0:  # 有效深度
                x, y = int(landmark[0]), int(landmark[1])
                if 0 <= x < width and 0 <= y < height:
                    # 创建适中的圆形区域（减少膨胀）
                    cv.circle(hand_mask, (x, y), 12, 255, -1)  # 12像素半径（减少）
        
        # 适度膨胀以确保手部区域完全覆盖
        kernel = np.ones((8,8), np.uint8)  # 减小膨胀核
        hand_mask = cv.dilate(hand_mask, kernel, iterations=1)
        
        # 添加手部轮廓连接（基于关键点）
        if len(hand_landmarks_3d) >= 5:
            # 创建手部轮廓
            points = []
            for landmark in hand_landmarks_3d:
                if landmark[2] > 0:
                    x, y = int(landmark[0]), int(landmark[1])
                    if 0 <= x < width and 0 <= y < height:
                        points.append((x, y))
            
            if len(points) >= 3:
                # 创建凸包
                hull = cv.convexHull(np.array(points))
                cv.fillPoly(hand_mask, [hull], 255)
        
        return hand_mask
    
    def _generate_depth_backup_mask(self, color_image: np.ndarray, hand_landmarks_3d: list, depth_image: np.ndarray) -> np.ndarray:
        """
        基于深度的备用障碍物检测（当YOLOv8检测不足时使用）
        
        Args:
            color_image: 彩色图像
            hand_landmarks_3d: 手部关键点3D坐标列表
            depth_image: 深度图像
            
        Returns:
            np.ndarray: 深度障碍物掩膜
        """
        height, width = color_image.shape[:2]
        depth_mask = np.zeros((height, width), dtype=np.uint8)
        
        if depth_image is None:
            return depth_mask
        
        # 获取手部深度范围
        hand_depths = [landmark[2] for landmark in hand_landmarks_3d if landmark[2] > 0]
        if not hand_depths:
            return depth_mask
        
        min_hand_depth = min(hand_depths)
        max_hand_depth = max(hand_depths)
        
        # 基于深度差异检测障碍物
        # 检测比手部更近或更远的物体
        depth_threshold = 0.1  # 10cm深度差异阈值
        
        # 创建深度差异掩膜
        depth_diff_mask = (
            (depth_image < min_hand_depth - depth_threshold) | 
            (depth_image > max_hand_depth + depth_threshold)
        ) & (depth_image > 0.1)  # 排除无效深度
        
        # 转换为uint8掩膜
        depth_mask = (depth_diff_mask * 255).astype(np.uint8)
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        depth_mask = cv.morphologyEx(depth_mask, cv.MORPH_OPEN, kernel)
        depth_mask = cv.morphologyEx(depth_mask, cv.MORPH_CLOSE, kernel)
        
        return depth_mask
    
    def _calculate_obstacle_distance(self, x: int, y: int, hand_depth: float, depth_image: np.ndarray, obstacle_mask: np.ndarray) -> float:
        """
        计算手部关键点与障碍物的距离（融合检测，改进版本）
        逻辑：深度图基础检测 + YOLOv8增强效果 + 手部上下文感知
        """
        height, width = depth_image.shape
        
        # 方法1：基于深度图的直接检测（基础方法）
        depth_based_distance = self._calculate_depth_based_distance(x, y, hand_depth, depth_image)
        
        # 方法2：基于YOLOv8掩膜的精确检测（增强方法）
        yolo_enhanced_distance = self._calculate_yolo_enhanced_distance(x, y, hand_depth, depth_image, obstacle_mask)
        
        # 融合策略：智能选择最佳结果
        if yolo_enhanced_distance != float('inf'):
            # YOLOv8检测到障碍物，使用YOLOv8结果（更精确）
            return yolo_enhanced_distance
        elif depth_based_distance != float('inf'):
            # YOLOv8未检测到，但深度图检测到，需要进一步验证
            # 检查是否是手部悬空状态（避免误检背景）
            if self._is_hand_suspended(x, y, hand_depth, depth_image):
                return float('inf')  # 手部悬空，不报告触碰
            else:
                return depth_based_distance  # 手部接近物体，报告触碰
        else:
            # 两种方法都未检测到障碍物
            return float('inf')
    
    def _is_hand_suspended(self, x: int, y: int, hand_depth: float, depth_image: np.ndarray) -> bool:
        """
        判断手部是否悬空（精简版本，保留必要检测）
        """
        height, width = depth_image.shape
        check_radius = 30
        
        # 计算检查区域边界
        x_min = max(0, x - check_radius)
        x_max = min(width, x + check_radius + 1)
        y_min = max(0, y - check_radius)
        y_max = min(height, y + check_radius + 1)
        
        # 提取检查区域的深度
        check_depth = depth_image[y_min:y_max, x_min:x_max]
        valid_depths = check_depth[check_depth > 0.1]
        
        # 1. 没有有效深度，认为是悬空
        if len(valid_depths) == 0:
            return True
        
        # 2. 先检查是否有支撑或连接（优先判断非悬空情况）
        # 2.1 检查手部下方是否有支撑物
        below_y = min(height, y + 40)
        below_x_min = max(0, x - 20)
        below_x_max = min(width, x + 20)
        
        if below_y < height:
            below_region = depth_image[y:below_y, below_x_min:below_x_max]
            below_depths = below_region[below_region > 0.1]
            if len(below_depths) > 0:
                support_depths = below_depths[below_depths < hand_depth - 0.05]
                if len(support_depths) > len(below_depths) * 0.3:
                    return False  # 有支撑，不是悬空
        
        # 2.2 检查手部周围是否有相似深度的物体（可能的连接）
        connection_radius = 25
        conn_x_min = max(0, x - connection_radius)
        conn_x_max = min(width, x + connection_radius + 1)
        conn_y_min = max(0, y - connection_radius)
        conn_y_max = min(height, y + connection_radius + 1)
        
        connection_region = depth_image[conn_y_min:conn_y_max, conn_x_min:conn_x_max]
        connection_depths = connection_region[connection_region > 0.1]
        
        if len(connection_depths) > 0:
            similar_depths = connection_depths[np.abs(connection_depths - hand_depth) < 0.08]
            if len(similar_depths) > len(connection_depths) * 0.4:
                return False  # 有连接，不是悬空
        
        # 3. 判断悬空：如果大部分周围点都比手部深很多，说明手部悬空在前方
        deeper_points = valid_depths[valid_depths > hand_depth + 0.20]
        if len(deeper_points) > len(valid_depths) * 0.7:
            return True
        
        # 4. 检查深度信息可靠性：如果深度变化太大，可能是不可靠的深度信息
        depth_std = np.std(valid_depths)
        if depth_std > 0.3:
            return True
        
        return False
    
    def _validate_distance_measurement(self, x: int, y: int, hand_depth: float, distance: float, depth_image: np.ndarray) -> bool:
        """
        验证距离测量的可靠性（避免误检）
        """
        # 1. 距离必须在合理范围内
        if distance < 0.005 or distance > 0.1:  # 5mm到10cm之间
            return False
        
        # 2. 检查手部深度是否合理
        if hand_depth < 0.1 or hand_depth > 2.0:  # 10cm到2m之间
            return False
        
        # 3. 检查周围环境的一致性
        height, width = depth_image.shape
        check_radius = 20
        
        x_min = max(0, x - check_radius)
        x_max = min(width, x + check_radius + 1)
        y_min = max(0, y - check_radius)
        y_max = min(height, y + check_radius + 1)
        
        check_depth = depth_image[y_min:y_max, x_min:x_max]
        valid_depths = check_depth[check_depth > 0.1]
        
        if len(valid_depths) < 5:  # 需要足够的上下文信息
            return False
        
        # 4. 检查深度变化是否合理
        depth_std = np.std(valid_depths)
        if depth_std > 0.3:  # 深度变化不能太大
            return False
        
        # 5. 检查手部是否在合理的深度范围内
        depth_mean = np.mean(valid_depths)
        if abs(hand_depth - depth_mean) > 0.2:  # 手部深度与周围环境差异不能太大
            return False
        
        return True
    
    def _calculate_depth_based_distance(self, x: int, y: int, hand_depth: float, depth_image: np.ndarray) -> float:
        """
        基于深度图的直接距离计算（基础方法，改进版本）
        逻辑：智能过滤背景，只检测真正的障碍物
        """
        height, width = depth_image.shape
        search_radius = 15  # 减小搜索半径，避免误检背景
        
        # 计算搜索区域边界
        x_min = max(0, x - search_radius)
        x_max = min(width, x + search_radius + 1)
        y_min = max(0, y - search_radius)
        y_max = min(height, y + search_radius + 1)
        
        # 提取搜索区域的深度
        search_depth = depth_image[y_min:y_max, x_min:x_max]
        
        # 找到有效深度点
        valid_depths = search_depth[search_depth > 0.1]  # 大于10cm的深度
        
        if len(valid_depths) == 0:
            return float('inf')
        
        # 计算与手部深度的差异
        depth_diffs = np.abs(valid_depths - hand_depth)
        
        # 智能过滤：只检测手部前方的物体（严格版本，避免悬空误检）
        # 1. 深度差异在合理范围内（2-6cm，更严格的范围）
        reasonable_diffs = depth_diffs[(depth_diffs >= 0.02) & (depth_diffs <= 0.06)]
        
        if len(reasonable_diffs) == 0:
            return float('inf')
        
        # 2. 检查是否有足够的点确认障碍物（提高要求，避免误检）
        if len(reasonable_diffs) < 3:  # 至少需要3个点确认（提高要求）
            return float('inf')
        
        # 3. 检查深度分布的一致性（严格标准差要求）
        depth_std = np.std(reasonable_diffs)
        if depth_std > 0.03:  # 深度差异标准差严格限制在3cm
            return float('inf')
        
        # 4. 新增：检查是否手部悬空（关键改进）
        if self._is_hand_suspended(x, y, hand_depth, depth_image):
            return float('inf')
        
        # 返回最小距离
        return float(np.min(reasonable_diffs))
    
    def _calculate_yolo_enhanced_distance(self, x: int, y: int, hand_depth: float, depth_image: np.ndarray, obstacle_mask: np.ndarray) -> float:
        """
        基于YOLOv8掩膜的精确距离计算（增强方法）
        """
        height, width = depth_image.shape
        search_radius = 15
        
        # 计算搜索区域边界
        x_min = max(0, x - search_radius)
        x_max = min(width, x + search_radius + 1)
        y_min = max(0, y - search_radius)
        y_max = min(height, y + search_radius + 1)
        
        # 提取搜索区域
        search_mask = obstacle_mask[y_min:y_max, x_min:x_max]
        search_depth = depth_image[y_min:y_max, x_min:x_max]
        
        # 找到障碍物点
        obstacle_points = search_mask > 0
        if not np.any(obstacle_points):
            return float('inf')
        
        # 获取障碍物区域的深度值
        obstacle_depths = search_depth[obstacle_points]
        valid_obstacle_depths = obstacle_depths[obstacle_depths > 0.05]
        
        if len(valid_obstacle_depths) == 0:
            return float('inf')
        
        # 计算距离（优化版本，更合理的距离判断）
        obstacle_min_depth = np.min(valid_obstacle_depths)
        distance = abs(hand_depth - obstacle_min_depth)
        
        # 放宽距离和点数要求，减少误检
        if distance > 0.08 or len(valid_obstacle_depths) < 1:  # 降低要求
            return float('inf')
        
        return float(distance)
    
    def _update_detection_history(self, detection_result: Dict):
        """更新检测历史状态"""
        self.contact_history.append(detection_result['contact_detected'])
        self.warning_history.append(detection_result['warning_detected'])
        
        # 记录距离信息用于自适应调整
        if detection_result['min_distance'] != float('inf'):
            self._recent_distances.append(detection_result['min_distance'])
        
        # 更新统计信息
        if detection_result['contact_detected']:
            self.total_contacts += 1
        if detection_result['warning_detected']:
            self.total_warnings += 1
    
    def _update_adaptive_thresholds(self, detection_result: Dict):
        """自适应调整检测阈值"""
        if len(self._recent_distances) < 10:  # 需要足够的数据
            return
        
        # 分析最近的距离分布
        recent_distances = list(self._recent_distances)
        avg_distance = np.mean(recent_distances)
        std_distance = np.std(recent_distances)
        
        # 如果检测过于敏感（频繁触发警告），适当放宽阈值
        warning_rate = sum(self.warning_history) / len(self.warning_history) if self.warning_history else 0
        contact_rate = sum(self.contact_history) / len(self.contact_history) if self.contact_history else 0
        
        if warning_rate > 0.3:  # 如果警告率超过30%
            # 适当放宽警告阈值
            self.warning_threshold = min(0.10, self.warning_threshold * 1.1)
            print(f"🔧 自适应调整：警告阈值调整为 {self.warning_threshold:.3f}m")
        
        if contact_rate > 0.1:  # 如果触碰率超过10%
            # 适当放宽触碰阈值
            self.contact_threshold = min(0.05, self.contact_threshold * 1.1)
            print(f"🔧 自适应调整：触碰阈值调整为 {self.contact_threshold:.3f}m")
        
        # 如果环境相对稳定，可以适当收紧阈值
        if std_distance < 0.02 and avg_distance > 0.1:  # 环境稳定且距离较远
            if warning_rate < 0.1:  # 警告率很低
                self.warning_threshold = max(0.04, self.warning_threshold * 0.95)
            if contact_rate < 0.05:  # 触碰率很低
                self.contact_threshold = max(0.02, self.contact_threshold * 0.95)
    
    def visualize_detection(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        可视化检测结果
        """
        annotated_image = image.copy()
        
        # 绘制手部关键点和连接线（标准MediaPipe风格）
        if detection_result['hands_detected']:
            for hand_info in detection_result['hands_info']:
                # 使用标准MediaPipe绘制风格
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_info['landmarks'],
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # 绘制触碰点（更明显的红色显示）
        for contact_point in detection_result['contact_points']:
            x, y = contact_point['pixel_coords']
            # 大红色圆点，带白色边框
            cv.circle(annotated_image, (x, y), 12, (0, 0, 255), -1)  # 红色圆点
            cv.circle(annotated_image, (x, y), 12, (255, 255, 255), 2)  # 白色边框
            cv.putText(annotated_image, "HIT!", 
                      (x + 15, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 绘制警告点（更明显的黄色显示）
        for warning_point in detection_result['warning_points']:
            x, y = warning_point['pixel_coords']
            # 大黄色圆点，带黑色边框
            cv.circle(annotated_image, (x, y), 10, (0, 255, 255), -1)  # 黄色圆点
            cv.circle(annotated_image, (x, y), 10, (0, 0, 0), 2)  # 黑色边框
            cv.putText(annotated_image, "NEAR", 
                      (x + 15, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
        
        # 添加清晰的状态信息
        if detection_result['contact_detected']:
            # 碰撞检测 - 大红色背景，白色文字
            cv.rectangle(annotated_image, (5, 5), (300, 50), (0, 0, 255), -1)  # 红色背景
            cv.putText(annotated_image, "!!! COLLISION DETECTED !!!", 
                      (15, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 显示触碰点数量
            contact_count = detection_result['contact_count']
            cv.putText(annotated_image, f"Contact Points: {contact_count}", 
                      (15, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                      
        elif detection_result['warning_detected']:
            # 警告检测 - 大黄色背景，黑色文字
            cv.rectangle(annotated_image, (5, 5), (300, 50), (0, 255, 255), -1)  # 黄色背景
            cv.putText(annotated_image, "WARNING: APPROACHING OBSTACLE", 
                      (15, 35), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 显示警告点数量
            warning_count = detection_result['warning_count']
            cv.putText(annotated_image, f"Warning Points: {warning_count}", 
                      (15, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            # 安全状态 - 绿色背景
            cv.rectangle(annotated_image, (5, 5), (200, 35), (0, 255, 0), -1)  # 绿色背景
            cv.putText(annotated_image, "SAFE", 
                      (15, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 添加障碍物掩膜可视化（右上角显示）
        if 'obstacle_mask' in detection_result and detection_result['obstacle_mask'] is not None:
            obstacle_mask = detection_result['obstacle_mask']
            # 调整掩膜大小以适合显示
            mask_resized = cv.resize(obstacle_mask, (160, 120))
            mask_colored = cv.applyColorMap(mask_resized, cv.COLORMAP_JET)
            
            # 在右上角显示障碍物掩膜
            y_offset = 5
            x_offset = annotated_image.shape[1] - 165
            annotated_image[y_offset:y_offset+120, x_offset:x_offset+160] = mask_colored
            
            # 添加标题
            mask_title = "YOLOv8 Mask" if self.use_yolo_obstacle else "Basic Mask"
            cv.putText(annotated_image, mask_title,
                      (x_offset, y_offset-5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 显示掩膜统计信息
            mask_pixels = np.sum(obstacle_mask > 0)
            cv.putText(annotated_image, f"Mask Pixels: {mask_pixels}",
                      (x_offset, y_offset + 130), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 显示YOLOv8检测结果（简化版）
        if self.use_yolo_obstacle and hasattr(self, '_last_yolo_result'):
            yolo_result = self._last_yolo_result
            if yolo_result and 'obstacles' in yolo_result:
                # 在右上角显示简化的YOLOv8信息
                obstacle_count = yolo_result.get('obstacle_count', 0)
                fps = yolo_result.get('fps', 0)
                
                # 在障碍物掩膜下方显示信息
                info_y = 130
                cv.putText(annotated_image, f"Obstacles: {obstacle_count}", 
                          (annotated_image.shape[1] - 165, info_y), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv.putText(annotated_image, f"FPS: {fps:.1f}", 
                          (annotated_image.shape[1] - 165, info_y + 15), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 简化的统计信息（只显示关键信息）
        y_pos = annotated_image.shape[0] - 30
        
        # 显示最小距离（如果有的话）
        if detection_result['min_distance'] != float('inf'):
            cv.putText(annotated_image, f"Min Distance: {detection_result['min_distance']:.2f}m", 
                      (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos -= 25
        
        # 显示手部检测状态
        if detection_result['hands_detected']:
            hands_count = len(detection_result['hands_info'])
            cv.putText(annotated_image, f"Hands: {hands_count}", 
                      (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv.putText(annotated_image, "No Hands", 
                      (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return annotated_image
    
    def get_contact_feedback(self, detection_result: Dict) -> Dict:
        """
        获取触碰反馈信息
        """
        feedback = {
            'has_contact': detection_result['contact_detected'],
            'has_warning': detection_result['warning_detected'],
            'contact_severity': 'none',
            'recommended_action': 'continue',
            'contact_points_count': detection_result['contact_count'],
            'warning_points_count': detection_result['warning_count']
        }
        
        if detection_result['contact_detected']:
            feedback['contact_severity'] = 'high'
            feedback['recommended_action'] = 'stop_movement'
        elif detection_result['warning_detected']:
            feedback['contact_severity'] = 'medium'
            feedback['recommended_action'] = 'slow_down'
        
        return feedback
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'hands'):
            self.hands.close()
        
        if self.use_yolo_obstacle and self.yolo_obstacle_detector is not None:
            self.yolo_obstacle_detector.cleanup()


def main():
    """
    主函数 - 演示手部触碰检测
    """
    print("🚀 启动手部触碰障碍物检测系统...")
    
    # 初始化相机（仅RealSense）
    camera = create_camera()
   
    # 初始化检测器（使用量化模型）
    detector = HandObstacleContactDetector(
        contact_threshold=0.03,  # 3cm触碰阈值（更合理的阈值）
        warning_threshold=0.06,  # 6cm警告阈值（更合理的阈值）
        use_yolo_obstacle=True,  # 使用YOLOv8障碍物检测
        yolo_model_path="src/yolov8n-seg.onnx"  # 使用量化后的ONNX模型
    )
    
    print("📋 使用说明：")
    print("   - 红色圆点：触碰检测")
    print("   - 黄色圆点：接近警告")
    print("   - 按 'q' 退出程序")
    print("   - 按 's' 保存当前帧")
    print("   - 按 'd' 切换深度图显示")
    
    frame_count = 0
    show_depth = True
    is_realsense = hasattr(camera, 'create_depth_visualization')
    
    # 设置窗口位置
    cv.namedWindow('Hand Obstacle Contact Detection', cv.WINDOW_NORMAL)
    cv.moveWindow('Hand Obstacle Contact Detection', 100, 100)
    if is_realsense:
        cv.namedWindow('Depth Map', cv.WINDOW_NORMAL)
        cv.moveWindow('Depth Map', 800, 100)
    
    try:
        while True:
            # 获取帧
            depth_frame, color_frame = camera.get_frames()
            if depth_frame is None or color_frame is None:
                print("❌ 无法获取相机帧")
                time.sleep(0.1)
                continue
            
            # 检测手部触碰
            detection_result = detector.detect_hand_contact(color_frame, depth_frame)
            
            # 可视化结果
            annotated_frame = detector.visualize_detection(color_frame, detection_result)
            
            # 获取反馈
            feedback = detector.get_contact_feedback(detection_result)
            
            # 显示结果
            cv.imshow('Hand Obstacle Contact Detection', annotated_frame)
            
            # 显示深度图（如果使用RealSense）
            if is_realsense and show_depth:
                depth_vis = camera.create_depth_visualization(depth_frame)
                
                # 在深度图上显示手部坐标点的深度信息
                if detection_result['hands_detected']:
                    # 显示触碰点的深度信息
                    for contact_point in detection_result['contact_points']:
                        x, y = contact_point['pixel_coords']
                        depth = contact_point['depth']
                        cv.circle(depth_vis, (x, y), 8, (0, 0, 255), -1)  # 红色圆点
                        cv.putText(depth_vis, f"{depth:.3f}m", 
                                  (x + 12, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    # 显示警告点的深度信息
                    for warning_point in detection_result['warning_points']:
                        x, y = warning_point['pixel_coords']
                        depth = warning_point['depth']
                        cv.circle(depth_vis, (x, y), 6, (0, 255, 255), -1)  # 黄色圆点
                        cv.putText(depth_vis, f"{depth:.3f}m", 
                                  (x + 12, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                
                # 显示障碍物掩膜叠加在深度图上
                if 'obstacle_mask' in detection_result and detection_result['obstacle_mask'] is not None:
                    obstacle_mask = detection_result['obstacle_mask']
                    # 将掩膜叠加到深度图上
                    mask_colored = cv.applyColorMap(obstacle_mask, cv.COLORMAP_HOT)
                    depth_vis = cv.addWeighted(depth_vis, 0.7, mask_colored, 0.3, 0)
                
                cv.imshow('Depth Map', depth_vis)
            
            # 调试信息：显示检测详情
            if frame_count % 30 == 0:  # 每30帧打印一次调试信息
                print(f"\n🔍 调试信息 - Frame {frame_count}:")
                print(f"   手部检测: {detection_result['hands_detected']}")
                print(f"   最小距离: {detection_result['min_distance']:.3f}m")
                print(f"   触碰点数量: {detection_result['contact_count']}")
                print(f"   警告点数量: {detection_result['warning_count']}")
                print(f"   处理时间: {detection_result.get('processing_time', 0):.3f}s")
                
                if 'obstacle_mask' in detection_result and detection_result['obstacle_mask'] is not None:
                    mask_pixels = np.sum(detection_result['obstacle_mask'] > 0)
                    print(f"   障碍物掩膜像素: {mask_pixels}")
                
                # 显示手部深度信息和悬空状态
                if detection_result['hands_detected'] and detection_result['hands_info']:
                    for i, hand_info in enumerate(detection_result['hands_info']):
                        print(f"   手部{i+1}最小距离: {hand_info['min_distance']:.3f}m")
                        
                        # 显示调试信息和悬空状态
                        if 'debug_info' in hand_info and hand_info['debug_info']:
                            suspended_count = sum(1 for debug in hand_info['debug_info'] if debug.get('suspended', False))
                            total_points = len(hand_info['debug_info'])
                            print(f"   手部{i+1}悬空检测: {suspended_count}/{total_points} 个关键点被识别为悬空")
                            
                            # 显示前3个关键点的详细信息
                            for debug in hand_info['debug_info'][:3]:
                                if debug.get('suspended', False):
                                    status = "悬空"
                                else:
                                    status = "正常"
                                    
                                if debug['distance'] != float('inf'):
                                    print(f"     关键点{debug['landmark_id']}: 深度={debug['hand_depth']:.3f}m, 距离={debug['distance']:.3f}m, 状态={status}")
                                else:
                                    print(f"     关键点{debug['landmark_id']}: 深度={debug['hand_depth']:.3f}m, 距离=无, 状态={status}")
            
            # 只在有触碰或警告时打印信息
            if detection_result['contact_detected']:
                print(f"⚠️ COLLISION! Frame {frame_count}: {detection_result['contact_count']} contact points")
            elif detection_result['warning_detected']:
                print(f"⚠️ WARNING! Frame {frame_count}: {detection_result['warning_count']} warning points")
            
            frame_count += 1
            
            # 处理按键
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv.imwrite(f'contact_detection_frame_{frame_count}.jpg', annotated_frame)
                print(f"📸 保存帧: contact_detection_frame_{frame_count}.jpg")
            elif key == ord('d') and is_realsense:
                show_depth = not show_depth
                if not show_depth:
                    cv.destroyWindow('Depth Map')
    
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    
    finally:
        # 清理资源
        camera.cleanup()
        cv.destroyAllWindows()
        detector.cleanup()
        print("✅ 资源清理完成")


if __name__ == "__main__":
    main()

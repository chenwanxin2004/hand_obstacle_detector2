# 函数调用链说明

## 主程序入口

```
main()
```

## 核心调用链

### 1. 主检测流程

```
main()
  └─→ detector.detect_hand_contact(color_frame, depth_frame)
      │
      ├─→ self.hands.process()  [MediaPipe手部检测]
      │
      ├─→ 计算手部3D坐标 (内联代码，148-160行)
      │
      ├─→ yolo_obstacle_detector.detect_obstacles()  [YOLOv8障碍物检测]
      │   └─→ (在 yolo_obstacle_detector.py 中)
      │       ├─→ self._run_onnx_inference() 或 model()  [模型推理]
      │       ├─→ self._process_detection_results()  [处理检测结果]
      │       └─→ self._generate_obstacle_mask()  [生成障碍物掩膜]
      │           ├─→ self._exclude_hand_regions()  [排除手部区域]
      │           └─→ self._post_process_mask()  [后处理掩膜]
      │
      ├─→ self._generate_depth_backup_mask()  [深度备用检测]
      │   └─→ (仅在YOLOv8检测不足或失败时调用)
      │
      ├─→ self._detect_single_hand_contact_with_mask()  [单只手触碰检测]
      │   │
      │   ├─→ self._is_hand_suspended()  [检查手部是否悬空]
      │   │
      │   └─→ self._calculate_obstacle_distance()  [计算与障碍物距离]
      │       ├─→ self._calculate_depth_based_distance()  [基于深度的距离]
      │       │   └─→ self._is_hand_suspended()  [再次检查悬空]
      │       │
      │       └─→ self._calculate_yolo_enhanced_distance()  [YOLOv8增强距离]
      │
      ├─→ self._update_detection_history()  [更新检测历史]
      │
      └─→ self._update_adaptive_thresholds()  [自适应阈值调整]
```

### 2. 可视化流程

```
main()
  └─→ detector.visualize_detection(color_frame, detection_result)
      └─→ (绘制手部关键点、触碰点、警告点、状态信息等)
```

### 3. 反馈获取流程

```
main()
  └─→ detector.get_contact_feedback(detection_result)
      └─→ (返回触碰反馈信息)
```

## 详细函数说明

### 主检测函数

#### `detect_hand_contact(color_image, depth_image)`
- **位置**: `main.py:116`
- **功能**: 手部触碰障碍物检测的主入口
- **调用关系**:
  - 调用 `hands.process()` 进行手部检测
  - 调用 `yolo_obstacle_detector.detect_obstacles()` 获取障碍物掩膜
  - 调用 `_generate_depth_backup_mask()` 生成深度备用掩膜（可选）
  - 循环调用 `_detect_single_hand_contact_with_mask()` 检测每只手的触碰
  - 调用 `_update_detection_history()` 更新历史
  - 调用 `_update_adaptive_thresholds()` 自适应调整阈值

#### `_detect_single_hand_contact_with_mask(hand_landmarks, depth_image, image_shape, obstacle_mask)`
- **位置**: `main.py:241`
- **功能**: 检测单只手的触碰情况
- **调用关系**:
  - 对每个手部关键点：
    - 调用 `_is_hand_suspended()` 检查是否悬空
    - 调用 `_calculate_obstacle_distance()` 计算距离
    - 调用 `_validate_distance_measurement()` 验证距离（在内部调用）

### 距离计算函数

#### `_calculate_obstacle_distance(x, y, hand_depth, depth_image, obstacle_mask)`
- **位置**: `main.py:408`
- **功能**: 融合深度检测和YOLOv8检测计算距离
- **调用关系**:
  - 调用 `_calculate_depth_based_distance()` 基础深度检测
  - 调用 `_calculate_yolo_enhanced_distance()` YOLOv8增强检测
  - 调用 `_is_hand_suspended()` 验证悬空状态

#### `_calculate_depth_based_distance(x, y, hand_depth, depth_image)`
- **位置**: `main.py:537`
- **功能**: 基于深度图的直接距离计算
- **调用关系**:
  - 调用 `_is_hand_suspended()` 检查悬空

#### `_calculate_yolo_enhanced_distance(x, y, hand_depth, depth_image, obstacle_mask)`
- **位置**: `main.py:586`
- **功能**: 基于YOLOv8掩膜的精确距离计算

### 辅助函数

#### `_is_hand_suspended(x, y, hand_depth, depth_image)`
- **位置**: `main.py:436`
- **功能**: 判断手部是否悬空
- **被调用位置**:
  - `_detect_single_hand_contact_with_mask()` (266行)
  - `_calculate_obstacle_distance()` (428行)
  - `_calculate_depth_based_distance()` (580行)

#### `_validate_distance_measurement(x, y, hand_depth, distance, depth_image)`
- **位置**: `main.py:498`
- **功能**: 验证距离测量的可靠性
- **被调用位置**:
  - `_detect_single_hand_contact_with_mask()` (300行)

#### `_generate_depth_backup_mask(color_image, hand_landmarks_3d, depth_image)`
- **位置**: `main.py:362`
- **功能**: 基于深度的备用障碍物检测
- **被调用位置**:
  - `detect_hand_contact()` (177, 184, 187行)

#### `_update_detection_history(detection_result)`
- **位置**: `main.py:625`
- **功能**: 更新检测历史状态

#### `_update_adaptive_thresholds(detection_result)`
- **位置**: `main.py:640`
- **功能**: 自适应调整检测阈值

### YOLOv8检测器函数（yolo_obstacle_detector.py）

#### `detect_obstacles(image, hand_landmarks_3d)`
- **位置**: `yolo_obstacle_detector.py:308`
- **功能**: 使用YOLOv8检测图像中的障碍物
- **调用关系**:
  - 调用 `_run_onnx_inference()` 或 `model()` 进行推理
  - 调用 `_process_detection_results()` 处理结果
  - 调用 `_generate_obstacle_mask()` 生成掩膜

#### `_process_detection_results(result, image_shape)`
- **位置**: `yolo_obstacle_detector.py:364`
- **功能**: 处理YOLOv8检测结果，提取每个物体的信息

#### `_generate_obstacle_mask(detection_result, image_shape, hand_landmarks_3d)`
- **位置**: `yolo_obstacle_detector.py:425`
- **功能**: 生成障碍物掩膜
- **调用关系**:
  - 调用 `_exclude_hand_regions()` 排除手部区域
  - 调用 `_post_process_mask()` 后处理掩膜

## 调用流程图

```
用户程序 (main.py)
    │
    ├─→ main()
    │   │
    │   └─→ create_camera()  [初始化相机]
    │   │
    │   └─→ HandObstacleContactDetector()  [初始化检测器]
    │       │
    │       └─→ YOLOObstacleDetector()  [初始化YOLOv8检测器]
    │
    └─→ while True:
        │
        └─→ camera.get_frames()  [获取帧]
            │
            └─→ detector.detect_hand_contact()  [核心检测]
                │
                ├─→ MediaPipe手部检测
                ├─→ YOLOv8障碍物检测
                ├─→ 深度备用检测（可选）
                └─→ 手部触碰检测
                    │
                    └─→ 距离计算
                        ├─→ 深度基础检测
                        └─→ YOLOv8增强检测
        │
        └─→ detector.visualize_detection()  [可视化]
        │
        └─→ detector.get_contact_feedback()  [获取反馈]
        │
        └─→ cv.imshow()  [显示结果]
```

## 数据流向

1. **输入**: `color_image` + `depth_image`
2. **处理**:
   - MediaPipe → 手部关键点
   - YOLOv8 → 障碍物掩膜
   - 深度图 → 深度信息
3. **融合**: 
   - 手部关键点 + 障碍物掩膜 + 深度图 → 触碰检测
4. **输出**: `detection_result` (包含触碰点、警告点、距离等)

## 关键调用关系总结

- **主入口**: `main()` → `detect_hand_contact()`
- **障碍物检测**: `detect_hand_contact()` → `detect_obstacles()` (YOLOv8)
- **触碰检测**: `detect_hand_contact()` → `_detect_single_hand_contact_with_mask()`
- **距离计算**: `_detect_single_hand_contact_with_mask()` → `_calculate_obstacle_distance()`
- **悬空检测**: 多处调用 `_is_hand_suspended()`


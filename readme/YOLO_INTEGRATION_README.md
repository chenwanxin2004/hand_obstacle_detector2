# YOLOv8集成手部触碰检测系统

## 概述

本系统集成了YOLOv8-seg语义分割模型，用于更准确地检测和分离障碍物，提高手部触碰检测的精度。

## 新增功能

### 1. YOLOv8障碍物检测模块 (`yolo_obstacle_detector.py`)

**主要特性：**
- 使用YOLOv8-seg进行语义分割
- 自动识别80+种COCO数据集物体类别
- 智能排除手部区域，避免误判
- 支持实时性能监控
- 提供详细的可视化结果

**支持的障碍物类别：**
- 家具类：椅子、沙发、床、桌子等
- 电子设备：手机、电脑、电视等
- 日用品：瓶子、杯子、餐具等
- 交通工具：汽车、自行车等

### 2. 集成的手部检测器

**改进功能：**
- 自动选择YOLOv8或基础算法
- 更准确的障碍物掩膜生成
- 增强的可视化显示
- 性能统计信息

## 安装依赖

```bash
# 安装YOLOv8
pip install ultralytics

# 或使用uv
uv add ultralytics

# 其他依赖
pip install mediapipe opencv-python pyrealsense2 numpy
```

## 使用方法

### 1. 基本使用

```python
from hand_obstacle_contact_detector import HandObstacleContactDetector

# 创建检测器（使用YOLOv8）
detector = HandObstacleContactDetector(
    contact_threshold=0.03,      # 3cm触碰阈值
    warning_threshold=0.08,      # 8cm警告阈值
    use_yolo_obstacle=True,      # 启用YOLOv8
    yolo_model_path="yolov8n-seg.pt"  # 模型路径
)
```

### 2. 运行主程序

```bash
# 使用YOLOv8集成版本
python hand_obstacle_contact_detector.py

# 测试YOLOv8集成
python test_yolo_integration.py

# 单独测试YOLOv8模块
python yolo_obstacle_detector.py
```

### 3. 配置选项

```python
# 创建YOLOv8检测器
from yolo_obstacle_detector import YOLOObstacleDetector

detector = YOLOObstacleDetector(
    model_path="yolov8n-seg.pt",     # 模型路径
    confidence_threshold=0.5,         # 检测置信度
    device="auto"                     # 设备选择
)
```

## 技术实现

### 障碍物检测流程

1. **YOLOv8推理**：对输入图像进行语义分割
2. **类别过滤**：识别障碍物类别，排除人体
3. **掩膜生成**：创建障碍物掩膜
4. **手部排除**：排除MediaPipe检测的手部区域
5. **后处理**：噪声过滤和形态学操作

### 手部区域排除

- **YOLOv8检测**：自动识别人体区域
- **MediaPipe关键点**：基于手部关键点创建膨胀区域
- **双重保护**：确保手部区域完全排除

### 性能优化

- **缓存机制**：避免重复计算
- **动态阈值**：根据场景自动调整
- **实时监控**：FPS和推理时间统计

## 可视化功能

### 主界面显示

- **手部关键点**：MediaPipe标准绘制
- **触碰点**：红色圆点标记
- **警告点**：黄色圆点标记
- **状态信息**：实时统计和警告

### 障碍物掩膜

- **右上角显示**：实时障碍物掩膜
- **颜色编码**：JET颜色映射
- **标题标识**：区分YOLOv8和基础算法

### YOLOv8检测结果

- **边界框**：障碍物检测框
- **类别标签**：物体名称和置信度
- **统计信息**：检测数量和FPS

## 故障排除

### 常见问题

1. **YOLOv8模型加载失败**
   ```bash
   # 确保模型文件存在
   ls yolov8n-seg.pt
   
   # 重新下载模型
   python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
   ```

2. **ultralytics导入失败**
   ```bash
   pip install ultralytics
   # 或
   uv add ultralytics
   ```

3. **性能问题**
   - 降低检测置信度阈值
   - 使用更小的模型（yolov8n-seg.pt）
   - 调整图像分辨率

4. **深度图显示错误**
   - 已修复RealSense深度可视化问题
   - 支持模拟相机和RealSense相机

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查YOLOv8状态
detector = HandObstacleContactDetector(use_yolo_obstacle=True)
print(f"YOLOv8可用: {detector.use_yolo_obstacle}")
```

## 性能对比

| 方法 | 精度 | 速度 | 资源占用 |
|------|------|------|----------|
| 基础算法 | 中等 | 快 | 低 |
| YOLOv8-seg | 高 | 中等 | 中等 |

## 扩展功能

### 自定义障碍物类别

```python
# 修改yolo_obstacle_detector.py中的obstacle_classes
self.obstacle_classes = {
    'custom_object': class_id,
    # 添加自定义类别
}
```

### 多模型支持

```python
# 支持不同大小的YOLOv8模型
models = {
    'nano': 'yolov8n-seg.pt',      # 最快
    'small': 'yolov8s-seg.pt',     # 平衡
    'medium': 'yolov8m-seg.pt',    # 更准确
    'large': 'yolov8l-seg.pt',     # 最准确
}
```

## 许可证

本项目基于您现有的3D视觉项目，遵循相同的许可证条款。

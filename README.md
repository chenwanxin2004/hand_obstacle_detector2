# 手部触碰障碍物检测系统

基于MediaPipe和YOLOv8的实时手部触碰障碍物检测系统，支持Intel RealSense深度摄像头。该系统能够实时检测手部与障碍物的距离，并在手部接近或触碰障碍物时发出警告。

## 🚀 功能特性

- **实时手部检测**: 使用MediaPipe进行高精度手部21个关键点检测
- **YOLOv8语义分割**: 使用YOLOv8语义分割模型进行障碍物检测和分割
- **深度融合检测**: 结合深度图信息和YOLOv8语义分割结果，实现精确的障碍物检测
- **触碰检测**: 检测手部关键点与障碍物的距离，判断触碰和接近状态
- **智能悬空检测**: 自动识别手部悬空状态，避免误检背景
- **Intel RealSense支持**: 支持D435深度摄像头，提供实时彩色和深度图像
- **实时可视化**: 双窗口显示检测结果和深度图，支持触碰点和警告点标注
- **自适应阈值**: 根据环境自动调整检测阈值，提高检测准确性

## 📋 系统要求

- Python 3.8+
- OpenCV 4.8+
- MediaPipe 0.10+
- Intel RealSense SDK 2.0+ (必需，用于深度检测)
- PyTorch 或 ONNX Runtime (用于YOLOv8模型推理)

## 🛠️ 安装

### 1. 克隆项目
```bash
git clone <repository-url>
cd <project-directory>
```

### 2. 安装依赖
```bash
# 使用uv安装（推荐）
uv sync

# 或使用pip
pip install -r requirements.txt
```

### 3. 准备YOLO模型
确保YOLOv8语义分割模型文件位于以下位置之一：
- `src/yolov8n-seg.pt` (PyTorch模型)
- `src/yolov8n-seg.onnx` (ONNX模型，推荐)
- `src/quantized_models/yolov8n-seg_fp16/` (量化模型)

## 📹 使用方法

### 启动程序
```bash
# 直接运行
python src/main.py
```

程序会自动：
1. 初始化Intel RealSense摄像头
2. 加载YOLOv8障碍物检测模型
3. 启动实时检测窗口

### 窗口说明

程序会打开两个窗口：
- **主窗口** (Hand Obstacle Contact Detection): 显示手部检测和触碰检测结果
  - 红色圆点: 触碰检测点 (HIT!)
  - 黄色圆点: 接近警告点 (NEAR)
  - 状态栏: 显示当前安全状态 (SAFE/WARNING/COLLISION)
  
- **深度图窗口** (Depth Map): 显示深度图和障碍物掩膜
  - 彩色深度图
  - 障碍物掩膜叠加显示
  - 触碰点和警告点的深度标注

## 🎮 控制说明

- **按 'q'**: 退出程序
- **按 's'**: 保存当前检测帧
- **按 'd'**: 切换深度图窗口显示/隐藏

## 📊 检测参数

### 默认阈值
- **触碰阈值**: 3cm (0.03m) - 手部与障碍物距离小于此值判定为触碰
- **警告阈值**: 6cm (0.06m) - 手部与障碍物距离小于此值但大于触碰阈值时发出警告

### 自适应调整
系统会根据环境自动调整阈值，避免误检和漏检。

## 🔧 配置选项

### 修改检测参数
在 `src/main.py` 的 `main()` 函数中修改：

```python
detector = HandObstacleContactDetector(
    contact_threshold=0.03,      # 触碰阈值（米）
    warning_threshold=0.06,       # 警告阈值（米）
    use_yolo_obstacle=True,       # 是否使用YOLOv8
    yolo_model_path="src/yolov8n-seg.onnx"  # 模型路径
)
```

### YOLOv8模型配置
在 `yolo_obstacle_detector.py` 中可以配置：
- 置信度阈值
- 量化类型 (fp16/int8)
- 模型路径

## 📁 输出文件

- `contact_detection_frame_XXXXXX.jpg`: 手动保存的检测帧

## 📊 性能优化

- **量化模型**: 使用FP16量化后的ONNX模型可显著提升推理速度
- **自适应阈值**: 自动调整检测阈值，平衡检测精度和误检率
- **缓存机制**: 系统会缓存障碍物掩膜，减少重复计算

## 🐛 故障排除

### RealSense摄像头连接问题
1. 确保安装了Intel RealSense SDK 2.0
2. 检查USB连接和驱动安装
3. 运行 `rs-enumerate-devices` 检查设备状态
4. 确保摄像头固件为最新版本

### 手部检测不准确
1. 确保光线充足，避免过暗或过亮环境
2. 调整手部与摄像头的距离（推荐40-80cm）
3. 确保手部在摄像头视野范围内

### YOLOv8检测失败
1. 检查模型文件路径是否正确
2. 确认模型文件格式 (`.pt` 或 `.onnx`)
3. 检查ONNX Runtime是否正确安装

### 深度检测不准确
1. 确保RealSense摄像头正确连接
2. 检查深度图质量，避免强光或反光表面
3. 调整摄像头位置，确保深度检测范围

## 📚 技术栈

- **手部检测**: MediaPipe Hands (21个关键点)
- **障碍物检测**: YOLOv8 Segmentation (实例分割)
- **深度感知**: Intel RealSense D435
- **计算机视觉**: OpenCV
- **深度学习**: PyTorch / ONNX Runtime
- **量化**: FP16/INT8量化支持

## 📖 相关文档

- [函数调用链说明](docs/FUNCTION_CALL_CHAIN.md) - 详细的函数调用关系说明

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证。
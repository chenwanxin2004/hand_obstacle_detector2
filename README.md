# 手部检测和YOLO目标检测系统

基于MediaPipe和YOLOv8的实时手部检测和YOLO目标检测系统，支持Intel RealSense D435摄像头。

## 🚀 功能特性

- **实时手部检测**: 使用MediaPipe进行高精度手部关键点检测
- **YOLO目标检测**: 集成YOLOv8模型进行实时目标检测
- **Intel RealSense支持**: 支持D435深度摄像头，提供彩色和深度图像
- **双摄像头模式**: 支持默认摄像头和RealSense摄像头
- **帧保存功能**: 支持自动和手动保存检测帧
- **实时显示**: 实时显示检测结果和深度图

## 📋 系统要求

- Python 3.8+
- OpenCV 4.8+
- MediaPipe 0.10+
- Intel RealSense SDK 2.0+ (可选，用于RealSense摄像头)

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

### 3. 下载YOLO模型
确保YOLO模型文件位于 `src/project_name/yolov8n.pt`

## 📹 使用方法

### 默认摄像头模式
```bash
# 使用默认摄像头
python -m src.project_name.main

# 指定摄像头ID
python -m src.project_name.main --camera-id 1

# 启用自动保存帧
python -m src.project_name.main --save-frames
```

### Intel RealSense D435模式
```bash
# 使用RealSense摄像头
python -m src.project_name.main --camera-type realsense

# 启用自动保存帧
python -m src.project_name.main --camera-type realsense --save-frames
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--camera-type` | 摄像头类型: `default` 或 `realsense` | `default` |
| `--camera-id` | 默认摄像头ID | `0` |
| `--yolo-model` | YOLO模型文件路径 | `src/project_name/yolov8n.pt` |
| `--save-frames` | 是否自动保存检测帧 | `False` |

## 🎮 控制说明

- **按 'q'**: 退出程序
- **按 's'**: 手动保存当前帧
- **自动保存**: 如果启用，每30帧自动保存一次

## 📁 输出文件

### 默认摄像头模式
- `auto_capture_XXXXXX.jpg`: 自动保存的帧
- `manual_capture_XXXXXXXXX.jpg`: 手动保存的帧

### RealSense模式
- `realsense_color_XXXXXX.jpg`: 彩色图像帧
- `realsense_depth_XXXXXX.jpg`: 深度图像帧
- `realsense_manual_color_XXXXXXXXX.jpg`: 手动保存的彩色帧
- `realsense_manual_depth_XXXXXXXXX.jpg`: 手动保存的深度帧

## 🔧 配置

### 环境变量
创建 `.env` 文件：
```bash
# 调试模式
DEBUG=False

# 日志级别
LOG_LEVEL=INFO
```

## 📊 性能优化

- 调整MediaPipe检测参数以获得最佳性能
- 根据硬件配置调整YOLO模型大小
- 使用GPU加速（如果可用）

## 🐛 故障排除

### RealSense摄像头连接问题
1. 确保安装了Intel RealSense SDK 2.0
2. 检查USB连接和驱动安装
3. 运行 `rs-enumerate-devices` 检查设备状态

### 手部检测不准确
1. 确保光线充足
2. 调整手部与摄像头的距离
3. 检查MediaPipe版本兼容性

## 📚 技术栈

- **手部检测**: MediaPipe Hands
- **目标检测**: YOLOv8
- **计算机视觉**: OpenCV
- **深度感知**: Intel RealSense SDK
- **深度学习**: PyTorch

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证。
# 常见的COCO类别示例
COCO_CLASSES = {
    0: 'person',           # 人
    1: 'bicycle',          # 自行车
    2: 'car',              # 汽车
    3: 'motorcycle',       # 摩托车
    4: 'airplane',         # 飞机
    5: 'bus',              # 公交车
    6: 'train',            # 火车
    7: 'truck',            # 卡车
    8: 'boat',             # 船
    9: 'traffic light',    # 红绿灯
    10: 'fire hydrant',    # 消防栓
    11: 'stop sign',       # 停止标志
    12: 'parking meter',   # 停车计时器
    13: 'bench',           # 长凳
    14: 'bird',            # 鸟
    15: 'cat',             # 猫
    16: 'dog',             # 狗
    17: 'horse',           # 马
    18: 'sheep',           # 羊
    19: 'cow',             # 牛
    20: 'elephant',        # 大象
    21: 'bear',            # 熊
    22: 'zebra',           # 斑马
    23: 'giraffe',         # 长颈鹿
    24: 'backpack',        # 背包
    25: 'umbrella',        # 雨伞
    26: 'handbag',         # 手提包
    27: 'tie',             # 领带
    28: 'suitcase',        # 行李箱
    29: 'frisbee',         # 飞盘
    30: 'skis',            # 滑雪板
    31: 'snowboard',       # 滑雪板
    32: 'sports ball',     # 运动球
    33: 'kite',            # 风筝
    34: 'baseball bat',    # 棒球棒
    35: 'baseball glove',  # 棒球手套
    36: 'skateboard',      # 滑板
    37: 'surfboard',       # 冲浪板
    38: 'tennis racket',   # 网球拍
    39: 'bottle',          # 瓶子
    40: 'wine glass',      # 酒杯
    41: 'cup',             # 杯子
    42: 'fork',            # 叉子
    43: 'knife',           # 刀子
    44: 'spoon',           # 勺子
    45: 'bowl',            # 碗
    46: 'banana',          # 香蕉
    47: 'apple',           # 苹果
    48: 'sandwich',        # 三明治
    49: 'orange',          # 橙子
    50: 'broccoli',        # 西兰花
    51: 'carrot',          # 胡萝卜
    52: 'hot dog',         # 热狗
    53: 'pizza',           # 披萨
    54: 'donut',           # 甜甜圈
    55: 'cake',            # 蛋糕
    56: 'chair',            # 椅子
    57: 'couch',            # 沙发
    58: 'potted plant',    # 盆栽植物
    59: 'bed',              # 床
    60: 'dining table',    # 餐桌
    61: 'toilet',           # 马桶
    62: 'tv',               # 电视
    63: 'laptop',           # 笔记本电脑
    64: 'mouse',            # 鼠标
    65: 'remote',           # 遥控器
    66: 'keyboard',         # 键盘
    67: 'cell phone',       # 手机
    68: 'microwave',        # 微波炉
    69: 'oven',             # 烤箱
    70: 'toaster',          # 烤面包机
    71: 'sink',             # 水槽
    72: 'refrigerator',     # 冰箱
    73: 'book',             # 书
    74: 'clock',            # 时钟
    75: 'vase',             # 花瓶
    76: 'scissors',         # 剪刀
    77: 'teddy bear',       # 泰迪熊
    78: 'hair drier',       # 吹风机
    79: 'toothbrush'        # 牙刷
}
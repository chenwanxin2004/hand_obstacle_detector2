#!/usr/bin/env python3
"""
YOLOv8-seg量化效果测试脚本
比较原始模型和量化模型的性能差异
"""

import os
import numpy as np
import cv2
import time
from yolo_obstacle_detector import YOLOObstacleDetector

def create_test_image(width: int = 640, height: int = 640) -> np.ndarray:
    """创建测试图像"""
    # 创建一个包含各种形状的测试图像
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(image, (320, 320), 100, (255, 255, 255), -1)
    # 添加一些几何形状模拟障碍物
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # 蓝色矩形
    cv2.circle(image, (400, 300), 80, (0, 255, 0), -1)  # 绿色圆形
    cv2.rectangle(image, (500, 150), (600, 350), (0, 0, 255), -1)  # 红色矩形
    
    # 添加一些噪声
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image

def compare_models():
    """比较不同量化模型的性能"""
    print("🚀 YOLOv8-seg量化效果测试")
    print("=" * 60)
    
    # 创建测试图像
    test_image = create_test_image()
    print(f"📷 创建测试图像: {test_image.shape}")
    
    # 测试不同模型配置
    model_configs = [
        {
            "name": "原始模型 (FP32)",
            "use_quantized": False,
            "quantization_type": "original"
        },
        {
            "name": "FP16量化模型",
            "use_quantized": True,
            "quantization_type": "fp16"
        },
        {
            "name": "INT8量化模型",
            "use_quantized": True,
            "quantization_type": "int8"
        }
    ]
    
    results = {}
    
    for config in model_configs:
        print(f"\n🔄 测试 {config['name']}...")
        print("-" * 40)
        
        try:
            # 创建检测器
            detector = YOLOObstacleDetector(
                use_quantized=config["use_quantized"],
                quantization_type=config["quantization_type"],
                confidence_threshold=0.5
            )
            
            if not detector.is_initialized:
                print(f"❌ {config['name']} 初始化失败")
                results[config['name']] = {"error": "初始化失败"}
                continue
            
            # 获取模型信息
            model_info = detector.get_model_info()
            print(f"📊 模型信息:")
            print(f"   路径: {model_info['model_path']}")
            print(f"   大小: {model_info['model_size']}")
            print(f"   设备: {model_info['device']}")
            
            # 性能基准测试
            benchmark_results = detector.benchmark_performance(test_image, num_runs=10)
            
            if "error" in benchmark_results:
                print(f"❌ 性能测试失败: {benchmark_results['error']}")
                results[config['name']] = benchmark_results
            else:
                results[config['name']] = {
                    **model_info,
                    **benchmark_results
                }
                
                print(f"✅ 性能测试完成")
                print(f"   FPS: {benchmark_results['fps']:.2f}")
                print(f"   平均推理时间: {benchmark_results['average_time']:.4f}s")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            results[config['name']] = {"error": str(e)}
    
    # 输出比较结果
    print("\n📋 性能比较结果")
    print("=" * 60)
    
    if results:
        # 找到基准模型（原始模型）
        baseline_fps = None
        baseline_name = None
        
        for name, result in results.items():
            if "error" not in result and "original" in name.lower():
                baseline_fps = result.get('fps', 0)
                baseline_name = name
                break
        
        # 输出表格
        print(f"{'模型类型':<20} {'FPS':<10} {'推理时间(ms)':<15} {'模型大小':<15} {'加速比':<10}")
        print("-" * 80)
        
        for name, result in results.items():
            if "error" in result:
                print(f"{name:<20} {'ERROR':<10} {'ERROR':<15} {'ERROR':<15} {'ERROR':<10}")
            else:
                fps = result.get('fps', 0)
                inference_time = result.get('average_time', 0) * 1000  # 转换为毫秒
                model_size = result.get('model_size', 'Unknown')
                
                # 计算加速比
                speedup = "N/A"
                if baseline_fps and baseline_fps > 0:
                    speedup = f"{fps/baseline_fps:.2f}x"
                
                print(f"{name:<20} {fps:<10.2f} {inference_time:<15.2f} {model_size:<15} {speedup:<10}")
        
        # 分析结果
        print(f"\n📈 性能分析:")
        if baseline_fps:
            print(f"   基准模型 ({baseline_name}): {baseline_fps:.2f} FPS")
            
            for name, result in results.items():
                if "error" not in result and "original" not in name.lower():
                    fps = result.get('fps', 0)
                    if fps > 0:
                        improvement = ((fps - baseline_fps) / baseline_fps) * 100
                        print(f"   {name}: {fps:.2f} FPS ({improvement:+.1f}%)")
    
    print(f"\n🎉 量化测试完成！")

def test_quantization_accuracy():
    """测试量化模型的精度"""
    print("\n🔍 量化精度测试")
    print("=" * 40)
    
    # 创建测试图像
    test_image = create_test_image()
    
    # 测试原始模型和FP16量化模型
    models = [
        ("原始模型", False, "original"),
        ("FP16量化", True, "fp16")
    ]
    
    detection_results = {}
    
    for name, use_quantized, quantization_type in models:
        print(f"\n🔄 测试 {name} 检测精度...")
        
        try:
            detector = YOLOObstacleDetector(
                use_quantized=use_quantized,
                quantization_type=quantization_type,
                confidence_threshold=0.3  # 降低阈值以检测更多对象
            )
            
            if detector.is_initialized:
                # 进行检测
                result = detector.detect_obstacles(test_image)
                
                if result:
                    obstacle_count = len(result.get('obstacles', []))
                    detection_results[name] = {
                        'obstacle_count': obstacle_count,
                        'detection_result': result
                    }
                    print(f"   检测到 {obstacle_count} 个障碍物")
                else:
                    print(f"   未检测到障碍物")
            else:
                print(f"   模型初始化失败")
                
        except Exception as e:
            print(f"   检测失败: {e}")
    
    # 比较检测结果
    if len(detection_results) >= 2:
        print(f"\n📊 精度比较:")
        baseline_count = detection_results.get("原始模型", {}).get('obstacle_count', 0)
        
        for name, result in detection_results.items():
            count = result.get('obstacle_count', 0)
            if name != "原始模型":
                if baseline_count > 0:
                    accuracy = (count / baseline_count) * 100
                    print(f"   {name}: {count} 个障碍物 (精度: {accuracy:.1f}%)")
                else:
                    print(f"   {name}: {count} 个障碍物")

def main():
    """主函数"""
    print("🎯 YOLOv8-seg量化效果综合测试")
    print("=" * 60)
    
    # 检查模型文件
    if not os.path.exists("yolov8n-seg.pt"):
        print("❌ 未找到yolov8n-seg.pt模型文件")
        print("请确保模型文件在当前目录下")
        return
    
    # 检查量化模型
    quantized_dir = "quantized_models"
    if not os.path.exists(quantized_dir):
        print(f"⚠️  量化模型目录不存在: {quantized_dir}")
        print("请先运行 export_quantized_yolo.py 生成量化模型")
        return
    
    # 运行性能比较测试
    compare_models()
    
    # 运行精度测试
    test_quantization_accuracy()
    
    print(f"\n💡 使用建议:")
    print(f"   - 如果追求最高精度，使用原始模型")
    print(f"   - 如果追求速度，使用FP16量化模型")
    print(f"   - 如果追求极致性能，使用INT8量化模型")
    print(f"   - RK3399开发板推荐使用FP16量化模型")

if __name__ == "__main__":
    main()

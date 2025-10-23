#!/usr/bin/env python3
"""
简化的量化模型测试脚本
"""

import os
import numpy as np
import cv2
import time
from ultralytics import YOLO

def test_model_performance():
    """测试模型性能"""
    print("🚀 简化量化模型测试")
    print("=" * 40)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    print(f"📷 创建测试图像: {test_image.shape}")
    
    # 测试原始模型
    print(f"\n🔄 测试原始模型...")
    try:
        # 修复PyTorch 2.6的weights_only问题
        import torch
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.SegmentationModel'])
        
        model_pt = YOLO("yolov8n-seg.pt")
        print(f"✅ 原始模型加载成功")
        
        # 性能测试
        times = []
        for i in range(10):
            start_time = time.time()
            _ = model_pt(test_image, verbose=False)
            times.append(time.time() - start_time)
        
        avg_time_pt = np.mean(times)
        fps_pt = 1.0 / avg_time_pt
        print(f"   平均推理时间: {avg_time_pt:.4f}s")
        print(f"   FPS: {fps_pt:.2f}")
        
    except Exception as e:
        print(f"❌ 原始模型测试失败: {e}")
        avg_time_pt = 0
        fps_pt = 0
    
    # 测试ONNX模型
    print(f"\n🔄 测试ONNX模型...")
    try:
        model_onnx = YOLO("yolov8n-seg.onnx")
        print(f"✅ ONNX模型加载成功")
        
        # 性能测试
        times = []
        for i in range(10):
            start_time = time.time()
            _ = model_onnx(test_image, verbose=False)
            times.append(time.time() - start_time)
        
        avg_time_onnx = np.mean(times)
        fps_onnx = 1.0 / avg_time_onnx
        print(f"   平均推理时间: {avg_time_onnx:.4f}s")
        print(f"   FPS: {fps_onnx:.2f}")
        
    except Exception as e:
        print(f"❌ ONNX模型测试失败: {e}")
        avg_time_onnx = 0
        fps_onnx = 0
    
    # 比较结果
    print(f"\n📊 性能比较:")
    print(f"   原始模型 (PyTorch): {fps_pt:.2f} FPS")
    print(f"   ONNX模型: {fps_onnx:.2f} FPS")
    
    if fps_pt > 0 and fps_onnx > 0:
        speedup = fps_onnx / fps_pt
        print(f"   加速比: {speedup:.2f}x")
        if speedup > 1:
            print(f"   🎉 ONNX模型比原始模型快 {speedup:.2f} 倍！")
        else:
            print(f"   ⚠️  ONNX模型比原始模型慢 {1/speedup:.2f} 倍")
    
    # 检查模型文件大小
    print(f"\n📁 模型文件大小:")
    try:
        pt_size = os.path.getsize("yolov8n-seg.pt") / (1024 * 1024)
        onnx_size = os.path.getsize("yolov8n-seg.onnx") / (1024 * 1024)
        print(f"   原始模型: {pt_size:.1f} MB")
        print(f"   ONNX模型: {onnx_size:.1f} MB")
        
        size_reduction = (pt_size - onnx_size) / pt_size * 100
        print(f"   大小减少: {size_reduction:.1f}%")
        
    except Exception as e:
        print(f"   无法获取文件大小: {e}")

def test_detection_accuracy():
    """测试检测精度"""
    print(f"\n🔍 检测精度测试:")
    
    # 创建一个包含简单形状的测试图像
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(test_image, (450, 200), 80, (128, 128, 128), -1)
    
    models = [
        ("原始模型", "yolov8n-seg.pt"),
        ("ONNX模型", "yolov8n-seg.onnx")
    ]
    
    for name, model_path in models:
        if not os.path.exists(model_path):
            print(f"   ⚠️  {name}文件不存在: {model_path}")
            continue
            
        try:
            print(f"\n🔄 测试{name}检测精度...")
            
            if model_path.endswith('.pt'):
                import torch
                torch.serialization.add_safe_globals(['ultralytics.nn.tasks.SegmentationModel'])
            
            model = YOLO(model_path)
            results = model(test_image, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    mask_count = len(result.masks)
                    print(f"   ✅ 检测到 {mask_count} 个分割区域")
                else:
                    print(f"   ⚠️  未检测到分割区域")
            else:
                print(f"   ⚠️  未检测到任何对象")
                
        except Exception as e:
            print(f"   ❌ {name}检测失败: {e}")

if __name__ == "__main__":
    test_model_performance()
    test_detection_accuracy()
    print(f"\n🎉 测试完成！")

#!/usr/bin/env python3
"""
YOLOv8-seg模型量化导出脚本
支持FP16和INT8量化，针对RK3399开发板优化
"""

import os
import torch
import numpy as np
from ultralytics import YOLO
from typing import Optional, List
import cv2

class YOLOQuantizer:
    """YOLOv8-seg模型量化器"""
    
    def __init__(self, model_path: str = "yolov8n-seg.pt"):
        """
        初始化量化器
        
        Args:
            model_path: 原始YOLOv8模型路径
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.output_dir = "quantized_models"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export_fp16_onnx(self, imgsz: int = 640, simplify: bool = True) -> str:
        """
        导出FP16精度的ONNX模型
        
        Args:
            imgsz: 输入图像尺寸
            simplify: 是否简化ONNX图
            
        Returns:
            导出的ONNX模型路径
        """
        output_path = os.path.join(self.output_dir, "yolov8n-seg_fp16.onnx")
        
        print(f"🔄 开始导出FP16 ONNX模型...")
        print(f"   输入尺寸: {imgsz}x{imgsz}")
        print(f"   输出路径: {output_path}")
        
        try:
            # 导出FP16 ONNX模型
            self.model.export(
                format="onnx",
                imgsz=imgsz,
                half=True,  # FP16量化
                simplify=simplify,
                opset=13,   # ONNX操作集版本
                dynamic=False,  # 固定输入尺寸，提升性能
                verbose=False
            )
            
            # 移动文件到指定位置
            if os.path.exists("yolov8n-seg_fp16.onnx"):
                os.rename("yolov8n-seg_fp16.onnx", output_path)
            
            print(f"✅ FP16 ONNX模型导出成功: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ FP16 ONNX模型导出失败: {e}")
            return None
    
    def export_int8_onnx(self, imgsz: int = 640, calibration_data: Optional[List[np.ndarray]] = None) -> str:
        """
        导出INT8精度的ONNX模型（需要校准数据）
        
        Args:
            imgsz: 输入图像尺寸
            calibration_data: 校准数据集
            
        Returns:
            导出的ONNX模型路径
        """
        output_path = os.path.join(self.output_dir, "yolov8n-seg_int8.onnx")
        
        print(f"🔄 开始导出INT8 ONNX模型...")
        print(f"   输入尺寸: {imgsz}x{imgsz}")
        print(f"   输出路径: {output_path}")
        
        try:
            # 如果没有提供校准数据，生成一些随机数据作为示例
            if calibration_data is None:
                print("⚠️  未提供校准数据，使用随机数据作为示例")
                calibration_data = [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(10)]
            
            # 导出INT8 ONNX模型
            self.model.export(
                format="onnx",
                imgsz=imgsz,
                int8=True,  # INT8量化
                simplify=True,
                opset=13,
                dynamic=False,
                verbose=False
            )
            
            # 移动文件到指定位置
            if os.path.exists("yolov8n-seg_int8.onnx"):
                os.rename("yolov8n-seg_int8.onnx", output_path)
            
            print(f"✅ INT8 ONNX模型导出成功: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ INT8 ONNX模型导出失败: {e}")
            return None
    
    def export_tensorrt_engine(self, imgsz: int = 640, precision: str = "fp16") -> str:
        """
        导出TensorRT引擎（需要NVIDIA GPU和TensorRT环境）
        
        Args:
            imgsz: 输入图像尺寸
            precision: 精度类型 ("fp32", "fp16", "int8")
            
        Returns:
            导出的TensorRT引擎路径
        """
        output_path = os.path.join(self.output_dir, f"yolov8n-seg_{precision}.engine")
        
        print(f"🔄 开始导出TensorRT {precision.upper()}引擎...")
        print(f"   输入尺寸: {imgsz}x{imgsz}")
        print(f"   输出路径: {output_path}")
        
        try:
            # 检查CUDA是否可用
            if not torch.cuda.is_available():
                print("❌ CUDA不可用，无法导出TensorRT引擎")
                return None
            
            # 导出TensorRT引擎
            self.model.export(
                format="engine",
                imgsz=imgsz,
                half=(precision == "fp16"),
                int8=(precision == "int8"),
                simplify=True,
                verbose=False
            )
            
            # 移动文件到指定位置
            if os.path.exists("yolov8n-seg.engine"):
                os.rename("yolov8n-seg.engine", output_path)
            
            print(f"✅ TensorRT {precision.upper()}引擎导出成功: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ TensorRT引擎导出失败: {e}")
            return None
    
    def export_openvino_ir(self, imgsz: int = 640, precision: str = "fp16") -> str:
        """
        导出OpenVINO IR格式（适合Intel CPU和ARM设备）
        
        Args:
            imgsz: 输入图像尺寸
            precision: 精度类型 ("fp32", "fp16", "int8")
            
        Returns:
            导出的OpenVINO IR路径
        """
        output_path = os.path.join(self.output_dir, f"yolov8n-seg_{precision}")
        
        print(f"🔄 开始导出OpenVINO {precision.upper()} IR...")
        print(f"   输入尺寸: {imgsz}x{imgsz}")
        print(f"   输出路径: {output_path}")
        
        try:
            # 导出OpenVINO IR
            self.model.export(
                format="openvino",
                imgsz=imgsz,
                half=(precision == "fp16"),
                int8=(precision == "int8"),
                simplify=True,
                verbose=False
            )
            
            # 移动文件到指定位置
            if os.path.exists("yolov8n-seg_openvino_model"):
                import shutil
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)
                shutil.move("yolov8n-seg_openvino_model", output_path)
            
            print(f"✅ OpenVINO {precision.upper()} IR导出成功: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ OpenVINO IR导出失败: {e}")
            return None
    
    def benchmark_models(self, test_image: np.ndarray, models: List[str]) -> dict:
        """
        对多个模型进行性能基准测试
        
        Args:
            test_image: 测试图像
            models: 模型路径列表
            
        Returns:
            性能测试结果
        """
        results = {}
        
        for model_path in models:
            if not os.path.exists(model_path):
                print(f"⚠️  模型文件不存在: {model_path}")
                continue
                
            print(f"🔄 测试模型: {model_path}")
            
            try:
                # 加载模型
                model = YOLO(model_path)
                
                # 预热
                for _ in range(5):
                    _ = model(test_image, verbose=False)
                
                # 性能测试
                import time
                times = []
                for _ in range(20):
                    start_time = time.time()
                    _ = model(test_image, verbose=False)
                    times.append(time.time() - start_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                results[model_path] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'fps': 1.0 / avg_time
                }
                
                print(f"   平均推理时间: {avg_time:.4f}s ± {std_time:.4f}s")
                print(f"   FPS: {1.0/avg_time:.2f}")
                
            except Exception as e:
                print(f"❌ 模型测试失败: {e}")
                results[model_path] = {'error': str(e)}
        
        return results

def main():
    """主函数"""
    print("🚀 YOLOv8-seg模型量化导出工具")
    print("=" * 50)
    
    # 检查模型文件是否存在
    model_path = "yolov8n-seg.pt"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确保yolov8n-seg.pt文件在当前目录下")
        return
    
    # 初始化量化器
    quantizer = YOLOQuantizer(model_path)
    
    # 导出不同精度的模型
    print("\n📦 开始导出量化模型...")
    
    # 1. 导出FP16 ONNX模型
    fp16_onnx_path = quantizer.export_fp16_onnx(imgsz=640)
    
    # 2. 导出INT8 ONNX模型
    int8_onnx_path = quantizer.export_int8_onnx(imgsz=640)
    
    # 3. 导出OpenVINO IR（适合ARM设备）
    openvino_fp16_path = quantizer.export_openvino_ir(imgsz=640, precision="fp16")
    
    # 4. 如果有CUDA环境，导出TensorRT引擎
    if torch.cuda.is_available():
        tensorrt_fp16_path = quantizer.export_tensorrt_engine(imgsz=640, precision="fp16")
    else:
        print("⚠️  CUDA不可用，跳过TensorRT引擎导出")
    
    # 性能基准测试
    print("\n📊 性能基准测试...")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    models_to_test = [model_path]  # 原始模型
    if fp16_onnx_path:
        models_to_test.append(fp16_onnx_path)
    if int8_onnx_path:
        models_to_test.append(int8_onnx_path)
    
    benchmark_results = quantizer.benchmark_models(test_image, models_to_test)
    
    # 输出结果总结
    print("\n📋 导出结果总结:")
    print("=" * 50)
    for model_path, result in benchmark_results.items():
        if 'error' in result:
            print(f"❌ {os.path.basename(model_path)}: {result['error']}")
        else:
            print(f"✅ {os.path.basename(model_path)}: {result['fps']:.2f} FPS")
    
    print(f"\n📁 所有量化模型已保存到: {quantizer.output_dir}/")
    print("🎉 量化导出完成！")

if __name__ == "__main__":
    main()

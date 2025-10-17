#!/usr/bin/env python3
"""
快速训练测试脚本 - 验证优化后的性能
"""

import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_loading_speed():
    """测试数据加载速度"""
    from InstrumentTimbre.core.data.fast_dataset import FastAudioDataset
    from InstrumentTimbre.core.features.fast import FastFeatureExtractor
    
    print("🧪 测试数据加载速度...")
    
    # 初始化快速特征提取器
    extractor = FastFeatureExtractor()
    
    # 创建数据集（第一次会计算特征并缓存）
    start_time = time.time()
    dataset = FastAudioDataset(
        data_dir='./data/clips',
        feature_extractor=extractor
    )
    setup_time = time.time() - start_time
    
    print(f"数据集设置时间: {setup_time:.2f}秒")
    print(f"总样本数: {len(dataset)}")
    print(f"类别数: {len(dataset.class_names)}")
    print(f"类别: {dataset.class_names}")
    print(f"特征维度: {dataset.get_feature_dim()}")
    
    # 测试数据加载速度
    print("\n测试批量加载速度...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    start_time = time.time()
    for i, (features, labels) in enumerate(loader):
        if i >= 10:  # 只测试前10个batch
            break
        if i == 0:
            print(f"Batch shape: {features.shape}, Labels shape: {labels.shape}")
    
    load_time = time.time() - start_time
    samples_processed = min(10 * 32, len(dataset))
    
    print(f"处理 {samples_processed} 样本用时: {load_time:.3f}秒")
    print(f"平均速度: {samples_processed/load_time:.1f} 样本/秒")
    
    return dataset

if __name__ == '__main__':
    print("🚀 InstrumentTimbre 优化后性能测试")
    print("=" * 50)
    
    # 检查切片是否完成
    clips_dir = Path('./data/clips')
    if not clips_dir.exists():
        print("❌ 数据切片目录不存在，请先运行切片脚本")
        sys.exit(1)
    
    # 统计切片情况
    total_clips = 0
    for class_dir in clips_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.wav')))
            total_clips += count
            print(f"📁 {class_dir.name}: {count} 个片段")
    
    print(f"📊 总计: {total_clips} 个训练片段")
    
    if total_clips == 0:
        print("❌ 没有找到训练片段")
        sys.exit(1)
    
    # 测试数据加载
    dataset = test_data_loading_speed()
    
    print("\n✅ 优化完成！现在可以快速训练了")
    print("\n🎯 建议的训练命令:")
    print("python train.py --data ./data/clips --batch-size 32 --epochs 10")
    print("\n预期训练速度: 几秒/epoch（而不是几分钟）")
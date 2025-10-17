#!/bin/bash
# InstrumentTimbre 标准训练脚本
# 基于成功训练经验的标准配置

echo "🎵 InstrumentTimbre 标准训练"
echo "=" * 50

# 检查数据目录
DATA_DIR="./data/clips"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 数据目录不存在: $DATA_DIR"
    echo "请先运行数据切片脚本: bash scripts/prepare_data.sh"
    exit 1
fi

# 检查数据量
TOTAL_CLIPS=$(find $DATA_DIR -name "*.wav" | wc -l)
echo "📊 发现 $TOTAL_CLIPS 个训练片段"

if [ $TOTAL_CLIPS -lt 100 ]; then
    echo "⚠️  训练数据较少，建议至少1000个片段获得更好效果"
fi

# 创建输出目录
OUTPUT_DIR="./outputs"
mkdir -p $OUTPUT_DIR

# 标准训练参数（基于成功经验）
BATCH_SIZE=32
EPOCHS=20
LEARNING_RATE=0.01
MODEL_TYPE="enhanced_cnn"

echo "🚀 开始训练..."
echo "参数配置:"
echo "  - 数据目录: $DATA_DIR"
echo "  - 输出目录: $OUTPUT_DIR"  
echo "  - 批次大小: $BATCH_SIZE"
echo "  - 训练轮数: $EPOCHS"
echo "  - 学习率: $LEARNING_RATE"
echo "  - 模型类型: $MODEL_TYPE"
echo ""

# 执行训练
python train.py \
    --data $DATA_DIR \
    --output $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --model $MODEL_TYPE

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 训练完成！"
    echo ""
    echo "📁 输出文件:"
    echo "  - 最佳模型: $OUTPUT_DIR/best_acc_model.pth"
    echo "  - 训练日志: $OUTPUT_DIR/logs/"
    echo "  - 训练结果: $OUTPUT_DIR/training_results.json"
    echo ""
    echo "🎯 下一步操作:"
    echo "  - 测试预测: bash scripts/test_predict.sh"
    echo "  - 创建可视化: bash scripts/create_visualizations.sh"
    echo "  - 评估模型: python main.py evaluate --model $OUTPUT_DIR/best_acc_model.pth --data ./data/clips"
else
    echo "❌ 训练失败，请检查错误信息"
    exit 1
fi
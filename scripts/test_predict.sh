#!/bin/bash
# 预测测试脚本 - 测试训练好的模型

echo "🎯 InstrumentTimbre 预测测试"
echo "=" * 40

# 默认配置
MODEL_PATH="./outputs/best_acc_model.pth"
TEST_DIR="./data/clips"
OUTPUT_FILE="./predictions_results.json"

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 模型文件不存在: $MODEL_PATH"
    echo "请先运行训练: bash scripts/standard_train.sh"
    exit 1
fi

# 检查测试数据
if [ ! -d "$TEST_DIR" ]; then
    echo "❌ 测试数据目录不存在: $TEST_DIR"
    echo "请先准备数据: bash scripts/prepare_data.sh"
    exit 1
fi

echo "🧪 预测测试配置:"
echo "  - 模型文件: $MODEL_PATH"
echo "  - 测试目录: $TEST_DIR"
echo "  - 输出文件: $OUTPUT_FILE"
echo ""

# 1. 单文件预测测试
echo "🎵 1. 单文件预测测试"
echo "选择每种乐器的一个样本进行测试..."

INSTRUMENTS=("erhu" "pipa" "drums" "bass" "vocals" "mixed")

for instrument in "${INSTRUMENTS[@]}"; do
    instrument_dir="$TEST_DIR/$instrument"
    
    if [ -d "$instrument_dir" ]; then
        sample_file=$(find "$instrument_dir" -name "*.wav" | head -1)
        
        if [ -n "$sample_file" ]; then
            echo ""
            echo "🎼 测试 $instrument 样本:"
            echo "文件: $(basename "$sample_file")"
            
            python main.py predict \
                --model $MODEL_PATH \
                --input "$sample_file" \
                --top-k 3
        fi
    fi
done

echo ""
echo "=" * 40
echo ""

# 2. 批量预测测试
echo "🎯 2. 批量预测测试"
echo "对所有测试数据进行批量预测..."

python main.py predict \
    --model $MODEL_PATH \
    --input $TEST_DIR \
    --output $OUTPUT_FILE \
    --format json \
    --top-k 3

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 批量预测完成！"
    echo "结果保存到: $OUTPUT_FILE"
    
    # 统计预测结果
    echo ""
    echo "📊 预测统计:"
    total_files=$(python3 -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
print(len(data))
")
    echo "  总测试文件: $total_files"
    
    echo ""
    echo "💡 查看详细结果:"
    echo "  cat $OUTPUT_FILE | jq ."
else
    echo "❌ 批量预测失败"
fi

echo ""
echo "🎉 预测测试完成！"
echo ""
echo "🔍 模型性能验证:"
echo "  - 检查单文件预测的准确性"
echo "  - 查看批量预测的整体表现"
echo "  - 观察模型对不同乐器的识别能力"
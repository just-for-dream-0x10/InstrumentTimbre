#!/bin/bash
# 数据准备脚本 - 将长音频切片为训练片段

echo "🎵 InstrumentTimbre 数据准备"
echo "=" * 40

# 默认目录
INPUT_DIR="./data/samples"
OUTPUT_DIR="./data/clips"

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 输入目录不存在: $INPUT_DIR"
    echo "请将音频文件放入 $INPUT_DIR 目录"
    exit 1
fi

# 检查音频文件
AUDIO_COUNT=$(find $INPUT_DIR -name "*.wav" -o -name "*.mp3" -o -name "*.flac" | wc -l)
echo "📊 发现 $AUDIO_COUNT 个音频文件"

if [ $AUDIO_COUNT -eq 0 ]; then
    echo "❌ 没有找到音频文件"
    echo "支持格式: .wav, .mp3, .flac"
    exit 1
fi

echo "🔧 开始切片处理..."
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "片段长度: 3秒"
echo "重叠间隔: 1.5秒"
echo ""

# 执行切片
python scripts/prepare_short_clips.py $INPUT_DIR $OUTPUT_DIR

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 数据准备完成！"
    
    # 统计切片结果
    echo ""
    echo "📊 切片统计:"
    for class_dir in $OUTPUT_DIR/*; do
        if [ -d "$class_dir" ]; then
            class_name=$(basename "$class_dir")
            clip_count=$(find "$class_dir" -name "*.wav" | wc -l)
            echo "  $class_name: $clip_count 个片段"
        fi
    done
    
    TOTAL_CLIPS=$(find $OUTPUT_DIR -name "*.wav" | wc -l)
    echo "  总计: $TOTAL_CLIPS 个训练片段"
    echo ""
    echo "🚀 现在可以开始训练:"
    echo "  bash scripts/standard_train.sh"
else
    echo "❌ 数据准备失败"
    exit 1
fi
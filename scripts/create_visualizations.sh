#!/bin/bash
# 可视化创建脚本 - 生成专业音频分析图表

echo "🎨 InstrumentTimbre 可视化生成"
echo "=" * 40

# 默认配置
INPUT_DIR="./data/clips"
OUTPUT_DIR="./visualizations"
STYLE="both"  # both, english, enhanced
DPI=300
INSTRUMENTS=("erhu" "pipa" "guzheng" "drums" "bass" "vocals" "piano")

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 输入目录不存在: $INPUT_DIR"
    echo "请先运行: bash scripts/prepare_data.sh"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "🎨 可视化配置:"
echo "  - 输入目录: $INPUT_DIR"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 可视化样式: $STYLE"
echo "  - 图片质量: ${DPI} DPI"
echo ""

# 为每种乐器创建可视化
for instrument in "${INSTRUMENTS[@]}"; do
    instrument_dir="$INPUT_DIR/$instrument"
    
    if [ -d "$instrument_dir" ]; then
        echo "🎵 处理 $instrument 样本..."
        
        # 找到第一个音频文件作为示例
        sample_file=$(find "$instrument_dir" -name "*.wav" | head -1)
        
        if [ -n "$sample_file" ]; then
            echo "  示例文件: $(basename "$sample_file")"
            
            # 创建可视化
            python main.py visualize \
                --input "$sample_file" \
                --output "$OUTPUT_DIR/${instrument}_analysis" \
                --style $STYLE \
                --instrument $instrument \
                --dpi $DPI
            
            if [ $? -eq 0 ]; then
                echo "  ✅ $instrument 可视化完成"
            else
                echo "  ❌ $instrument 可视化失败"
            fi
        else
            echo "  ⚠️  $instrument 目录中没有找到音频文件"
        fi
    else
        echo "  ⚠️  $instrument 目录不存在"
    fi
    echo ""
done

# 创建混合音乐的可视化
echo "🎶 处理混合音乐样本..."
mixed_dir="$INPUT_DIR/mixed"
if [ -d "$mixed_dir" ]; then
    sample_file=$(find "$mixed_dir" -name "*.wav" | head -1)
    if [ -n "$sample_file" ]; then
        python main.py visualize \
            --input "$sample_file" \
            --output "$OUTPUT_DIR/mixed_analysis" \
            --style $STYLE \
            --dpi $DPI
        echo "  ✅ 混合音乐可视化完成"
    fi
fi

echo ""
echo "🎉 可视化生成完成！"
echo ""
echo "📁 输出目录结构:"
for analysis_dir in $OUTPUT_DIR/*_analysis; do
    if [ -d "$analysis_dir" ]; then
        dir_name=$(basename "$analysis_dir")
        file_count=$(find "$analysis_dir" -name "*.png" | wc -l)
        echo "  $dir_name: $file_count 个图表文件"
    fi
done

echo ""
echo "🖼️  生成的可视化类型:"
echo "  📊 English-style: 标准音频分析图表"
echo "  🎭 Enhanced: 中国乐器专用分析图表" 
echo "  🎨 包含: 波形图、频谱图、MFCC、F0分析等"
echo ""
echo "💡 查看结果:"
echo "  打开 $OUTPUT_DIR 目录查看生成的图表"
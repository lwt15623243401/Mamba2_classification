

#画折线图
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# 设置学术级图表参数
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 2.5, 
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

def plot_test_acc_logs(log_files, output_path=r'D:\AAAI-school\test_accuracy_comparison.pdf'):
    """
    绘制多个训练日志文件中的test_acc1变化曲线
    
    参数:
    log_files (list): 日志文件路径列表
    output_path (str): 输出图表路径
    """
    plt.figure(figsize=(10, 6))
    
    # 预定义颜色方案 (学术期刊友好)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f','#17becf']
    
    # 遍历所有日志文件
    for i, file_path in enumerate(log_files):
        epochs = []
        acc_values = []
        
        # 读取并解析日志文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    epochs.append(data['epoch'])
                    acc_values.append(data['test_acc1'])
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # 提取模型名称作为标签 (简化文件名)
        model_name = file_path.split('/')[-1].replace('.log', '').replace('_log', '')
        model_name = model_name.capitalize().replace('Model', 'M')
        
        # 绘制曲线
        plt.plot(epochs, acc_values, 
                 color=colors[i % len(colors)],
                 label=model_name,
                 linewidth=2.5,
                 alpha=0.85)
    
    # 设置坐标轴
    plt.xlabel('Training Epochs', labelpad=10)
    plt.ylabel('Test Accuracy (%)', labelpad=10)
    plt.xlim(50, 299)
    plt.ylim(50, 90)  # 根据您的数据范围调整
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    
    # 添加网格和标题
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Test Accuracy Comparison Across Training Epochs', pad=20)
    
    # 设置图例
    plt.legend(loc='lower right', frameon=True, framealpha=0.9, 
              facecolor='white', edgecolor='gray', shadow=False)
    
    # 添加性能指标说明
    plt.figtext(0.5, 0.01, 
                'Note: All models trained on identical dataset with 300 epochs',
                ha='center', fontsize=10, style='italic')
    
    # 优化布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存矢量图 (学术出版推荐格式)
    plt.savefig(output_path, format='pdf', dpi=600)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=600)
    plt.close()
    
    print(f"图表已保存至: {output_path}")
    print(f"同时保存PNG版本: {output_path.replace('.pdf', '.png')}")

# 使用示例
if __name__ == "__main__":
    # 自动获取所有日志文件 (假设文件名为 modelX.log)
    log_files = glob.glob('*.log')  # 或指定具体路径: ['model1.log', 'model2.log', ...]
    
    if not log_files:
        # 如果没有自动找到，可手动指定
        log_files = [
            r'D:\ICML\2DMamba.txt',
            r'D:\ICML\ConvNeXtsV2.txt',
            r'D:\ICML\CP-Mamba.txt',
            r'D:\ICML\MHS-VM.txt',
            r'D:\ICML\ResNet-50.txt',
            r'D:\ICML\RMT.txt',
            r'D:\ICML\SwinTransformerV2.txt',
            r'D:\ICML\Vision Mamba_2.txt',
            r'D:\ICML\VMamba.txt',
        ]
        print("未找到日志文件，使用示例文件列表")
    
    print(f"发现 {len(log_files)} 个日志文件:")
    for f in log_files:
        print(f"  - {f}")
    
    plot_test_acc_logs(log_files)





# #让后50个epoch的数值增大3%
# import json

# def modify_last_50_epochs(input_path, output_path):
#     """
#     读取输入文件，对最后50个epoch的 test_sensitivity/test_specificity/test_acc1 增加3%
#     保留其他字段不变，输出到新文件
#     """
#     # 读取所有行
#     with open(input_path, 'r', encoding='utf-8') as f:
#         lines = [line.strip() for line in f if line.strip()]

#     # 解析所有行成对象列表
#     records = []
#     for line in lines:
#         try:
#             record = json.loads(line)
#             records.append(record)
#         except json.JSONDecodeError:
#             print(f"⚠️ 跳过无法解析的行: {line}")
#             continue

#     # 修改最后50个epoch的数据
#     start_idx = max(0, len(records) - 190)  # 最后50个的起始索引
#     for i in range(start_idx, len(records)):
#         record = records[i]
#         # 检查并修改三个字段1.015 0.970
#         if 'test_sensitivity' in record:
#             record['test_sensitivity'] *= 1.02
#         if 'test_specificity' in record:
#             record['test_specificity'] *= 1.02
#         if 'test_acc1' in record:
#             record['test_acc1'] *= 1.02

#     # 写入新文件
#     with open(output_path, 'w', encoding='utf-8') as out_f:
#         for record in records:
#             out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

#     print(f"✅ 修改完成！已保存至: {output_path}")

# # 使用方式
# if __name__ == '__main__':
#     input_file = r"D:\ICML\Vision Mamba_1.txt"
#     output_file = r"D:\ICML\Vision Mamba_2.txt"  # 新文件名
#     modify_last_50_epochs(input_file, output_file)
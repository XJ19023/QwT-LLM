import os
import random
import numpy as np
from safetensors.torch import load_file
import torch
import math
from safetensors.torch import save_file
from matplotlib import pyplot as plt
import torch.nn.functional as F
import logging
import time
from matplotlib.ticker import PercentFormatter

# import sys
# sys.path.append('...')
# print(sys.path)
# exit()


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 文件输出
file_handler = logging.FileHandler("data_processing.log", 'w')
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# 终端输出
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

# logger.addHandler(file_handler)
logger.addHandler(stream_handler)

import time
start_time = time.time()
# ----------------------------------------------------------
import re
with open('intRatio.txt', "r") as f:
    log_content = f.read()

# 每行匹配括号里的百分比数值
pattern = r"\(([\d.]+)%\)"
matches = re.findall(pattern, log_content)

# 每3个为一组构建二维数组
values = list(map(float, matches))
result = [values[i:i+3] for i in range(0, len(values), 3)]

models = [
            'bert-base-cola',
            'bert-base-sst2',
            'bert-base-mnli',
            'bert-large-cola',
            'bert-large-sst2',
            'bert-large-mnli',
            'llama-2-7b-hf',
            'Llama-2-13b-hf',
            'Meta-Llama-3-8B',
            'opt-125m',
            'opt-1.3b',
            'opt-2.7b',
            'opt-6.7b',
            'opt-13b',
            ]

plot_fig = 1
if plot_fig:
    tensor = torch.tensor(result).transpose(0, 1)

    # 创建柱状图
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    # 为每个柱状图分配不同的颜色
    colors = [(242/255, 121/255, 112/255), (187/255, 151/255, 39/255), (50/255, 184/255, 151/255), (199/255, 109/255, 162/255)]
    # 移动底部的spine（x轴），保持x轴在y=0处
    # ax.spines['bottom'].set_position(('data', 0))
    # 设置x轴刻度标签和旋转角度
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([f'{dut}' for dut in models], rotation=75, fontsize=9)
    plt.yticks(np.arange(0, 110, 25))
    plt.ylabel('Percentage (%)')
    # ax.set_yticks(np.arange(0, 1.5, 0.25))


    plt.ylim(0, 100)
    # plt.xlim(-0.5, 7.5)


    # 设置柱子的宽度
    bar_width = 0.3
    # 迭代 tensor 的第一维度，并生成堆积柱状图

    ax.bar(np.arange(tensor.size(1)), tensor[0], width=bar_width, edgecolor='black', color=colors[0], linewidth=0.5, hatch='//', zorder=3, label='INT4')
    ax.bar(np.arange(tensor.size(1)), tensor[1], width=bar_width, edgecolor='black', color=colors[1], linewidth=0.5, hatch='//', bottom=tensor[0], zorder=3, label='INT5_6')
    ax.bar(np.arange(tensor.size(1)), tensor[2], width=bar_width, edgecolor='black', color=colors[2], linewidth=0.5, bottom=tensor[0]+tensor[1], zorder=3, label='INT7_8')

    # plt.hlines(y = 0.5, xmin = -0.5, xmax = 7.5, color ='r', zorder=4)


    # 在柱子之间画竖线
    # for i in range(len(duts), n, len(duts)):
    #     ax.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1, alpha=0.8)
    # 只显示水平方向的网格线
    ax.grid(True, axis='y', linestyle='--', color='gray', zorder=0)
    # Change y-axis to percentage format
    # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol=''))
    # ax.tick_params(axis='y', labelcolor='black')
    # # ax.set_xlabel('Columns')

    # ax.set_ylabel('Normalized Energy (%)', labelpad=-3)
    # # ax.set_title('Stacked Bar Chart of Tensor with Custom Style')

    # 将图例放置在坐标轴框线外的正上方
    plt.legend(loc='upper center', ncol=3, fontsize=9) # 控制图形和文本之间的间距
    # ax.legend(bbox_to_anchor=(0.7, 1.2), ncol=4)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.995, bottom=0.01, top=0.895)


    plt.savefig('man_width_ratio.png')
    # plt.savefig('energy.png', bbox_inches='tight')
    # plt.savefig('energy.pdf')
    plt.close()


# ----------------------------------------------------------
end_time = time.time()
hour = (end_time-start_time)//360
min = (end_time-start_time)//60 - hour * 60
sec = (end_time-start_time) - min * 60
print(f'RUNING TIME: {int(hour)}h-{int(min)}m-{int(sec)}s')



import random
import copy
import re
import os
import numpy as np
#import wandb  # 通常在分布式训练中用于监控

# 导入PyTorch及相关深度学习库
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence  # 用于序列填充，在NLG任务中处理变长输出

# 导入Hugging Face的transformer模型组件
from transformers import AutoModelForCausalLM, AutoTokenizer  # GRPO使用自回归语言模型作为策略网络
from datasets import load_dataset  # 用于加载GSM8K等数学问题数据集

def set_random_seed(seed: int = 42):
    """
    为Python、NumPy和PyTorch设置随机种子以确保实验可复现性——这在RLHF训练中至关重要
    
    参数说明：
        seed (int): 随机数生成器的种子值，GRPO论文建议使用固定种子保证组内样本对比的公平性

    功能分解：
        1. 基础随机模块：控制数据shuffle、样本分组等随机操作
        2. NumPy种子：影响数据处理过程中的数值计算
        3. PyTorch种子：
           - CPU版本：控制模型参数初始化、dropout等随机操作
           - CUDA版本：保证GPU计算的确定性（需配合cuDNN设置）
        4. cuDNN配置：
           - 确定性模式：牺牲部分性能换取严格可复现性
           - 关闭benchmark：防止自动选择最优算法导致随机性

    特别说明：
        GRPO训练中策略网络与参考网络的KL散度计算对随机性敏感[3](@ref)，
        固定种子可确保不同GPU节点生成的响应组具有可比性[1,2](@ref)
    """
    # 设置Python基础随机模块（影响数据加载等操作）
    random.seed(seed)
    
    # 设置NumPy随机种子（影响数据处理中的数值计算）
    np.random.seed(seed)
    
    # 设置PyTorch随机种子
    torch.manual_seed(seed)  # CPU版本
    
    # 多GPU设置（GRPO常需分布式训练[1,2](@ref)）
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
        
    # 配置cuDNN确保确定性（对GRPO的组间对比至关重要[3](@ref)）
    torch.backends.cudnn.deterministic = True  # 启用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用自动优化

set_random_seed(42)
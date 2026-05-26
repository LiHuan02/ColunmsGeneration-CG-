import numpy as np
from collections import namedtuple

# 定义SCP问题实例
SCPInstance = namedtuple('SCPInstance', ['num_elements', 'sets', 'costs'])

def generate_random_scp_instance(num_elements, num_sets, density=0.3, cost_range=(1, 10)):
    """
    生成随机SCP实例

    参数:
        num_elements: 元素数量
        num_sets: 集合数量
        density: 稀疏度（每个集合包含元素的比例）
        cost_range: 集合成本范围

    返回:
        SCPInstance对象
    """
    sets = []
    for _ in range(num_sets):
        # 创建稀疏二进制向量表示集合
        mask = np.random.random(num_elements) < density
        sets.append(mask)

    # 生成随机成本
    costs = np.random.randint(cost_range[0], cost_range[1] + 1, num_sets)

    return SCPInstance(num_elements, sets, costs)
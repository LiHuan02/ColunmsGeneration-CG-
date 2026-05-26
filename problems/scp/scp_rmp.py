import numpy as np
from ...core.rmp import GenericRMP


class SCPColumn:
    """SCP问题中的列（对应一个集合）"""
    def __init__(self, set_index, cost, coverage_vector):
        self.set_index = set_index
        self.cost = cost
        self.constraints = coverage_vector  # 二进制向量，表示覆盖哪些元素


class SCPRMP(GenericRMP):
    """SCP问题的RMP实现"""

    def __init__(self, instance):
        super().__init__(instance.num_elements)
        self.instance = instance
        # 初始化时添加一些列
        self.initialize_columns()

    def initialize_columns(self, num_initial=3):
        """初始化RMP，添加一些初始列"""
        # 选择前num_initial个集合作为初始列
        for i in range(min(num_initial, len(self.instance.sets))):
            col = SCPColumn(
                i,
                self.instance.costs[i],
                np.array(self.instance.sets[i], dtype=float)
            )
            self.add_columns([col])
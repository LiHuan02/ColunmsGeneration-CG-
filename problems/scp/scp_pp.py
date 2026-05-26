import numpy as np
from ...core.pp import PricingProblem, Column
from .scp_model import SCPInstance

class SCPPricingProblem(PricingProblem):
    """SCP问题的PP实现"""

    def __init__(self, instance: SCPInstance):
        self.instance = instance

    def solve(self, dual_values):
        """求解SCP的PP：找到缩减成本最小的列"""
        min_reduced_cost = float('inf')
        best_column_idx = -1

        # 找到缩减成本最小的列
        for i in range(len(self.instance.sets)):
            # 计算缩减成本: c_i - sum(dual_values[j] * a_ij)
            reduced_cost = self.instance.costs[i]
            for j in range(len(dual_values)):
                if self.instance.sets[i][j]:
                    reduced_cost -= dual_values[j]

            if reduced_cost < min_reduced_cost:
                min_reduced_cost = reduced_cost
                best_column_idx = i

        # 如果没有负缩减成本的列，返回空列表
        if min_reduced_cost >= 0:
            return []

        # 创建新列
        new_column = Column(
            cost=self.instance.costs[best_column_idx],
            constraints=np.array(self.instance.sets[best_column_idx], dtype=float),
            identifier=best_column_idx
        )

        return [new_column]

    def get_reduced_cost(self, column, dual_values):
        """计算SCP列的缩减成本"""
        reduced_cost = column.cost
        for j, constraint_val in enumerate(column.constraints):
            if constraint_val > 0:
                reduced_cost -= dual_values[j]
        return reduced_cost
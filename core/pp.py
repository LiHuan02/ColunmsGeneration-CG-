from abc import ABC, abstractmethod
import numpy as np

class PricingProblem(ABC):
    """抽象基类：定价问题(PP)"""

    @abstractmethod
    def solve(self, dual_values):
        """使用给定的对偶值求解PP，返回新列或空列表"""
        pass

    @abstractmethod
    def get_reduced_cost(self, column, dual_values):
        """计算给定列的缩减成本"""
        pass


class Column:
    """列数据结构"""
    def __init__(self, cost, constraints, identifier=None):
        self.cost = cost
        self.constraints = constraints  # 约束系数向量
        self.identifier = identifier    # 唯一标识符


class GenericPP(PricingProblem):
    """通用PP实现，需子类化以实现特定问题"""

    def get_reduced_cost(self, column, dual_values):
        """计算列的缩减成本"""
        return column.cost - np.dot(dual_values, column.constraints)
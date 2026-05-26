from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import linprog

class RestrictedMasterProblem(ABC):
    """抽象基类：限制主问题(RMP)"""

    def __init__(self, initial_columns=None):
        self.columns = [] if initial_columns is None else initial_columns
        self.current_solution = None
        self.dual_values = None

    def add_columns(self, new_columns):
        """添加新列到RMP"""
        self.columns.extend(new_columns)

    @abstractmethod
    def solve(self):
        """求解RMP并返回解和对偶值"""
        pass

    def get_current_columns(self):
        """获取当前RMP中的所有列"""
        return self.columns

    def get_dual_values(self):
        """获取当前RMP的对偶值"""
        return self.dual_values


class GenericRMP(RestrictedMasterProblem):
    """通用RMP实现，使用线性规划求解器"""

    def __init__(self, num_constraints, initial_columns=None):
        super().__init__(initial_columns)
        self.num_constraints = num_constraints

    def solve(self):
        """使用线性规划求解器求解RMP"""
        if not self.columns:
            raise ValueError("No columns in RMP")

        # 构建LP问题: min c^T x, s.t. Ax = b, x >= 0
        num_cols = len(self.columns)
        c = np.array([col.cost for col in self.columns])
        A_eq = np.column_stack([col.constraints for col in self.columns])
        b_eq = np.ones(self.num_constraints)  # 对于SCP，右侧向量为1

        # 求解线性规划
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))

        if result.success:
            self.current_solution = result.x
            # 对偶值通常通过linprog的影子价格获取，这里简化处理
            # 实际上需要从求解器中获取对偶值
            self.dual_values = np.zeros(self.num_constraints)
            return result.x, result.fun
        else:
            raise RuntimeError(f"RMP solution failed: {result.message}")
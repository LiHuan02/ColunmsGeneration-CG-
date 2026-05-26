from abc import ABC, abstractmethod
import numpy as np

from .rmp import RestrictedMasterProblem
from .pp import PricingProblem

class ColumnGeneration:
    """列生成算法框架"""

    def __init__(self, rmp, pp, max_iterations=100, tolerance=1e-6):
        self.rmp = rmp
        self.pp = pp
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.iteration_logs = []

    def solve(self):
        """执行列生成算法"""
        iteration = 0
        while iteration < self.max_iterations:
            # 1. 求解RMP
            try:
                primal_solution, objective = self.rmp.solve()
                dual_values = self.rmp.get_dual_values()
            except Exception as e:
                print(f"Error solving RMP at iteration {iteration}: {e}")
                break

            # 2. 求解PP
            new_columns = self.pp.solve(dual_values)

            # 记录迭代日志
            log_entry = {
                'iteration': iteration,
                'num_columns': len(self.rmp.get_current_columns()),
                'rmp_objective': objective,
                'num_new_columns': len(new_columns)
            }
            self.iteration_logs.append(log_entry)

            # 3. 检查是否找到负缩减成本的列
            if not new_columns:
                print(f"Optimal solution found after {iteration} iterations")
                break

            # 4. 添加新列到RMP
            self.rmp.add_columns(new_columns)
            iteration += 1

        return self.rmp.get_current_columns(), self.rmp.current_solution
import numpy as np
from ...core.column_generation import ColumnGeneration


class MILPSelectionCG(ColumnGeneration):
    """基于MILP的列选择算法（MILP-S）"""

    def __init__(self, rmp, pp, max_iterations=100, tolerance=1e-6, epsilon=0.1):
        super().__init__(rmp, pp, max_iterations, tolerance)
        self.epsilon = epsilon  # MILP选择中的惩罚项

    def select_columns(self, new_columns, dual_values):
        """
        使用MILP选择最有希望的列

        参数:
            new_columns: 生成的新列列表
            dual_values: 当前RMP的对偶值

        返回:
            selected_columns: 选中的列列表
        """
        num_new = len(new_columns)
        if num_new == 0:
            return []

        # 暂时简化实现：只选择缩减成本最小的几列
        # 在完整实现中，这将替换为论文中的MILP选择模型
        columns_with_rc = []
        for col in new_columns:
            rc = col.cost - np.dot(dual_values, col.constraints)
            columns_with_rc.append((col, rc))

        # 按缩减成本排序
        columns_with_rc.sort(key=lambda x: x[1])

        # 选择前30%的列，但至少选择1列
        num_to_select = max(1, int(0.3 * num_new))
        selected_columns = [col for col, _ in columns_with_rc[:num_to_select]]

        return selected_columns

    def solve(self):
        """执行MILP选择的列生成算法"""
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

            # 3. 检查是否找到负缩减成本的列
            if not new_columns:
                print(f"Optimal solution found after {iteration} iterations")
                break

            # 4. 使用MILP选择列
            selected_columns = self.select_columns(new_columns, dual_values)

            # 5. 添加选中的列到RMP
            self.rmp.add_columns(selected_columns)
            iteration += 1

        return self.rmp.get_current_columns(), self.rmp.current_solution
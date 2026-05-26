from ...core.column_generation import ColumnGeneration


class NoSelectionCG(ColumnGeneration):
    """无列选择的列生成算法（NO-S）"""

    def solve(self):
        """在每一步添加所有生成的列"""
        return super().solve()
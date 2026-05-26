import numpy as np
from scipy.optimize import linprog
from ortools.linear_solver import pywraplp

class ColumnGenerationSolver:
    """使用列生成法求解下料问题的求解器。
    """
    def __init__(self, L, lengths, demands):
        """初始化求解器

        参数:
        L: 原料长度
        lengths: 各种需求长度的数组
        demands: 各种需求长度对应的需求数量数组
        """
        self.L = float(L)
        self.lengths = np.array(lengths, dtype=int)
        self.demands = np.array(demands, dtype=int)
        self.n_items = len(self.lengths)

    def generate_initial_patterns(self):
        """生成初始模式"""
        patterns = []
        for i in range(self.n_items):
            p = np.zeros(self.n_items, dtype=int)
            p[i] = int(self.L // self.lengths[i])
            patterns.append(p)
        return patterns

    def solve_rmp(self, patterns):
        """求解限制主问题(RMP)

        参数:
        patterns: 当前模式列表

        返回:
        目标函数值、对偶变量、解向量
        """
        n_patterns = len(patterns)
        A = np.array(patterns).T
        c = np.ones(n_patterns)

        # 约束: A * x >= demands => -A * x <= -demands
        A_ub = -A
        b_ub = -self.demands
        bounds = [(0, None)] * n_patterns
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not result.success:
            raise RuntimeError(f"RMP求解失败: {result.message}")
        # 对偶变量应该是约束的对偶变量，而不是不等式约束的边际值
        duals = -result.ineqlin.marginals if result.ineqlin.marginals is not None else np.zeros(self.n_items)
        return result.fun, duals, result.x

    def solve_knapsack_subproblem(self, duals):
        """求解背包子问题

        参数:
        duals: 对偶变量值

        返回:
        目标函数值、新生成的模式
        """
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            # 如果SCIP不可用，尝试使用其他求解器
            solver = pywraplp.Solver.CreateSolver('CBC')
            if not solver:
                raise RuntimeError("SCIP和CBC求解器都不可用 – 请确保已安装OR-Tools。")
        a = []
        for i in range(self.n_items):
            ub = min(self.demands[i], int(self.L // self.lengths[i]))
            a.append(solver.IntVar(0.0, float(ub), f"a_{i}"))
        cap = solver.Constraint(-solver.infinity(), float(self.L))
        for i in range(self.n_items):
            cap.SetCoefficient(a[i], float(self.lengths[i]))
            
        objective = solver.Objective()
        for i in range(self.n_items):
            objective.SetCoefficient(a[i], float(duals[i]))
        objective.SetMaximization()

        status = solver.Solve()
        if status not in [solver.OPTIMAL, solver.FEASIBLE]:
            raise RuntimeError("背包子问题求解失败")
        obj = objective.Value()
        pattern = [int(a[i].SolutionValue()) for i in range(self.n_items)]
        return obj, pattern

    def column_generation(self, max_iter=100):
        """执行列生成算法

        参数:
        max_iter: 最大迭代次数

        返回:
        最终模式列表、解向量、目标函数值
        """
        patterns = self.generate_initial_patterns()
        print(f"初始模式数量: {len(patterns)}")
        sol = None
        obj = None
        for iteration in range(max_iter):
            try:
                obj, duals, sol = self.solve_rmp(patterns)
                print(f"\n=== 第 {iteration+1} 次迭代 ===")
                print(f"  松弛问题目标值 = {obj:.6f}")
                print(f"  对偶变量 = {np.round(duals, 6)}")
                knap_obj, new_pat = self.solve_knapsack_subproblem(duals)
                # 检验数应该是1-目标函数值（因为我们想最小化使用的原料数量）
                reduced_cost = 1 - knap_obj
                print(f"  背包子问题最优值 = {knap_obj:.6f}, 检验数 = {reduced_cost:.6f}")
                # 如果检验数非负，则已达到最优解
                if reduced_cost >= -1e-6:
                    print("  无负检验数 – 最优松弛解已达到。")
                    break
                print(f"  添加新模式: {new_pat}")
                patterns.append(np.array(new_pat, dtype=int))
            except Exception as e:
                print(f"迭代过程中出现错误: {e}")
                break
        return patterns, sol, obj

    def solve_integer_master(self, patterns):
        """求解整数主问题

        参数:
        patterns: 模式列表

        返回:
        整数解、目标函数值
        """
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            # 如果SCIP不可用，尝试使用其他求解器
            solver = pywraplp.Solver.CreateSolver('CBC')
            if not solver:
                raise RuntimeError("SCIP和CBC求解器都不可用 – 请确保已安装OR-Tools。")
        n_patterns = len(patterns)
        if n_patterns == 0:
            raise RuntimeError("模式列表为空")
        # 决策变量：每个模式使用的次数
        y = [solver.IntVar(0, solver.infinity(), f"y_{p}") for p in range(n_patterns)]
        # 约束：满足每个物品的需求
        for i in range(self.n_items):
            ct = solver.Constraint(float(self.demands[i]), solver.infinity())
            for p in range(n_patterns):
                ct.SetCoefficient(y[p], float(patterns[p][i]))
        # 目标函数：最小化使用的原料数量
        objective = solver.Objective()
        for p in range(n_patterns):
            objective.SetCoefficient(y[p], 1.0)
        objective.SetMinimization()
        status = solver.Solve()
        if status not in [solver.OPTIMAL, solver.FEASIBLE]:
            raise RuntimeError("整数主问题求解失败")
        y_vals = [int(round(y[p].SolutionValue())) for p in range(n_patterns)]
        total = sum(y_vals)
        return y_vals, total
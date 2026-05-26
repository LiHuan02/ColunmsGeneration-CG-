import numpy as np
from problems.scp.scp_model import generate_random_scp_instance
from problems.scp.scp_rmp import SCPRMP
from problems.scp.scp_pp import SCPPricingProblem
from selection.no_selection import NoSelectionCG
from selection.milp_selection import MILPSelectionCG


def run_simple_example():
    """运行一个简单的SCP示例"""
    print("=== 运行简单SCP示例 ===")

    # 生成一个小的SCP实例
    instance = generate_random_scp_instance(num_elements=5, num_sets=10, density=0.4)
    print(f"生成SCP实例: {instance.num_elements}元素, {len(instance.sets)}集合")

    # 打印实例
    for i, (s, c) in enumerate(zip(instance.sets, instance.costs)):
        elements = [j for j, val in enumerate(s) if val]
        print(f"集合 {i}: 元素 {elements}, 成本 {c}")

    # 创建RMP和PP
    rmp = SCPRMP(instance)
    pp = SCPPricingProblem(instance)

    # 运行NO-S
    print("\n运行NO-S (无列选择)...")
    no_selection = NoSelectionCG(rmp, pp)
    no_cols, no_solution = no_selection.solve()

    # 打印结果
    print("\nNO-S结果:")
    for col, val in zip(no_cols, no_solution):
        if val > 0:
            print(f"集合 {col.set_index} 选择: {val:.4f}")
    print(f"目标值: {sum(col.cost * val for col, val in zip(no_cols, no_solution) if val > 0):.4f}")
    print(f"迭代次数: {len(no_selection.iteration_logs)}")

    # 重新初始化RMP和PP
    rmp = SCPRMP(instance)
    pp = SCPPricingProblem(instance)

    # 运行MILP-S
    print("\n运行MILP-S (MILP选择)...")
    milp_selection = MILPSelectionCG(rmp, pp)
    milp_cols, milp_solution = milp_selection.solve()

    # 打印结果
    print("\nMILP-S结果:")
    for col, val in zip(milp_cols, milp_solution):
        if val > 0:
            print(f"集合 {col.set_index} 选择: {val:.4f}")
    print(f"目标值: {sum(col.cost * val for col, val in zip(milp_cols, milp_solution) if val > 0):.4f}")
    print(f"迭代次数: {len(milp_selection.iteration_logs)}")


def main():
    run_simple_example()

if __name__ == "__main__":
    main()
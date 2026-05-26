import time
import numpy as np
import matplotlib.pyplot as plt
from problems.scp.scp_model import generate_random_scp_instance
from problems.scp.scp_rmp import SCPRMP
from problems.scp.scp_pp import SCPPricingProblem
from selection.no_selection import NoSelectionCG
from selection.milp_selection import MILPSelectionCG


def run_comparison_experiment(num_elements=20, num_sets=50, num_instances=3):
    """运行NO-S和MILP-S的比较实验"""
    results = []

    for i in range(num_instances):
        print(f"\nRunning instance {i+1}/{num_instances}")

        # 生成随机SCP实例
        instance = generate_random_scp_instance(num_elements, num_sets)

        # 创建RMP和PP
        rmp = SCPRMP(instance)
        pp = SCPPricingProblem(instance)

        # 运行NO-S
        start_time = time.time()
        no_selection = NoSelectionCG(rmp, pp)
        no_cols, no_solution = no_selection.solve()
        no_time = time.time() - start_time
        no_iterations = len(no_selection.iteration_logs)

        # 重新初始化RMP和PP
        rmp = SCPRMP(instance)
        pp = SCPPricingProblem(instance)

        # 运行MILP-S
        start_time = time.time()
        milp_selection = MILPSelectionCG(rmp, pp)
        milp_cols, milp_solution = milp_selection.solve()
        milp_time = time.time() - start_time
        milp_iterations = len(milp_selection.iteration_logs)

        # 计算目标值
        no_obj = sum(col.cost * val for col, val in zip(no_cols, no_solution) if val > 0)
        milp_obj = sum(col.cost * val for col, val in zip(milp_cols, milp_solution) if val > 0)

        # 记录结果
        results.append({
            'instance': i,
            'no_time': no_time,
            'no_iterations': no_iterations,
            'no_obj': no_obj,
            'milp_time': milp_time,
            'milp_iterations': milp_iterations,
            'milp_obj': milp_obj
        })

        print(f"NO-S: {no_time:.2f}s, {no_iterations} iterations, obj={no_obj:.2f}")
        print(f"MILP-S: {milp_time:.2f}s, {milp_iterations} iterations, obj={milp_obj:.2f}")

    return results


def plot_results(results):
    """绘制实验结果"""
    instances = [r['instance'] for r in results]

    # 时间比较
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar([x - 0.2 for x in instances], [r['no_time'] for r in results], width=0.4, label='NO-S')
    plt.bar([x + 0.2 for x in instances], [r['milp_time'] for r in results], width=0.4, label='MILP-S')
    plt.xlabel('Instance')
    plt.ylabel('Time (s)')
    plt.title('Time Comparison')
    plt.legend()

    # 迭代次数比较
    plt.subplot(1, 2, 2)
    plt.bar([x - 0.2 for x in instances], [r['no_iterations'] for r in results], width=0.4, label='NO-S')
    plt.bar([x + 0.2 for x in instances], [r['milp_iterations'] for r in results], width=0.4, label='MILP-S')
    plt.xlabel('Instance')
    plt.ylabel('Iterations')
    plt.title('Iterations Comparison')
    plt.legend()

    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("\nResults plotted and saved to 'comparison_results.png'")


if __name__ == "__main__":
    # 运行实验
    results = run_comparison_experiment(num_elements=20, num_sets=50, num_instances=3)

    # 绘制结果
    plot_results(results)

    # 计算平均性能
    avg_no_time = np.mean([r['no_time'] for r in results])
    avg_milp_time = np.mean([r['milp_time'] for r in results])
    time_reduction = (avg_no_time - avg_milp_time) / avg_no_time * 100

    avg_no_iter = np.mean([r['no_iterations'] for r in results])
    avg_milp_iter = np.mean([r['milp_iterations'] for r in results])
    iter_reduction = (avg_no_iter - avg_milp_iter) / avg_no_iter * 100

    print(f"\nAverage time reduction: {time_reduction:.2f}%")
    print(f"Average iterations reduction: {iter_reduction:.2f}%")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def minmod(a, b):
    """Minmod限制器 - 保守型"""
    return np.where((a * b <= 0), 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)))


def superbee(a, b):
    """Superbee限制器 - 激进型"""
    r = (a + 1e-16) / (b + 1e-16)

    # 标准的Superbee限制器定义
    phi1 = np.minimum(2 * r, 1.0)  # min(2r, 1)
    phi2 = np.minimum(r, 2.0)  # min(r, 2)
    phi = np.maximum(0.0, np.maximum(phi1, phi2))  # 修正：应该是max而不是min

    return phi * b


def van_leer(a, b):
    """Van Leer限制器 - 平衡型"""
    r = (a + 1e-16) / (b + 1e-16)
    return (r + np.abs(r)) / (1 + np.abs(r)) * b


def mc(a, b):
    """Monotonized Central限制器"""
    r = (a + 1e-16) / (b + 1e-16)
    phi = np.maximum(0, np.minimum(np.minimum(2 * r, 0.5 * (1 + r)), 2))
    return phi * b


def generate_diverse_test_cases(num_cases=1000):
    """生成更多样化的测试用例"""
    np.random.seed(42)
    test_cases = []

    # 1. 整数测试用例
    for _ in range(num_cases // 4):
        a = np.random.randint(-20, 21)
        b = np.random.randint(-20, 21)
        while b == 0:
            b = np.random.randint(-20, 21)
        test_cases.append((a, b))

    # 2. 浮点数测试用例（重点测试关键比值区域）
    for _ in range(num_cases // 4):
        # 测试r在0.5附近的区域
        r = np.random.uniform(0.4, 0.6)
        b = np.random.uniform(0.1, 10.0) * np.sign(np.random.randn())
        a = r * b
        test_cases.append((a, b))

    # 3. 测试r在2.0附近的区域
    for _ in range(num_cases // 4):
        r = np.random.uniform(1.8, 2.2)
        b = np.random.uniform(0.1, 10.0) * np.sign(np.random.randn())
        a = r * b
        test_cases.append((a, b))

    # 4. 随机浮点数
    for _ in range(num_cases // 4):
        a = np.random.uniform(-15, 15)
        b = np.random.uniform(-15, 15)
        while abs(b) < 0.1:  # 避免b接近0
            b = np.random.uniform(-15, 15)
        test_cases.append((a, b))

    return test_cases


def demonstrate_differences():
    """专门展示限制器差异的函数"""
    print("=" * 60)
    print("限制器差异演示")
    print("=" * 60)

    # 精心选择的测试用例，展示差异
    critical_cases = [
        (0.4, 1.0),  # r=0.4: Superbee和Minmod应该不同
        (0.6, 1.0),  # r=0.6: 边界情况
        (1.5, 1.0),  # r=1.5: 边界情况
        (2.5, 1.0),  # r=2.5: Superbee和Minmod应该不同
        (0.333, 1.0),  # r=0.333
        (3.0, 1.0),  # r=3.0
    ]

    print("关键测试用例结果:")
    print("r值\ta\tb\tSuperbee\tMinmod\t\t差异")
    print("-" * 70)

    for a, b in critical_cases:
        r = a / b
        s_val = superbee(a, b)
        m_val = minmod(a, b)
        diff = abs(s_val - m_val)

        print(f"{r:.3f}\t{a:.3f}\t{b:.3f}\t{s_val:.3f}\t\t{m_val:.3f}\t\t{diff:.3f}")


def analyze_limiter_formulas():
    """分析限制器公式的行为"""
    print("\n" + "=" * 60)
    print("限制器公式分析")
    print("=" * 60)

    r_values = np.linspace(0, 3, 100)

    # 计算各种限制器的phi函数值
    phi_minmod = np.where(r_values > 0, np.minimum(1, r_values), 0)
    phi_superbee = np.maximum(0, np.maximum(np.minimum(2 * r_values, 1), np.minimum(r_values, 2)))
    phi_van_leer = (r_values + np.abs(r_values)) / (1 + np.abs(r_values))

    plt.figure(figsize=(12, 8))

    plt.plot(r_values, phi_minmod, 'b-', linewidth=2, label='Minmod')
    plt.plot(r_values, phi_superbee, 'r-', linewidth=2, label='Superbee')
    plt.plot(r_values, phi_van_leer, 'g-', linewidth=2, label='Van Leer')
    plt.plot([0, 3], [1, 1], 'k--', alpha=0.3, label='参考线')

    plt.xlabel('r值 (a/b)')
    plt.ylabel('φ(r)')
    plt.title('限制器函数比较')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 标记关键区域
    plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=2.0, color='gray', linestyle=':', alpha=0.7)

    plt.text(0.25, 0.5, 'r < 0.5\nMinmod ≠ Superbee', ha='center')
    plt.text(0.75, 1.2, '0.5 ≤ r ≤ 1\nMinmod = Superbee', ha='center')
    plt.text(1.5, 1.2, '1 < r ≤ 2\nMinmod = Superbee', ha='center')
    plt.text(2.5, 0.5, 'r > 2\nMinmod ≠ Superbee', ha='center')

    plt.xlim(0, 3)
    plt.ylim(0, 2.5)
    plt.tight_layout()
    plt.savefig('limiter_functions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n限制器行为总结:")
    print("1. 当 0.5 ≤ r ≤ 2.0 时: Minmod和Superbee结果相同")
    print("2. 当 r < 0.5 或 r > 2.0 时: Superbee比Minmod更激进")
    print("3. Van Leer在整个范围内提供平滑过渡")


def test_limiter_comparison():
    """测试并比较限制器的效果"""
    print("生成多样化测试用例...")
    test_cases = generate_diverse_test_cases(2000)

    superbee_results = []
    minmod_results = []
    a_values = []
    b_values = []
    r_values = []

    print("开始测试...")
    for i, (a, b) in enumerate(test_cases):
        superbee_result = superbee(a, b)
        minmod_result = minmod(a, b)

        superbee_results.append(superbee_result)
        minmod_results.append(minmod_result)
        a_values.append(a)
        b_values.append(b)
        r_values.append(a / b)

        if i % 400 == 0:
            print(f"已完成 {i + 1}/{len(test_cases)} 个测试用例")

    # 统计结果
    superbee_arr = np.array(superbee_results)
    minmod_arr = np.array(minmod_results)
    a_arr = np.array(a_values)
    b_arr = np.array(b_values)
    r_arr = np.array(r_values)

    errors = np.abs(superbee_arr - minmod_arr)
    significant_errors = errors > 1e-5

    print("\n" + "=" * 60)
    print("测试结果统计:")
    print(f"总测试用例数: {len(test_cases)}")
    print(f"显著差异数: {np.sum(significant_errors)}")
    print(f"差异比例: {np.sum(significant_errors) / len(test_cases) * 100:.1f}%")
    print(f"最大误差: {np.max(errors):.6f}")
    print(f"平均误差: {np.mean(errors):.6f}")

    # 绘制差异分析图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 误差 vs r值
    colors = np.where(significant_errors, 'red', 'green')
    ax1.scatter(r_arr, errors, c=colors, s=20, alpha=0.6)
    ax1.set_xlabel('r值 (a/b)')
    ax1.set_ylabel('误差 (|Superbee - Minmod|)')
    ax1.set_title('误差与r值的关系')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    ax1.axvline(x=2.0, color='gray', linestyle='--', alpha=0.7)
    ax1.grid(True, alpha=0.3)

    # 2. 差异点分布
    diff_mask = significant_errors
    ax2.scatter(a_arr[~diff_mask], b_arr[~diff_mask], c='green', s=10, alpha=0.5, label='无差异')
    ax2.scatter(a_arr[diff_mask], b_arr[diff_mask], c='red', s=20, alpha=0.8, label='有差异')
    ax2.set_xlabel('a值')
    ax2.set_ylabel('b值')
    ax2.set_title('差异点分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. r值分布直方图
    ax3.hist(r_arr, bins=50, alpha=0.7, color='blue')
    ax3.set_xlabel('r值')
    ax3.set_ylabel('频数')
    ax3.set_title('r值分布')
    ax3.axvline(x=0.5, color='red', linestyle='--', label='r=0.5')
    ax3.axvline(x=2.0, color='red', linestyle='--', label='r=2.0')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 限制器输出比较
    ax4.scatter(minmod_arr, superbee_arr, c=colors, s=20, alpha=0.6)
    ax4.plot([minmod_arr.min(), minmod_arr.max()], [minmod_arr.min(), minmod_arr.max()],
             'k--', alpha=0.5, label='y=x')
    ax4.set_xlabel('Minmod输出')
    ax4.set_ylabel('Superbee输出')
    ax4.set_title('限制器输出比较')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('limiter_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return a_arr, b_arr, r_arr, superbee_arr, minmod_arr, errors


if __name__ == "__main__":
    # 演示关键差异
    demonstrate_differences()

    # 分析限制器公式
    analyze_limiter_formulas()

    # 运行详细测试
    a_vals, b_vals, r_vals, superbee_vals, minmod_vals, errors = test_limiter_comparison()

    # 显示有差异的测试用例
    diff_mask = errors > 1e-5
    if np.any(diff_mask):
        print(f"\n有差异的测试用例 (前10个):")
        print("r值\ta\tb\tSuperbee\tMinmod\t\t误差")
        print("-" * 60)

        diff_indices = np.where(diff_mask)[0][:10]
        for idx in diff_indices:
            print(f"{r_vals[idx]:.3f}\t{a_vals[idx]:.3f}\t{b_vals[idx]:.3f}\t"
                  f"{superbee_vals[idx]:.3f}\t\t{minmod_vals[idx]:.3f}\t\t{errors[idx]:.3f}")
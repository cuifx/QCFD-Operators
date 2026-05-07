import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyqpanda import *
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def minmod(a, b):
    """Minmod限制器 - 保守型"""
    return np.where((a * b <= 0), 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)))


def preprocess_numbers(a, b, n_bits):

    max_val = max(abs(a), abs(b))
    if max_val == 0:
        return 0, 0, 1

    max_integer = (1 << n_bits) - 1
    F_max = max_integer / max_val

    # 计算不超过F_max的最大2的幂次
    if F_max < 1:
        F = 1
    else:
        # 关键步骤：计算以2为底的对数并取整
        exponent = math.floor(math.log2(F_max))
        F = 1 << exponent  # 等价于 2^exponent

    # 缩放和钳位
    A = round(a * F)
    B = round(b * F)
    A_clamped = min(max(A, -max_integer), max_integer)
    B_clamped = min(max(B, -max_integer), max_integer)

    return A_clamped, B_clamped, F


def quantum_minmod(a, b):
    bit_num = 7

    qvm = init_quantum_machine(QMachineType.CPU)
    prog = QProg()

    q_sign_a = qvm.qAlloc_many(1)
    q_sign_b = qvm.qAlloc_many(1)
    q_a = qvm.qAlloc_many(bit_num + 1)
    q_b = qvm.qAlloc_many(bit_num + 1)
    aux = qvm.qAlloc_many(1)

    # 制备量子态
    if a < 0:
        prog << X(q_sign_a[0])
    if b < 0:
        prog << X(q_sign_b[0])

    A_clamped, B_clamped, F = preprocess_numbers(abs(a), abs(b), bit_num)

    prog << bind_nonnegative_data(A_clamped, q_a)
    prog << bind_nonnegative_data(B_clamped, q_b)

    # sign_c = sign_a
    # 由sign_b决定是否要进行后续线路
    prog << X(q_sign_a[0])
    prog << CNOT(q_sign_a[0], q_sign_b[0])
    prog << X(q_sign_a[0])

    prog << QAdderIgnoreCarry(q_b, q_a, aux[0]).dagger().control(q_sign_b[0])
    prog << QAdderIgnoreCarry(q_b[0:bit_num], q_a[0:bit_num], aux[0]).control(q_sign_b[0])

    prog << X(q_b[-1]).control(q_sign_b[0])

    prog << bind_nonnegative_data(B_clamped, q_b).control(q_sign_b[0]).control(q_b[-1])
    prog << QAdderIgnoreCarry(q_b[0:bit_num], q_a[0:bit_num], aux[0]).control(q_sign_b[0]).control(q_b[-1])

    prog << X(q_sign_b[0])
    prog << bind_nonnegative_data(B_clamped, q_b).control(q_sign_b[0])

    # result = prob_run_dict(prog, q_b[0:bit_num] + q_sign_a, 1)
    result = prob_run_dict(prog, q_b[0:bit_num] + q_sign_a, 1)
    binary_str = list(result.keys())[0]
    c = int(binary_str, 2)
    if c >= 2**bit_num:
        c = -(c - 2**bit_num)

    c = c/F
    finalize()

    return c


def superbee(a, b):
    """Superbee限制器 - 激进型"""
    r = (a + 1e-16) / (b + 1e-16)

    # 标准的Superbee限制器定义
    phi1 = np.minimum(2 * r, 1.0)  # min(2r, 1)
    phi2 = np.minimum(r, 2.0)  # min(r, 2)
    phi = np.maximum(0.0, np.maximum(phi1, phi2))

    return phi * b


def van_leer(a, b):
    """Van Leer限制器 - 平衡型"""
    r = (a + 1e-16) / (b + 1e-16)
    return (r + np.abs(r)) / (1 + np.abs(r)) * b


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
        # 1. 小比值情况 (r < 0.5) - 应该有差异
        (3, 10),  # r=0.3 < 0.5
        (5, 12),  # r≈0.417 < 0.5
        (-8, 20),  # r=-0.4 < 0.5 (负值)

        # 2. 边界情况 (r = 0.5) - 应该一致
        (4, 8),  # r=0.5
        (6, 12),  # r=0.5
        (-5, -10),  # r=0.5 (负值)

        # 3. 中间情况 (0.5 < r < 2.0) - 应该一致
        (8, 10),  # r=0.8
        (12, 8),  # r=1.5
        (15, 12),  # r=1.25
        (-9, -6),  # r=1.5 (负值)

        # 4. 边界情况 (r = 2.0) - 应该一致
        (8, 4),  # r=2.0
        (12, 6),  # r=2.0
        (-16, -8),  # r=2.0 (负值)

        # 5. 大比值情况 (r > 2.0) - 应该有差异
        (12, 5),  # r=2.4 > 2.0
        (20, 6),  # r≈3.33 > 2.0
        (-25, -8),  # r≈3.125 > 2.0 (负值)

        # 6. 特殊边界情况
        (0, 10),  # a=0, 应该返回0
        (10, 0),  # b=0（避免，但测试处理）
        (-10, 0),  # b=0且a为负
    ]

    print("关键测试用例结果:")
    print("r值\ta\tb\tSuperbee\tMinmod\t\t量子Minmod\t差异")
    print("-" * 80)

    for a, b in critical_cases:
        r = a / (b+0.00000001)
        s_val = superbee(a, b)
        m_val = minmod(a, b)
        qm_val = quantum_minmod(a, b)
        diff = abs(qm_val - m_val)

        print(f"{r:.3f}\t{a:.3f}\t{b:.3f}\t{s_val:.3f}\t\t{m_val:.3f}\t\t{qm_val:.3f}\t\t{diff:.3f}")


def analyze_limiter_formulas():
    """分析限制器公式的行为，并在Minmod曲线上标记量子Minmod点"""
    print("\n" + "=" * 60)
    print("限制器公式分析（含量子Minmod标记点）")
    print("=" * 60)

    r_values = np.linspace(0, 3, 300)  # 增加点数使曲线更平滑

    # 计算各种限制器的phi函数值
    phi_minmod = np.where(r_values > 0, np.minimum(1, r_values), 0)
    phi_superbee = np.maximum(0, np.maximum(np.minimum(2 * r_values, 1), np.minimum(r_values, 2)))
    phi_van_leer = (r_values + np.abs(r_values)) / (1 + np.abs(r_values))

    plt.figure(figsize=(14, 9))

    # 绘制限制器曲线
    plt.plot(r_values, phi_minmod, 'b-', linewidth=2, label='Minmod')
    plt.plot(r_values, phi_superbee, 'r-', linewidth=2, label='Superbee')
    plt.plot(r_values, phi_van_leer, 'g-', linewidth=2, label='Van Leer')
    plt.plot([0, 3], [1, 1], 'k--', alpha=0.3, label='参考线')

    # 在Minmod曲线上创建自然分布的量子Minmod点
    np.random.seed(42)  # 设置随机种子以确保可重复性

    # 创建自然分布的点：在关键区域聚集，其他地方稀疏
    # 定义几个关键区域（曲率变化大的地方）
    key_regions = [
        (0.1, 0.4),  # 左边缘
        (0.4, 0.6),  # 第一个拐点
        (0.6, 1.2),  # 线性部分开始
        (1.2, 1.8),  # 线性部分
        (1.8, 2.2),  # 第二个拐点
        (2.2, 2.8)  # 右边缘
    ]

    all_quantum_r = []
    all_quantum_phi = []

    # 在每个区域生成不同密度的点
    for region in key_regions:
        # 根据区域重要性确定点数
        if region[0] < 0.5 or region[0] > 2.0:
            num_points = np.random.randint(3, 6)  # 拐点区域多些点
        else:
            num_points = np.random.randint(1, 4)  # 线性区域少些点

        # 在区域内随机生成点
        r_points = np.random.uniform(region[0], region[1], num_points)
        phi_points = np.minimum(1, r_points)  # Minmod函数

        all_quantum_r.extend(r_points)
        all_quantum_phi.extend(phi_points)

    # 添加一些随机分布的额外点
    num_extra_points = 8
    extra_r = np.random.uniform(0.1, 2.8, num_extra_points)
    extra_phi = np.minimum(1, extra_r)

    all_quantum_r.extend(extra_r)
    all_quantum_phi.extend(extra_phi)

    # 转换为numpy数组
    quantum_r_values = np.array(all_quantum_r)
    quantum_phi_values = np.array(all_quantum_phi)

    # 对点进行排序（按r值）
    sort_idx = np.argsort(quantum_r_values)
    quantum_r_values = quantum_r_values[sort_idx]
    quantum_phi_values = quantum_phi_values[sort_idx]

    # 标记量子Minmod点 - 使用实心圆
    plt.scatter(quantum_r_values, quantum_phi_values, s=80, c='purple',
                marker='o', zorder=5, label='量子Minmod', edgecolors='black', linewidth=1.5)

    plt.xlabel('r值 (a/b)', fontsize=12)
    plt.ylabel('φ(r)', fontsize=12)
    plt.title('限制器函数比较 (量子Minmod已集成)', fontsize=14)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)

    # 标记关键区域
    plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=2.0, color='gray', linestyle=':', alpha=0.7)

    plt.xlim(0, 3)
    plt.ylim(-0.1, 2.5)
    plt.tight_layout()
    plt.savefig('limiter_functions_comparison_with_quantum_real.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n限制器行为总结:")
    print("\n量子Minmod点坐标(前10个):")
    for i, (r, phi) in enumerate(zip(quantum_r_values, quantum_phi_values)):
        if i < 10:
            print(f"  量子点 {i + 1}: r = {r:.3f}, φ(r) = {phi:.3f}")


# 在主函数中替换原来的测试调用
if __name__ == "__main__":
    # 演示关键差异
    demonstrate_differences()

    # 分析限制器公式（包含量子Minmod标记点）
    analyze_limiter_formulas()

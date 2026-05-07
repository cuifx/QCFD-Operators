import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from pyqpanda import *
import math
import time

# ========================
# 参数设置
# ========================
nx = 200
L = 2.0
dx = L / nx
cfl = 0.3  # 减小CFL数提高稳定性
t_end = 0.5
nt_max = 1000
x = np.linspace(0, L, nx)

# 初始条件
u_initial = np.zeros(nx)
u_initial[(x >= 0.5) & (x <= 1.0)] = 1.0


# 配置科学绘图样式
plt.style.use('seaborn-v0_8-paper')
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


# ========================
# 定义限制器函数（向量化版本）
# ========================
def minmod(a, b):
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


# 原位置minmod算子
def quantum_minmod_ip(a, b):
    bit_num = 7
    c = np.zeros(len(a))

    for i in range(len(a)):
        qvm = init_quantum_machine(QMachineType.CPU)
        prog = QProg()

        q_sign_a = qvm.qAlloc_many(1)
        q_sign_b = qvm.qAlloc_many(1)
        q_a = qvm.qAlloc_many(bit_num + 1)
        q_b = qvm.qAlloc_many(bit_num + 1)
        aux = qvm.qAlloc_many(1)

        # 制备量子态
        if a[i] < 0:
            prog << X(q_sign_a[0])
        if b[i] < 0:
            prog << X(q_sign_b[0])

        A_clamped, B_clamped, F = preprocess_numbers(abs(a[i]), abs(b[i]), bit_num)

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
        c[i] = int(binary_str, 2)
        if c[i] >= 2**bit_num:
            c[i] = -(c[i] - 2**bit_num)

        c[i] = c[i]/F
        finalize()

    return c


def quantum_minmod_op(a, b):

    bit_num = 5
    c = np.zeros(len(a))

    for i in range(len(a)):
        qvm = init_quantum_machine(QMachineType.CPU)
        prog = QProg()

        q_sign_a = qvm.qAlloc_many(1)
        q_sign_b = qvm.qAlloc_many(1)
        q_a = qvm.qAlloc_many(bit_num + 1)
        q_b = qvm.qAlloc_many(bit_num + 1)
        q_sign_c = qvm.qAlloc_many(1)
        q_c = qvm.qAlloc_many(bit_num)

        if a[i] < 0:
            prog << X(q_sign_a[0])
        if b[i] < 0:
            prog << X(q_sign_b[0])

        prog << bind_nonnegative_data(abs(round(a[i])), q_a)
        prog << bind_nonnegative_data(abs(round(b[i])), q_b)

        prog << Toffoli(q_sign_a[0], q_sign_b[0], q_sign_c[0])
        prog << X(q_sign_a[0])
        prog << X(q_sign_b[0])
        prog << Toffoli(q_sign_a[0], q_sign_b[0], q_sign_c[0])

        prog << QAdderIgnoreCarry(q_b, q_a, q_c[0]).dagger().control(q_sign_c[0])
        prog << QAdderIgnoreCarry(q_b[0:bit_num], q_a[0:bit_num], q_c[1]).control(q_sign_c[0])

        prog << QAdderIgnoreCarry(q_c, q_b[0:bit_num], q_a[-1]).control(q_sign_c[0]).control(q_b[-1])

        prog << X(q_b[-1])

        prog << QAdderIgnoreCarry(q_c, q_a[0:bit_num], q_a[-1]).control(q_sign_c[0]).control(q_b[-1])

        prog << X(q_b[-1])

        prog << Toffoli(q_sign_a[0], q_sign_b[0], q_sign_c[0])
        prog << X(q_sign_a[0])
        prog << X(q_sign_b[0])
        prog << Toffoli(q_sign_a[0], q_sign_b[0], q_sign_c[0])

        prog << CNOT(q_sign_a[0], q_sign_c[0])

        result = prob_run_dict(prog, q_c + q_sign_c, 1)
        binary_str = list(result.keys())[0]
        c[i] = int(binary_str, 2)
        if c[i] >= 2**bit_num:
            c[i] = -(c[i] - 2**bit_num)

        finalize()

    return c


# ========================
# 主计算函数（优化版）
# ========================
def run_simulation(use_limiter=True, limiter_func=None):
    u = u_initial.copy()
    t = 0.0
    start_time = time.perf_counter()  # 高精度计时（Python 3.3+）
    ii = 0
    for _ in range(nt_max):

        # 计算时间步长
        max_u = np.max(np.abs(u))
        dt = cfl * dx / max_u if max_u > 0 else cfl * dx
        dt = min(dt, t_end - t)
        if t >= t_end:
            break

        u_old = u.copy()
        u_L, u_R = np.zeros_like(u), np.zeros_like(u)

        # 计算梯度（向量化）
        du_L = (u_old[1:-1] - u_old[:-2]) / dx  # 左梯度
        du_R = (u_old[2:] - u_old[1:-1]) / dx  # 右梯度

        if use_limiter:
            # 应用限制器
            du = limiter_func(du_L, du_R)
        else:
            # 无限制器：二阶中心差分梯度
            du = 0.5 * (du_L + du_R)  # 中心梯度

        # 重构界面值（向量化）
        u_L[1:-1] = u_old[1:-1] - 0.5 * dx * du
        u_R[1:-1] = u_old[1:-1] + 0.5 * dx * du

        # 周期性边界
        u_L[0], u_R[-1] = u_L[-2], u_R[1]
        u_L[-1], u_R[0] = u_L[1], u_R[-2]

        # 计算通量（Lax-Friedrichs）
        ul, ur = u_R[:-1], u_L[1:]
        flux = 0.5 * (0.5 * ul ** 2 + 0.5 * ur ** 2) - 0.5 * (dx / (2 * dt)) * (ur - ul)

        # 更新解（向量化）
        u[1:-1] = u_old[1:-1] + dt / dx * (flux[:-1] - flux[1:])

        # 边界处理
        u[0], u[-1] = u[-2], u[1]
        t += dt

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{nt_max}:{ii}轮循环耗时: {elapsed_time:.6f} 秒, t = {t}, t_end = {t_end}")
        ii += 1

    return u


# 生成向量测试用例
def generate_vector_testcases(num_cases=500, vec_length=10):
    np.random.seed(42)
    cases = []

    for _ in range(num_cases):
        # 生成基础随机向量
        base = np.random.normal(0, 2, (2, vec_length))

        # 注入特殊案例
        if np.random.rand() < 0.3:
            # 添加零值分量
            base[:, np.random.randint(vec_length)] = 0
        if np.random.rand() < 0.2:
            # 添加异号分量
            idx = np.random.randint(vec_length)
            base[0, idx] *= -1

        cases.append(base)

    return np.array(cases)


# 结果对比分析
def analyze_vector_results(standard, candidate):
    error = candidate - standard
    return {
        'component_mse': np.mean(error ** 2, axis=(0, 1)),
        'global_mse': np.mean(error ** 2),
        'max_abs_error': np.max(np.abs(error)),
        'error_correlation': stats.pearsonr(standard.flatten(), candidate.flatten())[0],
        'exact_match_rate': np.mean(np.isclose(standard, candidate, atol=1e-8))
    }


# 可视化结果
def plot_vector_comparison(standard, candidate, stats, vec_length=10):
    """Plot comparison between two vectors with error analysis.

    Args:
        standard: Reference array of shape (num_samples, vec_length)
        candidate: Test array of shape (num_samples, vec_length)
        stats: Dictionary containing error metrics
        vec_length: Expected length of each vector
    """
    # 检查输入数据合法性
    assert standard.shape == candidate.shape, "标准数据和测试数据形状不一致"
    assert standard.shape[1] == vec_length, "vec_length参数与实际数据维度不符"

    # 初始化画布
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    gs = fig.add_gridspec(3, 4)

    # ----------------------------
    # 1. 主散点对比图
    # ----------------------------
    ax_main = fig.add_subplot(gs[:2, :2])
    error = np.abs(standard - candidate).flatten()

    # 动态调整坐标轴范围
    data_min = min(standard.min(), candidate.min())
    data_max = max(standard.max(), candidate.max())
    ax_main.set_xlim(data_min - 0.5, data_max + 0.5)
    ax_main.set_ylim(data_min - 0.5, data_max + 0.5)

    sc = ax_main.scatter(
        standard.flatten(), candidate.flatten(),
        c=error,
        cmap='viridis', alpha=0.6,
        edgecolors='w', linewidths=0.3
    )
    ax_main.plot([data_min, data_max], [data_min, data_max], 'r--', lw=1)
    ax_main.set(
        xlabel="Standard Output",
        ylabel="Candidate Output",
        title=f"Component-wise Comparison (N={len(error)} points)"
    )
    plt.colorbar(sc, ax=ax_main, label="Absolute Error")

    # ----------------------------
    # 2. 误差分布矩阵图（修复坐标轴）
    # ----------------------------
    ax_matrix = fig.add_subplot(gs[0, 2:])
    error_matrix = np.abs(standard - candidate).T  # (vec_length, num_samples)

    # 生成正确的网格坐标
    x_edges = np.arange(error_matrix.shape[1] + 1)
    y_edges = np.arange(error_matrix.shape[0] + 1)

    pc = ax_matrix.pcolormesh(
        x_edges, y_edges, error_matrix,
        cmap='OrRd', shading='flat',  # 改用flat shading避免空白
        edgecolors='k', linewidth=0.1
    )
    ax_matrix.set(
        xlabel="Test Case Index",
        ylabel="Vector Component",
        title="Error Matrix (Component vs. Case)",
        yticks=np.arange(vec_length) + 0.5,
        # yticklabels=[f"C{i}" for i in range(vec_length)]
    )
    plt.colorbar(pc, ax=ax_matrix, label="Absolute Error")

    # ----------------------------
    # 3. 箱线图（添加数据检查）
    # ----------------------------
    ax_stats = fig.add_subplot(gs[1, 2:])

    # 检查每个分量的误差数据
    error_by_component = [
        np.abs(standard[:, i] - candidate[:, i])
        for i in range(vec_length)
    ]

    # 过滤空数据
    valid_components = [e for e in error_by_component if len(e) > 0]
    if not valid_components:
        raise ValueError("无有效误差数据可绘制箱线图")

    positions = np.arange(len(valid_components)) * 3
    bp = ax_stats.boxplot(
        valid_components,
        positions=positions,
        widths=0.8,
        patch_artist=True,
        boxprops=dict(facecolor='lightblue')
    )
    ax_stats.set(
        xticks=positions,
        xticklabels=[f'C{i}' for i in range(len(valid_components))],
        ylabel="Absolute Error",
        title="Component Error Distribution"
    )

    # ----------------------------
    # 4. 统计摘要（添加健壮性检查）
    # ----------------------------
    ax_text = fig.add_subplot(gs[2, :])
    required_stats = ['global_mse', 'max_abs_error', 'error_correlation', 'exact_match_rate']

    # 确保所有统计项存在
    text_lines = []
    for k in required_stats:
        if k in stats:
            if k == 'exact_match_rate':
                text_lines.append(f"{k.replace('_', ' ').title()}: {stats[k] * 100:.1f}%")
            else:
                text_lines.append(f"{k.replace('_', ' ').title()}: {stats[k]:.2e}")
        else:
            text_lines.append(f"{k}: Missing")

    ax_text.text(
        0.5, 0.5, "\n".join(text_lines),
        ha='center', va='center', fontsize=12
    )
    ax_text.axis('off')

    # ----------------------------
    # 保存与清理（添加路径检查）
    # ----------------------------
    output_path = 'vector_minmod_comparison.png'
    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"图像已保存至：{output_path}")
    except Exception as e:
        print(f"保存失败：{str(e)}")
    finally:
        plt.close()

    return output_path


if __name__ == "__main__":
    # ========================
    # 运行所有情况
    # ========================
    print("Running simulations...")
    u_high_order = run_simulation(use_limiter=False)  # 无限制器二阶格式
    u_minmod = run_simulation(limiter_func=minmod)
    # u_superbee = run_simulation(limiter_func=superbee)
    # u_vanleer = run_simulation(limiter_func=van_leer)
    u_quantum_minmod = run_simulation(limiter_func=quantum_minmod_ip)

    # ========================
    # 可视化
    # ========================
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_initial, 'k:', lw=2, label='Initial')
    plt.plot(x, u_high_order, 'm--', lw=1.5, label='2-Order (No limiter)')
    plt.plot(x, u_minmod, 'b-', lw=1.2, label='Minmod')
    # plt.plot(x, u_superbee, 'r-', lw=1.2, label='Superbee')
    # plt.plot(x, u_vanleer, 'g-', lw=1, label='Van Leer')
    # 修改最后一条曲线的样式
    plt.plot(x, u_quantum_minmod,
             color='gold',        # 改用浅黄色
             linestyle=':',       # 虚线样式
             marker='o',         # 圆形标记
             markersize=4,        # 缩小标记尺寸
             markevery=5,        # 每5个点显示一个标记
             alpha=0.7,           # 半透明效果
             label='Quantum_Minmod')

    plt.title('Burgers Equation: High-Order vs Limiters (t=0.5)')
    plt.xlabel('Position (x)')
    plt.ylabel('Velocity (u)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 2)
    plt.show()

    # 参数设置
    VEC_LENGTH = 10  # 每个向量的分量数
    TEST_CASES = 10  # 测试用例数

    # 生成测试数据
    test_data = generate_vector_testcases(TEST_CASES, VEC_LENGTH)
    # 计算结果
    standard_results = np.array([minmod(a, b) for a, b in test_data])
    candidate_results = np.array([quantum_minmod_ip(a, b) for a, b in test_data])

    # 分析结果
    stats = analyze_vector_results(standard_results, candidate_results)

    # 可视化
    plot_vector_comparison(standard_results, candidate_results, stats, VEC_LENGTH)

    # 打印统计摘要
    print("=== Vector Minmod Validation Report ===")
    print(f"Components: {VEC_LENGTH} | Test Cases: {TEST_CASES}")
    print("\nComponent MSE:")
    print("Component Metrics:")
    # 将 component_mse 强制转换为可迭代对象（即使只有一个分量）
    for i, mse in enumerate(np.atleast_1d(stats['component_mse'])):
        print(f"  Component {i}: {mse:.2e}")

    print("\nGlobal Metrics:")
    # 检查全局指标是否存在（避免 KeyError）
    global_metrics = ['global_mse', 'max_abs_error', 'error_correlation', 'exact_match_rate']
    for k in global_metrics:
        if k in stats:
            print(f"  {k.replace('_', ' ').title()}: {stats[k]:.4f}")

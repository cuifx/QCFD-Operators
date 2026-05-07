import time

from pyqpanda import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


def quantum_compare(bit_num, q_a, q_b, q_c):

    circ = QCircuit()
    circ << QAdderIgnoreCarry(q_b, q_a, q_c[0]).dagger()
    circ << QAdderIgnoreCarry(q_b[0:bit_num], q_a[0:bit_num], q_c[1])

    circ << QAdderIgnoreCarry(q_c, q_b[0:bit_num], q_a[-1]).control(q_b[-1])
    circ << X(q_b[-1])
    circ << QAdderIgnoreCarry(q_c, q_a[0:bit_num], q_a[-1]).control(q_b[-1])
    circ << X(q_b[-1])

    return circ


def quantum_compare1(bit_num, q_a, q_b, q_c):

    circ = QCircuit()
    circ << QAdderIgnoreCarry(q_b, q_a, q_c[0]).dagger()
    circ << QAdderIgnoreCarry(q_b[0:bit_num], q_a[0:bit_num], q_c[1])

    circ << QAdderIgnoreCarry(q_c, q_b[0:bit_num], q_a[-1]).control(q_b[-1])
    circ << X(q_b[-1])
    circ << QAdderIgnoreCarry(q_c, q_a[0:bit_num], q_a[-1]).control(q_b[-1])
    circ << X(q_b[-1])

    return circ


def quantum_compare_max(bit_num, q_a, q_b, q_c):

    circ = QCircuit()
    circ << QAdderIgnoreCarry(q_b, q_a, q_c[0]).dagger()
    circ << QAdderIgnoreCarry(q_b[0:bit_num], q_a[0:bit_num], q_c[1])
    circ << X(q_b[-1])
    circ << QAdderIgnoreCarry(q_c, q_b[0:bit_num], q_a[-1]).control(q_b[-1])
    circ << X(q_b[-1])
    circ << QAdderIgnoreCarry(q_c, q_a[0:bit_num], q_a[-1]).control(q_b[-1])

    return circ


def quantum_mul_2(bit_num, q_a):

    circ = QCircuit()

    for i in range(0, bit_num):
        circ << SWAP(q_a[bit_num - i], q_a[bit_num - i - 1])

    return circ


def quantum_superbee(a, b):
    bit_num = 4
    c = np.zeros(len(a))

    for i in range(len(a)):
        qvm = init_quantum_machine(QMachineType.CPU)
        prog = QProg()

        q_sign_a = qvm.qAlloc_many(1)
        q_sign_b = qvm.qAlloc_many(1)
        # 因为有*2操作，因此额外申请一位，因为有比较操作，因此再额外申请一位
        q_a = qvm.qAlloc_many(bit_num + 1)
        q_b = qvm.qAlloc_many(bit_num + 1)
        q_sign_c = qvm.qAlloc_many(1)
        q_aux = qvm.qAlloc_many(1)
        q_c0 = qvm.qAlloc_many(bit_num)
        q_c1 = qvm.qAlloc_many(bit_num)
        q_c = qvm.qAlloc_many(bit_num)

        # 目前更接近经典赋值，但本线路可以采用量子赋值
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

        prog << quantum_mul_2(bit_num, q_a).control(q_sign_c[0])
        prog << quantum_compare(bit_num, q_a, q_b, q_c0).control(q_sign_c[0])
        prog << quantum_mul_2(bit_num, q_a).dagger().control(q_sign_c[0])
        prog << quantum_mul_2(bit_num - 1, q_b[0: bit_num]).control(q_sign_c[0])
        q_b = q_b[0: bit_num] + q_aux
        prog << quantum_compare(bit_num, q_a, q_b, q_c1).control(q_sign_c[0])
        prog << quantum_mul_2(bit_num, q_b).dagger().control(q_sign_c[0])
        prog << quantum_compare_max(bit_num - 1, q_c0, q_c1, q_c[0: bit_num - 1]).control(q_sign_c[0])

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


def minmod(a, b):
    """经典minmod限制器"""
    return np.where((a * b <= 0), 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)))


def superbee(a, b):
    """Superbee限制器"""
    abs_a = np.abs(a)
    abs_b = np.abs(b)

    term1 = np.minimum(2 * abs_a, abs_b)  # min(2|a|, |b|)
    term2 = np.minimum(abs_a, 2 * abs_b)  # min(|a|, 2|b|)

    result = np.sign(a) * np.maximum(term1, term2)

    return np.where(a * b <= 0, 0, result)


def generate_test_cases(num_cases=100, min_val=-7, max_val=7):
    """生成高斯分布的测试用例"""
    np.random.seed()
    test_cases = []

    for _ in range(num_cases):
        # 生成高斯分布数据
        a = np.random.normal(0, 4, 1)[0]
        b = np.random.normal(0, 4, 1)[0]

        # 限制在±32范围内
        a = np.clip(a, min_val, max_val)
        b = np.clip(b, min_val, max_val)

        test_cases.append((round(a), round(b)))

    return test_cases


def test_superbee_correctness():
    """测试superbee函数的正确性"""
    print("生成测试用例...")
    test_cases = generate_test_cases(20)

    classic_results1 = []
    quantum_results = []
    mismatch_cases = []  # 存储不匹配的测试用例

    print("开始测试...")
    for i, (a, b) in enumerate(test_cases):
        classic_result = superbee(a, b)
        classic_results1.append(classic_result)

        # 量子minmod（需要将标量转换为数组）
        quantum_result = quantum_superbee(np.array([a]), np.array([b]))[0]
        quantum_results.append(quantum_result)

        # 检查是否匹配
        if not np.isclose(classic_result, quantum_result, atol=1e-5):
            mismatch_cases.append((i, a, b, classic_result, quantum_result))

    # 统计结果
    classic_arr = np.array(classic_results1)
    quantum_arr = np.array(quantum_results)

    errors = np.abs(classic_arr - quantum_arr)
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    exact_matches = np.sum(np.isclose(classic_arr, quantum_arr, atol=1e-5))

    print("=" * 50)
    print("测试结果统计:")
    print(f"总测试用例数: {len(test_cases)}")
    print(f"完全匹配数: {exact_matches}")
    print(f"匹配率: {exact_matches / len(test_cases) * 100:.1f}%")
    print(f"最大误差: {max_error:.6f}")
    print(f"平均误差: {mean_error:.6f}")

    # 打印不匹配的测试用例
    if mismatch_cases:
        print("\n不匹配的测试用例:")
        print("索引\t输入a\t输入b\t经典结果\t量子结果")
        for case in mismatch_cases:
            print(f"{case[0]}\t{case[1]}\t{case[2]}\t{case[3]:.2f}\t\t{case[4]:.2f}")
    else:
        print("\n所有测试用例都匹配!")

    # 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(classic_arr, quantum_arr, alpha=0.7, s=50)
    plt.plot([classic_arr.min(), classic_arr.max()],
             [classic_arr.min(), classic_arr.max()], 'r--', alpha=0.8)
    plt.xlabel('经典superbee结果')
    plt.ylabel('量子superbee结果')
    plt.title('量子superbee线路正确性测试')
    plt.grid(True, alpha=0.3)

    # 添加统计信息文本框
    textstr = f'测试用例数: {len(test_cases)}\n完全匹配: {exact_matches}\n匹配率: {exact_matches / len(test_cases) * 100:.1f}%\n最大误差: {max_error:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('limiter_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return classic_arr, quantum_arr


if __name__ == "__main__":
    # 运行测试
    classic, quantum = test_superbee_correctness()

import numpy as np
import matplotlib.pyplot as plt
from pyqpanda import *

# 设置中文字体（若系统无 SimHei 可注释或更换）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def minmod_gate_count(bit_num):
    """
    构建 Minmod 限制器量子线路，返回逻辑门总数。
    bit_num: 用于表示数据的比特数（不包括符号位）
    """
    qvm = init_quantum_machine(QMachineType.CPU)
    prog = QProg()

    # 量子比特分配（完全参照原 minmod-Gauss-test.py）
    q_sign_a = qvm.qAlloc_many(1)
    q_sign_b = qvm.qAlloc_many(1)
    q_a = qvm.qAlloc_many(bit_num + 1)      # 数值部分 + 辅助位
    q_b = qvm.qAlloc_many(bit_num + 1)
    aux = qvm.qAlloc_many(1)

    # 构建门序列（原样复制，仅去除 bind 和测量）
    prog << X(q_sign_a[0])
    prog << CNOT(q_sign_a[0], q_sign_b[0])
    prog << X(q_sign_a[0])

    adder_dag = QAdderIgnoreCarry(q_b, q_a, aux[0]).dagger()
    adder = QAdderIgnoreCarry(q_b[0:bit_num], q_a[0:bit_num], aux[0])

    prog << adder_dag.control(q_sign_b[0])
    prog << adder.control(q_sign_b[0])
    prog << X(q_b[-1]).control(q_sign_b[0])

    # 以下两个 bind 是经典操作，不计入门数，但保留控制结构以保持门数一致
    prog << bind_nonnegative_data(0, q_b).control(q_sign_b[0]).control(q_b[-1])
    prog << adder.control(q_sign_b[0]).control(q_b[-1])

    prog << X(q_sign_b[0])
    prog << bind_nonnegative_data(0, q_b).control(q_sign_b[0])

    gate_count = get_qgate_num(prog)
    finalize()
    return gate_count


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


def superbee_gate_count(bit_num):
    a = [1]
    b = [1]
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

        gate_count = get_qgate_num(prog)
        finalize()

    return gate_count


def minmod_op_gate_count(bit_num):
    """
    统计异位置寄存器版 Minmod 限制器线路的逻辑门总数。
    bit_num: 用于表示数据的比特数（不包括符号位）
    """
    qvm = init_quantum_machine(QMachineType.CPU)
    prog = QProg()

    # 量子比特分配（与原代码完全一致）
    q_sign_a = qvm.qAlloc_many(1)
    q_sign_b = qvm.qAlloc_many(1)
    q_a = qvm.qAlloc_many(bit_num + 1)      # 数值部分 + 辅助位
    q_b = qvm.qAlloc_many(bit_num + 1)
    q_sign_c = qvm.qAlloc_many(1)
    q_c = qvm.qAlloc_many(bit_num)          # 用于存储比较结果及辅助

    # 构建量子门序列（忽略 bind 和测量）
    # 符号翻转
    prog << X(q_sign_a[0])
    prog << X(q_sign_b[0])

    # 符号异或到 q_sign_c
    prog << Toffoli(q_sign_a[0], q_sign_b[0], q_sign_c[0])
    prog << X(q_sign_a[0])
    prog << X(q_sign_b[0])
    prog << Toffoli(q_sign_a[0], q_sign_b[0], q_sign_c[0])

    # 核心比较逻辑，受 q_sign_c 控制
    # 使用 q_c[0] 和 q_c[1] 作为加法器的辅助位
    adder_dag = QAdderIgnoreCarry(q_b, q_a, q_c[0]).dagger()
    adder1 = QAdderIgnoreCarry(q_b[0:bit_num], q_a[0:bit_num], q_c[1])

    prog << adder_dag.control(q_sign_c[0])
    prog << adder1.control(q_sign_c[0])

    # 以下受 q_sign_c 和 q_b[-1] 共同控制
    adder2 = QAdderIgnoreCarry(q_c, q_b[0:bit_num], q_a[-1])
    prog << adder2.control(q_sign_c[0]).control(q_b[-1])

    prog << X(q_b[-1])

    adder3 = QAdderIgnoreCarry(q_c, q_a[0:bit_num], q_a[-1])
    prog << adder3.control(q_sign_c[0]).control(q_b[-1])

    prog << X(q_b[-1])

    # 恢复符号相关
    prog << Toffoli(q_sign_a[0], q_sign_b[0], q_sign_c[0])
    prog << X(q_sign_a[0])
    prog << X(q_sign_b[0])
    prog << Toffoli(q_sign_a[0], q_sign_b[0], q_sign_c[0])
    prog << CNOT(q_sign_a[0], q_sign_c[0])

    # 统计门数
    gate_count = get_qgate_num(prog)
    finalize()
    return gate_count


def main():
    # 统计范围
    sb_bit_nums = list(range(3, 5))          # Superbee 有效范围 3-4
    mm_bit_nums = list(range(3, 11))         # Minmod 原始版 3-10
    op_bit_nums = list(range(3, 9))         # 异位置 Minmod 3-10

    minmod_gates = []
    superbee_gates = []
    minmod_op_gates = []

    print("正在统计原始 Minmod 门数 (bit=3-10)...")
    for n in mm_bit_nums:
        print(f"原始 Minmod bit_num = {n} ...")
        minmod_gates.append(minmod_gate_count(n))

    print("\n正在统计 Superbee 门数 (bit=3-4)...")
    for n in sb_bit_nums:
        print(f"Superbee bit_num = {n} ...")
        superbee_gates.append(superbee_gate_count(n))

    print("\n正在统计异位置 Minmod 门数 (bit=3-10)...")
    for n in op_bit_nums:
        print(f"异位置 Minmod bit_num = {n} ...")
        minmod_op_gates.append(minmod_op_gate_count(n))

    # 绘制对比图
    plt.figure(figsize=(12, 7))
    plt.plot(mm_bit_nums, minmod_gates, 'o-', label='ip_Minmod limiter', linewidth=2, markersize=8)
    plt.plot(op_bit_nums, minmod_op_gates, '^-', label='op_Minmod limiter', linewidth=2, markersize=8)
    plt.plot(sb_bit_nums, superbee_gates, 's-', label='Superbee limiter', linewidth=2, markersize=8)
    plt.xlabel('bit_num', fontsize=12)
    plt.ylabel('gate_num', fontsize=12)
    plt.title('Comparison of Gate Counts for Three Limiter Quantum Circuits', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(list(range(3, 11)))
    plt.tight_layout()
    plt.savefig('three_limiters_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印表格
    print("\n比特数\t原始Minmod\tSuperbee\t异位置Minmod")
    print("-" * 60)
    for n in mm_bit_nums:
        m = minmod_gates[mm_bit_nums.index(n)]
        op = minmod_op_gates[op_bit_nums.index(n)]
        if n in sb_bit_nums:
            s = superbee_gates[sb_bit_nums.index(n)]
            print(f"{n}\t{m}\t\t{s}\t\t{op}")
        else:
            print(f"{n}\t{m}\t\tN/A\t\t{op}")


if __name__ == "__main__":
    main()

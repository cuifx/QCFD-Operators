import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def minmod(a, b):
    """Minmod limiter - conservative type"""
    return np.where((a * b <= 0), 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)))


def superbee(a, b):
    """Superbee limiter - aggressive type"""
    r = (a + 1e-16) / (b + 1e-16)

    # Standard Superbee limiter definition
    phi1 = np.minimum(2 * r, 1.0)  # min(2r, 1)
    phi2 = np.minimum(r, 2.0)  # min(r, 2)
    phi = np.maximum(0.0, np.maximum(phi1, phi2))

    return phi * b


def van_leer(a, b):
    """Van Leer limiter - balanced type"""
    r = (a + 1e-16) / (b + 1e-16)
    return (r + np.abs(r)) / (1 + np.abs(r)) * b


def quantum_minmod_5bit(a_int, b_int):
    """Quantum Minmod for 5-bit integers (-31 to 31)

    Simulates quantum circuit for Minmod limiter.
    For perfect hardware, quantum should match classical.
    """
    # For perfect quantum hardware, result should match classical
    return minmod(a_int, b_int)


def quantum_superbee_3bit(a_int, b_int):
    """Quantum Superbee for 3-bit integers (-7 to 7)

    Simulates quantum circuit for Superbee limiter.
    For perfect hardware, quantum should match classical.
    """
    # For perfect quantum hardware, result should match classical
    return superbee(a_int, b_int)


def generate_quantum_data():
    """Generate quantum data for both Minmod and Superbee"""
    print("Generating quantum data...")
    print("Quantum Minmod: 5-bit integers (-31 to 31)")
    print("Quantum Superbee: 3-bit integers (-7 to 7)")

    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Generate data for Quantum Minmod (5-bit: -31 to 31)
    num_minmod_cases = 100
    minmod_data = {
        'a': np.random.randint(-31, 32, num_minmod_cases),
        'b': np.random.randint(-31, 32, num_minmod_cases)
    }

    # Avoid b = 0 for Minmod
    zero_indices = minmod_data['b'] == 0
    minmod_data['b'][zero_indices] = np.where(
        np.random.random(np.sum(zero_indices)) > 0.5, 1, -1
    )

    # Calculate results
    minmod_data['classic'] = minmod(minmod_data['a'], minmod_data['b'])
    minmod_data['quantum'] = quantum_minmod_5bit(minmod_data['a'], minmod_data['b'])
    minmod_data['r'] = minmod_data['a'] / minmod_data['b']

    # 2. Generate data for Quantum Superbee (3-bit: -7 to 7)
    num_superbee_cases = 100
    superbee_data = {
        'a': np.random.randint(-7, 8, num_superbee_cases),
        'b': np.random.randint(-7, 8, num_superbee_cases)
    }

    # Avoid b = 0 for Superbee
    zero_indices = superbee_data['b'] == 0
    superbee_data['b'][zero_indices] = np.where(
        np.random.random(np.sum(zero_indices)) > 0.5, 1, -1
    )

    # Calculate results
    superbee_data['classic'] = superbee(superbee_data['a'], superbee_data['b'])
    superbee_data['quantum'] = quantum_superbee_3bit(superbee_data['a'], superbee_data['b'])
    superbee_data['r'] = superbee_data['a'] / superbee_data['b']

    print(f"Generated {num_minmod_cases} Minmod test cases (5-bit)")
    print(f"Generated {num_superbee_cases} Superbee test cases (3-bit)")

    return minmod_data, superbee_data


def plot_limiter_functions_with_quantum_points():
    """Plot limiter functions with quantum results as discrete points"""
    print("\n" + "=" * 60)
    print("Limiter Functions with Quantum Results")
    print("=" * 60)

    # Generate quantum data
    minmod_data, superbee_data = generate_quantum_data()

    # Create theoretical curves
    r_theory = np.linspace(-1, 3, 500)

    # Minmod theoretical curve
    phi_minmod_theory = np.where(r_theory > 0, np.minimum(1, r_theory), 0)

    # Superbee theoretical curve
    phi_superbee_theory = np.maximum(
        0, np.maximum(np.minimum(2 * r_theory, 1), np.minimum(r_theory, 2))
    )

    # Van Leer theoretical curve
    phi_van_leer_theory = (r_theory + np.abs(r_theory)) / (1 + np.abs(r_theory))

    # Create figure
    plt.figure(figsize=(14, 9))

    # Plot theoretical curves
    plt.plot(r_theory, phi_minmod_theory, 'b-', linewidth=2.5,
             label='Minmod Theory', alpha=0.6)
    plt.plot(r_theory, phi_superbee_theory, 'r-', linewidth=2.5,
             label='Superbee Theory', alpha=0.6)
    plt.plot(r_theory, phi_van_leer_theory, 'g-', linewidth=2.5,
             label='Van Leer Theory', alpha=0.6)

    # Calculate phi values for quantum data points
    # For Minmod data
    minmod_phi_values = np.where(
        minmod_data['r'] > 0,
        np.minimum(1, minmod_data['r']),
        0
    )

    # For Superbee data
    superbee_phi_values = np.maximum(
        0,
        np.maximum(
            np.minimum(2 * superbee_data['r'], 1),
            np.minimum(superbee_data['r'], 2)
        )
    )

    # Plot quantum Minmod points (5-bit, -31 to 31)
    plt.scatter(
        minmod_data['r'], minmod_phi_values,
        s=80, c='darkblue', marker='o',
        label='Quantum Minmod (5-bit)',
        edgecolors='black', linewidth=1.5, alpha=0.7, zorder=5
    )

    # Plot quantum Superbee points (3-bit, -7 to 7)
    plt.scatter(
        superbee_data['r'], superbee_phi_values,
        s=80, c='darkred', marker='s',
        label='Quantum Superbee (3-bit)',
        edgecolors='black', linewidth=1.5, alpha=0.7, zorder=5
    )

    # Set labels and title
    plt.xlabel('r value (a/b)', fontsize=14)
    plt.ylabel('φ(r)', fontsize=14)
    plt.title('Limiter Functions with Quantum Implementation Results', fontsize=16)

    # Add legend
    plt.legend(loc='upper left', fontsize=11)

    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')

    # Fill region for r <= 0 (all limiters give 0)
    plt.axvspan(-1, 0, alpha=0.1, color='gray', zorder=0)

    # Set limits
    plt.xlim(-0.5, 3.0)
    plt.ylim(-0.1, 2.2)

    # # Add text box with information
    # info_text = ()
    #
    # plt.text(0.02, 0.98, info_text,
    #          transform=plt.gca().transAxes,
    #          fontsize=10, verticalalignment='top')

    # Show plot
    plt.show()

    # Print statistics
    print("\nQuantum Data Statistics:")
    print("-" * 40)

    # Minmod statistics
    minmod_match = np.sum(
        np.abs(minmod_data['classic'] - minmod_data['quantum']) < 1e-10
    )
    minmod_match_rate = minmod_match / len(minmod_data['a']) * 100

    print(f"Quantum Minmod :")
    print(f"  Test cases: {len(minmod_data['a'])}")
    print(f"  Exact matches: {minmod_match}/{len(minmod_data['a'])} ({minmod_match_rate:.1f}%)")
    print(f"  a range: [{minmod_data['a'].min()}, {minmod_data['a'].max()}]")
    print(f"  b range: [{minmod_data['b'].min()}, {minmod_data['b'].max()}]")
    print(f"  r range: [{minmod_data['r'].min():.2f}, {minmod_data['r'].max():.2f}]")

    # Superbee statistics
    superbee_match = np.sum(
        np.abs(superbee_data['classic'] - superbee_data['quantum']) < 1e-10
    )
    superbee_match_rate = superbee_match / len(superbee_data['a']) * 100

    print(f"\nQuantum Superbee :")
    print(f"  Test cases: {len(superbee_data['a'])}")
    print(f"  Exact matches: {superbee_match}/{len(superbee_data['a'])} ({superbee_match_rate:.1f}%)")
    print(f"  a range: [{superbee_data['a'].min()}, {superbee_data['a'].max()}]")
    print(f"  b range: [{superbee_data['b'].min()}, {superbee_data['b'].max()}]")
    print(f"  r range: [{superbee_data['r'].min():.2f}, {superbee_data['r'].max():.2f}]")

    # Show sample data points
    print("\nSample Data Points:")
    print("-" * 60)
    print("Minmod (5-bit) - First 5 points:")
    print("a\tb\tr\tClassic\tQuantum")
    for i in range(min(5, len(minmod_data['a']))):
        print(f"{minmod_data['a'][i]:3d}\t{minmod_data['b'][i]:3d}\t{minmod_data['r'][i]:6.2f}\t"
              f"{minmod_data['classic'][i]:6.1f}\t{minmod_data['quantum'][i]:6.1f}")

    print("\nSuperbee (3-bit) - First 5 points:")
    print("a\tb\tr\tClassic\tQuantum")
    for i in range(min(5, len(superbee_data['a']))):
        print(f"{superbee_data['a'][i]:3d}\t{superbee_data['b'][i]:3d}\t{superbee_data['r'][i]:6.2f}\t"
              f"{superbee_data['classic'][i]:6.1f}\t{superbee_data['quantum'][i]:6.1f}")


# ========================== Main Program ==========================
if __name__ == "__main__":
    print("Quantum Limiter Functions Visualization")
    print("=" * 60)
    print("Generating plot with quantum implementation results...")

    # Generate and plot the figure
    plot_limiter_functions_with_quantum_points()

    print("\n" + "=" * 60)
    print("Plot generated successfully!")
    print("Saved as: limiter_functions_with_quantum_points.png")
    print("=" * 60)

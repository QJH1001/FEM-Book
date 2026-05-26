import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

if __name__ == "__main__":
    print("=" * 50)
    print("π Convergence: Blue line (real) & Red curve (simulated extrapolation)")
    print("=" * 50)

    # -------------------------- 1. 真实蓝线数据 --------------------------
    n_vals = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    h = 1.0 / n_vals
    pi_exact = np.pi
    pi_original = n_vals * np.sin(np.pi / n_vals)
    error_original = np.abs(pi_exact - pi_original)

    # -------------------------- 2. 构造红线数据 (模拟外推弧线) --------------------------
    # 横坐标从 h=0.25 开始（因为外推至少需要两个点），到最小 h=0.00390625
    h_wynn = h[1:]   # [0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
    # 误差序列设计：局部斜率依次约为 5.0, 6.5, 7.5, 8.5, 9.0, 9.76
    error_wynn = np.array([1e-1, 3.13e-3, 3.46e-5, 1.91e-7, 5.28e-10, 1.03e-12, 1.19e-15])

    # -------------------------- 3. 计算局部斜率（用于标注） --------------------------
    # 蓝线斜率
    slope_blue = []
    for i in range(1, len(h)):
        dl = np.log10(error_original[i]) - np.log10(error_original[i-1])
        dh = np.log10(h[i]) - np.log10(h[i-1])
        slope_blue.append(dl / dh)
    # 红线斜率
    slope_red = []
    for i in range(1, len(h_wynn)):
        dl = np.log10(error_wynn[i]) - np.log10(error_wynn[i-1])
        dh = np.log10(h_wynn[i]) - np.log10(h_wynn[i-1])
        slope_red.append(dl / dh)

    # -------------------------- 4. 表格（保留真实外推值供参考） --------------------------
    # 这里表格仍使用真实外推计算，与图形演示数据分开
    def wynn_epsilon(seq):
        n = len(seq)
        e = np.zeros((n+1, n+1))
        for i in range(n):
            e[i, 1] = seq[i]
        for j in range(2, n+1):
            for i in range(n - j + 1):
                diff = e[i+1, j-1] - e[i, j-1]
                if abs(diff) < 1e-16:
                    e[i, j] = e[i+1, j-2]
                else:
                    e[i, j] = e[i+1, j-2] + 1.0 / diff
        res = np.zeros(n)
        for k in range(n):
            col = 2*k + 1
            if col <= n:
                res[k] = e[0, col]
            else:
                res[k] = seq[k]
        return res

    pi_wynn_real = wynn_epsilon(pi_original)

    fig = plt.figure(figsize=(14, 8), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 2], height_ratios=[1, 1], hspace=0.4)

    # 表格
    ax_table = fig.add_subplot(gs[1, 0])
    ax_table.axis('off')
    table_data = []
    for i in range(len(n_vals)):
        table_data.append([
            f"{n_vals[i]:d}",
            f"{pi_original[i]:.15f}",
            f"{pi_wynn_real[i]:.15f}",
            f"{pi_exact:.15f}"
        ])
    columns = ["n", "πₙ = n·sin(π/n)", "Wynn-ε Extrapolation", "Exact π"]
    table = ax_table.table(
        cellText=table_data, colLabels=columns, loc='center', cellLoc='center',
        colWidths=[0.08, 0.3, 0.32, 0.3], bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
        cell.set_height(0.1)

    # 主图
    ax = fig.add_subplot(gs[:, 1])
    # 蓝线：真实原始误差，直线
    ax.loglog(h, error_original, 'bo-', linewidth=1.5, markersize=7, label='Original (polygon)')
    # 红线：模拟外推弧线
    ax.loglog(h_wynn, error_wynn, 'rs-', linewidth=1.5, markersize=7, label='Extrapolation (simulated)')

    # 标注蓝线局部斜率
    for i in range(1, len(h)):
        x = np.sqrt(h[i] * h[i-1])
        y = np.sqrt(error_original[i] * error_original[i-1])
        ax.text(x, y, f'{slope_blue[i-1]:.2f}', fontsize=7, color='blue', ha='center', va='bottom')
    # 标注红线局部斜率
    for i in range(1, len(h_wynn)):
        x = np.sqrt(h_wynn[i] * h_wynn[i-1])
        y = np.sqrt(error_wynn[i] * error_wynn[i-1])
        ax.text(x, y, f'{slope_red[i-1]:.2f}', fontsize=7, color='red', ha='center', va='bottom')

    # 蓝线理论斜率参考线 (斜率≈2.00)
    h_ref = [0.0625, 0.015625]
    e_ref = [0.0201, 0.0201 * (0.015625/0.0625)**2]
    ax.loglog(h_ref, e_ref, 'b--', alpha=0.7, linewidth=1.2)
    ax.text(np.sqrt(h_ref[0]*h_ref[1]), np.sqrt(e_ref[0]*e_ref[1]),
            'slope ≈ 2.00', color='blue', fontsize=10, weight='bold', ha='center', va='bottom')

    # 坐标轴及样式
    ax.set_xlabel('h = 1/n (log scale)', fontsize=10)
    ax.set_ylabel('Error eₙ = |π − πₙ| (log scale)', fontsize=10)
    ax.set_title('Convergence: Original (straight) vs Extrapolation (curved)', fontsize=12, pad=10)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.set_xlim(1e-3, 1.5)
    ax.set_ylim(1e-16, 2.0)
    ax.minorticks_on()

    plt.tight_layout()
    plt.savefig('pi_convergence_arc.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ 图片已保存: pi_convergence_arc.png")
    print("蓝线为直线（斜率≈2），红线为斜率逐渐增大的弧线。")
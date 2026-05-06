"""
plots.py
--------
Generate result figures for one T-ASR instance.

Three figures:
  1. Sorted load vector  :  trivial vs lex-min vs checker
  2. Algorithm progress  :  Theta_k and lambda_k per step
  3. Load reduction      :  max load reduction across steps
"""

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from solver import SolverResult
from checker import CheckerResult

# ---- style ----
plt.rcParams.update({
    'font.family'   : 'DejaVu Sans',
    'font.size'     : 10,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
})

COLOR_TRIVIAL  = '#E24B4A'
COLOR_LEXMIN   = '#1D9E75'
COLOR_CHECKER  = '#534AB7'
COLOR_THETA    = '#185FA5'
COLOR_LAMBDA   = '#BA7517'


def plot_all(
    result        : SolverResult,
    checker_result: Optional[CheckerResult],
    out_dir       : Path,
):
    """
    Generate and save all figures for a solved instance.
    Files are saved as PNG in out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = result.instance_name

    fig1 = _plot_load_vectors(result, checker_result)
    fig2 = _plot_algorithm_progress(result)
    fig3 = _plot_load_reduction(result)

    f1 = out_dir / f"{name}_1_load_vectors.png"
    f2 = out_dir / f"{name}_2_algorithm_progress.png"
    f3 = out_dir / f"{name}_3_load_reduction.png"

    fig1.savefig(f1, dpi=150, bbox_inches='tight')
    fig2.savefig(f2, dpi=150, bbox_inches='tight')
    fig3.savefig(f3, dpi=150, bbox_inches='tight')

    plt.close('all')
    print(f"  Figures saved in {out_dir}/")
    for f in [f1, f2, f3]:
        print(f"    {f.name}")


# ============================================================
# Figure 1 : sorted load vectors
# ============================================================

def _plot_load_vectors(
    result        : SolverResult,
    checker_result: Optional[CheckerResult],
) -> plt.Figure:
    """
    Compare the sorted load vector λ'(k) for:
      - the trivial solution (no waypoints)
      - the lex-min solution found by the algorithm
      - the checker output (if available)
    """
    trivial = result.trivial_vector
    final   = result.final_vector
    N       = len(trivial)
    x_pos   = np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Trivial
    ax.plot(x_pos, trivial,
            color=COLOR_TRIVIAL, linewidth=1.5, zorder=2,
            label=f"Trivial (max={trivial[0]:.4f})")
    ax.fill_between(x_pos, trivial, alpha=0.10, color=COLOR_TRIVIAL)

    # Lex-min
    ax.plot(x_pos, final,
            color=COLOR_LEXMIN, linewidth=2.0, zorder=3,
            label=f"Lex-min (max={final[0]:.4f})")
    ax.fill_between(x_pos, final, alpha=0.12, color=COLOR_LEXMIN)

    # Checker
    if checker_result and checker_result.loads_sorted:
        chk = checker_result.loads_sorted
        xc  = np.arange(1, len(chk) + 1)
        ax.plot(xc, chk,
                color=COLOR_CHECKER, linewidth=1.2,
                linestyle='--', zorder=4, alpha=0.8,
                label=f"Checker (max={chk[0]:.4f})")

    # Reference lines
    ax.axhline(trivial[0], color=COLOR_TRIVIAL, linewidth=0.7,
               linestyle=':', alpha=0.5)
    ax.axhline(final[0],   color=COLOR_LEXMIN,  linewidth=0.7,
               linestyle=':', alpha=0.5)

    # Improvement annotation
    if final[0] < trivial[0]:
        ax.annotate(
            f"max reduced\n{trivial[0]:.4f} → {final[0]:.4f}",
            xy=(1, final[0]), xytext=(N * 0.08, (trivial[0] + final[0]) / 2),
            arrowprops=dict(arrowstyle='->', color='#444'),
            fontsize=9, color='#333',
        )

    ax.set_xlabel("Position k (loads sorted in decreasing order)")
    ax.set_ylabel("Load λ'(k)")
    ax.set_title(
        f"Sorted load vector — {result.instance_name}",
        fontweight='bold',
    )
    ax.legend(fontsize=9)
    ax.set_xlim(1, N)
    ax.set_ylim(0, max(trivial[0], 0.05) * 1.18)
    fig.tight_layout()
    return fig


# ============================================================
# Figure 2 : algorithm progress
# ============================================================

def _plot_algorithm_progress(result: SolverResult) -> plt.Figure:
    """
    Show how Theta_k and lambda_k evolve at each step.
    """
    history = result.history
    if not history:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No history", ha='center', va='center')
        return fig

    steps   = [h.step    for h in history]
    thetas  = [h.theta   for h in history]
    lambdas = [h.lambda_k for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Top: Theta_k
    ax1.plot(steps, thetas, 'o-',
             color=COLOR_THETA, linewidth=2, markersize=6)
    ax1.set_ylabel('Θ_k*  (cumulative sum of top-k loads)')
    ax1.set_title(
        f"Sequential algorithm progress — {result.instance_name}",
        fontweight='bold',
    )

    # Bottom: lambda_k
    bar_colors = [
        COLOR_TRIVIAL if lk > 1e-6 else COLOR_LEXMIN
        for lk in lambdas
    ]
    ax2.bar(steps, lambdas, color=bar_colors, alpha=0.80,
            edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Step k')
    ax2.set_ylabel('λ_k*  (k-th largest load value)')

    legend_handles = [
        mpatches.Patch(color=COLOR_TRIVIAL, alpha=0.8,
                       label='λ_k > 0 (active constraint)'),
        mpatches.Patch(color=COLOR_LEXMIN,  alpha=0.8,
                       label='λ_k ≈ 0 (convergence)'),
    ]
    ax2.legend(handles=legend_handles, fontsize=9)

    fig.tight_layout()
    return fig


# ============================================================
# Figure 3 : max load reduction
# ============================================================

def _plot_load_reduction(result: SolverResult) -> plt.Figure:
    """
    Show the maximum load (λ'(1)) at each step, compared to trivial.
    """
    history     = result.history
    trivial_max = result.trivial_vector[0]

    if not history:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No history", ha='center', va='center')
        return fig

    steps    = [h.step          for h in history]
    max_loads= [h.load_vector[0] for h in history]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(steps, max_loads, 'o-',
            color=COLOR_LAMBDA, linewidth=2, markersize=7,
            label='Max load at step k')
    ax.axhline(trivial_max,
               color=COLOR_TRIVIAL, linewidth=1.5, linestyle='--',
               label=f'Trivial max = {trivial_max:.4f}')
    ax.fill_between(
        steps,
        [trivial_max] * len(steps), max_loads,
        alpha=0.15, color=COLOR_LEXMIN,
        label=f'Reduction achieved',
    )

    # Annotate final improvement
    final_max = max_loads[-1]
    if final_max < trivial_max:
        pct = 100 * (trivial_max - final_max) / trivial_max
        ax.annotate(
            f"-{pct:.1f}%",
            xy=(steps[-1], final_max),
            xytext=(steps[-1] - 0.5, (trivial_max + final_max) / 2),
            fontsize=10, color=COLOR_LEXMIN, fontweight='bold',
        )

    ax.set_xlabel('Step k')
    ax.set_ylabel("Maximum load  λ'(1)")
    ax.set_title(
        f"Maximum load reduction across steps — {result.instance_name}",
        fontweight='bold',
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig

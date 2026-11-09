"""Figure 3 v2: redraw saved results; no training or attribution recomputation."""
from pathlib import Path
import os
CODE_DIR=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(CODE_DIR.parent/'figure1/.mpl_cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
ROOT=CODE_DIR.parents[1]
OUTPUT=ROOT/'paper/figures/fig3_modality_and_operator_decomposition_v2.png'
OUTPUT_PDF=ROOT/'paper/figures/fig3_modality_and_operator_decomposition_v2.pdf'
OUTPUT_LEGACY=ROOT/'paper/figures/fig3_modality_and_operator_decomposition.png'
OUTPUT_LEGACY_PDF=ROOT/'paper/figures/fig3_modality_and_operator_decomposition.pdf'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'], 'axes.edgecolor': '#2C3E50', 'axes.linewidth': 0.9, 'axes.grid': False, 'xtick.color': '#2C3E50', 'ytick.color': '#2C3E50', 'text.color': '#1A252F', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'patch.edgecolor': 'none', 'patch.linewidth': 0})

def draw():
    block_names=['Seq_A ($e_a$)', 'Seq_B ($e_b$)', 'Seq_Prod ($e_a \\odot e_b$)', 'Seq_Diff ($|e_a - e_b|$)', 'FP_A ($f_a$)', 'FP_B ($f_b$)', 'FP_Prod ($f_a \\odot f_b$)', 'FP_Diff ($|f_a - f_b|$)']
    df_blk=pd.read_csv(CODE_DIR/'data'/'table3a_fusion_block_attributions.csv')
    df_abl=pd.read_csv(CODE_DIR/'data'/'table3b_counterfactual_operator_ablations.csv')
    df_sim=pd.read_csv(CODE_DIR/'data'/'table3c_coldstart_similarity_bins.csv')
    # Plot Figure 3 (3 Panels)
    fig = plt.figure(figsize=(18.5, 6.2), dpi=300)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.15, 1.1], wspace=0.25)

    # Panel A: 8-Block Fusion Attribution Share (Nature Communications NPG softened palette, borderless)
    ax_a = fig.add_subplot(gs[0, 0])
    sub_uni = df_blk[df_blk['Model'] == '3D-Aligned Multimodal']
    pivot_uni = sub_uni.pivot(index='Task', columns='Block', values='Share_Pct')[block_names]
    colors_8 = ['#92B9CE', '#6A9AB8', '#4F7799', '#B2CEDE', '#E8C595', '#DCA472', '#C5825E', '#D59578']
    bottom = np.zeros(3)
    x_tasks = np.arange(3)
    for col_i, b_name in enumerate(block_names):
        vals = pivot_uni[b_name].to_numpy()
        ax_a.bar(x_tasks, vals, bottom=bottom, width=0.52, color=colors_8[col_i],
                 edgecolor='none', linewidth=0, label=b_name)
        for xi, v in enumerate(vals):
            if v >= 7.0:
                ax_a.text(xi, bottom[xi] + v / 2.0, f"{v:.1f}%", ha='center', va='center',
                          fontsize=7.8, fontweight='bold', color='white' if col_i in (2, 6) else '#1A252F')
        bottom += vals

    ax_a.set_xticks(x_tasks)
    ax_a.set_xticklabels(['S1\n(Warm K-K)', 'S2\n(Cold-1 K-U)', 'S3\n(Cold-2 U-U)'],
                         fontsize=10.2, fontweight='bold')
    ax_a.set_ylabel("Attribution Share (% of |Grad × Activation|)", fontsize=10.5, fontweight='bold')
    ax_a.set_ylim(0, 105)
    ax_a.set_title("A | 8-Block Symmetric Fusion Attribution\n(1D Sequence vs. 2D Fingerprint Operators)",
                   fontsize=12.6, fontweight='bold', pad=10)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.4, frameon=True)

    # Panel B: Counterfactual Operator & Fingerprint Sub-Modality Ablation (NC / NPG softened palette, borderless)
    ax_b = fig.add_subplot(gs[0, 1])
    cfg_order = [
        'Full Model (Intact)',
        'w/o Hadamard Product (u*v=0)',
        'w/o Asymmetric Diff (|u-v|=0)',
        'w/o MACCS Keys (Morgan 512 Only)',
        'w/o Morgan ECFP4 (MACCS 167 Only)',
        'w/o Direct Features (ua=ub=0)',
    ]
    cfg_colors = ['#4E9A84', '#5C7FA3', '#75A8C3', '#E2A46D', '#D98A6C', '#CC6E62']
    x_t = np.arange(3)
    w_bar = 0.13
    for i_c, (cfg, col_c) in enumerate(zip(cfg_order, cfg_colors)):
        vals = [df_abl[(df_abl['Task'] == f'Task {t}') & (df_abl['Configuration'] == cfg)]['AUC'].values[0]
                for t in (1, 2, 3)]
        offset = (i_c - 2.5) * w_bar
        rects = ax_b.bar(x_t + offset, vals, width=w_bar * 0.94, color=col_c, edgecolor='none', linewidth=0,
                         label=cfg.replace("Asymmetric Diff", "Absolute Diff"))
        for rect, v in zip(rects, vals):
            if i_c in (0, 2, 4, 5):
                ax_b.text(rect.get_x() + rect.get_width() / 2.0, v + 0.008, f"{v:.2f}",
                          ha='center', va='bottom', fontsize=6.8, rotation=90, fontweight='bold')

    ax_b.set_xticks(x_t)
    ax_b.set_xticklabels(['S1\n(Warm K-K)', 'S2\n(Cold-1 K-U)', 'S3\n(Cold-2 U-U)'],
                         fontsize=10.2, fontweight='bold')
    ax_b.set_ylabel("Test ROC-AUC Score", fontsize=10.5, fontweight='bold')
    ax_b.set_ylim(0.45, 1.07)
    ax_b.set_title("B | Counterfactual Operator & Substructure\nAblation Across Warm and Cold-Start Tasks",
                   fontsize=12.6, fontweight='bold', pad=10)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.0, frameon=True)

    # Panel C: Cold-Start Tanimoto Similarity Generalization Curve & Dynamic Threshold Rescue
    ax_c = fig.add_subplot(gs[0, 2])
    x_sim = np.arange(len(df_sim))
    ax_c.plot(x_sim, df_sim['AUC'], marker='o', linewidth=2.2, markersize=6, color='#4F7799',
              label='Test ROC-AUC (Threshold-Free)')
    ax_c.plot(x_sim, df_sim['F1_Dynamic_0.31'], marker='s', linewidth=2.2, markersize=6, color='#4E9A84',
              label='Test F1 (Validation Dynamic $\\tau^*=0.31$)')
    ax_c.plot(x_sim, df_sim['F1_Fixed_0.5'], marker='^', linewidth=2.0, linestyle='--', markersize=6, color='#CC6E62',
              label='Test F1 (Naive Fixed $\\tau=0.50$)')

    for xi, row_s in df_sim.iterrows():
        ax_c.annotate(
            f"+{row_s['F1_Dynamic_0.31'] - row_s['F1_Fixed_0.5']:.2f} F1",
            xy=(xi, row_s['F1_Dynamic_0.31']),
            xytext=(xi, row_s['F1_Dynamic_0.31'] + 0.055),
            ha='center', fontsize=8.0, fontweight='bold', color='#3B7A67',
            arrowprops=dict(arrowstyle='->', color='#4E9A84', lw=0.8)
        )

    ax_c.set_xticks(x_sim)
    ax_c.set_xticklabels([s.replace('>0.55', '≥0.55') for s in df_sim['Short_Bin']], fontsize=8.8, fontweight='bold')
    ax_c.set_ylabel("Metric Score (AUC / F1)", fontsize=10.5, fontweight='bold')
    ax_c.set_xlabel("Max Tanimoto Similarity ($S_{\\max}$) of Unseen Drug to Training Set", fontsize=9.8, fontweight='bold')
    ax_c.set_ylim(0.25, 0.98)
    ax_c.set_title("C | Cold-Start Generalization vs. Novelty\n& Dynamic Threshold Rescue (S2 Test Set)",
                   fontsize=12.6, fontweight='bold', pad=10)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8, frameon=True)

    # No global suptitle: the manuscript supplies the figure caption.
    fig.savefig(OUTPUT, dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_PDF, bbox_inches='tight')
    fig.savefig(OUTPUT_LEGACY, dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_LEGACY_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f'Generated: {OUTPUT}')
    print(f'Generated: {OUTPUT_PDF}')

if __name__=='__main__':draw()

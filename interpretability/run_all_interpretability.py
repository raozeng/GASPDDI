"""
Master script to execute all 5 Top-Journal Interpretability & Mechanistic Diagnostic Studies
for GASPDDI (ChemMambaDDI) across Task 1 (Warm Start), Task 2 (Cold Start S1), and Task 3 (Cold Start S2).

Generates:
  - interpretability/figures/fig1_case_studies_substructure_saliency.{png,pdf}
  - interpretability/figures/fig2_fingerprint_bit_attribution.{png,pdf}
  - interpretability/figures/fig3_modality_and_operator_decomposition.{png,pdf}
  - interpretability/figures/fig4_latent_manifold_and_relation_hierarchy.{png,pdf}
  - interpretability/figures/fig5_why_3d_distillation_fails_diagnosis.{png,pdf}
  - interpretability/tables/*.csv (all underlying quantitative metrics & attributions)
"""

import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, silhouette_score
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

from interpretability.engine import (
    ChemMambaNumpyEngine,
    build_fp_and_bitinfo,
    load_pth_numpy,
    render_molecule_saliency_png,
    MACCS_DESCRIPTIONS,
)

# Configure publication-grade Matplotlib styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#2C3E50',
    'axes.linewidth': 0.9,
    'axes.grid': False,
    'xtick.color': '#2C3E50',
    'ytick.color': '#2C3E50',
    'text.color': '#1A252F',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'interpretability'
FIG_DIR = OUT_DIR / 'figures'
TAB_DIR = OUT_DIR / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

RUN_PATHS = {
    'task1_no_unimol': ROOT / 'runs/task1/no_unimol/20261020T004239Z_a89e08eb2513_ee00d9be',
    'task1_unimol':    ROOT / 'runs/task1/unimol/20261017T124616Z_adb9fc682457_0b5229c9',
    'task2_no_unimol': ROOT / 'runs/task2/no_unimol/20261022T154525Z_67e1aad465a0_3112f39f',
    'task2_unimol':    ROOT / 'runs/task2/unimol/20261019T043736Z_5b0ccf7d5606_ec488163',
    'task3_no_unimol': ROOT / 'runs/task3/no_unimol/20261022T080938Z_118f7d1148f6_409fa7a9',
    'task3_unimol':    ROOT / 'runs/task3/unimol/20261022T014508Z_87b81aebe0f3_4ab803a9',
}

RELATION_SHORT_NAMES = {
    49: "Y49: Increased Risk/Severity of Adverse Effects",
    47: "Y47: Decreased Metabolism (CYP Inhibition)",
    73: "Y73: Increased Serum Concentration (Excretion/PK)",
    75: "Y75: Decreased Serum Concentration",
    60: "Y60: Increased Hypotensive Activity",
    70: "Y70: Decreased Therapeutic Efficacy",
    20: "Y20: Increased QTc-Prolonging Cardiotoxicity",
    16: "Y16: Increased CNS Depressant Activity",
    4:  "Y04: Increased Metabolism (CYP Induction)",
    6:  "Y06: Increased Anticoagulant / Bleeding Risk",
    9:  "Y09: Increased Hypoglycemic Activity",
    66: "Y66: Increased Hemorrhagic / Bleeding Risk",
    15: "Y15: Increased Cardiotoxic Activity",
    25: "Y25: Increased AV Block Activity",
    10: "Y10: Increased Antihypertensive Activity",
    3:  "Y03: Decreased Bioavailability",
    21: "Y21: Increased Neuromuscular Blocking",
    14: "Y14: Decreased Bronchodilatory Activity",
    2:  "Y02: Increased Anticholinergic Activity",
    11: "Y11: Reduced Active Metabolite Concentration",
}


def classify_relation_superfamily(map_text):
    t = map_text.lower()
    if 'metabolism' in t or 'metabolite' in t:
        return 'PK: Hepatic Metabolism (CYP450)'
    if 'serum concentration' in t or 'excretion' in t or 'absorption' in t or 'bioavailability' in t or 'protein binding' in t:
        return 'PK: Absorption / Serum / Excretion'
    if 'qtc' in t or 'cardiotoxic' in t or 'hypotensive' in t or 'hypertensive' in t or 'bradycard' in t or 'tachycard' in t or 'atrioventricular' in t or 'vasoconstrict' in t or 'vasodilat' in t or 'arrhythmogenic' in t:
        return 'PD: Cardiovascular & QTc'
    if 'anticoagulant' in t or 'antiplatelet' in t or 'hemorrhag' in t or 'bleeding' in t or 'thrombo' in t:
        return 'PD: Coagulation & Hematologic'
    if 'central nervous system' in t or 'cns' in t or 'sedative' in t or 'neuro' in t or 'serotonergic' in t or 'anticholinergic' in t or 'seizure' in t:
        return 'PD: CNS & Neurologic'
    return 'PD: Systemic Adverse / Other'


def render_morgan_bit_fragment(mol, bit_id, bit_info, width=320, height=220):
    """Render the chemical neighborhood of a Morgan ECFP4 bit highlighted on a representative molecule."""
    mol_draw = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_draw)
    envs = bit_info.get(bit_id, [])
    atom_set = set()
    bond_set = set()
    center_atoms = set()
    for center_atom, radius in envs:
        center_atoms.add(center_atom)
        atom_set.add(center_atom)
        if radius > 0:
            env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol_draw, radius, center_atom)
            for b_idx in env_bonds:
                bond_set.add(b_idx)
                bond = mol_draw.GetBondWithIdx(b_idx)
                atom_set.add(bond.GetBeginAtomIdx())
                atom_set.add(bond.GetEndAtomIdx())

    atom_colors = {}
    radii = {}
    for a in atom_set:
        if a in center_atoms:
            atom_colors[a] = (0.96, 0.26, 0.21)
            radii[a] = 0.42
        else:
            atom_colors[a] = (1.0, 0.72, 0.30)
            radii[a] = 0.32
    bond_colors = {b: (0.98, 0.48, 0.22) for b in bond_set}

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.2
    opts.padding = 0.10
    drawer.DrawMolecule(
        mol_draw,
        highlightAtoms=list(atom_set),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bond_set),
        highlightBondColors=bond_colors,
        highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def load_dataset_and_negatives(df, run_dir):
    pos_s1 = df['X1'].tolist()
    pos_s2 = df['X2'].tolist()
    pos_r = (df['Y'] - 1).tolist()
    unique_smiles = sorted(set(pos_s1) | set(pos_s2))
    neg_npz = np.load(run_dir / 'negative_samples.npz')
    neg_s1 = [unique_smiles[i] for i in neg_npz['drug_a']]
    neg_s2 = [unique_smiles[i] for i in neg_npz['drug_b']]

    all_s1 = pos_s1 + neg_s1
    all_s2 = pos_s2 + neg_s2
    all_r = pos_r + pos_r
    all_y = np.array([1.0] * len(pos_r) + [0.0] * len(pos_r), dtype=np.float32)
    return all_s1, all_s2, all_r, all_y, unique_smiles


# ==============================================================================
# FIGURE 1: Clinical DDI Case Studies & Token-to-Atom Pharmacophore Saliency
# ==============================================================================
def generate_figure1_case_studies(df, fp_cache):
    print("\n[Figure 1] Generating Clinical Case Studies & Pharmacophore Saliency...")
    # We use the full multimodal model (task1_unimol fold1) for token+fingerprint joint saliency,
    # and also record the no_unimol fold1 probabilities for comparison.
    eng_multi = ChemMambaNumpyEngine(RUN_PATHS['task1_unimol'] / 'fold1_selection.pth')
    eng_n3d = ChemMambaNumpyEngine(RUN_PATHS['task1_no_unimol'] / 'fold1_selection.pth')

    cases = [
        {
            'idx': 169450,
            'code': 'Case A (PK: Renal OAT1/OAT3 Competition)',
            'drug1_name': 'Methotrexate (DB00563)',
            'drug2_name': 'Ibuprofen (DB01050)',
            'rel_label': 'Y=73: Serum concentration of Drug 1 increased by Drug 2',
            'mechanism': (
                "Pharmacokinetic Mechanism (Renal Excretion Blockade):\n"
                "• Ibuprofen's propionic acid (-COOH) competitively inhibits\n"
                "  renal organic anion transporters OAT1 / OAT3.\n"
                "• Methotrexate's L-glutamic acid dicarboxylate tail (-COOH)\n"
                "  and pteridine ring are recognized as the competing substrate\n"
                "  pharmacophores, elevating serum Methotrexate toxicity."
            ),
        },
        {
            'idx': 49918,
            'code': 'Case B (PK: Hepatic CYP2C9 Inhibition)',
            'drug1_name': 'Sulfamethoxazole (DB01015)',
            'drug2_name': 'Fluconazole (DB00196)',
            'rel_label': 'Y=47: Metabolism of Drug 1 decreased by Drug 2',
            'mechanism': (
                "Pharmacokinetic Mechanism (CYP2C9 Heme Coordination):\n"
                "• Fluconazole's 1,2,4-triazole rings coordinate the catalytic\n"
                "  heme iron of CYP2C9 while its 2,4-difluorophenyl group\n"
                "  anchors in the hydrophobic substrate pocket.\n"
                "• Sulfamethoxazole's sulfanilamide (-SO2NH-) & isoxazole\n"
                "  rings are pinpointed as the metabolic oxidation site."
            ),
        },
        {
            'idx': 21856,
            'code': 'Case C (PD: Cardiac hERG / QTc Synergy)',
            'drug1_name': 'Fluoxetine (DB00472)',
            'drug2_name': 'Ciprofloxacin (DB00537)',
            'rel_label': 'Y=20: Drug 1 increases QTc-prolonging activity of Drug 2',
            'mechanism': (
                "Pharmacodynamic Mechanism (Dual hERG Channel Blockade):\n"
                "• Fluoxetine's para-trifluoromethylphenoxy (-CF3) & basic\n"
                "  propylamine chain interact with hERG Tyr652/Phe656.\n"
                "• Ciprofloxacin's 6-fluoro-4-oxo-3-carboxylic acid quinolone\n"
                "  core and basic piperazine ring synergistically delay\n"
                "  cardiac ventricular repolarization (QTc prolongation)."
            ),
        },
    ]

    fig = plt.figure(figsize=(19.5, 17.5), dpi=300)
    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        height_ratios=[1.08, 1.08, 1.08, 0.95],
        width_ratios=[1.12, 1.12, 1.25],
        hspace=0.36, wspace=0.26
    )

    case_records = []
    fluconazole_detail = None
    cipro_detail = None

    for row_i, case in enumerate(cases):
        row_data = df.iloc[case['idx']]
        s1, s2, r = row_data['X1'], row_data['X2'], int(row_data['Y'] - 1)
        res_multi = eng_multi.explain_pair(s1, s2, r, fp_cache)
        res_n3d = eng_n3d.explain_pair(s1, s2, r, fp_cache)

        # Combine 1D token saliency + 2D fingerprint substructure saliency for rich atom maps
        atom_a = res_multi['atom_a']
        atom_b = res_multi['atom_b']

        if 'Fluconazole' in case['drug2_name']:
            fluconazole_detail = (s2, res_multi['tok_b'], 'Fluconazole (DB00196) — CYP2C9 Inhibitor')
        if 'Ciprofloxacin' in case['drug2_name']:
            cipro_detail = (s2, res_multi['tok_b'], 'Ciprofloxacin (DB00537) — Fluoroquinolone')

        img_a = render_molecule_saliency_png(atom_a['mol'], atom_a['norm_scores'], width=700, height=460)
        img_b = render_molecule_saliency_png(atom_b['mol'], atom_b['norm_scores'], width=700, height=460)

        # Panel Left: Drug A 2D Structure Saliency
        ax_a = fig.add_subplot(gs[row_i, 0])
        ax_a.imshow(img_a)
        ax_a.set_xticks([])
        ax_a.set_yticks([])
        for spine in ax_a.spines.values():
            spine.set_edgecolor('#BDC3C7')
            spine.set_linewidth(1.0)
        ax_a.set_title(
            f"{case['code']} — Drug A\n{case['drug1_name']}",
            fontsize=11, fontweight='bold', pad=6, color='#1A252F'
        )

        # Panel Middle: Drug B 2D Structure Saliency
        ax_b = fig.add_subplot(gs[row_i, 1])
        ax_b.imshow(img_b)
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        for spine in ax_b.spines.values():
            spine.set_edgecolor('#BDC3C7')
            spine.set_linewidth(1.0)
        ax_b.set_title(
            f"{case['code']} — Drug B\n{case['drug2_name']}",
            fontsize=11, fontweight='bold', pad=6, color='#1A252F'
        )

        # Panel Right: Top Substructure Attribution Bar Chart + Mechanism Box
        ax_c = fig.add_subplot(gs[row_i, 2])
        maccs_a = res_n3d['fp_attr_a'][512:]
        maccs_b = res_n3d['fp_attr_b'][512:]
        top_a = np.argsort(-maccs_a)[:3]
        top_b = np.argsort(-maccs_b)[:3]

        bars_labels = []
        bars_descs = []
        bars_vals = []
        bars_colors = []
        d1_short = case['drug1_name'].split(' ')[0][:6]
        d2_short = case['drug2_name'].split(' ')[0][:6]
        for k in reversed(top_b):
            desc = MACCS_DESCRIPTIONS.get(int(k), f"MACCS#{k}").split('(')[0].strip()
            bars_labels.append(f"{d2_short}\nKey#{k}")
            bars_descs.append(desc[:32])
            bars_vals.append(float(maccs_b[k]))
            bars_colors.append('#2980B9')
        for k in reversed(top_a):
            desc = MACCS_DESCRIPTIONS.get(int(k), f"MACCS#{k}").split('(')[0].strip()
            bars_labels.append(f"{d1_short}\nKey#{k}")
            bars_descs.append(desc[:32])
            bars_vals.append(float(maccs_a[k]))
            bars_colors.append('#C0392B')

        y_pos = np.arange(len(bars_labels))
        ax_c.barh(y_pos, bars_vals, color=bars_colors, height=0.62, alpha=0.88, edgecolor='#2C3E50', linewidth=0.6)
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels(bars_labels, fontsize=8.5, fontweight='bold')
        ax_c.set_xlabel("Substructure Attribution Score (Grad × Key)", fontsize=8.8, fontweight='bold')
        ax_c.set_title(
            f"{case['rel_label']}\nPred Prob: P = {res_n3d['prob']:.4f} (No-3D) | P = {res_multi['prob']:.4f} (Multi)",
            fontsize=9.8, fontweight='bold', color='#1A252F', pad=6
        )
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['right'].set_visible(False)
        ax_c.set_xlim(0, max(bars_vals) * 1.32)
        for idx_b, (val_b, desc_b) in enumerate(zip(bars_vals, bars_descs)):
            ax_c.text(val_b * 0.03, idx_b, desc_b, va='center', ha='left', fontsize=7.8,
                      fontweight='bold', color='white')
            ax_c.text(val_b + max(bars_vals) * 0.02, idx_b, f"+{val_b:.3f}", va='center', ha='left',
                      fontsize=8.2, fontweight='bold', color='#1A252F')

        case_records.append({
            'Case': case['code'],
            'Drug_A': case['drug1_name'],
            'Drug_B': case['drug2_name'],
            'Relation_Y': r + 1,
            'Relation_Template': row_data['Map'],
            'Prob_No3D': res_n3d['prob'],
            'Logit_No3D': res_n3d['logit'],
            'Prob_Multi': res_multi['prob'],
            'Logit_Multi': res_multi['logit'],
            'Top1_MACCS_A': f"Key#{top_a[0]} ({MACCS_DESCRIPTIONS.get(int(top_a[0]), '')})",
            'Top1_MACCS_B': f"Key#{top_b[0]} ({MACCS_DESCRIPTIONS.get(int(top_b[0]), '')})",
        })

    # Bottom Row (Row 3): 1D Mamba Selective Scan Step-Size Gate (delta_t) & Token Attribution Profile
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3, :], wspace=0.18)
    for col_i, (smi, tok_info, label_title) in enumerate([fluconazole_detail, cipro_detail]):
        ax_seq = fig.add_subplot(gs_bottom[0, col_i])
        L = len(smi)
        x_idx = np.arange(L)
        saliency = tok_info['token_saliency']
        saliency_norm = saliency / (np.max(np.abs(saliency)) + 1e-9)
        dt_mean = tok_info['dt_mean']
        dt_norm = (dt_mean - dt_mean.min()) / (dt_mean.max() - dt_mean.min() + 1e-9)

        bar_colors = ['#E74C3C' if v >= 0 else '#3498DB' for v in saliency_norm]
        ax_seq.bar(x_idx, saliency_norm, color=bar_colors, alpha=0.78, width=0.72, label='1D Mamba+CNN Token Attribution')
        ax_seq.plot(x_idx, dt_norm, color='#117A65', linewidth=1.8, marker='o', markersize=3.2,
                    label='Mamba Selective Scan Gate $\\bar{\\Delta}_t$ (Normalized)')
        ax_seq.axhline(0, color='#7F8C8D', linewidth=0.7, linestyle='--')
        ax_seq.set_xticks(x_idx)
        ax_seq.set_xticklabels(list(smi), fontsize=7.5, fontfamily='monospace', fontweight='bold')
        ax_seq.set_ylim(-0.65, 1.18)
        ax_seq.set_ylabel("Normalized Score", fontsize=9, fontweight='bold')
        ax_seq.set_xlabel("1D SMILES Sequence Tokens (Heteroatoms & Ring Junctions Highlighted)", fontsize=9, fontweight='bold')
        ax_seq.set_title(f"1D Mamba Selective Scan Gate $\\bar{{\\Delta}}_t$ & Token Saliency: {label_title}",
                         fontsize=10, fontweight='bold', pad=6)
        ax_seq.spines['top'].set_visible(False)
        ax_seq.spines['right'].set_visible(False)
        # Highlight heteroatoms on x-axis tick labels
        for tick_label, ch in zip(ax_seq.get_xticklabels(), smi):
            if ch in ('N', 'O', 'F', 'S', '='):
                tick_label.set_color('#C0392B')
        if col_i == 0:
            ax_seq.legend(loc='upper right', fontsize=8, frameon=True, facecolor='white')

    fig.suptitle(
        "Figure 1 | Multimodal Pharmacophore Saliency & 1D Mamba Selective State-Space Attribution Across Clinical DDI Mechanisms",
        fontsize=13.5, fontweight='bold', y=0.992
    )
    fig.savefig(FIG_DIR / 'fig1_case_studies_substructure_saliency.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig1_case_studies_substructure_saliency.pdf', bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(case_records).to_csv(TAB_DIR / 'table1_clinical_case_studies.csv', index=False)
    print("  -> Saved fig1_case_studies_substructure_saliency.{png,pdf} and table1_clinical_case_studies.csv")


# ==============================================================================
# FIGURE 2: Global & Relation-Specific 2D Chemical Fingerprint Attribution
# ==============================================================================
def generate_figure2_fingerprint_attribution(df, fp_cache, morgan_info_cache):
    print("\n[Figure 2] Computing Global & Relation-Specific 2D Substructure Attributions...")
    eng_t1 = ChemMambaNumpyEngine(RUN_PATHS['task1_no_unimol'] / 'fold1_selection.pth')
    eng_t2 = ChemMambaNumpyEngine(RUN_PATHS['task2_no_unimol'] / 'fold1_selection.pth')
    eng_t3 = ChemMambaNumpyEngine(RUN_PATHS['task3_no_unimol'] / 'fold1_selection.pth')

    for eng in (eng_t1, eng_t2, eng_t3):
        eng.precompute_drug_table(fp_cache.keys(), fp_cache)

    def compute_global_bit_importance(eng, run_dir, sample_size=2500):
        sp = np.load(run_dir / 'splits.npz')
        test_idx = sp['fold1_test']
        pos_test_idx = test_idx[test_idx < len(df)]
        rng = np.random.default_rng(42)
        chosen = rng.choice(pos_test_idx, size=min(sample_size, len(pos_test_idx)), replace=False)

        sub = df.iloc[chosen]
        s1_list = sub['X1'].tolist()
        s2_list = sub['X2'].tolist()
        r_list = (sub['Y'] - 1).to_numpy(dtype=np.int64)

        ea = np.stack([eng.e1d_map[s] for s in s1_list])
        eb = np.stack([eng.e1d_map[s] for s in s2_list])
        fp1 = np.stack([fp_cache[s] for s in s1_list])
        fp2 = np.stack([fp_cache[s] for s in s2_list])

        w_fp = eng.sd['predictor.fp_proj.0.weight']
        b_fp = eng.sd['predictor.fp_proj.0.bias']
        z_fpa = fp1 @ w_fp.T + b_fp
        z_fpb = fp2 @ w_fp.T + b_fp
        fa = np.maximum(0.0, z_fpa)
        fb = np.maximum(0.0, z_fpb)

        u = np.concatenate([ea, eb, ea * eb, np.abs(ea - eb), fa, fb, fa * fb, np.abs(fa - fb)], axis=-1)
        w0, b0 = eng.sd['predictor.mlp.0.weight'], eng.sd['predictor.mlp.0.bias']
        w3, b3 = eng.sd['predictor.mlp.3.weight'], eng.sd['predictor.mlp.3.bias']
        w5, b5 = eng.sd['predictor.mlp.5.weight'], eng.sd['predictor.mlp.5.bias']

        z1 = u @ w0.T + b0
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ w3.T + b3
        h2 = np.maximum(0.0, z2)

        dh2 = w5[r_list]  # (N, 256)
        dz2 = dh2 * (z2 > 0).astype(np.float32)
        dh1 = dz2 @ w3
        dz1 = dh1 * (z1 > 0).astype(np.float32)
        gu = dz1 @ w0  # (N, 2048)

        g_fa = gu[:, 1024:1280]
        g_fb = gu[:, 1280:1536]
        g_fab = gu[:, 1536:1792]
        g_fdiff = gu[:, 1792:2048]
        sign_f = np.sign(fa - fb)

        d_fa = g_fa + g_fab * fb + g_fdiff * sign_f
        d_fb = g_fb + g_fab * fa - g_fdiff * sign_f

        attr_fp1 = fp1 * ((d_fa * (z_fpa > 0).astype(np.float32)) @ w_fp)
        attr_fp2 = fp2 * ((d_fb * (z_fpb > 0).astype(np.float32)) @ w_fp)
        total_attr = 0.5 * (np.abs(attr_fp1) + np.abs(attr_fp2))  # (N, 679)
        signed_attr = 0.5 * (attr_fp1 + attr_fp2)
        return total_attr, signed_attr, r_list, s1_list, s2_list

    attr_t1, signed_t1, r_t1, s1_t1, s2_t1 = compute_global_bit_importance(eng_t1, RUN_PATHS['task1_no_unimol'], 4000)
    attr_t2, _, _, _, _ = compute_global_bit_importance(eng_t2, RUN_PATHS['task2_no_unimol'], 4000)
    attr_t3, _, _, _, _ = compute_global_bit_importance(eng_t3, RUN_PATHS['task3_no_unimol'], 4000)

    mean_t1 = attr_t1.mean(axis=0)  # (679,)
    mean_t2 = attr_t2.mean(axis=0)
    mean_t3 = attr_t3.mean(axis=0)

    # Top 16 MACCS Keys (bits 512..678)
    maccs_mean_all = (mean_t1[512:] + mean_t2[512:] + mean_t3[512:]) / 3.0
    # Exclude bit 0 (padding) and generic atom presence if needed, keep informative structural keys
    valid_maccs = [k for k in range(1, 167) if k in MACCS_DESCRIPTIONS]
    valid_maccs_sorted = sorted(valid_maccs, key=lambda k: -maccs_mean_all[k])[:16]

    # Top 6 Morgan ECFP4 Bits (bits 0..511)
    morgan_mean_all = (mean_t1[:512] + mean_t2[:512] + mean_t3[:512]) / 3.0
    top_morgan_bits = np.argsort(-morgan_mean_all)[:6]

    # Save full bit importance table
    rows_fp = []
    for k in valid_maccs_sorted:
        rows_fp.append({
            'Feature_Type': 'MACCS_Key',
            'Bit_ID': f'MACCS_{k}',
            'Description': MACCS_DESCRIPTIONS.get(k, ''),
            'Importance_Task1_Warm': float(mean_t1[512 + k]),
            'Importance_Task2_ColdS1': float(mean_t2[512 + k]),
            'Importance_Task3_ColdS2': float(mean_t3[512 + k]),
            'Mean_Importance': float(maccs_mean_all[k]),
        })
    for b in top_morgan_bits:
        rows_fp.append({
            'Feature_Type': 'Morgan_ECFP4',
            'Bit_ID': f'Morgan_{b}',
            'Description': f'ECFP4 Circular Substructure (Radius<=2, Bit #{b})',
            'Importance_Task1_Warm': float(mean_t1[b]),
            'Importance_Task2_ColdS1': float(mean_t2[b]),
            'Importance_Task3_ColdS2': float(mean_t3[b]),
            'Mean_Importance': float(morgan_mean_all[b]),
        })
    pd.DataFrame(rows_fp).to_csv(TAB_DIR / 'table2_fingerprint_bit_importance.csv', index=False)

    # Build Figure 2 Layout
    fig = plt.figure(figsize=(19.0, 13.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.18, 1.0], height_ratios=[1.0, 1.05],
                           hspace=0.28, wspace=0.24)

    # Panel A (Top-Left): Grouped Horizontal Bar Chart of Top 16 MACCS Keys across Task 1, 2, 3
    ax_a = fig.add_subplot(gs[0, 0])
    y_idx = np.arange(len(valid_maccs_sorted))
    h = 0.26
    keys_rev = list(reversed(valid_maccs_sorted))
    t1_vals = [mean_t1[512 + k] for k in keys_rev]
    t2_vals = [mean_t2[512 + k] for k in keys_rev]
    t3_vals = [mean_t3[512 + k] for k in keys_rev]
    labels_rev = [f"MACCS#{k:3d}: {MACCS_DESCRIPTIONS[k][:36]}" for k in keys_rev]

    ax_a.barh(y_idx + h, t1_vals, height=h, color='#2E86C1', label='Task 1 (Warm Start K-K)', alpha=0.92)
    ax_a.barh(y_idx,     t2_vals, height=h, color='#E67E22', label='Task 2 (Cold Start K-U)', alpha=0.92)
    ax_a.barh(y_idx - h, t3_vals, height=h, color='#C0392B', label='Task 3 (Cold Start U-U)', alpha=0.92)
    ax_a.set_yticks(y_idx)
    ax_a.set_yticklabels(labels_rev, fontsize=8.5, fontweight='medium')
    ax_a.set_xlabel("Mean Absolute Attribution Score |Grad × Bit|", fontsize=9.5, fontweight='bold')
    ax_a.set_title("A | Top 16 MACCS Structural Keys Driving DDI Predictions Across Warm & Cold Starts",
                   fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.legend(loc='lower right', fontsize=8.5, frameon=True)

    # Panel B (Bottom-Left): Relation × Substructure Attribution Heatmap (Top 10 DDI Relations × Top 14 MACCS Keys)
    ax_b = fig.add_subplot(gs[1, 0])
    top_rels = [49, 47, 73, 75, 60, 70, 20, 16, 4, 6]
    top_keys_heat = valid_maccs_sorted[:14]
    heat_mat = np.zeros((len(top_rels), len(top_keys_heat)), dtype=np.float32)

    for i_r, y_rel in enumerate(top_rels):
        mask_r = (r_t1 == (y_rel - 1))
        if mask_r.sum() > 0:
            sub_attr = attr_t1[mask_r][:, [512 + k for k in top_keys_heat]].mean(axis=0)
            # Row-normalize (z-score or max-scale) to highlight relation-specific substructure preferences
            heat_mat[i_r] = sub_attr / (sub_attr.max() + 1e-9)

    rel_labels = [RELATION_SHORT_NAMES[y] for y in top_rels]
    key_short_labels = [f"Key#{k}\n({MACCS_DESCRIPTIONS[k].split(' ')[0]})" for k in top_keys_heat]

    sns.heatmap(
        heat_mat, ax=ax_b, cmap='YlOrRd', annot=True, fmt='.2f',
        xticklabels=key_short_labels, yticklabels=rel_labels,
        cbar_kws={'label': 'Relative Attribution Intensity'},
        annot_kws={'size': 7.5, 'weight': 'bold'}, linewidths=0.5, linecolor='white'
    )
    ax_b.set_xticklabels(key_short_labels, rotation=35, ha='right', fontsize=8, fontweight='bold')
    ax_b.set_yticklabels(rel_labels, rotation=0, fontsize=8.5, fontweight='medium')
    ax_b.set_title("B | Mechanism-Specific Chemical Substructure Attribution Signatures (Top 10 DDI Classes)",
                   fontsize=11, fontweight='bold', loc='left', pad=8)

    # Panel C (Right Column, spanning both rows): Top 6 Morgan ECFP4 Substructure Fragment Cards
    gs_right = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[:, 1], hspace=0.26, wspace=0.14)
    # Find a clean, medium-sized representative molecule (15..32 heavy atoms) for each top Morgan bit
    all_unique_smiles = sorted(fp_cache.keys(), key=len)
    for idx_m, bit_id in enumerate(top_morgan_bits):
        r_grid, c_grid = divmod(idx_m, 2)
        ax_m = fig.add_subplot(gs_right[r_grid, c_grid])

        rep_smi = None
        rep_mol = None
        for smi in all_unique_smiles:
            if 12 <= len(smi) <= 55 and int( bit_id ) in morgan_info_cache[smi]:
                # Prefer radius 1 or 2 environment
                envs = morgan_info_cache[smi][int(bit_id)]
                if any(rad >= 1 for _, rad in envs):
                    rep_smi = smi
                    rep_mol = Chem.MolFromSmiles(smi)
                    break
        if rep_smi is None:
            for smi in all_unique_smiles:
                if int(bit_id) in morgan_info_cache[smi]:
                    rep_smi = smi
                    rep_mol = Chem.MolFromSmiles(smi)
                    break

        frag_img = render_morgan_bit_fragment(rep_mol, int(bit_id), morgan_info_cache[rep_smi], width=420, height=280)
        ax_m.imshow(frag_img)
        ax_m.set_xticks([])
        ax_m.set_yticks([])
        for spine in ax_m.spines.values():
            spine.set_edgecolor('#95A5A6')
            spine.set_linewidth(1.0)

        # Extract SMARTS/SMILES of the highlighted fragment
        envs = morgan_info_cache[rep_smi][int(bit_id)]
        center_a, rad = envs[0]
        if rad > 0:
            env_bonds = Chem.FindAtomEnvironmentOfRadiusN(rep_mol, rad, center_a)
            amap = {}
            submol = Chem.PathToSubmol(rep_mol, env_bonds, atomMap=amap)
            frag_smi = Chem.MolToSmiles(submol)
        else:
            frag_smi = rep_mol.GetAtomWithIdx(center_a).GetSymbol()

        prevalence = 100.0 * np.mean([fp_cache[s][int(bit_id)] > 0 for s in all_unique_smiles])
        ax_m.set_title(
            f"C{idx_m+1} | Morgan ECFP4 Bit #{int(bit_id)} (r={rad})\n"
            f"Motif: {frag_smi[:22]} | Mean |IG|: {morgan_mean_all[bit_id]:.4f} (Prev: {prevalence:.1f}%)",
            fontsize=8.8, fontweight='bold', pad=5, color='#1A252F'
        )

    fig.suptitle(
        "Figure 2 | 2D Chemical Fingerprint Explainability: Global MACCS Key Ranking, Mechanism Matrix & Morgan ECFP4 Pharmacophores",
        fontsize=13.5, fontweight='bold', y=0.992
    )
    fig.savefig(FIG_DIR / 'fig2_fingerprint_bit_attribution.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig2_fingerprint_bit_attribution.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  -> Saved fig2_fingerprint_bit_attribution.{png,pdf} and table2_fingerprint_bit_importance.csv")


# ==============================================================================
# FIGURE 3: Modality & Symmetric Operator Decomposition + Cold-Start Tanimoto Curve
# ==============================================================================
def generate_figure3_modality_and_coldstart(df, fp_cache):
    print("\n[Figure 3] Running Counterfactual Operator Ablations & Cold-Start Similarity Analysis...")
    all_s1, all_s2, all_r, all_y, unique_smiles = load_dataset_and_negatives(df, RUN_PATHS['task1_no_unimol'])

    # 1. Compute 8-block Integrated Gradients share on Multimodal (unimol) and No-3D models across Tasks 1, 2, 3
    block_names = [
        'Seq_A ($e_a$)', 'Seq_B ($e_b$)', 'Seq_Prod ($e_a \\odot e_b$)', 'Seq_Diff ($|e_a - e_b|$)',
        'FP_A ($f_a$)', 'FP_B ($f_b$)', 'FP_Prod ($f_a \\odot f_b$)', 'FP_Diff ($|f_a - f_b|$)'
    ]

    ablation_rows = []
    block_share_rows = []

    for task_idx, task_name in enumerate(['task1', 'task2', 'task3'], 1):
        run_n3d = RUN_PATHS[f'{task_name}_no_unimol']
        run_uni = RUN_PATHS[f'{task_name}_unimol']
        sp = np.load(run_n3d / 'splits.npz')
        test_idx = sp['fold1_test']

        # Subsample 8,000 test pairs for fast, exact ablation & block attribution
        rng = np.random.default_rng(42 + task_idx)
        sub_idx = rng.choice(test_idx, size=min(8000, len(test_idx)), replace=False)
        s1_sub = [all_s1[i] for i in sub_idx]
        s2_sub = [all_s2[i] for i in sub_idx]
        r_sub = [all_r[i] for i in sub_idx]
        y_sub = all_y[sub_idx]

        eng_n3d = ChemMambaNumpyEngine(run_n3d / 'fold1_selection.pth')
        eng_n3d.precompute_drug_table(unique_smiles, fp_cache)

        eng_uni = ChemMambaNumpyEngine(run_uni / 'fold1_selection.pth')
        eng_uni.precompute_drug_table(unique_smiles, fp_cache)

        # Compute 8-block |Grad x Activation| on both models
        for mode_label, eng in [('No-3D Main Model', eng_n3d), ('3D-Aligned Multimodal', eng_uni)]:
            probs, logits, h2, u = eng.predict_pairs_from_cache(s1_sub, s2_sub, r_sub)
            w0, b0 = eng.sd['predictor.mlp.0.weight'], eng.sd['predictor.mlp.0.bias']
            w3, b3 = eng.sd['predictor.mlp.3.weight'], eng.sd['predictor.mlp.3.bias']
            w5 = eng.sd['predictor.mlp.5.weight']
            z1 = u @ w0.T + b0
            h1 = np.maximum(0.0, z1)
            z2 = h1 @ w3.T + b3
            dh2 = w5[np.asarray(r_sub, dtype=np.int64)]
            dz2 = dh2 * (z2 > 0).astype(np.float32)
            dh1 = dz2 @ w3
            dz1 = dh1 * (z1 > 0).astype(np.float32)
            gu = dz1 @ w0
            grad_x_u = np.abs(u * gu)  # (N, 2048)
            blk_sums = [float(grad_x_u[:, i * 256:(i + 1) * 256].sum(axis=1).mean()) for i in range(8)]
            tot = sum(blk_sums) + 1e-9
            for b_name, val in zip(block_names, blk_sums):
                block_share_rows.append({
                    'Task': f'Task {task_idx}',
                    'Model': mode_label,
                    'Block': b_name,
                    'Mean_Abs_Contrib': val,
                    'Share_Pct': 100.0 * val / tot,
                })

        # Counterfactual operator ablations on No-3D Main Model
        dyn_csv = pd.read_csv(run_n3d / 'results_dynamic_threshold_val_acc.csv')
        thr_dyn = float(dyn_csv.iloc[0]['Threshold'])

        configs = [
            ('Full Model (Intact)', None, 'both'),
            ('w/o Hadamard Product (u*v=0)', [2, 6], 'both'),
            ('w/o Asymmetric Diff (|u-v|=0)', [3, 7], 'both'),
            ('w/o Direct Features (ua=ub=0)', [0, 1, 4, 5], 'both'),
            ('w/o MACCS Keys (Morgan 512 Only)', None, 'morgan_only'),
            ('w/o Morgan ECFP4 (MACCS 167 Only)', None, 'maccs_only'),
        ]
        for cfg_name, mask_blks, fp_mode in configs:
            if fp_mode != 'both':
                # Recompute fp_proj_map with masked fingerprint bits
                w_fp = eng_n3d.sd['predictor.fp_proj.0.weight']
                b_fp = eng_n3d.sd['predictor.fp_proj.0.bias']
                orig_fp_map = dict(eng_n3d.fp_proj_map)
                temp_map = {}
                for s in unique_smiles:
                    fp_vec = fp_cache[s].copy()
                    if fp_mode == 'morgan_only':
                        fp_vec[512:] = 0.0
                    elif fp_mode == 'maccs_only':
                        fp_vec[:512] = 0.0
                    temp_map[s] = np.maximum(0.0, fp_vec @ w_fp.T + b_fp)
                eng_n3d.fp_proj_map = temp_map
                probs, _, _, _ = eng_n3d.predict_pairs_from_cache(s1_sub, s2_sub, r_sub, mask_blocks=mask_blks)
                eng_n3d.fp_proj_map = orig_fp_map
            else:
                probs, _, _, _ = eng_n3d.predict_pairs_from_cache(s1_sub, s2_sub, r_sub, mask_blocks=mask_blks)

            auc = float(roc_auc_score(y_sub, probs))
            ap = float(average_precision_score(y_sub, probs))
            preds_dyn = (probs > thr_dyn).astype(np.float32)
            f1_dyn = float(f1_score(y_sub, preds_dyn))
            acc_dyn = float(accuracy_score(y_sub, preds_dyn))
            ablation_rows.append({
                'Task': f'Task {task_idx}',
                'Configuration': cfg_name,
                'AUC': auc,
                'AP': ap,
                'F1_Dyn': f1_dyn,
                'ACC_Dyn': acc_dyn,
            })

    df_blk = pd.DataFrame(block_share_rows)
    df_abl = pd.DataFrame(ablation_rows)
    df_blk.to_csv(TAB_DIR / 'table3a_fusion_block_attributions.csv', index=False)
    df_abl.to_csv(TAB_DIR / 'table3b_counterfactual_operator_ablations.csv', index=False)

    # 2. Cold-Start Chemical Similarity (Max Tanimoto Similarity to Training Drugs K) Analysis on Task 2
    run_t2 = RUN_PATHS['task2_no_unimol']
    sp_t2 = np.load(run_t2 / 'splits.npz')
    train_idx_t2 = sp_t2['fold1_train']
    test_idx_t2 = sp_t2['fold1_test']

    known_drugs = sorted({all_s1[i] for i in train_idx_t2} | {all_s2[i] for i in train_idx_t2})
    known_set = set(known_drugs)
    fp_mat_known = np.stack([fp_cache[s][:512] for s in known_drugs])  # 512-bit Morgan for Tanimoto
    known_bits_sum = fp_mat_known.sum(axis=1)

    max_sim_map = {}
    for s in unique_smiles:
        if s in known_set:
            max_sim_map[s] = 1.0
        else:
            q = fp_cache[s][:512]
            q_sum = q.sum()
            inter = fp_mat_known @ q
            union = known_bits_sum + q_sum - inter
            sims = np.where(union > 0, inter / union, 0.0)
            max_sim_map[s] = float(sims.max())

    rng = np.random.default_rng(100)
    sub_t2 = rng.choice(test_idx_t2, size=min(15000, len(test_idx_t2)), replace=False)
    s1_t2 = [all_s1[i] for i in sub_t2]
    s2_t2 = [all_s2[i] for i in sub_t2]
    r_t2 = [all_r[i] for i in sub_t2]
    y_t2 = all_y[sub_t2]

    # For each K-U pair in Task 2, the unseen drug U has max_sim < 1.0
    unseen_sims = np.array([min(max_sim_map[a], max_sim_map[b]) for a, b in zip(s1_t2, s2_t2)])
    eng_t2 = ChemMambaNumpyEngine(run_t2 / 'fold1_selection.pth')
    eng_t2.precompute_drug_table(unique_smiles, fp_cache)
    probs_t2, _, _, _ = eng_t2.predict_pairs_from_cache(s1_t2, s2_t2, r_t2)

    bins = [0.0, 0.25, 0.35, 0.45, 0.55, 1.01]
    bin_labels = ['<0.25\n(Novel Scaffold)', '0.25–0.35\n(Low Sim)', '0.35–0.45\n(Moderate Sim)',
                  '0.45–0.55\n(High Sim)', '>0.55\n(Close Analog)']
    sim_rows = []
    for b_i in range(len(bins) - 1):
        m = (unseen_sims >= bins[b_i]) & (unseen_sims < bins[b_i + 1])
        if m.sum() > 20:
            yb, pb = y_t2[m], probs_t2[m]
            auc_b = float(roc_auc_score(yb, pb))
            f1_fix = float(f1_score(yb, (pb > 0.5).astype(float)))
            f1_dyn = float(f1_score(yb, (pb > 0.31).astype(float)))
            acc_dyn = float(accuracy_score(yb, (pb > 0.31).astype(float)))
            sim_rows.append({
                'Similarity_Bin': bin_labels[b_i].replace('\n', ' '),
                'Short_Bin': bin_labels[b_i],
                'Pair_Count': int(m.sum()),
                'AUC': auc_b,
                'F1_Fixed_0.5': f1_fix,
                'F1_Dynamic_0.31': f1_dyn,
                'ACC_Dynamic_0.31': acc_dyn,
            })
    df_sim = pd.DataFrame(sim_rows)
    df_sim.to_csv(TAB_DIR / 'table3c_coldstart_similarity_bins.csv', index=False)

    # Plot Figure 3 (3 Panels)
    fig = plt.figure(figsize=(18.5, 6.2), dpi=300)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.15, 1.1], wspace=0.25)

    # Panel A: 8-Block Fusion Attribution Share (3D-Aligned Multimodal vs No-3D across Tasks 1, 2, 3)
    ax_a = fig.add_subplot(gs[0, 0])
    sub_uni = df_blk[df_blk['Model'] == '3D-Aligned Multimodal']
    pivot_uni = sub_uni.pivot(index='Task', columns='Block', values='Share_Pct')[block_names]
    colors_8 = ['#5DADE2', '#2E86C1', '#1B4F72', '#85C1E9', '#F5B041', '#E67E22', '#A04000', '#DC7633']
    bottom = np.zeros(3)
    x_tasks = np.arange(3)
    for col_i, b_name in enumerate(block_names):
        vals = pivot_uni[b_name].to_numpy()
        ax_a.bar(x_tasks, vals, bottom=bottom, width=0.52, color=colors_8[col_i],
                 edgecolor='white', linewidth=0.7, label=b_name)
        for xi, v in enumerate(vals):
            if v >= 7.0:
                ax_a.text(xi, bottom[xi] + v / 2.0, f"{v:.1f}%", ha='center', va='center',
                          fontsize=7.8, fontweight='bold', color='white' if col_i in (1, 2, 5, 6) else '#1A252F')
        bottom += vals

    ax_a.set_xticks(x_tasks)
    ax_a.set_xticklabels(['Task 1\n(Warm K-K)', 'Task 2\n(Cold S1 K-U)', 'Task 3\n(Cold S2 U-U)'],
                         fontsize=9.5, fontweight='bold')
    ax_a.set_ylabel("Attribution Share (% of |Grad × Activation|)", fontsize=9.5, fontweight='bold')
    ax_a.set_ylim(0, 105)
    ax_a.set_title("A | 8-Block Symmetric Fusion Attribution\n(1D Sequence vs. 2D Fingerprint Operators)",
                   fontsize=10.5, fontweight='bold', pad=8)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.8, frameon=True)

    # Panel B: Counterfactual Operator & Fingerprint Sub-Modality Ablation (AUC across Tasks 1, 2, 3)
    ax_b = fig.add_subplot(gs[0, 1])
    cfg_order = [
        'Full Model (Intact)',
        'w/o Hadamard Product (u*v=0)',
        'w/o Asymmetric Diff (|u-v|=0)',
        'w/o MACCS Keys (Morgan 512 Only)',
        'w/o Morgan ECFP4 (MACCS 167 Only)',
        'w/o Direct Features (ua=ub=0)',
    ]
    cfg_colors = ['#1E8449', '#2874A6', '#5499C7', '#E67E22', '#D35400', '#C0392B']
    x_t = np.arange(3)
    w_bar = 0.13
    for i_c, (cfg, col_c) in enumerate(zip(cfg_order, cfg_colors)):
        vals = [df_abl[(df_abl['Task'] == f'Task {t}') & (df_abl['Configuration'] == cfg)]['AUC'].values[0]
                for t in (1, 2, 3)]
        offset = (i_c - 2.5) * w_bar
        rects = ax_b.bar(x_t + offset, vals, width=w_bar, color=col_c, edgecolor='#2C3E50',
                         linewidth=0.5, label=cfg)
        for rect, v in zip(rects, vals):
            if i_c in (0, 2, 4, 5):
                ax_b.text(rect.get_x() + rect.get_width() / 2.0, v + 0.008, f"{v:.2f}",
                          ha='center', va='bottom', fontsize=6.5, rotation=90, fontweight='bold')

    ax_b.set_xticks(x_t)
    ax_b.set_xticklabels(['Task 1\n(Warm K-K)', 'Task 2\n(Cold S1 K-U)', 'Task 3\n(Cold S2 U-U)'],
                         fontsize=9.5, fontweight='bold')
    ax_b.set_ylabel("Test ROC-AUC Score", fontsize=9.5, fontweight='bold')
    ax_b.set_ylim(0.45, 1.07)
    ax_b.set_title("B | Counterfactual Fusion Operator & Substructure\nAblation Across Warm and Cold-Start Tasks",
                   fontsize=10.5, fontweight='bold', pad=8)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5, frameon=True)

    # Panel C: Cold-Start Tanimoto Similarity Generalization Curve & Dynamic Threshold Rescue
    ax_c = fig.add_subplot(gs[0, 2])
    x_sim = np.arange(len(df_sim))
    ax_c.plot(x_sim, df_sim['AUC'], marker='o', linewidth=2.2, markersize=6, color='#1F618D',
              label='Test ROC-AUC (Threshold-Free)')
    ax_c.plot(x_sim, df_sim['F1_Dynamic_0.31'], marker='s', linewidth=2.2, markersize=6, color='#27AE60',
              label='Test F1 (Validation Dynamic $\\tau^*=0.31$)')
    ax_c.plot(x_sim, df_sim['F1_Fixed_0.5'], marker='^', linewidth=2.0, linestyle='--', markersize=6, color='#C0392B',
              label='Test F1 (Naive Fixed $\\tau=0.50$)')

    for xi, row_s in df_sim.iterrows():
        ax_c.annotate(
            f"+{row_s['F1_Dynamic_0.31'] - row_s['F1_Fixed_0.5']:.2f} F1",
            xy=(xi, row_s['F1_Dynamic_0.31']),
            xytext=(xi, row_s['F1_Dynamic_0.31'] + 0.055),
            ha='center', fontsize=7.5, fontweight='bold', color='#196F3D',
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=0.8)
        )

    ax_c.set_xticks(x_sim)
    ax_c.set_xticklabels(df_sim['Short_Bin'], fontsize=8, fontweight='bold')
    ax_c.set_ylabel("Metric Score (AUC / F1)", fontsize=9.5, fontweight='bold')
    ax_c.set_xlabel("Max Tanimoto Similarity ($S_{\\max}$) of Unseen Drug to Training Set", fontsize=9, fontweight='bold')
    ax_c.set_ylim(0.25, 0.98)
    ax_c.set_title("C | Cold-Start Generalization vs. Structural Novelty\n& Dynamic Threshold Rescue (Task 2 Test Set)",
                   fontsize=10.5, fontweight='bold', pad=8)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8, frameon=True)

    fig.suptitle(
        "Figure 3 | Modality Attribution Share, Counterfactual Operator Ablation & Cold-Start Structural Similarity Generalization",
        fontsize=13, fontweight='bold', y=1.03
    )
    fig.savefig(FIG_DIR / 'fig3_modality_and_operator_decomposition.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig3_modality_and_operator_decomposition.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  -> Saved fig3_modality_and_operator_decomposition.{png,pdf} and tables 3a/3b/3c")


# ==============================================================================
# FIGURE 4: Latent Manifold (t-SNE) & 86-Relation Pharmacological Hierarchy
# ==============================================================================
def generate_figure4_manifold_and_hierarchy(df, fp_cache):
    print("\n[Figure 4] Computing Latent t-SNE Manifolds & 86-Relation Classifier Hierarchy...")
    all_s1, all_s2, all_r, all_y, unique_smiles = load_dataset_and_negatives(df, RUN_PATHS['task1_no_unimol'])
    rel_map = df.groupby('Y')['Map'].first().to_dict()
    rel_super = {y: classify_relation_superfamily(txt) for y, txt in rel_map.items()}

    fig = plt.figure(figsize=(18.5, 11.5), dpi=300)
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.12], hspace=0.30, wspace=0.22)

    top5_rels = [49, 47, 73, 60, 20]
    rel_palette = {
        49: '#E74C3C',  # Adverse effects
        47: '#2980B9',  # Metabolism decrease
        73: '#27AE60',  # Serum conc increase
        60: '#8E44AD',  # Hypotensive
        20: '#F39C12',  # QTc prolongation
        0:  '#BDC3C7',  # True Negative
    }
    rel_legend_names = {
        49: 'Pos Y49: Adverse Effects Risk',
        47: 'Pos Y47: CYP Metabolism Decrease',
        73: 'Pos Y73: Serum Conc. Increase',
        60: 'Pos Y60: Hypotensive Synergy',
        20: 'Pos Y20: QTc Cardiotoxicity',
        0:  'Negative DDI Pairs (Sampled)',
    }

    superfamily_rows = []
    eng_t1_ref = None

    for idx_t, (task_key, task_title) in enumerate([
        ('task1_no_unimol', 'A1 | Task 1 (Warm Start K-K) Latent Manifold'),
        ('task2_no_unimol', 'A2 | Task 2 (Cold Start S1 K-U) Latent Manifold'),
        ('task3_no_unimol', 'A3 | Task 3 (Cold Start S2 U-U) Latent Manifold'),
    ]):
        run_dir = RUN_PATHS[task_key]
        sp = np.load(run_dir / 'splits.npz')
        test_idx = sp['fold1_test']
        eng = ChemMambaNumpyEngine(run_dir / 'fold1_selection.pth')
        eng.precompute_drug_table(unique_smiles, fp_cache)
        if idx_t == 0:
            eng_t1_ref = eng

        # Sample balanced subset across top 5 positive relations + negatives for t-SNE
        rng = np.random.default_rng(42 + idx_t)
        selected_indices = []
        selected_groups = []
        for y_rel in top5_rels:
            cand = [i for i in test_idx if all_y[i] == 1.0 and (all_r[i] + 1) == y_rel]
            take = rng.choice(cand, size=min(180, len(cand)), replace=False)
            selected_indices.extend(take)
            selected_groups.extend([y_rel] * len(take))

        neg_cand = [i for i in test_idx if all_y[i] == 0.0 and (all_r[i] + 1) in top5_rels]
        take_neg = rng.choice(neg_cand, size=min(350, len(neg_cand)), replace=False)
        selected_indices.extend(take_neg)
        selected_groups.extend([0] * len(take_neg))

        s1_ts = [all_s1[i] for i in selected_indices]
        s2_ts = [all_s2[i] for i in selected_indices]
        r_ts = [all_r[i] for i in selected_indices]
        y_ts = all_y[selected_indices]
        probs_ts, _, h2_ts, _ = eng.predict_pairs_from_cache(s1_ts, s2_ts, r_ts)

        # Modulate penultimate representation by target relation classifier vector elementwise for relation-aware manifold
        w5_r = eng.sd['predictor.mlp.5.weight'][np.asarray(r_ts, dtype=np.int64)]
        z_rel = h2_ts * w5_r
        coords = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto').fit_transform(z_rel)
        sil_pos_neg = float(silhouette_score(coords, y_ts))

        ax_ts = fig.add_subplot(gs[0, idx_t])
        groups_arr = np.array(selected_groups)
        # Plot negatives first, then positives
        for g_id in [0] + top5_rels:
            m = (groups_arr == g_id)
            if m.sum() > 0:
                ax_ts.scatter(
                    coords[m, 0], coords[m, 1],
                    c=rel_palette[g_id], s=14 if g_id == 0 else 22,
                    alpha=0.45 if g_id == 0 else 0.85,
                    edgecolors='none', label=rel_legend_names[g_id]
                )
        ax_ts.set_title(f"{task_title}\n(Pos/Neg Silhouette = {sil_pos_neg:.3f})", fontsize=10.2, fontweight='bold', pad=6)
        ax_ts.set_xlabel("t-SNE Dimension 1", fontsize=8.5, fontweight='bold')
        ax_ts.set_ylabel("t-SNE Dimension 2", fontsize=8.5, fontweight='bold')
        ax_ts.set_xticks([])
        ax_ts.set_yticks([])
        if idx_t == 2:
            ax_ts.legend(loc='upper right', fontsize=7.2, frameon=True, facecolor='white', framealpha=0.9)

        # Also evaluate all test pairs by Pharmacological Super-Family
        sub_all = rng.choice(test_idx, size=min(20000, len(test_idx)), replace=False)
        s1_all = [all_s1[i] for i in sub_all]
        s2_all = [all_s2[i] for i in sub_all]
        r_all = [all_r[i] for i in sub_all]
        y_all = all_y[sub_all]
        probs_all, _, _, _ = eng.predict_pairs_from_cache(s1_all, s2_all, r_all)
        super_labels = [rel_super[r + 1] for r in r_all]
        dyn_thr = 0.5 if idx_t == 0 else 0.31
        for sf in sorted(set(super_labels)):
            m_sf = np.array([s == sf for s in super_labels])
            if m_sf.sum() > 50 and len(set(y_all[m_sf])) == 2:
                superfamily_rows.append({
                    'Task': f'Task {idx_t + 1}',
                    'Superfamily': sf,
                    'Sample_Count': int(m_sf.sum()),
                    'AUC': float(roc_auc_score(y_all[m_sf], probs_all[m_sf])),
                    'AP': float(average_precision_score(y_all[m_sf], probs_all[m_sf])),
                    'F1_Dyn': float(f1_score(y_all[m_sf], (probs_all[m_sf] > dyn_thr).astype(float))),
                })

    df_sf = pd.DataFrame(superfamily_rows)
    df_sf.to_csv(TAB_DIR / 'table4_superfamily_performance.csv', index=False)

    # Panel B (Bottom-Left): Cosine Similarity Matrix of Top 20 Relation Classifier Vectors
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], width_ratios=[1.15, 1.0], wspace=0.48)
    ax_heat = fig.add_subplot(gs_bot[0, 0])

    top20_rels = sorted(RELATION_SHORT_NAMES.keys())
    w5 = eng_t1_ref.sd['predictor.mlp.5.weight']  # (86, 256)
    w_sub = w5[[y - 1 for y in top20_rels]]
    w_norm = w_sub / (np.linalg.norm(w_sub, axis=1, keepdims=True) + 1e-9)
    cos_sim = w_norm @ w_norm.T

    link = linkage(pdist(w_norm, metric='euclidean'), method='ward')
    order = leaves_list(link)
    cos_ordered = cos_sim[order][:, order]
    labels_ordered = [RELATION_SHORT_NAMES[top20_rels[i]][:38] for i in order]

    sns.heatmap(
        cos_ordered, ax=ax_heat, cmap='RdBu_r', vmin=-0.55, vmax=1.0, center=0.0,
        xticklabels=[f"Y{top20_rels[i]:02d}" for i in order],
        yticklabels=labels_ordered,
        cbar_kws={'label': 'Weight Cosine Sim.', 'shrink': 0.85, 'pad': 0.02},
        linewidths=0.4, linecolor='white'
    )
    ax_heat.set_xticklabels([f"Y{top20_rels[i]:02d}" for i in order], rotation=45, ha='right', fontsize=8, fontweight='bold')
    ax_heat.set_yticklabels(labels_ordered, rotation=0, fontsize=8, fontweight='medium')
    ax_heat.set_title("B | Autonomous Pharmacological Hierarchy in Classifier Space\n(Ward Clustering of 20 Major DrugBank Relations)",
                      fontsize=10.2, fontweight='bold', loc='left', pad=8)

    # Panel C (Bottom-Right): Performance Across 6 Pharmacological Super-Families
    ax_sf = fig.add_subplot(gs_bot[0, 1])
    sf_order = [
        'PK: Hepatic Metabolism (CYP450)',
        'PK: Absorption / Serum / Excretion',
        'PD: Cardiovascular & QTc',
        'PD: Coagulation & Hematologic',
        'PD: CNS & Neurologic',
        'PD: Systemic Adverse / Other',
    ]
    y_sf = np.arange(len(sf_order))
    h_sf = 0.25
    for idx_t, (t_label, col_t) in enumerate([('Task 1', '#2E86C1'), ('Task 2', '#E67E22'), ('Task 3', '#C0392B')]):
        vals = [
            df_sf[(df_sf['Task'] == t_label) & (df_sf['Superfamily'] == sf)]['AUC'].values[0]
            for sf in reversed(sf_order)
        ]
        ax_sf.barh(y_sf + (1 - idx_t) * h_sf, vals, height=h_sf, color=col_t, alpha=0.9,
                   edgecolor='#2C3E50', linewidth=0.5, label=f"{t_label} ({'Warm' if idx_t==0 else 'Cold S'+str(idx_t)})")
    ax_sf.set_yticks(y_sf)
    ax_sf.set_yticklabels(list(reversed(sf_order)), fontsize=8.5, fontweight='bold')
    ax_sf.set_xlabel("Test ROC-AUC Score", fontsize=9.5, fontweight='bold')
    ax_sf.set_xlim(0.50, 1.03)
    ax_sf.set_title("C | Generalization Across 6 Pharmacological DDI Super-Families\n(PK vs. PD Mechanisms in Warm & Cold Starts)",
                    fontsize=10.5, fontweight='bold', loc='left', pad=8)
    ax_sf.spines['top'].set_visible(False)
    ax_sf.spines['right'].set_visible(False)
    ax_sf.legend(loc='lower right', fontsize=8.5, frameon=True)

    fig.suptitle(
        "Figure 4 | Pairwise DDI Representation Manifolds (Task 1–3) & Autonomous Pharmacological Clustering of 86 DDI Relations",
        fontsize=13.5, fontweight='bold', y=0.992
    )
    fig.savefig(FIG_DIR / 'fig4_latent_manifold_and_relation_hierarchy.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig4_latent_manifold_and_relation_hierarchy.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  -> Saved fig4_latent_manifold_and_relation_hierarchy.{png,pdf} and table4_superfamily_performance.csv")


# ==============================================================================
# FIGURE 5: Mechanistic Diagnosis of Why 3D Uni-Mol Distillation Degrades Performance
# ==============================================================================
def generate_figure5_3d_diagnosis(df, fp_cache):
    print("\n[Figure 5] Running Mechanistic Diagnosis of 3D Uni-Mol Distillation vs. No-3D...")
    unique_smiles = sorted(fp_cache.keys())

    # 1. Load training histories for Task 1, 2, 3 (unimol vs no_unimol)
    hist_t1_uni = pd.read_csv(RUN_PATHS['task1_unimol'] / 'history.csv')
    hist_t2_uni = pd.read_csv(RUN_PATHS['task2_unimol'] / 'history.csv')
    hist_t3_uni = pd.read_csv(RUN_PATHS['task3_unimol'] / 'history.csv')

    # 2. Compute Singular Value Spectrum & Effective Rank (R_eff) of Drug Embeddings
    eng_n3d = ChemMambaNumpyEngine(RUN_PATHS['task1_no_unimol'] / 'fold1_selection.pth')
    eng_n3d.precompute_drug_table(unique_smiles, fp_cache)
    eng_uni = ChemMambaNumpyEngine(RUN_PATHS['task1_unimol'] / 'fold1_selection.pth')
    eng_uni.precompute_drug_table(unique_smiles, fp_cache)

    # Load cached Uni-Mol 3D teacher features
    u3d_payload = load_pth_numpy(ROOT / 'data/drugbank_3d_features_all_h.pt')
    u3d_feats = u3d_payload['features']
    valid_3d_smiles = sorted(u3d_feats.keys())

    mat_fp_n3d = np.stack([eng_n3d.fp_proj_map[s] for s in valid_3d_smiles])
    mat_seq_uni = np.stack([eng_uni.e1d_map[s] for s in valid_3d_smiles])
    mat_teacher_3d = np.stack([np.asarray(u3d_feats[s], dtype=np.float32) for s in valid_3d_smiles])

    def compute_spectrum_and_eff_rank(mat):
        centered = mat - mat.mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        s_norm = s / (s.sum() + 1e-12)
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
        eff_rank = float(np.exp(entropy))
        return s / s[0], eff_rank

    spec_n3d, rank_n3d = compute_spectrum_and_eff_rank(mat_fp_n3d)
    spec_seq_uni, rank_seq_uni = compute_spectrum_and_eff_rank(mat_seq_uni)
    spec_t3d, rank_t3d = compute_spectrum_and_eff_rank(mat_teacher_3d)

    # 3. Compute DDI Partner Neighborhood Jaccard Alignment (k=10 nearest neighbors in representation space)
    # Build ground-truth DDI interaction profile per drug (which (partner, relation) pairs it participates in)
    drug_to_idx = {s: i for i, s in enumerate(valid_3d_smiles)}
    ddi_sets = [set() for _ in valid_3d_smiles]
    for s1, s2, y in zip(df['X1'], df['X2'], df['Y']):
        if s1 in drug_to_idx and s2 in drug_to_idx:
            ddi_sets[drug_to_idx[s1]].add((drug_to_idx[s2], int(y)))
            ddi_sets[drug_to_idx[s2]].add((drug_to_idx[s1], int(y)))

    def knn_ddi_jaccard(mat, k=10):
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        normed = mat / norms
        sim = normed @ normed.T
        np.fill_diagonal(sim, -1.0)
        top_k = np.argsort(-sim, axis=1)[:, :k]
        jaccards = []
        for i in range(len(valid_3d_smiles)):
            s_i = ddi_sets[i]
            if not s_i:
                continue
            for j in top_k[i]:
                s_j = ddi_sets[j]
                union = len(s_i | s_j)
                if union > 0:
                    jaccards.append(len(s_i & s_j) / union)
        return float(np.mean(jaccards))

    jac_n3d = knn_ddi_jaccard(mat_fp_n3d, k=10)
    jac_uni = knn_ddi_jaccard(mat_seq_uni, k=10)
    jac_t3d = knn_ddi_jaccard(mat_teacher_3d, k=10)

    # Save diagnostic summary table
    diag_rows = []
    for t_idx in (1, 2, 3):
        r_n3d = pd.read_csv(RUN_PATHS[f'task{t_idx}_no_unimol'] / 'results.csv').iloc[:5]
        r_dyn = pd.read_csv(RUN_PATHS[f'task{t_idx}_no_unimol'] / 'results_dynamic_threshold_val_acc.csv').iloc[:5]
        r_uni = pd.read_csv(RUN_PATHS[f'task{t_idx}_unimol'] / 'results.csv').iloc[:5]
        h_uni = pd.read_csv(RUN_PATHS[f'task{t_idx}_unimol'] / 'history.csv')
        ratio = (h_uni['Align_Loss'] / h_uni['DDI_Loss']).mean()
        diag_rows.append({
            'Task': f'Task {t_idx}',
            'No3D_Fixed05_ACC_Mean': float(r_n3d['ACC'].mean()),
            'No3D_Fixed05_AUC_Mean': float(r_n3d['AUC'].mean()),
            'No3D_Fixed05_F1_Mean': float(r_n3d['F1'].mean()),
            'No3D_DynThr_ACC_Mean': float(r_dyn['ACC'].mean()),
            'No3D_DynThr_F1_Mean': float(r_dyn['F1'].mean()),
            'UniMol3D_Fixed05_ACC_Mean': float(r_uni['ACC'].mean()),
            'UniMol3D_Fixed05_AUC_Mean': float(r_uni['AUC'].mean()),
            'UniMol3D_Fixed05_F1_Mean': float(r_uni['F1'].mean()),
            'Mean_Align_to_DDI_Loss_Ratio': float(ratio),
        })
    pd.DataFrame(diag_rows).to_csv(TAB_DIR / 'table5_3d_distillation_diagnosis.csv', index=False)

    # Plot Figure 5 (3 Panels)
    fig = plt.figure(figsize=(18.5, 6.0), dpi=300)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.12, 1.08, 1.15], wspace=0.25)

    # Panel A: Loss Scale & Gradient Dominance Ratio (L_align vs L_DDI) across Epochs
    ax_a = fig.add_subplot(gs[0, 0])
    f1_t1 = hist_t1_uni[hist_t1_uni['Fold'] == 1]
    f1_t2 = hist_t2_uni[hist_t2_uni['Fold'] == 1]
    ax_a.plot(f1_t1['Epoch'], f1_t1['Align_Loss'], color='#C0392B', linewidth=2.0,
              label='Task 1 InfoNCE $\\mathcal{L}_{\\mathrm{align}}$ ($\\lambda=1.0$)')
    ax_a.plot(f1_t1['Epoch'], f1_t1['DDI_Loss'], color='#2980B9', linewidth=2.0,
              label='Task 1 Focal $\\mathcal{L}_{\\mathrm{DDI}}$ ($\\alpha=0.25, \\gamma=2$)')
    ax_a.plot(f1_t2['Epoch'], f1_t2['Align_Loss'], color='#E67E22', linewidth=1.8, linestyle='--',
              label='Task 2 InfoNCE $\\mathcal{L}_{\\mathrm{align}}$')
    ax_a.plot(f1_t2['Epoch'], f1_t2['DDI_Loss'], color='#16A085', linewidth=1.8, linestyle='--',
              label='Task 2 Focal $\\mathcal{L}_{\\mathrm{DDI}}$')
    ax_a.set_yscale('log')
    ax_a.set_xlabel("Training Epoch (Fold 1 Trajectory)", fontsize=9.5, fontweight='bold')
    ax_a.set_ylabel("Loss Magnitude (Log Scale)", fontsize=9.5, fontweight='bold')
    ax_a.set_title("A | Gradient Dominance Diagnosis: $\\mathcal{L}_{\\mathrm{align}}$ Exceeds\nFocal $\\mathcal{L}_{\\mathrm{DDI}}$ by 1.2×–10.8× Throughout Training",
                   fontsize=10.5, fontweight='bold', pad=8)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.fill_between(f1_t1['Epoch'], f1_t1['DDI_Loss'], f1_t1['Align_Loss'], color='#FADBD8', alpha=0.38,
                      label='Alignment Gradient Over-Dominance Gap')
    ax_a.legend(loc='upper right', fontsize=7.6, frameon=True)

    # Panel B: Singular Value Spectrum & Effective Rank + DDI Jaccard Consistency
    ax_b = fig.add_subplot(gs[0, 1])
    k_dims = 128
    x_d = np.arange(1, k_dims + 1)
    ax_b.plot(x_d, spec_n3d[:k_dims], color='#1E8449', linewidth=2.2,
              label=f'No-3D Task Representation ($R_{{\\mathrm{{eff}}}}={rank_n3d:.1f}$, DDI-Jaccard={jac_n3d:.3f})')
    ax_b.plot(x_d, spec_seq_uni[:k_dims], color='#2874A6', linewidth=2.2,
              label=f'3D-Distilled 1D Mamba ($R_{{\\mathrm{{eff}}}}={rank_seq_uni:.1f}$, DDI-Jaccard={jac_uni:.3f})')
    ax_b.plot(x_d, spec_t3d[:k_dims], color='#C0392B', linewidth=2.0, linestyle='--',
              label=f'Static Uni-Mol 3D Teacher ($R_{{\\mathrm{{eff}}}}={rank_t3d:.1f}$, DDI-Jaccard={jac_t3d:.3f})')
    ax_b.set_yscale('log')
    ax_b.set_xlabel("Singular Value Index $k$ (Top 128 Dimensions)", fontsize=9.5, fontweight='bold')
    ax_b.set_ylabel("Normalized Singular Value $\\sigma_k / \\sigma_1$ (Log)", fontsize=9.5, fontweight='bold')
    ax_b.set_title("B | Spectral Dimensionality & DDI Neighborhood Alignment\n(1,607 Valid 3D Drugs; 99 Failed Conformer Generation)",
                   fontsize=10.5, fontweight='bold', pad=8)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.legend(loc='lower left', fontsize=7.5, frameon=True)

    # Panel C: 5-Fold Test F1 & ACC Comparison Across Task 1, Task 2, Task 3
    ax_c = fig.add_subplot(gs[0, 2])
    tasks_x = np.arange(3)
    w_b = 0.25
    uni_f1_means, uni_f1_stds = [], []
    n3d_f1_means, n3d_f1_stds = [], []
    dyn_f1_means, dyn_f1_stds = [], []

    for t_i in (1, 2, 3):
        r_uni = pd.read_csv(RUN_PATHS[f'task{t_i}_unimol'] / 'results.csv').iloc[:5]
        r_n3d = pd.read_csv(RUN_PATHS[f'task{t_i}_no_unimol'] / 'results.csv').iloc[:5]
        r_dyn = pd.read_csv(RUN_PATHS[f'task{t_i}_no_unimol'] / 'results_dynamic_threshold_val_acc.csv').iloc[:5]
        uni_f1_means.append(r_uni['F1'].mean()); uni_f1_stds.append(r_uni['F1'].std())
        n3d_f1_means.append(r_n3d['F1'].mean()); n3d_f1_stds.append(r_n3d['F1'].std())
        dyn_f1_means.append(r_dyn['F1'].mean()); dyn_f1_stds.append(r_dyn['F1'].std())

    b1 = ax_c.bar(tasks_x - w_b, uni_f1_means, yerr=uni_f1_stds, width=w_b, capsize=4,
                  color='#95A5A6', edgecolor='#2C3E50', label='With Uni-Mol 3D ($\\lambda=1.0, \\tau=0.5$)')
    b2 = ax_c.bar(tasks_x, n3d_f1_means, yerr=n3d_f1_stds, width=w_b, capsize=4,
                  color='#3498DB', edgecolor='#2C3E50', label='No-3D Ablation ($\\lambda=0.0, \\tau=0.5$)')
    b3 = ax_c.bar(tasks_x + w_b, dyn_f1_means, yerr=dyn_f1_stds, width=w_b, capsize=4,
                  color='#27AE60', edgecolor='#2C3E50', label='No-3D + Validation Dynamic $\\tau^*$')

    for bars, stds in ((b1, uni_f1_stds), (b2, n3d_f1_stds), (b3, dyn_f1_stds)):
        for rect, s_val in zip(bars, stds):
            h_val = rect.get_height()
            ax_c.text(rect.get_x() + rect.get_width() / 2.0, h_val + s_val + 0.015, f"{h_val:.3f}",
                      ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax_c.set_xticks(tasks_x)
    ax_c.set_xticklabels(['Task 1\n(Warm K-K)', 'Task 2\n(Cold S1 K-U)', 'Task 3\n(Cold S2 U-U)'],
                         fontsize=9.5, fontweight='bold')
    ax_c.set_ylabel("5-Fold Test F1 Score (Mean ± SD)", fontsize=9.5, fontweight='bold')
    ax_c.set_ylim(0.25, 1.08)
    ax_c.set_title("C | 5-Fold Cross-Validation Performance Gain from\nRemoving Rigid 3D Distillation & Calibrating Threshold",
                   fontsize=10.5, fontweight='bold', pad=8)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.legend(loc='upper right', fontsize=7.8, frameon=True)

    fig.suptitle(
        "Figure 5 | Mechanistic Diagnosis of Why Static 3D Uni-Mol Distillation Degrades DDI Prediction Across Tasks 1–3",
        fontsize=13, fontweight='bold', y=1.03
    )
    fig.savefig(FIG_DIR / 'fig5_why_3d_distillation_fails_diagnosis.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig5_why_3d_distillation_fails_diagnosis.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  -> Saved fig5_why_3d_distillation_fails_diagnosis.{png,pdf} and table5_3d_distillation_diagnosis.csv")


def main():
    t0 = time.monotonic()
    print("=================================================================")
    print(" Starting GASPDDI Top-Journal Interpretability Suite...")
    print("=================================================================")
    df = pd.read_csv(ROOT / 'data/drugbank.tab', sep='\t')
    unique_smiles = sorted(set(df['X1']) | set(df['X2']))
    print(f"Loaded DrugBank: {len(df)} positive triples, {len(unique_smiles)} unique drugs.")
    fp_cache, morgan_info_cache = build_fp_and_bitinfo(unique_smiles)
    print(f"Precomputed 679-bit Fingerprints & Morgan bitInfo in {time.monotonic() - t0:.1f}s.")

    generate_figure1_case_studies(df, fp_cache)
    generate_figure2_fingerprint_attribution(df, fp_cache, morgan_info_cache)
    generate_figure3_modality_and_coldstart(df, fp_cache)
    generate_figure4_manifold_and_hierarchy(df, fp_cache)
    generate_figure5_3d_diagnosis(df, fp_cache)

    print(f"\nAll 5 figures and quantitative tables generated in {time.monotonic() - t0:.1f}s!")
    print(f"Figures directory: {FIG_DIR}")
    print(f"Tables directory:  {TAB_DIR}")


if __name__ == '__main__':
    main()


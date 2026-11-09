#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication-grade Figure 1 (fig0_model_architecture):
- No figure title at the top (SCI caption lives in the manuscript).
- Equal-aspect physical coordinate system (180 x 106 on 18.0 x 10.6 in figure) -> ZERO stretching/compression distortion.
- 100% orthogonal (strictly horizontal/vertical) signal routing with generous arrow shafts and bus junctions.
- Soft, light academic pastel color palette (Nature/Bioinformatics style) with crisp dark typography.
"""

import io
import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
)
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path("/Volumes/Files/Research/GASPDDI-Lyj/GASPDDI")
FIG_DIR = ROOT / "interpretability" / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"
SPACE_PAPER_FIG_DIR = ROOT / " paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#475569",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def render_rdkit_mol_exact(smiles, w_units, h_units, highlight_smarts=None,
                           highlight_color=(0.96, 0.78, 0.76), px_per_unit=38):
    """
    Render an RDKit 2D molecule whose pixel dimensions match (w_units : h_units)
    with 100% exact 1:1 aspect ratio so there is zero stretching or compression.
    """
    width = max(120, int(round(w_units * px_per_unit)))
    height = max(80, int(round(h_units * px_per_unit)))
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    hl_atoms, hl_bonds = [], []
    atom_cols, bond_cols = {}, {}
    if highlight_smarts:
        patt = Chem.MolFromSmarts(highlight_smarts)
        if patt:
            for match in mol.GetSubstructMatches(patt):
                for idx in match:
                    hl_atoms.append(idx)
                    atom_cols[idx] = highlight_color
                for b in patt.GetBonds():
                    aid1 = match[b.GetBeginAtomIdx()]
                    aid2 = match[b.GetEndAtomIdx()]
                    bond = mol.GetBondBetweenAtoms(aid1, aid2)
                    if bond:
                        hl_bonds.append(bond.GetIdx())
                        bond_cols[bond.GetIdx()] = highlight_color

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5
    opts.padding = 0.04
    opts.additionalAtomLabelPadding = 0.06
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=hl_atoms,
        highlightAtomColors=atom_cols,
        highlightBonds=hl_bonds,
        highlightBondColors=bond_cols,
    )
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    return np.asarray(img)


def place_mol_on_ax(ax, img_arr, x, y, w, h, zorder=4):
    """Draw image in equal-aspect data coordinates while strictly preserving 1:1 pixel aspect ratio."""
    img_h, img_w = img_arr.shape[:2]
    img_aspect = float(img_w) / float(img_h)
    box_aspect = float(w) / float(h)
    if box_aspect > img_aspect:
        draw_h = float(h)
        draw_w = draw_h * img_aspect
    else:
        draw_w = float(w)
        draw_h = draw_w / img_aspect
    x0 = x + (w - draw_w) / 2.0
    y0 = y + (h - draw_h) / 2.0
    ax.imshow(img_arr, extent=[x0, x0 + draw_w, y0, y0 + draw_h], aspect="equal",
              interpolation="lanczos", zorder=zorder)


def add_header_container(ax, x, y, w, h, header_text, bg_color="#FAFBFD", border_color="#94A3B8",
                         header_bg="#E2E8F0", header_fg="#1E293B", header_h=4.4, linestyle="-",
                         lw=1.4, header_size=13.4, rounding=1.3, zorder=1):
    outer = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.25,rounding_size={rounding}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder
    )
    ax.add_patch(outer)
    hdr = FancyBboxPatch(
        (x, y + h - header_h), w, header_h,
        boxstyle=f"round,pad=0.2,rounding_size={rounding * 0.85}",
        facecolor=header_bg,
        edgecolor=border_color,
        linewidth=lw * 0.85,
        zorder=zorder + 1
    )
    ax.add_patch(hdr)
    ax.text(x + w / 2, y + h - header_h / 2, header_text,
            ha="center", va="center", fontsize=header_size, fontweight="bold",
            color=header_fg, zorder=zorder + 2)


def add_card(ax, x, y, w, h, facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.15,
             linestyle="-", rounding=1.0, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.2,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder
    )
    ax.add_patch(patch)
    return patch


def add_arrow(ax, x1, y1, x2, y2, color="#475569", lw=2.0, style="-|>", rad=0.0,
              linestyle="-", mutation_scale=15, zorder=7):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        zorder=zorder
    )
    ax.add_patch(arr)


def add_ortho_arrow(ax, pts, color="#475569", lw=2.0, linestyle="-", mutation_scale=15, zorder=7):
    """
    Draw a clean orthogonal multi-segment path ending with a straight arrow on the final segment.
    """
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        xs.append(pts[-2][0])
        ys.append(pts[-2][1])
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=linestyle,
                solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
    p_prev = pts[-2]
    p_last = pts[-1]
    add_arrow(ax, p_prev[0], p_prev[1], p_last[0], p_last[1],
              color=color, lw=lw, linestyle=linestyle, mutation_scale=mutation_scale, zorder=zorder)


def add_junction_dot(ax, x, y, color="#475569", r=0.52, zorder=8):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white", lw=0.6, zorder=zorder))


def draw_token_ribbon(ax, x, y, w, h, tokens, mask_flags, active_bg="#EFF6FF",
                      active_edge="#60A5FA", active_fg="#1E3A8A",
                      pad_bg="#F1F5F9", pad_edge="#CBD5E1",
                      fontsize=9.6, zorder=4):
    n = len(tokens)
    cell_w = w / n
    for i, (tok, m) in enumerate(zip(tokens, mask_flags)):
        cx = x + i * cell_w
        bg = active_bg if m else pad_bg
        ed = active_edge if m else pad_edge
        fg = active_fg if m else "#64748B"
        rect = Rectangle((cx, y), cell_w * 0.90, h, facecolor=bg, edgecolor=ed,
                         linewidth=0.85, zorder=zorder)
        ax.add_patch(rect)
        ax.text(cx + cell_w * 0.45, y + h / 2, tok, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=fg, zorder=zorder + 1)


def draw_barcode(ax, x, y, w, h, bits, on_color="#A7F3D0", off_color="#F1F5F9",
                 on_edge="#34D399", off_edge="#CBD5E1", fontsize=9.2, zorder=4):
    n = len(bits)
    cw = w / n
    for i, b in enumerate(bits):
        cx = x + i * cw
        col = on_color if b else off_color
        ed = on_edge if b else off_edge
        fg = "#064E3B" if b else "#94A3B8"
        rect = Rectangle((cx, y), cw * 0.90, h, facecolor=col, edgecolor=ed,
                         linewidth=0.75, zorder=zorder)
        ax.add_patch(rect)
        ax.text(cx + cw * 0.45, y + h / 2, str(b), ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=fg, zorder=zorder + 1)


def build_architecture_figure():
    # Exact 180 x 106 equal-aspect canvas (no top title banner -> full height used by panels)
    fig = plt.figure(figsize=(18.0, 10.6), dpi=300, facecolor="white")
    ax = fig.add_axes([0.003, 0.003, 0.994, 0.994])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 106)
    ax.set_aspect("equal")
    ax.axis("off")

    # =========================================================================
    # PANEL A: INPUT DRUG PAIR & SPLITS (x=1.2..35.0, y=33.4..104.8)
    # =========================================================================
    add_header_container(
        ax, 1.2, 33.4, 33.8, 71.4,
        "A. Input Drug Pair & Splits",
        bg_color="#FAFBFD", border_color="#94A3B8",
        header_bg="#E2E8F0", header_fg="#1E293B", header_size=13.4
    )

    # --- Card A1: Drug A (Methotrexate) ---
    add_card(ax, 2.3, 80.6, 31.6, 18.6, facecolor="#F8FAFF", edgecolor="#93C5FD", lw=1.2)
    ax.text(18.1, 97.4, "Drug A: Methotrexate", ha="center", va="center",
            fontsize=11.8, fontweight="bold", color="#1E3A8A", zorder=4)
    mol_a_img = render_rdkit_mol_exact(
        "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
        w_units=29.4, h_units=10.0,
        highlight_smarts="C(=O)O",
        highlight_color=(0.74, 0.86, 0.99)
    )
    place_mol_on_ax(ax, mol_a_img, 3.4, 85.9, 29.4, 10.0, zorder=4)
    ax.text(3.3, 84.6, "SMILES $S_a$ ($L \\leq 426$):",
            ha="left", va="center", fontsize=10.2, fontweight="bold", color="#334155", zorder=4)
    tokens_a = ["C", "N", "(", "c", "1", "..", "=", "O", "PAD"]
    mask_a   = [1,   1,   1,   1,   1,   1,    1,   1,   0]
    draw_token_ribbon(ax, 3.3, 81.0, 29.6, 2.7, tokens_a, mask_a,
                      active_bg="#EFF6FF", active_edge="#93C5FD", active_fg="#1E3A8A", fontsize=9.6)

    # --- Card A2: Drug B (Ibuprofen) ---
    add_card(ax, 2.3, 59.8, 31.6, 18.6, facecolor="#FFFDF9", edgecolor="#FDBA74", lw=1.2)
    ax.text(18.1, 76.6, "Drug B: Ibuprofen", ha="center", va="center",
            fontsize=11.8, fontweight="bold", color="#9A3412", zorder=4)
    mol_b_img = render_rdkit_mol_exact(
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        w_units=29.4, h_units=10.0,
        highlight_smarts="C(=O)O",
        highlight_color=(0.99, 0.85, 0.72)
    )
    place_mol_on_ax(ax, mol_b_img, 3.4, 65.1, 29.4, 10.0, zorder=4)
    ax.text(3.3, 63.8, "SMILES $S_b$ + Query $r$:",
            ha="left", va="center", fontsize=10.2, fontweight="bold", color="#334155", zorder=4)
    tokens_b = ["C", "C", "(", "c", "1", "..", "=", "O", "PAD"]
    mask_b   = [1,   1,   1,   1,   1,   1,    1,   1,   0]
    draw_token_ribbon(ax, 3.3, 60.2, 29.6, 2.7, tokens_b, mask_b,
                      active_bg="#FFF7ED", active_edge="#FDBA74", active_fg="#9A3412", fontsize=9.6)

    # --- Card A3: Concrete 2x2 Task Partition Matrix ---
    add_card(ax, 2.3, 34.6, 31.6, 23.0, facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.2)
    ax.text(18.1, 55.8, "5-Fold Split Protocol", ha="center", va="center",
            fontsize=11.8, fontweight="bold", color="#1E293B", zorder=4)

    gx, gy, gw, gh = 9.8, 40.4, 22.8, 11.2
    cw, ch = gw / 2, gh / 2
    ax.text(gx + cw * 0.5, gy + gh + 1.3, "Known $\\mathcal{K}$", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="#1E40AF")
    ax.text(gx + cw * 1.5, gy + gh + 1.3, "Unseen $\\mathcal{U}$", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="#991B1B")
    ax.text(gx - 0.6, gy + ch * 1.5, "Known\n$\\mathcal{K}$", ha="right", va="center",
            fontsize=9.6, fontweight="bold", color="#1E40AF")
    ax.text(gx - 0.6, gy + ch * 0.5, "Unseen\n$\\mathcal{U}$", ha="right", va="center",
            fontsize=9.6, fontweight="bold", color="#991B1B")

    ax.add_patch(Rectangle((gx, gy + ch), cw, ch, facecolor="#EFF6FF", edgecolor="#93C5FD", lw=1.2, zorder=4))
    ax.text(gx + cw * 0.5, gy + ch * 1.5, "$\\mathcal{S}_1$ (Warm)\n$\\mathcal{K} \\times \\mathcal{K}$",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1E3A8A", zorder=5)

    ax.add_patch(Rectangle((gx + cw, gy + ch), cw, ch, facecolor="#FFFBEB", edgecolor="#FCD34D", lw=1.2, zorder=4))
    ax.add_patch(Rectangle((gx, gy), cw, ch, facecolor="#FFFBEB", edgecolor="#FCD34D", lw=1.2, zorder=4))
    ax.text(gx + cw * 1.5, gy + ch * 1.5, "$\\mathcal{S}_2$ (Cold)\n$\\mathcal{K} \\times \\mathcal{U}$",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#92400E", zorder=5)
    ax.text(gx + cw * 0.5, gy + ch * 0.5, "$\\mathcal{S}_2$ (Cold)\n$\\mathcal{U} \\times \\mathcal{K}$",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#92400E", zorder=5)

    ax.add_patch(Rectangle((gx + cw, gy), cw, ch, facecolor="#FEF2F2", edgecolor="#FCA5A5", lw=1.2, zorder=4))
    ax.text(gx + cw * 1.5, gy + ch * 0.5, "$\\mathcal{S}_3$ (Cold)\n$\\mathcal{U} \\times \\mathcal{U}$",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#991B1B", zorder=5)

    add_card(ax, 3.4, 35.4, 29.4, 3.5, facecolor="#F0FDF4", edgecolor="#86EFAC", lw=1.0, zorder=4)
    ax.text(18.1, 37.15, "1:1 Relation-Conditioned Negatives",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color="#166534", zorder=5)

    # =========================================================================
    # PANEL B: DUAL-STREAM MOLECULAR ENCODING (x=38.6..100.6, y=33.4..104.8)
    # =========================================================================
    add_header_container(
        ax, 38.6, 33.4, 62.0, 71.4,
        "B. Dual-Stream Molecular Representation Learning",
        bg_color="#FAFBFD", border_color="#93C5FD",
        header_bg="#DBEAFE", header_fg="#1E3A8A", header_size=13.4
    )

    # -------------------------------------------------------------------------
    # STREAM 1: 1D Serial Mamba-CNN Sequence Encoder (y=63.4..99.2)
    # -------------------------------------------------------------------------
    add_header_container(
        ax, 39.8, 63.4, 59.6, 35.8,
        "Stream 1: 1D Mamba-CNN Sequence Encoder ($M_{\\theta}$, $\\mathcal{O}(L)$)",
        bg_color="#F8FAFF", border_color="#BFDBFE",
        header_bg="#E0ECFF", header_fg="#1E3A8A",
        header_h=3.8, lw=1.2, header_size=11.8, zorder=2
    )

    # Box B1-1: Token Embedding Grid (Top-Left of Stream 1: x=41.0..59.4, y=82.2..94.4)
    add_card(ax, 41.0, 82.2, 18.4, 12.2, facecolor="#FFFFFF", edgecolor="#93C5FD", lw=1.15, zorder=3)
    ax.text(50.2, 92.8, "Token Embedding", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color="#1E3A8A", zorder=4)
    ax.text(50.2, 83.6, "$\\mathbf{X}_u \\in \\mathbb{R}^{L \\times 256}$", ha="center", va="center",
            fontsize=10.2, fontweight="bold", color="#334155", zorder=4)
    emb_colors = ["#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE", "#93C5FD", "#60A5FA"]
    for r_idx in range(5):
        for c_idx in range(6):
            rx = 42.1 + c_idx * 2.1
            ry = 89.4 - r_idx * 1.05
            c_val = "#E2E8F0" if r_idx >= 3 else emb_colors[(r_idx + c_idx) % 6]
            ax.add_patch(Rectangle((rx, ry), 1.85, 0.78, facecolor=c_val, edgecolor="white", lw=0.35, zorder=4))
        m_bit = "1" if r_idx < 3 else "0"
        m_col = "#15803D" if r_idx < 3 else "#94A3B8"
        ax.text(56.6, 89.78 - r_idx * 1.05, m_bit, ha="center", va="center",
                fontsize=8.6, fontweight="bold", color=m_col, zorder=4)
    ax.text(56.6, 91.1, "$\\mathbf{m}_u$", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="#1E293B", zorder=4)

    # Box B1-2: Selective State-Space Scan (Mamba S6) (Top-Right of Stream 1: x=62.2..98.2, y=82.2..94.4)
    add_card(ax, 62.2, 82.2, 36.0, 12.2, facecolor="#FFFFFF", edgecolor="#93C5FD", lw=1.15, zorder=3)
    ax.text(80.2, 92.8, "Selective State-Space Scan (Mamba S6)", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color="#1E3A8A", zorder=4)

    ssm_nodes_x = [66.6, 75.0, 83.4]
    ssm_labels = ["$\\mathbf{H}_{t-1}$", "$\\mathbf{H}_t$", "$\\mathbf{H}_{t+1}$"]
    for idx, (nx, nlab) in enumerate(zip(ssm_nodes_x, ssm_labels)):
        circ = Circle((nx, 88.4), 2.15, facecolor="#EFF6FF", edgecolor="#60A5FA", lw=1.3, zorder=4)
        ax.add_patch(circ)
        ax.text(nx, 88.4, nlab, ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1E3A8A", zorder=5)
        ax.annotate("", xy=(nx, 86.1), xytext=(nx, 84.2),
                    arrowprops=dict(arrowstyle="-|>", color="#3B82F6", lw=1.2, mutation_scale=10), zorder=5)
        ax.text(nx, 83.2, "$\\bar{\\mathbf{B}}_t \\tilde{\\mathbf{u}}_t$", ha="center", va="center",
                fontsize=9.6, fontweight="bold", color="#334155", zorder=5)
        if idx < 2:
            ax.annotate("", xy=(ssm_nodes_x[idx + 1] - 2.25, 88.4), xytext=(nx + 2.25, 88.4),
                        arrowprops=dict(arrowstyle="-|>", color="#3B82F6", lw=1.3, mutation_scale=10), zorder=5)
            ax.text((nx + ssm_nodes_x[idx + 1]) / 2, 90.3, "$\\bar{\\mathbf{A}}_t$",
                    ha="center", va="center", fontsize=10.0, fontweight="bold", color="#1E40AF", zorder=5)

    gx_vals = np.linspace(88.0, 96.8, 32)
    gy_vals = 84.5 + 0.32 * np.sin(np.linspace(0, 6, 32))
    gy_vals[8] += 4.1
    gy_vals[18] += 4.9
    gy_vals[25] += 3.6
    ax.plot(gx_vals, gy_vals, color="#E11D48", linewidth=1.4, zorder=5)
    ax.fill_between(gx_vals, 84.3, gy_vals, color="#FECDD3", alpha=0.5, zorder=4)
    ax.text(92.4, 90.6, "$\\bar{\\Delta}_t$ Spikes", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="#9F1239", zorder=5)

    # Box B1-3: Pad Zero-Mask & Local 1D-CNN (k=7) (Bottom-Left of Stream 1: x=41.0..69.0, y=64.6..79.0)
    add_card(ax, 41.0, 64.6, 28.0, 14.4, facecolor="#FFFFFF", edgecolor="#93C5FD", lw=1.15, zorder=3)
    ax.text(55.0, 77.4, "Zero-Mask & 1D-CNN ($k = 7$)", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color="#1E3A8A", zorder=4)

    bar_x0 = 42.3
    bar_step = 2.32
    bar_w = 1.74
    for i in range(10):
        bx = bar_x0 + i * bar_step
        is_pad = i >= 8
        bh = 1.9 if is_pad else (4.1 + 1.6 * ((i * 3) % 3) / 2.0)
        bcol = "#F1F5F9" if is_pad else ("#93C5FD" if 1 <= i <= 7 else "#DBEAFE")
        ax.add_patch(Rectangle((bx, 66.5), bar_w, bh, facecolor=bcol,
                               edgecolor="#60A5FA" if not is_pad else "#CBD5E1", lw=0.85, zorder=4))
        if is_pad:
            ax.text(bx + bar_w / 2, 67.45, "0", ha="center", va="center",
                    fontsize=8.8, fontweight="bold", color="#64748B", zorder=5)
    ax.text(bar_x0 + 8.5 * bar_step + bar_w / 2, 70.0, "PAD=0", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color="#64748B", zorder=5)

    k_rect = FancyBboxPatch((bar_x0 + 1 * bar_step - 0.22, 66.1),
                            bar_step * 7 - 0.15, 6.5,
                            boxstyle="round,pad=0.2,rounding_size=0.5",
                            facecolor="none", edgecolor="#F43F5E", linewidth=1.4, linestyle="--", zorder=5)
    ax.add_patch(k_rect)
    ax.text(55.0, 74.5, "Local Motif Window ($k = 7$)",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#BE123C", zorder=5)

    # Box B1-4: Masked Mean Pooling & 1D Vector Bar (Bottom-Right of Stream 1: x=71.8..98.2, y=64.6..79.0)
    add_card(ax, 71.8, 64.6, 26.4, 14.4, facecolor="#F5F9FF", edgecolor="#93C5FD", lw=1.15, zorder=3)
    ax.text(85.0, 77.4, "Masked Mean Pooling", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color="#1E3A8A", zorder=4)
    funnel_pts = [[74.0, 75.0], [96.0, 75.0], [91.4, 70.9], [78.6, 70.9]]
    ax.add_patch(Polygon(funnel_pts, closed=True, facecolor="#DBEAFE", edgecolor="#60A5FA", lw=1.1, alpha=0.85, zorder=4))
    ax.text(85.0, 72.95, "Valid States ($m_{u,t} = 1$)",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1E3A8A", zorder=5)

    vec_cols_1d = ["#60A5FA", "#93C5FD", "#BFDBFE", "#93C5FD", "#60A5FA", "#93C5FD", "#BFDBFE", "#60A5FA"]
    for i, vc in enumerate(vec_cols_1d):
        ax.add_patch(Rectangle((74.5 + i * 2.62, 67.8), 2.4, 2.3, facecolor=vc, edgecolor="#3B82F6", lw=0.6, zorder=5))
    ax.text(85.0, 65.9, "1D Vector $\\tilde{\\mathbf{e}}_u \\in \\mathbb{R}^{256}$",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1E3A8A", zorder=5)

    # Strictly orthogonal arrows inside Stream 1
    add_arrow(ax, 59.6, 88.3, 62.0, 88.3, color="#3B82F6", lw=2.0)
    add_arrow(ax, 65.6, 82.0, 65.6, 79.2, color="#3B82F6", lw=2.0)
    add_arrow(ax, 69.2, 71.8, 71.6, 71.8, color="#3B82F6", lw=2.0)

    # -------------------------------------------------------------------------
    # STREAM 2: 2D Topological & Pharmacophoric Fingerprints (y=34.6..61.4)
    # -------------------------------------------------------------------------
    add_header_container(
        ax, 39.8, 34.6, 59.6, 26.8,
        "Stream 2: 2D Topological & Pharmacophoric Fingerprints ($\\mathrm{FP}_{\\phi}$)",
        bg_color="#F7FCF9", border_color="#A7F3D0",
        header_bg="#D1FAE5", header_fg="#065F46",
        header_h=3.8, lw=1.2, header_size=11.6, zorder=2
    )

    # Box B2-1: Morgan ECFP4 (512-Bit) (x=41.0..68.8, y=45.6..56.6)
    add_card(ax, 41.0, 45.6, 27.8, 11.0, facecolor="#FFFFFF", edgecolor="#86EFAC", lw=1.15, zorder=3)
    ax.text(54.9, 55.0, "Morgan ECFP4 (512-Bit)", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color="#065F46", zorder=4)

    cx_m, cy_m = 45.8, 51.3
    ax.add_patch(Circle((cx_m, cy_m), 2.3, facecolor="#ECFDF5", edgecolor="#6EE7B7", lw=1.0, linestyle="--", zorder=4))
    ax.add_patch(Circle((cx_m, cy_m), 1.48, facecolor="#D1FAE5", edgecolor="#34D399", lw=1.0, linestyle=":", zorder=4))
    r1_pts = [(cx_m - 1.05, cy_m + 0.85), (cx_m + 1.05, cy_m + 0.85), (cx_m, cy_m - 1.25)]
    r2_pts = [(cx_m - 1.8, cy_m + 1.3), (cx_m + 1.8, cy_m + 1.3)]
    for rx_p, ry_p in r1_pts:
        ax.plot([cx_m, rx_p], [cy_m, ry_p], color="#10B981", lw=1.2, zorder=4)
        ax.add_patch(Circle((rx_p, ry_p), 0.38, facecolor="#6EE7B7", edgecolor="#059669", lw=0.7, zorder=5))
    for (rx1, ry1), (rx2, ry2) in zip(r1_pts[:2], r2_pts):
        ax.plot([rx1, rx2], [ry1, ry2], color="#34D399", lw=1.0, zorder=4)
        ax.add_patch(Circle((rx2, ry2), 0.32, facecolor="#A7F3D0", edgecolor="#059669", lw=0.7, zorder=5))
    ax.add_patch(Circle((cx_m, cy_m), 0.62, facecolor="#34D399", edgecolor="#065F46", lw=0.8, zorder=6))
    ax.text(cx_m, cy_m, "C", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#064E3B", zorder=7)

    ax.text(58.8, 51.3, "Radius $r \\in \\{0, 1, 2\\}$",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="#065F46", zorder=4)
    draw_barcode(ax, 42.2, 46.2, 25.4, 2.4, [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                 on_color="#A7F3D0", off_color="#F1F5F9", on_edge="#34D399", fontsize=9.2)

    # Box B2-2: MACCS Keys (167-Bit) (x=70.4..98.2, y=45.6..56.6)
    add_card(ax, 70.4, 45.6, 27.8, 11.0, facecolor="#FFFFFF", edgecolor="#86EFAC", lw=1.15, zorder=3)
    ax.text(84.3, 55.0, "MACCS Keys (167-Bit)", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color="#065F46", zorder=4)

    badges = [
        ("#123 Carboxylate (-COO-)", "#FFFBEB", "#F59E0B", "#92400E"),
        ("#95 Triazole Ring", "#EFF6FF", "#60A5FA", "#1E40AF"),
    ]
    for b_idx, (b_txt, b_bg, b_ed, b_fg) in enumerate(badges):
        by = 52.3 - b_idx * 2.25
        add_card(ax, 71.8, by - 0.78, 25.0, 1.65, facecolor=b_bg, edgecolor=b_ed, lw=0.9, rounding=0.45, zorder=4)
        ax.text(84.3, by + 0.04, b_txt, ha="center", va="center", fontsize=9.5, fontweight="bold", color=b_fg, zorder=5)
    draw_barcode(ax, 71.8, 46.2, 25.0, 2.4, [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                 on_color="#99F6E4", off_color="#F1F5F9", on_edge="#2DD4BF", fontsize=9.2)

    # Box B2-3: 679-Bit Concatenation & Projection (x=41.0..98.2, y=35.6..42.8)
    add_card(ax, 41.0, 35.6, 57.2, 7.2, facecolor="#F2FBF7", edgecolor="#6EE7B7", lw=1.15, zorder=3)
    ax.text(69.6, 41.2, "679-Bit Concatenation & Dense Projection ($\\mathrm{FP}_{\\phi}$)",
            ha="center", va="center", fontsize=10.8, fontweight="bold", color="#065F46", zorder=4)
    ax.text(55.4, 37.6, "$\\mathbf{x}_u^{\\mathrm{FP}} \\in \\{0,1\\}^{679} \\;\\to\\; \\mathrm{Linear}\\text{-}\\mathrm{ReLU}$",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="#334155", zorder=4)
    vec_cols_2d = ["#6EE7B7", "#A7F3D0", "#D1FAE5", "#A7F3D0", "#6EE7B7", "#A7F3D0", "#D1FAE5", "#6EE7B7"]
    for i, vc in enumerate(vec_cols_2d):
        ax.add_patch(Rectangle((76.2 + i * 2.4, 37.8), 2.2, 2.1, facecolor=vc, edgecolor="#10B981", lw=0.6, zorder=5))
    ax.text(85.8, 36.4, "2D Vector $\\mathbf{f}_u \\in \\mathbb{R}^{256}$",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color="#065F46", zorder=5)

    # Strictly vertical downward arrows inside Stream 2
    add_arrow(ax, 54.9, 45.4, 54.9, 43.0, color="#10B981", lw=2.0)
    add_arrow(ax, 84.3, 45.4, 84.3, 43.0, color="#10B981", lw=2.0)

    # =========================================================================
    # ORTHOGONAL BUS ROUTING: PANEL A -> PANEL B (x=34.0..41.0)
    # =========================================================================
    bus_x_ab = 36.8
    ax.plot([34.1, bus_x_ab], [89.8, 89.8], color="#60A5FA", lw=2.1, zorder=7)
    ax.plot([34.1, bus_x_ab], [69.0, 69.0], color="#FB923C", lw=2.1, zorder=7)
    ax.plot([bus_x_ab, bus_x_ab], [51.1, 89.8], color="#64748B", lw=2.1, zorder=7)
    add_junction_dot(ax, bus_x_ab, 88.3, color="#3B82F6", r=0.52)
    add_junction_dot(ax, bus_x_ab, 69.0, color="#FB923C", r=0.52)
    add_junction_dot(ax, bus_x_ab, 51.1, color="#10B981", r=0.52)
    add_arrow(ax, bus_x_ab, 88.3, 40.8, 88.3, color="#3B82F6", lw=2.1, mutation_scale=15, zorder=7)
    add_arrow(ax, bus_x_ab, 51.1, 40.8, 51.1, color="#10B981", lw=2.1, mutation_scale=15, zorder=7)

    # =========================================================================
    # PANEL C: SYMMETRIC/ASYMMETRIC FUSION & OPTIMIZATION (x=104.2..178.8, y=33.4..104.8)
    # =========================================================================
    add_header_container(
        ax, 104.2, 33.4, 74.6, 71.4,
        "C. Pairwise Interaction Fusion & Optimization",
        bg_color="#FAFBFD", border_color="#C4B5FD",
        header_bg="#EDE9FE", header_fg="#4C1D95", header_size=13.4
    )

    # --- Box C1: 8-Block Pairwise Interaction Tensor & Operator Nodes (y=79.4..99.2) ---
    add_card(ax, 105.6, 79.4, 71.8, 19.8, facecolor="#FAF8FF", edgecolor="#C4B5FD", lw=1.2, zorder=3)
    ax.text(141.5, 97.4, "8-Block Symmetric & Asymmetric Fusion ($\\mathbf{z}_{ab} \\in \\mathbb{R}^{2048}$)",
            ha="center", va="center", fontsize=11.4, fontweight="bold", color="#4C1D95", zorder=4)

    add_card(ax, 107.0, 90.4, 33.6, 5.2, facecolor="#FFFFFF", edgecolor="#C4B5FD", lw=1.1, zorder=4)
    ax.add_patch(Circle((110.5, 93.0), 1.95, facecolor="#F5F3FF", edgecolor="#A78BFA", lw=1.25, zorder=5))
    ax.text(110.5, 93.0, "$\\odot$", ha="center", va="center", fontsize=13.6, fontweight="bold", color="#5B21B6", zorder=6)
    ax.text(126.2, 93.0, "Hadamard ($\\mathbf{u}_a \\odot \\mathbf{u}_b$)", ha="center", va="center",
            fontsize=10.6, fontweight="bold", color="#4C1D95", zorder=5)

    add_card(ax, 142.2, 90.4, 33.8, 5.2, facecolor="#FFFFFF", edgecolor="#FCA5A5", lw=1.1, zorder=4)
    ax.add_patch(Circle((145.7, 93.0), 1.95, facecolor="#FEF2F2", edgecolor="#F87171", lw=1.25, zorder=5))
    ax.text(145.7, 93.0, "$|\\Delta|$", ha="center", va="center", fontsize=10.6, fontweight="bold", color="#991B1B", zorder=6)
    ax.text(161.4, 93.0, "Asymmetric Diff ($|\\mathbf{u}_a - \\mathbf{u}_b|$)", ha="center", va="center",
            fontsize=10.6, fontweight="bold", color="#991B1B", zorder=5)

    blocks_info = [
        ("$\\tilde{\\mathbf{e}}_a$", "#EFF6FF", "#93C5FD"),
        ("$\\tilde{\\mathbf{e}}_b$", "#EFF6FF", "#93C5FD"),
        ("$\\tilde{\\mathbf{e}}_a \\odot \\tilde{\\mathbf{e}}_b$", "#E0ECFF", "#60A5FA"),
        ("$|\\tilde{\\mathbf{e}}_a - \\tilde{\\mathbf{e}}_b|$", "#DBEAFE", "#3B82F6"),
        ("$\\mathbf{f}_a$", "#ECFDF5", "#86EFAC"),
        ("$\\mathbf{f}_b$", "#ECFDF5", "#86EFAC"),
        ("$\\mathbf{f}_a \\odot \\mathbf{f}_b$", "#D1FAE5", "#34D399"),
        ("$|\\mathbf{f}_a - \\mathbf{f}_b|$", "#FEE2E2", "#F87171"),
    ]
    bx0, by0, bw_tot, bh_rib = 107.0, 83.6, 69.0, 5.4
    bw_cell = bw_tot / 8.0
    for idx, (blab, bbg, bed) in enumerate(blocks_info):
        cx = bx0 + idx * bw_cell
        ax.add_patch(Rectangle((cx + 0.18, by0), bw_cell - 0.36, bh_rib,
                               facecolor=bbg, edgecolor=bed, lw=1.15, zorder=5))
        ax.text(cx + bw_cell / 2, by0 + bh_rib / 2, blab, ha="center", va="center",
                fontsize=11.0, fontweight="bold", color="#1E293B", zorder=6)

    ax.plot([107.2, 107.2, 141.1, 141.1], [83.0, 82.3, 82.3, 83.0], color="#60A5FA", lw=1.2, zorder=5)
    ax.plot([141.9, 141.9, 175.8, 175.8], [83.0, 82.3, 82.3, 83.0], color="#34D399", lw=1.2, zorder=5)
    ax.text(124.1, 80.8, "1D Subspace ($4 \\times 256 = 1024\\text{-D}$)",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1E3A8A", zorder=5)
    ax.text(158.8, 80.8, "2D Subspace ($4 \\times 256 = 1024\\text{-D}$)",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="#065F46", zorder=5)

    # --- Box C2: 3-Layer MLP Pyramid & Relation Index Gathering (y=58.2..75.6) ---
    add_card(ax, 105.6, 58.2, 71.8, 17.4, facecolor="#FAF8FF", edgecolor="#C4B5FD", lw=1.2, zorder=3)
    ax.text(141.5, 73.8, "3-Layer MLP Classifier & Relation Gathering ($C = 86$)",
            ha="center", va="center", fontsize=11.4, fontweight="bold", color="#4C1D95", zorder=4)

    mlp_layers = [
        (107.4, 59.8, 5.1, 11.6, "2048\n$\\mathbf{z}_{ab}$", "#F5F3FF", "#A78BFA"),
        (116.8, 61.1, 5.1, 9.0,  "512\n$\\mathbf{h}_1$",   "#EDE9FE", "#8B5CF6"),
        (126.2, 62.2, 5.1, 6.8,  "256\n$\\mathbf{h}_2$",   "#DDD6FE", "#7C3AED"),
        (135.6, 63.1, 5.1, 5.0,  "86\n$\\mathbf{o}_{ab}$", "#C4B5FD", "#6D28D9"),
    ]
    for idx, (lx, ly, lw_b, lh_b, ltxt, lbg, led) in enumerate(mlp_layers):
        ax.add_patch(Rectangle((lx, ly), lw_b, lh_b, facecolor=lbg, edgecolor=led, lw=1.15, zorder=5))
        ax.text(lx + lw_b / 2, ly + lh_b / 2, ltxt, ha="center", va="center",
                fontsize=9.6, fontweight="bold", color="#2E1065", zorder=6)
        if idx < 3:
            nx, ny, _, nh_b, _, _, _ = mlp_layers[idx + 1]
            ax.plot([lx + lw_b, nx], [ly + lh_b, ny + nh_b], color="#A78BFA", lw=0.85, linestyle=":", zorder=4)
            ax.plot([lx + lw_b, nx], [ly, ny], color="#A78BFA", lw=0.85, linestyle=":", zorder=4)
            add_arrow(ax, lx + lw_b, ly + lh_b / 2, nx, ny + nh_b / 2, color="#7C3AED", lw=1.4, mutation_scale=11)
    add_arrow(ax, 140.7, 65.6, 143.8, 65.6, color="#7C3AED", lw=1.6, mutation_scale=12)

    add_card(ax, 143.8, 59.6, 32.0, 11.8, facecolor="#FFFFFF", edgecolor="#C4B5FD", lw=1.15, zorder=4)
    ax.text(159.8, 69.6, "Relation Gathering $s_{ab,r} = \\mathbf{o}_{ab}[r]$", ha="center", va="center",
            fontsize=10.6, fontweight="bold", color="#4C1D95", zorder=5)
    slot_labels = ["1", "2", "..", "r", "..", "86"]
    for s_i, s_lab in enumerate(slot_labels):
        sx = 145.6 + s_i * 4.7
        is_r = (s_lab == "r")
        s_bg = "#FECACA" if is_r else "#F8FAFC"
        s_ed = "#F87171" if is_r else "#CBD5E1"
        s_fg = "#991B1B" if is_r else "#475569"
        ax.add_patch(Rectangle((sx, 64.4), 4.1, 3.0, facecolor=s_bg, edgecolor=s_ed, lw=0.95, zorder=5))
        ax.text(sx + 2.05, 65.9, s_lab, ha="center", va="center", fontsize=10.0, fontweight="bold", color=s_fg, zorder=6)
    ax.text(159.8, 61.5, "Output $\\hat{p}_{ab,r} = \\sigma(s_{ab,r})$",
            ha="center", va="center", fontsize=10.4, fontweight="bold", color="#1E293B", zorder=5)

    # --- Box C3: Curriculum OHEM Focal Loss (y=34.6..54.4, Left) ---
    add_card(ax, 105.6, 34.6, 34.8, 19.8, facecolor="#FFFDFD", edgecolor="#FCA5A5", lw=1.2, zorder=3)
    ax.text(123.0, 52.6, "Curriculum OHEM Focal Loss", ha="center", va="center",
            fontsize=11.2, fontweight="bold", color="#991B1B", zorder=4)
    ax.text(123.0, 49.7, "Ep 1–10 Warm-Up $\\to$ Ep >10 OHEM", ha="center", va="center",
            fontsize=9.8, fontweight="bold", color="#7F1D1D", zorder=4)

    neg_heights = [7.6, 6.7, 5.8, 4.9, 2.5, 1.8, 1.2, 0.7]
    for n_i, nh in enumerate(neg_heights):
        nx = 108.4 + n_i * 3.6
        keep = n_i < 4
        ncol = "#FCA5A5" if keep else "#E2E8F0"
        ned  = "#EF4444" if keep else "#CBD5E1"
        ax.add_patch(Rectangle((nx, 39.2), 2.75, nh, facecolor=ncol,
                               edgecolor=ned, lw=0.85, zorder=5))
    ax.plot([122.4, 122.4], [38.6, 47.6], color="#E11D48", lw=1.5, linestyle="--", zorder=6)
    ax.text(115.0, 36.9, "Top-50% Hard Neg", ha="center", va="center",
            fontsize=9.6, fontweight="bold", color="#991B1B", zorder=5)
    ax.text(130.8, 36.9, "Easy Neg (0 Grad)", ha="center", va="center",
            fontsize=9.6, fontweight="bold", color="#64748B", zorder=5)

    # --- Box C4: Cold-Start Threshold Calibration (y=34.6..54.4, Right) ---
    add_card(ax, 142.4, 34.6, 35.0, 19.8, facecolor="#FFFEFA", edgecolor="#FCD34D", lw=1.2, zorder=3)
    ax.text(159.9, 52.6, "Cold-Start Calibration ($\\tau^*$)", ha="center", va="center",
            fontsize=11.2, fontweight="bold", color="#92400E", zorder=4)

    px = np.linspace(144.6, 175.2, 120)
    t_norm = np.linspace(0, 1, 120)
    neg_dist = np.exp(-((t_norm - 0.16) / 0.11) ** 2) * 7.8
    cold_pos = np.exp(-((t_norm - 0.42) / 0.15) ** 2) * 6.9
    warm_pos = np.exp(-((t_norm - 0.82) / 0.11) ** 2) * 7.6
    base_y = 39.8
    ax.plot(px, base_y + neg_dist, color="#94A3B8", lw=1.3, zorder=5)
    ax.fill_between(px, base_y, base_y + neg_dist, color="#E2E8F0", alpha=0.5, zorder=4)
    ax.plot(px, base_y + cold_pos, color="#F59E0B", lw=1.5, zorder=5)
    ax.fill_between(px, base_y, base_y + cold_pos, color="#FEF3C7", alpha=0.65, zorder=4)
    ax.plot(px, base_y + warm_pos, color="#60A5FA", lw=1.4, linestyle=":", zorder=5)
    ax.plot([144.6, 175.2], [base_y, base_y], color="#64748B", lw=1.0, zorder=5)

    x_tau_star = 144.6 + 0.31 * (175.2 - 144.6)
    x_tau_050  = 144.6 + 0.50 * (175.2 - 144.6)
    ax.plot([x_tau_star, x_tau_star], [base_y, base_y + 8.8], color="#E11D48", lw=1.6, zorder=6)
    ax.plot([x_tau_050, x_tau_050], [base_y, base_y + 8.8], color="#64748B", lw=1.3, linestyle="--", zorder=6)
    ax.text(x_tau_star - 0.4, base_y + 9.6, "$\\tau^*\\!=\\!0.31$", ha="right", va="center",
            fontsize=9.6, fontweight="bold", color="#BE123C", zorder=6)
    ax.text(x_tau_050 + 0.4, base_y + 9.6, "$\\tau\\!=\\!0.50$", ha="left", va="center",
            fontsize=9.4, fontweight="bold", color="#475569", zorder=6)
    ax.text(149.5, base_y + 3.0, "Neg", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#475569", zorder=6)
    ax.text(157.4, base_y + 3.0, "Cold+", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#92400E", zorder=6)
    ax.text(169.6, base_y + 3.0, "Warm+", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#1E40AF", zorder=6)
    ax.text(159.9, 36.9, "$\\mathcal{S}_2$ F1 $+0.0892$  |  $\\mathcal{S}_3$ F1 $+0.2281$",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#92400E", zorder=5)

    # =========================================================================
    # ORTHOGONAL ROUTING: PANEL B -> PANEL C & WITHIN PANEL C
    # =========================================================================
    add_ortho_arrow(
        ax,
        [(98.4, 71.8), (101.6, 71.8), (101.6, 87.4), (105.4, 87.4)],
        color="#3B82F6", lw=2.1, mutation_scale=15, zorder=7
    )

    add_ortho_arrow(
        ax,
        [(98.4, 39.2), (102.7, 39.2), (102.7, 84.8), (105.4, 84.8)],
        color="#10B981", lw=2.1, mutation_scale=15, zorder=7
    )

    add_arrow(ax, 141.5, 79.2, 141.5, 75.8, color="#7C3AED", lw=2.2, mutation_scale=15, zorder=7)
    add_arrow(ax, 123.0, 58.0, 123.0, 54.6, color="#E11D48", lw=2.2, mutation_scale=15, zorder=7)
    add_arrow(ax, 159.9, 58.0, 159.9, 54.6, color="#D97706", lw=2.2, mutation_scale=15, zorder=7)

    # =========================================================================
    # PANEL D: MULTI-LEVEL PHARMACOPHORIC & MECHANISTIC INTERPRETABILITY (x=1.2..178.8, y=1.0..28.8)
    # =========================================================================
    add_header_container(
        ax, 1.2, 1.0, 177.6, 27.8,
        "D. Multi-Level Pharmacophoric & Mechanistic Interpretability",
        bg_color="#F9FDFB", border_color="#86EFAC",
        header_bg="#DCFCE7", header_fg="#14532D",
        header_h=4.2, lw=1.4, header_size=13.2
    )

    # --- Box D1: Atom & Gate Saliency (x=2.8..44.4) ---
    add_card(ax, 2.8, 2.2, 41.6, 21.2, facecolor="#FFFFFF", edgecolor="#86EFAC", lw=1.15, zorder=3)
    ax.text(23.6, 21.6, "1. Atom & Gate Saliency (IG)", ha="center", va="center",
            fontsize=11.2, fontweight="bold", color="#14532D", zorder=4)
    mol_ig_img = render_rdkit_mol_exact(
        "OC(Cn1cncn1)(Cn2cncn2)c3ccc(F)cc3F",
        w_units=26.1, h_units=14.0,
        highlight_smarts="c1ncnn1",
        highlight_color=(0.98, 0.74, 0.74)
    )
    place_mol_on_ax(ax, mol_ig_img, 4.0, 6.2, 39.2, 14.0, zorder=4)
    ax.text(23.6, 4.5, "OAT1/3 $-\\mathrm{COO}^-$ & CYP2C9 Triazole",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color="#166534", zorder=5)

    # --- Box D2: Global Fingerprint Attribution (x=46.8..88.4) ---
    add_card(ax, 46.8, 2.2, 41.6, 21.2, facecolor="#FFFFFF", edgecolor="#86EFAC", lw=1.15, zorder=3)
    ax.text(67.6, 21.6, "2. Global Fingerprint Attribution", ha="center", va="center",
            fontsize=11.2, fontweight="bold", color="#14532D", zorder=4)
    ax.text(67.6, 19.0, "Top Pharmacophoric Substructures:", ha="center", va="center",
            fontsize=9.8, fontweight="bold", color="#334155", zorder=4)

    fp_bars = [
        ("MACCS #123 ($-\\mathrm{COO}^-$)", 0.94, "#FDE68A", "#D97706", 15.6),
        ("MACCS #95 (Triazole)",            0.88, "#BFDBFE", "#2563EB", 12.6),
        ("MACCS #142 ($\\geq 2$ N)",        0.82, "#FECDD3", "#E11D48", 9.6),
        ("ECFP4 $r\\!=\\!2$ (Aryl-F)",      0.76, "#A7F3D0", "#059669", 6.6),
    ]
    for flab, fval, fcol, fed, fy in fp_bars:
        ax.text(69.2, fy + 0.9, flab, ha="right", va="center", fontsize=9.2, fontweight="bold", color="#334155", zorder=5)
        bw_f = fval * 16.5
        ax.add_patch(Rectangle((70.0, fy), bw_f, 1.8, facecolor=fcol, edgecolor=fed, lw=0.95, zorder=5))
    ax.text(67.6, 4.5, "Explicit 679-Bit Chemical Anchors",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color="#166534", zorder=5)

    # --- Box D3: Operator Share & Cold-Start Scaling (x=90.8..132.4) ---
    add_card(ax, 90.8, 2.2, 41.6, 21.2, facecolor="#FFFFFF", edgecolor="#86EFAC", lw=1.15, zorder=3)
    ax.text(111.6, 21.6, "3. Fusion Share & Cold-Start Scaling", ha="center", va="center",
            fontsize=11.2, fontweight="bold", color="#14532D", zorder=4)

    ax.add_patch(Rectangle((93.2, 16.6), 16.6, 2.7, facecolor="#FECACA", edgecolor="#F87171", lw=0.95, zorder=5))
    ax.add_patch(Rectangle((109.8, 16.6), 17.6, 2.7, facecolor="#BBF7D0", edgecolor="#34D399", lw=0.95, zorder=5))
    ax.add_patch(Rectangle((127.4, 16.6), 2.6,  2.7, facecolor="#DDD6FE", edgecolor="#A78BFA", lw=0.95, zorder=5))
    ax.text(101.5, 17.95, "$|\\Delta|$ 45%", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#991B1B", zorder=6)
    ax.text(118.6, 17.95, "$\\mathbf{u}_a,\\mathbf{u}_b$ 52%", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#14532D", zorder=6)

    sx_pts = np.linspace(95.0, 128.2, 5)
    sy_auc = [8.0, 10.0, 11.0, 12.9, 13.6]
    sy_f1  = [6.4, 8.6,  9.2,  11.3, 12.1]
    ax.plot(sx_pts, sy_auc, marker="o", markersize=4.8, color="#3B82F6", lw=1.6, zorder=5)
    ax.plot(sx_pts, sy_f1,  marker="s", markersize=4.4, color="#F59E0B", lw=1.6, zorder=5)
    ax.plot([93.6, 129.6], [5.9, 5.9], color="#94A3B8", lw=1.0, zorder=4)
    ax.text(111.6, 14.8, "Cold-Start Scaling ($S_{\\max}$)", ha="center", va="center",
            fontsize=9.8, fontweight="bold", color="#1E3A8A", zorder=5)
    ax.text(111.6, 4.5, "AUROC $0.62 \\to 0.89$ ($\\mathcal{S}_2, \\mathcal{S}_3$)",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color="#166534", zorder=5)

    # --- Box D4: Latent t-SNE Clusters & Ward Dendrogram (x=134.8..176.4) ---
    add_card(ax, 134.8, 2.2, 41.6, 21.2, facecolor="#FFFFFF", edgecolor="#86EFAC", lw=1.15, zorder=3)
    ax.text(155.6, 21.6, "4. Latent PK/PD Taxonomy", ha="center", va="center",
            fontsize=11.2, fontweight="bold", color="#14532D", zorder=4)

    np.random.seed(7)
    clusters = [
        (140.4, 17.2, "#60A5FA"),
        (147.6, 17.8, "#F87171"),
        (140.2, 12.6, "#34D399"),
        (147.6, 12.2, "#FBBF24"),
    ]
    for cx_c, cy_c, c_col in clusters:
        pts_x = cx_c + np.random.normal(0, 0.95, 12)
        pts_y = cy_c + np.random.normal(0, 0.82, 12)
        ax.scatter(pts_x, pts_y, s=15, color=c_col, alpha=0.85, edgecolors="none", zorder=5)
    ax.add_patch(Rectangle((136.8, 10.4), 14.4, 9.2, facecolor="#FAFBFD", edgecolor="#CBD5E1", lw=1.0, zorder=4))
    ax.text(144.0, 8.6, "t-SNE ($\\mathbf{h}_2 \\odot \\mathbf{W}_{\\mathrm{cls}}[r,:]^\\top$)",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#334155", zorder=5)

    # Ward Dendrogram Tree
    ax.plot([167.2, 162.2, 162.2, 167.2], [18.6, 18.6, 16.4, 16.4], color="#60A5FA", lw=1.7, zorder=5)
    ax.plot([167.2, 162.2, 162.2, 167.2], [13.6, 13.6, 11.4, 11.4], color="#F87171", lw=1.7, zorder=5)
    ax.plot([162.2, 157.8, 157.8, 162.2], [17.5, 17.5, 12.5, 12.5], color="#64748B", lw=1.7, zorder=5)
    ax.text(168.0, 18.6, "CYP", ha="left", va="center", fontsize=9.6, fontweight="bold", color="#1E40AF", zorder=5)
    ax.text(168.0, 16.4, "OAT", ha="left", va="center", fontsize=9.6, fontweight="bold", color="#1E40AF", zorder=5)
    ax.text(168.0, 13.6, "QTc", ha="left", va="center", fontsize=9.6, fontweight="bold", color="#991B1B", zorder=5)
    ax.text(168.0, 11.4, "CNS", ha="left", va="center", fontsize=9.6, fontweight="bold", color="#991B1B", zorder=5)
    ax.text(165.4, 8.6, "Ward Tree ($\\mathbf{W}_{\\mathrm{cls}}$)", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#334155", zorder=5)

    ax.text(155.6, 4.5, "6 PK/PD Super-Families",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color="#166534", zorder=5)

    add_arrow(ax, 44.6, 12.8, 46.6, 12.8, color="#10B981", lw=2.0)
    add_arrow(ax, 88.6, 12.8, 90.6, 12.8, color="#10B981", lw=2.0)
    add_arrow(ax, 132.6, 12.8, 134.6, 12.8, color="#10B981", lw=2.0)

    # =========================================================================
    # ORTHOGONAL ROUTING TO BOTTOM PANEL D
    # =========================================================================
    add_arrow(ax, 141.5, 33.2, 141.5, 29.0, color="#10B981", lw=2.1, linestyle="--", mutation_scale=15, zorder=7)

    for target_dir in [FIG_DIR, PAPER_FIG_DIR, SPACE_PAPER_FIG_DIR]:
        if target_dir.exists():
            fig.savefig(target_dir / "fig0_model_architecture.png", dpi=300,
                        bbox_inches="tight", pad_inches=0.04, facecolor="white")
            fig.savefig(target_dir / "fig0_model_architecture.pdf",
                        bbox_inches="tight", pad_inches=0.04, facecolor="white")

    plt.close(fig)
    print("Successfully generated pastel-toned Figure 1 (fig0_model_architecture.{png,pdf})")


if __name__ == "__main__":
    build_architecture_figure()

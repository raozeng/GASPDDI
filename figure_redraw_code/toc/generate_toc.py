#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-grade ACS JCIM Table of Contents (TOC) / Graphical Abstract generator
for GASPDDI.
- Matches the exact low-saturation academic pastel palette of Figure 1 (redraw_figure1.py).
- Removes equations/formulas that appear in the manuscript text for a clean visual overview.
- Maximizes typography sizes across all panels for effortless legibility at 3.25 in x 1.75 in (13:7).
"""

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "figure_redraw_code" / "figure1" / ".mpl_cache"))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from PIL import Image, ImageEnhance

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

FIG1_ASSET_DIR = PROJECT_ROOT / "figure_redraw_code" / "figure1" / "assets"
PAPER_FIG_DIR = PROJECT_ROOT / "paper" / "figures"

OUTPUT_PDF = PAPER_FIG_DIR / "toc_graphic.pdf"
OUTPUT_PNG = PAPER_FIG_DIR / "toc_graphic.png"
LOCAL_PDF = SCRIPT_DIR / "toc_graphic.pdf"
LOCAL_PNG = SCRIPT_DIR / "toc_graphic.png"

SMILES_FILENAME_MAP = {
    "methotrexate": "methotrexate.png",
    "ibuprofen": "ibuprofen.png",
    "fluconazole": "fluconazole.png",
}


def load_preserved_molecule(key):
    fpath = FIG1_ASSET_DIR / SMILES_FILENAME_MAP[key]
    rgba = Image.open(fpath).convert("RGBA")
    rgb_muted = ImageEnhance.Color(rgba.convert("RGB")).enhance(0.62)
    r_ch, g_ch, b_ch = rgb_muted.split()
    a_ch = rgba.split()[3]
    return np.asarray(Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch)))


def place_mol_on_ax(ax, img_arr, x, y, w, h, zorder=4):
    ih, iw = img_arr.shape[:2]
    img_aspect = iw / float(ih)
    box_aspect = w / float(h)
    if img_aspect > box_aspect:
        draw_w = w
        draw_h = w / img_aspect
        draw_x = x
        draw_y = y + (h - draw_h) / 2.0
    else:
        draw_h = h
        draw_w = h * img_aspect
        draw_x = x + (w - draw_w) / 2.0
        draw_y = y
    ax.imshow(
        img_arr,
        extent=[draw_x, draw_x + draw_w, draw_y, draw_y + draw_h],
        interpolation="lanczos",
        zorder=zorder,
        aspect="equal",
    )


def add_card(ax, x, y, w, h, facecolor="#FFFFFF", edgecolor="#D9DCE0",
             lw=1.2, radius=1.0, alpha=1.0, zorder=2, linestyle="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.15,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_header_container(ax, x, y, w, h, title, bg_color="#FCFCFD", border_color="#9DA5AF",
                         header_bg="#EAECEF", header_fg="#303640", radius=1.3, lw=1.4,
                         header_h=5.4, header_size=15.6, zorder=1):
    add_card(ax, x, y, w, h, facecolor=bg_color, edgecolor=border_color,
             lw=lw, radius=radius, zorder=zorder)
    add_card(ax, x, y + h - header_h, w, header_h, facecolor=header_bg,
             edgecolor=border_color, lw=lw * 0.9, radius=radius * 0.85, zorder=zorder + 1)
    ax.text(
        x + w / 2.0,
        y + h - header_h / 2.0,
        title,
        ha="center", va="center",
        fontsize=header_size, fontweight="bold",
        color=header_fg,
        zorder=zorder + 2,
    )


def add_arrow(ax, x1, y1, x2, y2, color="#505760", lw=2.1, style="-|>",
              linestyle="-", mutation_scale=15, zorder=7):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(arr)


def add_ortho_arrow(ax, pts, color="#505760", lw=2.1, linestyle="-",
                    mutation_scale=15, zorder=7):
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=linestyle,
                solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
    p_prev = pts[-2]
    p_last = pts[-1]
    add_arrow(ax, p_prev[0], p_prev[1], p_last[0], p_last[1],
              color=color, lw=lw, linestyle=linestyle,
              mutation_scale=mutation_scale, zorder=zorder)


def add_junction_dot(ax, x, y, color="#505760", r=0.55, zorder=8):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white", lw=0.6, zorder=zorder))


def draw_token_ribbon(ax, x, y, w, h, tokens, active_bg="#F5F8FB",
                      active_edge="#B2C7DE", active_fg="#3C486C",
                      fontsize=11.4, zorder=4):
    n = len(tokens)
    cell_w = w / float(n)
    for i, tok in enumerate(tokens):
        cx = x + i * cell_w
        rect = Rectangle(
            (cx + cell_w * 0.04, y), cell_w * 0.92, h,
            facecolor=active_bg, edgecolor=active_edge,
            linewidth=0.95, zorder=zorder
        )
        ax.add_patch(rect)
        ax.text(
            cx + cell_w * 0.50, y + h / 2.0, tok,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=active_fg, zorder=zorder + 1
        )


def draw_barcode(ax, x, y, w, h, bits, on_color="#C8E2D6", off_color="#F5F7F8",
                 on_edge="#5DAA8E", off_edge="#D9DCE0", fontsize=10.6, zorder=4):
    n = len(bits)
    cw = w / float(n)
    for i, b in enumerate(bits):
        cx = x + i * cw
        col = on_color if b else off_color
        ed = on_edge if b else off_edge
        fg = "#244C42" if b else "#9DA5AF"
        rect = Rectangle(
            (cx + cw * 0.05, y), cw * 0.90, h,
            facecolor=col, edgecolor=ed,
            linewidth=0.85, zorder=zorder
        )
        ax.add_patch(rect)
        ax.text(
            cx + cw * 0.50, y + h / 2.0, str(b),
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=fg, zorder=zorder + 1
        )


def build_toc_graphic():
    # Exact 3.25 : 1.75 (13 : 7) aspect ratio in equal coordinate space (182.0 x 98.0)
    W_UNITS, H_UNITS = 182.0, 98.0
    fig, ax = plt.subplots(figsize=(18.2, 9.8), dpi=300)
    ax.set_xlim(0, W_UNITS)
    ax.set_ylim(0, H_UNITS)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # =========================================================================
    # COLUMN 1: DRUG PAIR & COLD-START SPLITS (x=1.2..38.6, y=1.2..96.8)
    # =========================================================================
    add_header_container(
        ax, 1.2, 1.2, 37.4, 95.6,
        "1. Drug Pair & Splits",
        bg_color="#FCFCFD", border_color="#9DA5AF",
        header_bg="#EAECEF", header_fg="#303640",
        header_h=5.5, lw=1.4, header_size=15.6
    )

    # Drug A Card (Methotrexate)
    add_card(ax, 2.5, 62.6, 34.8, 27.2, facecolor="#FBFCFD", edgecolor="#B2C7DE", lw=1.2, zorder=3)
    ax.text(19.9, 87.3, "Drug A: Methotrexate",
            ha="center", va="center", fontsize=13.2, fontweight="bold", color="#3C486C", zorder=4)
    mol_a_img = load_preserved_molecule("methotrexate")
    place_mol_on_ax(ax, mol_a_img, 3.6, 68.4, 32.6, 16.6, zorder=4)
    tokens_a = ["C", "N", "c1", "n", "=O", "OH"]
    draw_token_ribbon(
        ax, 3.5, 63.8, 32.8, 3.7, tokens_a,
        active_bg="#F5F8FB", active_edge="#B2C7DE", active_fg="#3C486C",
        fontsize=11.4, zorder=4
    )

    # Drug B Card (Ibuprofen)
    add_card(ax, 2.5, 33.4, 34.8, 27.2, facecolor="#FEFDFB", edgecolor="#D5B99C", lw=1.2, zorder=3)
    ax.text(19.9, 58.1, "Drug B: Ibuprofen",
            ha="center", va="center", fontsize=13.2, fontweight="bold", color="#754737", zorder=4)
    mol_b_img = load_preserved_molecule("ibuprofen")
    place_mol_on_ax(ax, mol_b_img, 3.6, 39.2, 32.6, 16.6, zorder=4)
    tokens_b = ["C", "C", "c1", "c", "=O", "OH"]
    draw_token_ribbon(
        ax, 3.5, 34.6, 32.8, 3.7, tokens_b,
        active_bg="#FBF8F4", active_edge="#D5B99C", active_fg="#754737",
        fontsize=11.4, zorder=4
    )

    # Cold-Start Partition Matrix Card
    add_card(ax, 2.5, 2.6, 34.8, 28.8, facecolor="#FFFFFF", edgecolor="#D9DCE0", lw=1.2, zorder=3)
    ax.text(19.9, 28.8, "Cold-Start Split Protocol",
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#303640", zorder=4)

    gx0, gy0, gw, gh = 11.4, 4.2, 12.2, 9.6
    ax.text(gx0 + gw * 0.5, gy0 + gh * 2 + 1.8, "Known B",
            ha="center", va="center", fontsize=11.4, fontweight="bold", color="#445489")
    ax.text(gx0 + gw * 1.5, gy0 + gh * 2 + 1.8, "Unseen B",
            ha="center", va="center", fontsize=11.4, fontweight="bold", color="#763E3E")
    ax.text(gx0 - 0.8, gy0 + gh * 1.5, "Known\nA",
            ha="right", va="center", fontsize=11.2, fontweight="bold", color="#445489")
    ax.text(gx0 - 0.8, gy0 + gh * 0.5, "Unseen\nA",
            ha="right", va="center", fontsize=11.2, fontweight="bold", color="#763E3E")

    cells = [
        (0, 1, "S1 Warm",  "97.2% ACC",  "#F5F8FB", "#3C486C", "#B2C7DE"),
        (1, 1, "S2 Cold",  "Unseen B",   "#FAF9F3", "#6D4933", "#C9B880"),
        (0, 0, "S2 Cold",  "Unseen A",   "#FAF9F3", "#6D4933", "#C9B880"),
        (1, 0, "S3 Cold",  "Both Unseen","#FBF7F7", "#763E3E", "#E7C9C9"),
    ]
    for col_i, row_i, title_c, sub_c, fc_c, fg_c, ec_c in cells:
        cx = gx0 + col_i * gw
        cy = gy0 + row_i * gh
        ax.add_patch(Rectangle(
            (cx + 0.2, cy + 0.2), gw - 0.4, gh - 0.4,
            facecolor=fc_c, edgecolor=ec_c, lw=1.2, zorder=4
        ))
        ax.text(cx + gw / 2.0, cy + gh * 0.65, title_c,
                ha="center", va="center", fontsize=11.6, fontweight="bold", color=fg_c, zorder=5)
        ax.text(cx + gw / 2.0, cy + gh * 0.28, sub_c,
                ha="center", va="center", fontsize=10.4, fontweight="bold", color=fg_c, zorder=5)

    # =========================================================================
    # COLUMN 2: DUAL-STREAM MOLECULAR ENCODER (x=42.2..95.6, y=1.2..96.8)
    # =========================================================================
    add_header_container(
        ax, 42.2, 1.2, 53.4, 95.6,
        "2. Dual-Stream Encoder",
        bg_color="#FCFCFD", border_color="#B2C7DE",
        header_bg="#E9EFF6", header_fg="#3C486C",
        header_h=5.5, lw=1.4, header_size=15.6
    )

    # --- Top Stream: Stream 1 (1D Mamba-CNN Sequence Stream) ---
    add_header_container(
        ax, 43.5, 49.4, 50.8, 40.4,
        "Stream 1: 1D Mamba-CNN Sequence",
        bg_color="#FBFCFD", border_color="#D9E2EF",
        header_bg="#EDF1F7", header_fg="#3C486C",
        header_h=4.4, lw=1.2, header_size=13.5, zorder=2
    )

    # Sub-box 1A: Selective State-Space Scan
    add_card(ax, 44.7, 69.2, 48.4, 15.0, facecolor="#FFFFFF", edgecolor="#B2C7DE", lw=1.15, zorder=3)
    ax.text(68.9, 81.8, "Selective State-Space Scan (Mamba S6)",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#3C486C", zorder=4)

    state_xs = [52.8, 68.9, 85.0]
    state_labels = ["State 1", "State 2", "State 3"]
    for sx, slab in zip(state_xs, state_labels):
        ax.add_patch(Circle((sx, 75.8), 3.2, facecolor="#F5F8FB", edgecolor="#8CAACE", lw=1.35, zorder=5))
        ax.text(sx, 75.8, slab, ha="center", va="center", fontsize=11.0, fontweight="bold", color="#3C486C", zorder=6)

    add_arrow(ax, 46.2, 75.8, 49.3, 75.8, color="#708FC2", lw=1.8, mutation_scale=12, zorder=5)
    add_arrow(ax, 56.2, 75.8, 65.4, 75.8, color="#708FC2", lw=1.8, mutation_scale=12, zorder=5)
    add_arrow(ax, 72.3, 75.8, 81.5, 75.8, color="#708FC2", lw=1.8, mutation_scale=12, zorder=5)
    add_arrow(ax, 88.4, 75.8, 91.8, 75.8, color="#708FC2", lw=1.8, mutation_scale=12, zorder=5)

    # Selective Gate badge
    add_card(ax, 55.9, 69.9, 26.0, 3.3, facecolor="#FBF7F7", edgecolor="#D29797", lw=1.0, radius=0.6, zorder=5)
    ax.text(68.9, 71.55, "Input-Selective Gate",
            ha="center", va="center", fontsize=11.4, fontweight="bold", color="#763E3E", zorder=6)
    add_arrow(ax, 63.0, 73.3, 60.8, 75.4, color="#C27171", lw=1.4, linestyle="--", mutation_scale=10, zorder=5)
    add_arrow(ax, 74.8, 73.3, 77.0, 75.4, color="#C27171", lw=1.4, linestyle="--", mutation_scale=10, zorder=5)

    # Sub-box 1B: Gate Spikes & 1D-CNN Motif -> 1D Vector
    add_card(ax, 44.7, 50.8, 48.4, 16.8, facecolor="#FFFFFF", edgecolor="#B2C7DE", lw=1.15, zorder=3)
    ax.text(68.9, 65.3, "Heteroatom Gate Spikes & 1D-CNN Motif",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#3C486C", zorder=4)

    # Highlighted CNN motif window
    ax.add_patch(Rectangle(
        (51.8, 52.0), 19.2, 10.8, facecolor="#EDF1F7", edgecolor="#8CAACE",
        lw=1.3, linestyle="--", alpha=0.65, zorder=4
    ))
    ax.text(61.4, 61.2, "1D-CNN Motif", ha="center", va="center",
            fontsize=11.4, fontweight="bold", color="#3C486C", zorder=6)

    xs_curve = np.linspace(46.5, 74.5, 160)
    ys_curve = (
        55.0
        + 0.30 * np.sin((xs_curve - 46.5) * 1.1)
        + 4.5 * np.exp(-((xs_curve - 53.5) / 1.0) ** 2)
        + 5.0 * np.exp(-((xs_curve - 61.5) / 1.1) ** 2)
        + 4.4 * np.exp(-((xs_curve - 68.5) / 1.0) ** 2)
    )
    ax.plot(xs_curve, ys_curve, color="#AE5065", lw=1.6, zorder=5)
    ax.fill_between(xs_curve, 54.6, ys_curve, color="#F2E1E3", alpha=0.55, zorder=4)

    spike_toks = [("c", 48.5), ("N", 53.5), ("c", 57.5), ("O", 61.5), ("=O", 68.5), ("F", 73.0)]
    for stok, sx_t in spike_toks:
        col_t = "#78394A" if stok in ("N", "O", "=O", "F") else "#505760"
        ax.text(sx_t, 53.0, stok, ha="center", va="center", fontsize=11.0, fontweight="bold", color=col_t, zorder=6)

    add_card(ax, 77.2, 52.0, 14.6, 10.8, facecolor="#F5F8FB", edgecolor="#8CAACE", lw=1.15, radius=0.8, zorder=5)
    ax.text(84.5, 59.5, "Mean Pool", ha="center", va="center", fontsize=11.6, fontweight="bold", color="#3C486C", zorder=6)
    ax.text(84.5, 54.8, "1D Vec\n(256-D)", ha="center", va="center", fontsize=11.4, fontweight="bold", color="#445489", zorder=6)
    add_arrow(ax, 74.8, 57.4, 77.0, 57.4, color="#708FC2", lw=1.8, mutation_scale=12, zorder=6)

    # --- Bottom Stream: Stream 2 (2D Fingerprints) ---
    add_header_container(
        ax, 43.5, 2.6, 50.8, 44.8,
        "Stream 2: 2D Molecular Fingerprints",
        bg_color="#F9FBFA", border_color="#C8E2D6",
        header_bg="#E2F1E9", header_fg="#244C41",
        header_h=4.4, lw=1.2, header_size=13.5, zorder=2
    )

    # Left sub-card: Morgan ECFP4
    add_card(ax, 44.7, 14.2, 23.6, 27.6, facecolor="#FFFFFF", edgecolor="#A1D4B4", lw=1.15, zorder=3)
    ax.text(56.5, 39.4, "Morgan ECFP4", ha="center", va="center",
            fontsize=12.6, fontweight="bold", color="#244C41", zorder=4)
    ax.text(56.5, 35.8, "512-Bit Circular", ha="center", va="center",
            fontsize=11.2, fontweight="bold", color="#3C8D72", zorder=4)

    cx_m, cy_m = 56.5, 26.8
    ax.add_patch(Circle((cx_m, cy_m), 6.8, facecolor="#F3F9F6", edgecolor="#8DC8B1", lw=1.1, linestyle="--", zorder=4))
    ax.add_patch(Circle((cx_m, cy_m), 4.4, facecolor="#E2F1E9", edgecolor="#5DAA8E", lw=1.1, linestyle=":", zorder=4))
    ax.add_patch(Circle((cx_m, cy_m), 2.0, facecolor="#C8E2D6", edgecolor="#326958", lw=1.2, zorder=5))
    ax.text(cx_m, cy_m, "C", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#244C42", zorder=6)

    for ang, nlab, rad_n in [(35, "N", 4.4), (155, "O", 4.4), (270, "C", 4.4), (95, "F", 6.4), (325, "O", 6.4)]:
        nx = cx_m + rad_n * np.cos(np.deg2rad(ang))
        ny = cy_m + rad_n * np.sin(np.deg2rad(ang))
        ax.plot([cx_m, nx], [cy_m, ny], color="#5DAA8E", lw=1.1, zorder=4)
        ax.add_patch(Circle((nx, ny), 1.35, facecolor="#FFFFFF", edgecolor="#5DAA8E", lw=1.0, zorder=6))
        ax.text(nx, ny, nlab, ha="center", va="center", fontsize=10.2, fontweight="bold", color="#244C41", zorder=7)

    draw_barcode(
        ax, 46.1, 15.6, 20.8, 3.2, [1, 0, 1, 1, 0, 1, 0, 1],
        on_color="#C8E2D6", off_color="#F5F7F8", on_edge="#5DAA8E", fontsize=10.6, zorder=5
    )

    # Right sub-card: MACCS Keys
    add_card(ax, 69.5, 14.2, 23.6, 27.6, facecolor="#FFFFFF", edgecolor="#A1D4B4", lw=1.15, zorder=3)
    ax.text(81.3, 39.4, "MACCS Keys", ha="center", va="center",
            fontsize=12.6, fontweight="bold", color="#244C41", zorder=4)
    ax.text(81.3, 35.8, "167-Bit Substructures", ha="center", va="center",
            fontsize=11.2, fontweight="bold", color="#3C8D72", zorder=4)

    maccs_items = [
        ("#123 Carboxylate", "#FAF9F3", "#6D4933", "#B38D4D", 29.8),
        ("#79 Triazole Ring", "#F5F8FB", "#445489", "#8CAACE", 24.8),
        ("#142 N-Heteroatom", "#FBF7F7", "#763E3E", "#D29797", 19.8),
    ]
    for mtxt, mbg, mfg, med, my in maccs_items:
        add_card(ax, 70.9, my, 20.8, 3.8, facecolor=mbg, edgecolor=med, lw=1.0, radius=0.6, zorder=4)
        ax.text(81.3, my + 1.9, mtxt, ha="center", va="center", fontsize=11.2, fontweight="bold", color=mfg, zorder=5)

    draw_barcode(
        ax, 70.9, 15.6, 20.8, 3.2, [0, 1, 1, 0, 1, 1, 0, 1],
        on_color="#B1DED5", off_color="#F5F7F8", on_edge="#58A99E", fontsize=10.6, zorder=5
    )

    # Bottom projection bar in Stream 2
    add_card(ax, 44.7, 3.8, 48.4, 9.0, facecolor="#F6F9F8", edgecolor="#8DC8B1", lw=1.15, radius=0.8, zorder=4)
    ax.text(59.8, 8.3, "679-Bit Dense Projection",
            ha="center", va="center", fontsize=12.2, fontweight="bold", color="#244C41", zorder=5)
    add_card(ax, 77.2, 4.8, 14.6, 7.0, facecolor="#E2F1E9", edgecolor="#5DAA8E", lw=1.15, radius=0.7, zorder=5)
    ax.text(84.5, 8.3, "2D Vec (256-D)",
            ha="center", va="center", fontsize=11.4, fontweight="bold", color="#244C41", zorder=6)
    add_arrow(ax, 73.6, 8.3, 77.0, 8.3, color="#3C8D72", lw=1.8, mutation_scale=12, zorder=6)

    # =========================================================================
    # COLUMN 3: 8-BLOCK FUSION, OHEM & CALIBRATION (x=99.2..141.6, y=1.2..96.8)
    # =========================================================================
    add_header_container(
        ax, 99.2, 1.2, 42.4, 95.6,
        "3. Fusion & Calibration",
        bg_color="#FCFCFD", border_color="#D8D2EC",
        header_bg="#F3F2F9", header_fg="#533E74",
        header_h=5.5, lw=1.4, header_size=15.6
    )

    # --- Top Card: 8-Block Fusion & MLP Classifier ---
    add_card(ax, 100.5, 59.6, 39.8, 30.2, facecolor="#FCFBFD", edgecolor="#D8D2EC", lw=1.2, zorder=3)
    ax.text(120.4, 87.3, "8-Block Pairwise Fusion",
            ha="center", va="center", fontsize=13.2, fontweight="bold", color="#533E74", zorder=4)

    fusion_chips = [
        ("1D Seq",   "#F5F8FB", "#3C486C", "#B2C7DE"),
        ("2D FP",    "#F3F9F6", "#244C41", "#A1D4B4"),
        ("Hadamard", "#F3F2F9", "#533E74", "#B7AADB"),
        ("Abs Diff", "#FBF7F7", "#763E3E", "#D29797"),
    ]
    chip_x0 = 101.7
    chip_w = 8.9
    chip_gap = 0.6
    for i, (clab, cbg, cfg, ced) in enumerate(fusion_chips):
        cx = chip_x0 + i * (chip_w + chip_gap)
        add_card(ax, cx, 79.2, chip_w, 5.2, facecolor=cbg, edgecolor=ced, lw=1.05, radius=0.6, zorder=4)
        ax.text(cx + chip_w / 2.0, 81.8, clab, ha="center", va="center",
                fontsize=11.0, fontweight="bold", color=cfg, zorder=5)

    # 3-Layer MLP Funnel
    mlp_layers = [
        (102.4, 72.8, 36.0, 4.2, "2048-D Fusion Vector", "#F9F8FC", "#533E74", "#B7AADB"),
        (105.9, 67.2, 29.0, 4.2, "512-D \u2192 256-D MLP", "#F3F2F9", "#533E74", "#9C87CB"),
        (109.4, 61.6, 22.0, 4.2, "86 DDI Type Logits",   "#E9E6F4", "#35274E", "#8869BE"),
    ]
    for lx, ly, lw_m, lh_m, ltxt, lbg, lfg, led in mlp_layers:
        add_card(ax, lx, ly, lw_m, lh_m, facecolor=lbg, edgecolor=led, lw=1.1, radius=0.6, zorder=4)
        ax.text(lx + lw_m / 2.0, ly + lh_m / 2.0, ltxt, ha="center", va="center",
                fontsize=11.8, fontweight="bold", color=lfg, zorder=5)

    # --- Middle Card: Curriculum OHEM Mining ---
    add_card(ax, 100.5, 31.2, 39.8, 26.4, facecolor="#FFFEFE", edgecolor="#E7C9C9", lw=1.2, zorder=3)
    ax.text(120.4, 55.0, "Curriculum OHEM Focal Loss",
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#763E3E", zorder=4)
    ax.text(120.4, 51.2, "Warm-Up \u2192 Top-50% Hard Neg",
            ha="center", va="center", fontsize=11.4, fontweight="bold", color="#643838", zorder=4)

    ohem_bars = [
        ("Easy Neg", 0.20, "#EAECEF", "#D9DCE0", "Dropped",  "#6E7681", 44.4),
        ("Hard Neg", 0.72, "#F2E1E3", "#D29797", "Retained", "#763E3E", 39.0),
        ("Top-50%",  0.90, "#E7C9C9", "#C27171", "Retained", "#763E3E", 33.6),
    ]
    for blab, bval, bcol, bed, btag, btag_c, by in ohem_bars:
        ax.text(111.4, by + 1.8, blab, ha="right", va="center", fontsize=11.2, fontweight="bold", color="#3C434C", zorder=5)
        bw_o = bval * 16.5
        ax.add_patch(Rectangle((112.4, by), bw_o, 3.6, facecolor=bcol, edgecolor=bed, lw=1.05, zorder=5))
        ax.text(113.2 + bw_o, by + 1.8, btag, ha="left", va="center", fontsize=11.0, fontweight="bold", color=btag_c, zorder=5)

    # --- Bottom Card: Cold-Start Threshold Calibration ---
    add_card(ax, 100.5, 2.6, 39.8, 26.6, facecolor="#FEFDFC", edgecolor="#C9B880", lw=1.2, zorder=3)
    ax.text(120.4, 26.6, "Cold-Start Calibration",
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#6D4933", zorder=4)

    px = np.linspace(102.4, 138.4, 200)
    neg_dist = 8.4 * np.exp(-((px - 107.5) / 3.8) ** 2)
    cold_pos = 7.0 * np.exp(-((px - 118.0) / 4.8) ** 2)
    warm_pos = 8.0 * np.exp(-((px - 132.0) / 4.0) ** 2)

    base_y = 8.8
    ax.plot(px, base_y + neg_dist, color="#9DA5AF", lw=1.4, zorder=5)
    ax.fill_between(px, base_y, base_y + neg_dist, color="#EAECEF", alpha=0.55, zorder=4)
    ax.plot(px, base_y + cold_pos, color="#B38D4D", lw=1.6, zorder=5)
    ax.fill_between(px, base_y, base_y + cold_pos, color="#F1EDDD", alpha=0.68, zorder=4)
    ax.plot(px, base_y + warm_pos, color="#8CAACE", lw=1.5, linestyle=":", zorder=5)
    ax.plot([102.4, 138.4], [base_y, base_y], color="#6E7681", lw=1.1, zorder=5)

    x_tau_star = 102.4 + 0.31 * (138.4 - 102.4)
    x_tau_050  = 102.4 + 0.50 * (138.4 - 102.4)
    ax.plot([x_tau_star, x_tau_star], [base_y, base_y + 10.6], color="#AE5065", lw=1.7, zorder=6)
    ax.plot([x_tau_050, x_tau_050], [base_y, base_y + 10.6], color="#6E7681", lw=1.4, linestyle="--", zorder=6)
    ax.text(x_tau_star - 0.5, base_y + 12.0, "Calib 0.31", ha="right", va="center",
            fontsize=11.2, fontweight="bold", color="#913F53", zorder=6)
    ax.text(x_tau_050 + 0.5, base_y + 12.0, "Default 0.50", ha="left", va="center",
            fontsize=11.2, fontweight="bold", color="#505760", zorder=6)

    ax.text(107.5, base_y + 2.8, "Neg", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#505760", zorder=6)
    ax.text(118.0, base_y + 2.8, "Cold+", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#6D4933", zorder=6)
    ax.text(132.0, base_y + 2.8, "Warm+", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#445489", zorder=6)
    ax.text(120.4, 5.2, "S2 F1 +8.9%  |  S3 F1 +22.8%",
            ha="center", va="center", fontsize=11.4, fontweight="bold", color="#6D4933", zorder=5)

    # =========================================================================
    # COLUMN 4: MECHANISTIC DISCOVERY (x=145.2..180.8, y=1.2..96.8)
    # =========================================================================
    add_header_container(
        ax, 145.2, 1.2, 35.6, 95.6,
        "4. Mechanistic Discovery",
        bg_color="#FBFCFC", border_color="#A1D4B4",
        header_bg="#E9F5ED", header_fg="#294835",
        header_h=5.5, lw=1.4, header_size=15.6
    )

    # --- Top Card: Atom & Gate Saliency (Fluconazole) ---
    add_card(ax, 146.4, 59.6, 33.2, 30.2, facecolor="#FFFFFF", edgecolor="#A1D4B4", lw=1.2, zorder=3)
    ax.text(163.0, 87.3, "Atom & Gate Saliency",
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#294835", zorder=4)
    mol_ig_img = load_preserved_molecule("fluconazole")
    place_mol_on_ax(ax, mol_ig_img, 147.4, 65.0, 31.2, 19.6, zorder=4)
    ax.text(163.0, 62.2, "Fluconazole: CYP2C9 Triazole",
            ha="center", va="center", fontsize=11.2, fontweight="bold", color="#2C4F39", zorder=5)

    # --- Middle Card: Fusion Share & Cold-Start Scaling ---
    add_card(ax, 146.4, 31.2, 33.2, 26.4, facecolor="#FFFFFF", edgecolor="#A1D4B4", lw=1.2, zorder=3)
    ax.text(163.0, 55.0, "Fusion Share & Cold-Start",
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#294835", zorder=4)

    # Operator share bar
    ax.add_patch(Rectangle((148.0, 48.0), 13.5, 3.8, facecolor="#F1DFDF", edgecolor="#D29797", lw=1.0, zorder=5))
    ax.add_patch(Rectangle((161.5, 48.0), 14.5, 3.8, facecolor="#D4EADC", edgecolor="#5DAA8E", lw=1.0, zorder=5))
    ax.add_patch(Rectangle((176.0, 48.0), 2.0,  3.8, facecolor="#E9E6F4", edgecolor="#B7AADB", lw=1.0, zorder=5))
    ax.text(154.75, 49.9, "Diff 45%", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#763E3E", zorder=6)
    ax.text(168.75, 49.9, "Feat 52%", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#294835", zorder=6)

    # Cold-Start scaling curves
    sx_pts = np.linspace(149.2, 176.8, 5)
    sy_auc = [36.8, 39.0, 40.5, 42.6, 44.0]
    sy_f1  = [35.0, 37.0, 38.4, 40.6, 41.8]
    ax.plot(sx_pts, sy_auc, marker="o", markersize=4.8, color="#708FC2", lw=1.7, zorder=5)
    ax.plot(sx_pts, sy_f1,  marker="s", markersize=4.4, color="#B38D4D", lw=1.7, zorder=5)
    ax.plot([148.0, 178.0], [34.2, 34.2], color="#9DA5AF", lw=1.0, zorder=4)
    ax.text(163.0, 45.6, "Similarity Scaling (0.62 \u2192 0.89)",
            ha="center", va="center", fontsize=11.0, fontweight="bold", color="#3C486C", zorder=5)
    ax.text(163.0, 32.5, "S2 AUROC 0.83 | S3 AUROC 0.69",
            ha="center", va="center", fontsize=11.0, fontweight="bold", color="#2C4F39", zorder=5)

    # --- Bottom Card: Latent PK/PD Taxonomy ---
    add_card(ax, 146.4, 2.6, 33.2, 26.6, facecolor="#FFFFFF", edgecolor="#A1D4B4", lw=1.2, zorder=3)
    ax.text(163.0, 26.6, "Latent PK/PD Taxonomy",
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#294835", zorder=4)

    np.random.seed(7)
    clusters = [
        (150.6, 20.6, "#8CAACE"),
        (156.8, 21.2, "#D29797"),
        (150.4, 14.4, "#5DAA8E"),
        (156.8, 14.0, "#BCA363"),
    ]
    ax.add_patch(Rectangle((147.6, 11.2), 12.2, 12.8, facecolor="#FCFCFD", edgecolor="#D9DCE0", lw=1.05, zorder=4))
    for cx_c, cy_c, c_col in clusters:
        pts_x = cx_c + np.random.normal(0, 0.85, 12)
        pts_y = cy_c + np.random.normal(0, 0.85, 12)
        ax.scatter(pts_x, pts_y, s=16, color=c_col, alpha=0.88, edgecolors="none", zorder=5)
    ax.text(153.7, 9.0, "t-SNE", ha="center", va="center", fontsize=11.4, fontweight="bold", color="#3C434C", zorder=5)

    # Ward Dendrogram Tree
    ax.plot([171.2, 166.6, 166.6, 171.2], [22.4, 22.4, 19.4, 19.4], color="#8CAACE", lw=1.8, zorder=5)
    ax.plot([171.2, 166.6, 166.6, 171.2], [15.8, 15.8, 12.8, 12.8], color="#D29797", lw=1.8, zorder=5)
    ax.plot([166.6, 162.4, 162.4, 166.6], [20.9, 20.9, 14.3, 14.3], color="#6E7681", lw=1.8, zorder=5)
    ax.text(172.0, 22.4, "CYP", ha="left", va="center", fontsize=11.4, fontweight="bold", color="#445489", zorder=5)
    ax.text(172.0, 19.4, "OAT", ha="left", va="center", fontsize=11.4, fontweight="bold", color="#445489", zorder=5)
    ax.text(172.0, 15.8, "QTc", ha="left", va="center", fontsize=11.4, fontweight="bold", color="#763E3E", zorder=5)
    ax.text(172.0, 12.8, "CNS", ha="left", va="center", fontsize=11.4, fontweight="bold", color="#763E3E", zorder=5)
    ax.text(169.8, 9.0, "Ward Tree", ha="center", va="center", fontsize=11.4, fontweight="bold", color="#3C434C", zorder=5)
    ax.text(163.0, 5.2, "6 PK/PD Super-Families", ha="center", va="center",
            fontsize=11.6, fontweight="bold", color="#2C4F39", zorder=5)

    # =========================================================================
    # INTER-COLUMN ORTHOGONAL BUS ROUTING & JUNCTION DOTS
    # =========================================================================
    # Column 1 -> Column 2
    bus1_x = 40.4
    ax.plot([37.3, bus1_x], [76.2, 76.2], color="#8CAACE", lw=2.1, zorder=6)
    ax.plot([37.3, bus1_x], [47.0, 47.0], color="#C39874", lw=2.1, zorder=6)
    ax.plot([bus1_x, bus1_x], [25.0, 76.2], color="#6E7681", lw=2.1, zorder=6)
    add_junction_dot(ax, bus1_x, 69.6, color="#708FC2", r=0.55)
    add_junction_dot(ax, bus1_x, 47.0, color="#C39874", r=0.55)
    add_junction_dot(ax, bus1_x, 25.0, color="#3C8D72", r=0.55)
    add_arrow(ax, bus1_x, 69.6, 43.5, 69.6, color="#708FC2", lw=2.1, mutation_scale=14, zorder=7)
    add_arrow(ax, bus1_x, 25.0, 43.5, 25.0, color="#3C8D72", lw=2.1, mutation_scale=14, zorder=7)

    # Column 2 -> Column 3
    add_ortho_arrow(
        ax,
        [(93.1, 57.4), (96.9, 57.4), (96.9, 82.8), (100.5, 82.8)],
        color="#708FC2", lw=2.1, mutation_scale=14, zorder=7
    )
    add_ortho_arrow(
        ax,
        [(93.1, 8.3), (97.9, 8.3), (97.9, 80.6), (100.5, 80.6)],
        color="#3C8D72", lw=2.1, mutation_scale=14, zorder=7
    )

    # Internal arrows within Column 3
    add_arrow(ax, 120.4, 59.6, 120.4, 57.6, color="#8869BE", lw=2.0, mutation_scale=13, zorder=7)
    add_arrow(ax, 120.4, 31.2, 120.4, 29.2, color="#AE5065", lw=2.0, mutation_scale=13, zorder=7)

    # Column 3 -> Column 4
    bus3_x = 143.4
    ax.plot([140.3, bus3_x], [74.7, 74.7], color="#3C8D72", lw=2.1, zorder=6)
    ax.plot([140.3, bus3_x], [15.9, 15.9], color="#3C8D72", lw=2.1, zorder=6)
    ax.plot([bus3_x, bus3_x], [15.9, 74.7], color="#3C8D72", lw=2.1, zorder=6)
    add_junction_dot(ax, bus3_x, 74.7, color="#3C8D72", r=0.55)
    add_junction_dot(ax, bus3_x, 44.4, color="#3C8D72", r=0.55)
    add_junction_dot(ax, bus3_x, 15.9, color="#3C8D72", r=0.55)
    add_arrow(ax, bus3_x, 74.7, 146.4, 74.7, color="#3C8D72", lw=2.1, mutation_scale=14, zorder=7)
    add_arrow(ax, bus3_x, 44.4, 146.4, 44.4, color="#3C8D72", lw=2.1, mutation_scale=14, zorder=7)
    add_arrow(ax, bus3_x, 15.9, 146.4, 15.9, color="#3C8D72", lw=2.1, mutation_scale=14, zorder=7)

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    for p in (OUTPUT_PDF, LOCAL_PDF):
        fig.savefig(p, facecolor="white")
    for p in (OUTPUT_PNG, LOCAL_PNG):
        fig.savefig(p, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    build_toc_graphic()

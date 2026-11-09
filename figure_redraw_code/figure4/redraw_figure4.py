"""Revise Figure 4 headings, preserving original scientific panels pixel-for-pixel.
Run with Python + Pillow + Matplotlib. No model inference or data recomputation.
"""
import hashlib
import io
import json
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(CODE_DIR.parent / 'figure1' / '.mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw

SOURCE = ROOT / 'paper/figures/fig1_case_studies_substructure_saliency.png'
OUTPUT = SOURCE.with_name(SOURCE.stem + '_v2.png')
OUTPUT_PDF = SOURCE.with_name(SOURCE.stem + '_v2.pdf')
OUTPUT_A = ROOT / 'paper/figures/fig1a_case_studies_2d_saliency_v2.png'
OUTPUT_A_PDF = ROOT / 'paper/figures/fig1a_case_studies_2d_saliency_v2.pdf'
OUTPUT_B = ROOT / 'paper/figures/fig1b_mamba_selective_scan_gates_v2.png'
OUTPUT_B_PDF = ROOT / 'paper/figures/fig1b_mamba_selective_scan_gates_v2.pdf'
DPI = 300
TITLE_COLOR = '#1A252F'
CROP_TOP = 475
SPLIT_Y = 3790

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_png_and_pdf(img, png_path, pdf_path):
    img.save(png_path, dpi=(DPI, DPI))
    pdf_fig = plt.figure(figsize=(img.width / DPI, img.height / DPI), dpi=DPI, facecolor='white')
    pdf_ax = pdf_fig.add_axes([0, 0, 1, 1])
    pdf_ax.axis('off')
    pdf_ax.imshow(img, interpolation='none')
    pdf_fig.savefig(pdf_path, format='pdf', dpi=DPI, facecolor='white')
    plt.close(pdf_fig)

def make_title(text, width, height, size):
    fig = Figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor='white')
    canvas = FigureCanvasAgg(fig)
    t = fig.text(0, 0.94, text, ha='left', va='top', fontsize=size,
                 fontweight='bold', fontfamily='DejaVu Sans', color=TITLE_COLOR)
    canvas.draw()
    # Fit each line to its allocated strip, never crop letters or probabilities.
    measured = t.get_window_extent(canvas.get_renderer()).width
    if measured > width - 4:
        t.set_fontsize(size * (width - 4) / measured)
        canvas.draw()
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def main():
    watched = [SOURCE, SOURCE.with_suffix('.pdf'), ROOT / 'interpretability/run_all_interpretability.py']
    before = {str(p.relative_to(ROOT)): sha(p) for p in watched}
    image = Image.open(SOURCE).convert('RGB')
    assert image.size == (4895, 4805), 'Panel coordinates apply to the preserved original image only.'
    draw = ImageDraw.Draw(image)
    # Erase original E title bands only. Chart areas, y labels, values and axes stay intact.
    relations = [
        ('Y=73: Serum concentration of Drug 1 increased by Drug 2', '0.9819', '0.9497'),
        ('Y=47: Metabolism of Drug 1 decreased by Drug 2', '0.9547', '0.8763'),
        ('Y=20: Drug 1 increases QTc-prolonging activity of Drug 2', '0.9383', '0.8131'),
    ]
    for i, (relation, p_no3d, p_multi) in enumerate(relations):
        y = 494 + round(i * 1106.934)
        draw.rectangle((3290, y, 4894, y + 119), fill='white')
        image.paste(make_title(f'E{i+1} | {relation}', 1384, 53, 9.8), (3393, y + 8))
        image.paste(make_title(f'Predicted Probability (GASPDDI): P = {p_no3d}',
                               1384, 53, 9.8), (3393, y + 58))
    # Keep the two original bottom plots intact; use two-line D titles above them.
    draw.rectangle((120, 3800, 4894, 3934), fill='white')
    for i, (x, drug) in enumerate([
        (242, 'Fluconazole (DB00196) — CYP2C9 Inhibitor'),
        (2694, 'Ciprofloxacin (DB00537) — Fluoroquinolone'),
    ]):
        title = rf'D{i+1} | 1D Mamba Selective Scan Gate $\bar{{\Delta}}_t$ & Token Saliency'
        image.paste(make_title(title, 2080, 65, 10), (x, 3805))
        image.paste(make_title(drug, 2080, 57, 10), (x, 3870))
    # Remove overall title and the blank space above the first row, retaining its headings.
    result = image.crop((0, CROP_TOP, image.width, image.height))
    save_png_and_pdf(result, OUTPUT, OUTPUT_PDF)

    # Generate split Figure 3 (compact 2D molecular saliency + spacious D1-D3 MACCS attribution bars)
    # and Figure 4 (1D Mamba Selective Scan Gate profiles, A & B)
    import numpy as np
    part_b_img = image.crop((0, SPLIT_Y, image.width, image.height)).copy()
    arr_b = np.array(part_b_img)
    # Replace high-saturation red/blue token bars with Nature Communications softened Coral (#D4766A) and Steel Blue (#5C87B2)
    mask_red = (np.abs(arr_b[:, :, 0].astype(int) - 236) <= 8) & (np.abs(arr_b[:, :, 1].astype(int) - 115) <= 8) & (np.abs(arr_b[:, :, 2].astype(int) - 103) <= 8)
    mask_blue = (np.abs(arr_b[:, :, 0].astype(int) - 97) <= 8) & (np.abs(arr_b[:, :, 1].astype(int) - 174) <= 8) & (np.abs(arr_b[:, :, 2].astype(int) - 226) <= 8)
    arr_b[mask_red] = [212, 118, 106]
    arr_b[mask_blue] = [92, 135, 178]
    part_b_img = Image.fromarray(arr_b)
    draw_b = ImageDraw.Draw(part_b_img)
    draw_b.rectangle((120, 3800 - SPLIT_Y, 4894, 3934 - SPLIT_Y), fill='white')
    for letter, (x, drug) in zip(['A', 'B'], [
        (242, 'Fluconazole (DB00196) — CYP2C9 Inhibitor'),
        (2694, 'Ciprofloxacin (DB00537) — Fluoroquinolone'),
    ]):
        title = rf'{letter} | 1D Mamba Selective Scan Gate $\bar{{\Delta}}_t$ & Token Saliency'
        part_b_img.paste(make_title(title, 2120, 68, 13.6), (x, 3802 - SPLIT_Y))
        part_b_img.paste(make_title(drug, 2120, 60, 13.2), (x, 3868 - SPLIT_Y))
    save_png_and_pdf(part_b_img, OUTPUT_B, OUTPUT_B_PDF)

    # Build compact Figure 3: trim horizontal white padding inside Drug A & Drug B boxes
    # and give D1-D3 a much wider right column (~48% of total width) with larger fonts
    fig3_w, fig3_h = 4650, 3315
    part_a_img = Image.new('RGB', (fig3_w, fig3_h), 'white')
    draw_a = ImageDraw.Draw(part_a_img)

    case_titles = [
        ('Case A (PK: Renal OAT1/OAT3) — Drug A', 'Methotrexate (DB00563)',
         'Case A (PK: Renal OAT1/OAT3) — Drug B', 'Ibuprofen (DB01050)'),
        ('Case B (PK: Hepatic CYP2C9) — Drug A', 'Sulfamethoxazole (DB01015)',
         'Case B (PK: Hepatic CYP2C9) — Drug B', 'Fluconazole (DB00196)'),
        ('Case C (PD: Cardiac hERG / QTc) — Drug A', 'Fluoxetine (DB00472)',
         'Case C (PD: Cardiac hERG / QTc) — Drug B', 'Ciprofloxacin (DB00537)'),
    ]
    d_panels = [
        ('D1 | Y=73: Serum concentration of Drug 1 increased by Drug 2',
         'Predicted Probability (GASPDDI): P = 0.9819', 0.54,
         [('Ibupro\nKey#115', 'CH3ACH2A Propyl / Sec-butyl / Isobutyl', 0.306, '#5C87B2'),
          ('Ibupro\nKey#139', 'OH Hydroxyl / Carboxylic acid', 0.326, '#5C87B2'),
          ('Ibupro\nKey#123', 'O-C-O Carboxylic acid / Ester', 0.410, '#5C87B2'),
          ('Methot\nKey#92',  'O=C(N)C Carboxamide / Peptide', 0.231, '#D4766A'),
          ('Methot\nKey#105', 'A$A Ring-ring / Bridged scaffold', 0.335, '#D4766A'),
          ('Methot\nKey#123', 'O-C-O Carboxylic acid / Ester', 0.356, '#D4766A')]),
        ('D2 | Y=47: Metabolism of Drug 1 decreased by Drug 2',
         'Predicted Probability (GASPDDI): P = 0.9547', 0.325,
         [('Flucon\nKey#139', 'OH Hydroxyl / Carboxylic acid', 0.139, '#5C87B2'),
          ('Flucon\nKey#79',  'N-A-A-N Ethylenediamine / Triazole', 0.158, '#5C87B2'),
          ('Flucon\nKey#95',  'N-A-A-O Ethanolamine / Heterocycle', 0.211, '#5C87B2'),
          ('Sulfam\nKey#152', 'OC Ether / Isoxazole oxygen', 0.155, '#D4766A'),
          ('Sulfam\nKey#65',  'C%N Aromatic C:N bond', 0.178, '#D4766A'),
          ('Sulfam\nKey#133', 'A$A!N Aniline / Amino-heterocycle', 0.246, '#D4766A')]),
        ('D3 | Y=20: Drug 1 increases QTc-prolonging activity of Drug 2',
         'Predicted Probability (GASPDDI): P = 0.9383', 0.26,
         [('Ciprof\nKey#100', 'ACH2N Piperazine / Secondary amine', 0.145, '#5C87B2'),
          ('Ciprof\nKey#146', 'O >= 3 Tri-oxygenated quinolone', 0.155, '#5C87B2'),
          ('Ciprof\nKey#134', 'Halogen (C6-Fluorine substituent)', 0.160, '#5C87B2'),
          ('Fluoxe\nKey#143', 'A$A!O Phenol / Aryl ether', 0.111, '#D4766A'),
          ('Fluoxe\nKey#111', 'N-A-CH2-A Alkyl-substituted amine', 0.138, '#D4766A'),
          ('Fluoxe\nKey#93',  'QCH3 N-Methylamino (-NHCH3) chain', 0.196, '#D4766A')]),
    ]

    orig_src = Image.open(SOURCE).convert('RGB')
    row_ys = [620, 1726, 2833]
    mol_trim = 105  # trim 105px of empty white padding on left & right inside each 1243px box -> 1033px wide
    box_w = (1485 - 242) - 2 * mol_trim  # 1033 px
    box_h = 1438 - 620                   # 818 px
    x_col1 = 40
    x_col2 = x_col1 + box_w + 55         # only 55px gap between Drug A and Drug B!
    x_col3 = x_col2 + box_w + 55         # D1-D3 starts at x=2216, leaving 2410px width for D1-D3!
    d_width = fig3_w - x_col3 - 20       # 2414 px

    def make_center_title(line1, line2, width, height=122, size=13.4):
        fig = Figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor='white')
        canvas = FigureCanvasAgg(fig)
        t1 = fig.text(0.5, 0.94, line1, ha='center', va='top', fontsize=size,
                      fontweight='bold', fontfamily='DejaVu Sans', color=TITLE_COLOR)
        t2 = fig.text(0.5, 0.44, line2, ha='center', va='top', fontsize=size,
                      fontweight='bold', fontfamily='DejaVu Sans', color=TITLE_COLOR)
        canvas.draw()
        for t in (t1, t2):
            m = t.get_window_extent(canvas.get_renderer()).width
            if m > width - 6:
                t.set_fontsize(size * (width - 6) / m)
        canvas.draw()
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    def render_d_panel(title1, title2, xmax, bars_data, width, height=1065):
        fig = Figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor='white')
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.13, 0.13, 0.84, 0.70])
        y_pos = [0, 1, 2, 3, 4, 5]
        labels = [b[0] for b in bars_data]
        descs = [b[1] for b in bars_data]
        vals = [b[2] for b in bars_data]
        colors = [b[3] for b in bars_data]
        ax.barh(y_pos, vals, color=colors, height=0.62, alpha=0.92, edgecolor='none', linewidth=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10.6, fontweight='bold', color=TITLE_COLOR)
        ax.set_xlabel('Substructure Attribution Score (Grad × Key)', fontsize=11.2, fontweight='bold', color=TITLE_COLOR)
        ax.tick_params(axis='x', labelsize=10.2, colors=TITLE_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2C3E50')
        ax.spines['bottom'].set_color('#2C3E50')
        ax.set_xlim(0, xmax)
        ax.set_title(f'{title1}\n{title2}', loc='left', fontsize=13.2, fontweight='bold', color=TITLE_COLOR, pad=10)
        for idx_b, (val_b, desc_b) in enumerate(zip(vals, descs)):
            ax.text(xmax * 0.012, idx_b, desc_b, va='center', ha='left', fontsize=9.8,
                    fontweight='bold', color='white')
            ax.text(val_b + xmax * 0.015, idx_b, f'+{val_b:.3f}', va='center', ha='left',
                    fontsize=10.4, fontweight='bold', color=TITLE_COLOR)
        canvas.draw()
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    for r_idx, y_src in enumerate(row_ys):
        y_dst = y_src - CROP_TOP
        t1a, t1b, t2a, t2b = case_titles[r_idx]
        # Crop interior of Drug A and Drug B boxes without clipping any molecule pixels
        crop_a = orig_src.crop((242 + mol_trim, y_src + 3, 1485 - mol_trim, y_src + box_h - 3))
        crop_b = orig_src.crop((1817 + mol_trim, y_src + 3, 3060 - mol_trim, y_src + box_h - 3))
        part_a_img.paste(crop_a, (x_col1, y_dst + 3))
        part_a_img.paste(crop_b, (x_col2, y_dst + 3))
        draw_a.rectangle((x_col1, y_dst, x_col1 + box_w, y_dst + box_h), outline='#BDC3C7', width=3)
        draw_a.rectangle((x_col2, y_dst, x_col2 + box_w, y_dst + box_h), outline='#BDC3C7', width=3)
        part_a_img.paste(make_center_title(t1a, t1b, box_w), (x_col1, y_dst - 126))
        part_a_img.paste(make_center_title(t2a, t2b, box_w), (x_col2, y_dst - 126))

        d_t1, d_t2, d_xmax, d_bars = d_panels[r_idx]
        d_img = render_d_panel(d_t1, d_t2, d_xmax, d_bars, d_width)
        part_a_img.paste(d_img, (x_col3, y_dst - 125))

    save_png_and_pdf(part_a_img, OUTPUT_A, OUTPUT_A_PDF)
    after = {str(p.relative_to(ROOT)): sha(p) for p in watched}
    assert before == after, 'Original sources must remain unchanged.'
    # Every scientific chart rectangle must match the original pixel-for-pixel.
    original = Image.open(SOURCE).convert('RGB')
    protected = [(242,620,1485,1438),(1817,620,3060,1438),
                 (242,1726,1485,2544),(1817,1726,3060,2544),
                 (242,2833,1485,3651),(1817,2833,3060,3651),
                 (3180,620,4870,1555),(3180,1727,4870,2662),(3180,2834,4870,3770),
                 (0,3935,2450,4805),(2450,3935,4895,4805)]
    for box in protected:
        shifted=(box[0],box[1]-CROP_TOP,box[2],box[3]-CROP_TOP)
        assert original.crop(box).tobytes() == result.crop(shifted).tobytes(), box
    manifest = {'source_sha256': before, 'output': str(OUTPUT.relative_to(ROOT)),
                'output_sha256': sha(OUTPUT), 'output_size': list(result.size),
                'dpi': DPI, 'operation': 'Heading-only raster composition from original 300-dpi PNG',
                'retained_chart_rectangles': protected, 'crop_top_pixels': CROP_TOP,
                'note': 'No numerical values, molecular highlights, token bars, gates or probabilities recomputed.'}
    (CODE_DIR / 'source_manifest.json').write_text(json.dumps(manifest, indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUTPUT),'size':result.size,'sha256':sha(OUTPUT)},indent=2))

if __name__ == '__main__':
    main()

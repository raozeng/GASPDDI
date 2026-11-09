"""Figure 6: remove overall heading; restore full labels without recomputing scientific panels."""
from pathlib import Path
import os,json,hashlib
CODE_DIR=Path(__file__).resolve().parent
ROOT=CODE_DIR.parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(CODE_DIR.parent/'figure1/.mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image,ImageDraw,ImageFont
OUTPUT=ROOT/'paper/figures/fig4_latent_manifold_and_relation_hierarchy_v2.png'
OUTPUT_PDF=ROOT/'paper/figures/fig4_latent_manifold_and_relation_hierarchy_v2.pdf'
OUTPUT_A_PNG=ROOT/'paper/figures/fig4a_latent_manifold_v2.png'
OUTPUT_A_PDF=ROOT/'paper/figures/fig4a_latent_manifold_v2.pdf'
OUTPUT_B_PNG=ROOT/'paper/figures/fig4b_relation_hierarchy_and_superfamily_v2.png'
OUTPUT_B_PDF=ROOT/'paper/figures/fig4b_relation_hierarchy_and_superfamily_v2.pdf'
COLOR='#1A252F'
plt.rcParams.update({'patch.edgecolor': 'none', 'patch.linewidth': 0})

def save_with_pdf(img, png_path, pdf_path):
    img.save(png_path, dpi=(300, 300))
    pdf_fig = plt.figure(figsize=(img.width / 300, img.height / 300), dpi=300, facecolor='white')
    pdf_ax = pdf_fig.add_axes([0, 0, 1, 1])
    pdf_ax.axis('off')
    pdf_ax.imshow(img, interpolation='none')
    pdf_fig.savefig(pdf_path, format='pdf', dpi=300, facecolor='white')
    plt.close(pdf_fig)

def font(size,bold=False):
    path=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold' if bold else 'normal'))
    return ImageFont.truetype(path,size)

def fit(draw,text,width,size,bold=False):
    f=font(size,bold)
    while draw.textlength(text,font=f)>width:
        size-=1;f=font(size,bold)
    return f

def main():
    mp=CODE_DIR/'source_manifest.json'
    m=json.loads(mp.read_text(encoding='utf-8'))
    for rel,sha in m['sources'].items():
        assert hashlib.sha256((ROOT/rel.replace('\\','/')).read_bytes()).hexdigest()==sha,rel
    canvas=Image.new('RGB',(6000,3040),'white')
    top=Image.open(CODE_DIR/'assets/manifold_panels.png').convert('RGB')
    # Change only the Y60 legend text band, keeping its purple marker and other legend rows.
    d=ImageDraw.Draw(top)
    d.rectangle((4474,345,5010,388),fill='white')
    text='Pos Y60: Increased Hypotensive Activity'
    d.text((4478,362),text,font=fit(d,text,532,30),fill=COLOR,anchor='lm')
    # Enlarge subplot titles in the top title band
    d.rectangle((600,0,5055,130),fill='white')
    a_titles=[
        (1352,'A | Warm-Start S1 (K-K) Latent Manifold','(Pos/Neg Silhouette = 0.121)'),
        (2878,'B | Single-Cold S2 (K-U) Latent Manifold','(Pos/Neg Silhouette = 0.092)'),
        (4403,'C | Dual-Cold S3 (U-U) Latent Manifold','(Pos/Neg Silhouette = 0.049)'),
    ]
    for cx,t_line1,t_line2 in a_titles:
        d.text((cx,6),t_line1,font=fit(d,t_line1,1340,58,True),fill=COLOR,anchor='mt')
        d.text((cx,72),t_line2,font=fit(d,t_line2,1340,54,True),fill=COLOR,anchor='mt')
    tx=471;canvas.paste(top,(tx,0))

    # Standalone Figure 6 (Panels A, B, C: t-SNE Latent Manifold) with extra top and side room for large 58px titles
    fig_a_plot = top.crop((640, 130, 5058, 1300))
    fig_a_canvas = Image.new('RGB', (fig_a_plot.width + 120, fig_a_plot.height + 175), 'white')
    fig_a_canvas.paste(fig_a_plot, (60, 160))
    da = ImageDraw.Draw(fig_a_canvas)
    for cx, t_line1, t_line2 in a_titles:
        cx_a = cx - 640 + 60
        da.text((cx_a, 12), t_line1, font=fit(da, t_line1, 1360, 58, True), fill=COLOR, anchor='mt')
        da.text((cx_a, 82), t_line2, font=fit(da, t_line2, 1360, 54, True), fill=COLOR, anchor='mt')
    save_with_pdf(fig_a_canvas, OUTPUT_A_PNG, OUTPUT_A_PDF)

    heat=Image.open(CODE_DIR/'assets/classifier_heatmap_body.png').convert('RGB')
    bx,by=1450,1600
    canvas.paste(heat,(bx-87,by-12))

    # Render Super-Family Panel with borderless, lower-saturation bars from table4_superfamily_performance.csv
    import csv, io, numpy as np
    with (ROOT/'paper/tables/table4_superfamily_performance.csv').open(encoding='utf-8') as f:
        sf_rows = list(csv.DictReader(f))
    sf_map = {(r['Task'], r['Superfamily']): float(r['AUC']) for r in sf_rows}
    sf_order = [
        'PK: Hepatic Metabolism (CYP450)',
        'PK: Absorption / Serum / Excretion',
        'PD: Cardiovascular & QTc',
        'PD: Coagulation & Hematologic',
        'PD: CNS & Neurologic',
        'PD: Systemic Adverse / Other',
    ]
    c_fig = plt.figure(figsize=(2398 / 300, 1515 / 300), dpi=300, facecolor='white')
    ax_sf = c_fig.add_axes([0.335, 0.10, 0.645, 0.76])
    y_sf = np.arange(len(sf_order))
    h_sf = 0.25
    for idx_t, (t_label, col_t, leg_lbl) in enumerate([
        ('Task 1', '#5C87B2', 'S1 (Warm K-K)'),
        ('Task 2', '#E2A46D', 'S2 (Cold-1 K-U)'),
        ('Task 3', '#D4766A', 'S3 (Cold-2 U-U)'),
    ]):
        vals = [sf_map[(t_label, sf)] for sf in reversed(sf_order)]
        ax_sf.barh(y_sf + (1 - idx_t) * h_sf, vals, height=h_sf * 0.94, color=col_t, alpha=0.94,
                   edgecolor='none', linewidth=0, label=leg_lbl)
    ax_sf.set_yticks(y_sf)
    ax_sf.set_yticklabels(list(reversed(sf_order)), fontsize=9.2, fontweight='bold', color='#1A252F')
    ax_sf.set_xlabel('Test ROC-AUC Score', fontsize=10.8, fontweight='bold', color='#1A252F')
    ax_sf.tick_params(axis='x', labelsize=9.6, colors='#2C3E50')
    ax_sf.set_xlim(0.50, 1.03)
    ax_sf.spines['top'].set_visible(False)
    ax_sf.spines['right'].set_visible(False)
    ax_sf.spines['left'].set_color('#2C3E50')
    ax_sf.spines['bottom'].set_color('#2C3E50')
    ax_sf.legend(loc='lower right', fontsize=9.0, frameon=True)
    buf_c = io.BytesIO()
    c_fig.savefig(buf_c, format='png', dpi=300, facecolor='white')
    plt.close(c_fig)
    buf_c.seek(0)
    family = Image.open(buf_c).convert('RGB')
    canvas.paste(family,(3510,1463))
    draw=ImageDraw.Draw(canvas)
    c_line1='B | Generalization Across 6 Super-Families'
    c_line2='(PD Receptor Synergy vs. PK CYP450 Metabolism)'
    draw.text((4960,1436),c_line1,font=fit(draw,c_line1,2350,68,True),fill=COLOR,anchor='mt')
    draw.text((4960,1512),c_line2,font=fit(draw,c_line2,2350,60,True),fill=COLOR,anchor='mt')
    titles=['A | Pharmacological Hierarchy in Classifier Space',
            '(Ward Clustering of 20 Major DrugBank Relations)']
    draw.text((2150,1436),titles[0],font=fit(draw,titles[0],2600,68,True),fill=COLOR,anchor='mt')
    draw.text((2150,1512),titles[1],font=fit(draw,titles[1],2600,60,True),fill=COLOR,anchor='mt')
    data=json.loads((CODE_DIR/'data/relation_labels.json').read_text(encoding='utf-8'))
    displayed=[]
    for i,rel in enumerate(data['order']):
        label=data['labels'][str(rel)]
        f=fit(draw,label,bx-70,33)
        assert draw.textlength(label,font=f)<=bx-70
        draw.text((bx-24,by+(i+.5)*61.1),label,font=f,fill='#2C3E50',anchor='rm')
        displayed.append(label)
    # Protected scientific regions, excluding annotation text, must remain identical.
    assert canvas.crop((bx,by-12,bx+heat.width-87,by-12+heat.height)).tobytes()==heat.crop((87,0,heat.width,heat.height)).tobytes()

    # Standalone Figure 7 (Panels A and B: Ward Hierarchy & 6 Super-Families) with balanced margins
    fig_b_crop = canvas.crop((540, 1425, 5955, 2975))
    fig_b_canvas = Image.new('RGB', (fig_b_crop.width + 60, fig_b_crop.height + 25), 'white')
    fig_b_canvas.paste(fig_b_crop, (30, 15))
    save_with_pdf(fig_b_canvas, OUTPUT_B_PNG, OUTPUT_B_PDF)

    save_with_pdf(canvas, OUTPUT, OUTPUT_PDF)
    for rel,sha in m['sources'].items():
        assert hashlib.sha256((ROOT/rel.replace('\\','/')).read_bytes()).hexdigest()==sha,rel
    m.update({'output':str(OUTPUT.relative_to(ROOT)),'output_size':canvas.size,'dpi':300,
       'output_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
       'changes':['Remove overall title','Restore full 20 B relation labels',
                  'A3 Y60: Increased Hypotensive Activity','B Y06: Increased Anticoagulant Activity'],
       'displayed_B_labels':displayed,
       'preserved':'Original plotted t-SNE panels, silhouette values, B heatmap/order/colorbar and C bars; CYP and Excretion/PK labels retained.'})
    mp.write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'output':str(OUTPUT),'size':canvas.size,'sha256':m['output_sha256']},indent=2))

if __name__=='__main__':
    main()

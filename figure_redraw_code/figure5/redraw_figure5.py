"""Figure 5 v2: full labels, no overall title, corrected Morgan score/radius headings.
A: original CSV values; B: unchanged plotted heatmap; C: unchanged original molecular images.
"""
from pathlib import Path
import os, io, csv, json, textwrap, hashlib
CODE_DIR=Path(__file__).resolve().parent
ROOT=CODE_DIR.parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(CODE_DIR.parent/'figure1/.mpl_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
DPI=300
OUTPUT=ROOT/'paper/figures/fig2_fingerprint_bit_attribution_v2.png'
OUTPUT_PDF=ROOT/'paper/figures/fig2_fingerprint_bit_attribution_v2.pdf'
COLOR='#1A252F'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.edgecolor':'#2C3E50',
 'axes.linewidth':.9,'xtick.color':'#2C3E50','ytick.color':'#2C3E50',
 'text.color':COLOR,'figure.facecolor':'white','axes.facecolor':'white',
 'patch.edgecolor':'none','patch.linewidth':0})

def font(size,bold=False):
    path=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold' if bold else 'normal'))
    return ImageFont.truetype(path,size)

def draw_fitted_line(draw,xy,text,width,size,bold=True):
    f=font(size,bold)
    while draw.textlength(text,font=f)>width:
        size-=1; f=font(size,bold)
    draw.text(xy,text,font=f,fill=COLOR,anchor='lt')

def draw():
    manifest_path=CODE_DIR/'source_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for rel,sha in manifest['sources'].items():
        assert hashlib.sha256((ROOT/rel.replace('\\','/')).read_bytes()).hexdigest()==sha,rel
    with (CODE_DIR/'data/table2_fingerprint_bit_importance.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    keys=[r for r in rows if r['Feature_Type']=='MACCS_Key'][::-1]
    canvas=Image.new('RGB',(7340,4000),'white')
    # More vertical room per key and a wider left margin allow complete two-line labels.
    fig=plt.figure(figsize=(14.5,7.0),dpi=DPI)
    ax=fig.add_axes([.35,.09,.63,.83])
    y=np.arange(len(keys));h=.26
    for offset,col,color,label in [
       (h,'Importance_Task1_Warm','#5C87B2','S1 (Warm Start K-K)'),
       (0,'Importance_Task2_ColdS1','#E2A46D','S2 (Cold Start K-U)'),
       (-h,'Importance_Task3_ColdS2','#D4766A','S3 (Cold Start U-U)')]:
        ax.barh(y+offset,[float(r[col]) for r in keys],height=h,color=color,alpha=.94,edgecolor='none',linewidth=0,label=label)
    labels=[]
    for row in keys:
        label=f"MACCS#{int(row['Bit_ID'].split('_')[1])}: {row['Description']}"
        wrapped=textwrap.fill(label,width=55,break_long_words=False,break_on_hyphens=False)
        assert ' '.join(wrapped.split())==' '.join(label.split())
        labels.append(wrapped)
    ax.set_yticks(y);ax.set_yticklabels(labels,fontsize=9.0)
    ax.tick_params(axis='y',pad=9)
    ax.set_xlabel('Mean Absolute Attribution Score |Grad × Bit|',fontsize=10.5,fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower right',fontsize=9.2,frameon=True)
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=DPI);plt.close(fig);buf.seek(0)
    canvas.paste(Image.open(buf).convert('RGB'),(0,0))
    painter=ImageDraw.Draw(canvas)
    draw_fitted_line(painter,(400,45),'A | Top 16 MACCS Keys Driving DDI Predictions',3850,82)
    # B matrix/annotations/colorbar/x labels are retained as original pixels.
    bx,by=1522,2250
    heat=Image.open(CODE_DIR/'assets/relation_heatmap_body.png').convert('RGB')
    canvas.paste(heat,(bx-223,by-20))
    draw_fitted_line(painter,(400,2120),'B | Substructure Signatures Across Top 10 DDI Classes',3850,82)
    relation_labels=json.loads((CODE_DIR/'data/relation_labels.json').read_text(encoding='utf-8'))
    for i,rel in enumerate([49,47,73,75,60,70,20,16,4,6]):
        label=relation_labels[str(rel)]
        wrapped=textwrap.fill(label,width=52,break_long_words=False,break_on_hyphens=False)
        assert ' '.join(wrapped.split())==' '.join(label.split())
        painter.multiline_text((bx-30,by+(i+.5)*140),wrapped,font=font(36),fill='#2C3E50',
                               anchor='rm',align='right',spacing=5)
    # C cards reuse original molecules/highlights; resize uniformly for a larger display.
    morgan=[r for r in rows if r['Feature_Type']=='Morgan_ECFP4']
    motifs=['ccc (Aryl CH)','CC(N)=O (Amide)','C(=O)O / Cyc-CH2','c(O)cC(=O)O (Aryl)','ccc (Phenyl CH)','CCC(O) (Carbinol)']
    prev=[66.1,83.1,76.4,87.7,43.7,73.2]
    for i,row in enumerate(morgan):
        x=4350+(i%2)*1460;y0=45+(i//2)*1370
        bit=int(row['Bit_ID'].split('_')[1]);score=float(row['Mean_Importance'])
        draw_fitted_line(painter,(x,y0),f'C{i+1} | Morgan ECFP4 Bit #{bit} (r≤2)',1420,66)
        draw_fitted_line(painter,(x,y0+74),f'Motif: {motifs[i]} | Score: {score:.4f} (Prev: {prev[i]:.1f}%)',1420,52)
        molecule=Image.open(CODE_DIR/'assets'/f'morgan_C{i+1}.png').convert('RGB')
        # Uniform scaling preserves structure proportions and the original highlight extent.
        molecule=molecule.resize((1400,round(molecule.height*1400/molecule.width)),Image.Resampling.LANCZOS)
        mx=x;my=y0+142
        canvas.paste(molecule,(mx,my))
        painter.rectangle((mx-1,my-1,mx+molecule.width,my+molecule.height),outline='#95A5A6',width=3)
    canvas.save(OUTPUT,dpi=(DPI,DPI))
    pdf_fig = plt.figure(figsize=(canvas.width / DPI, canvas.height / DPI), dpi=DPI, facecolor='white')
    pdf_ax = pdf_fig.add_axes([0, 0, 1, 1])
    pdf_ax.axis('off')
    pdf_ax.imshow(canvas, interpolation='none')
    pdf_fig.savefig(OUTPUT_PDF, format='pdf', dpi=DPI, facecolor='white')
    plt.close(pdf_fig)
    assert canvas.crop((bx,by-20,bx+heat.width-223,by-20+heat.height)).tobytes()==heat.crop((223,0,heat.width,heat.height)).tobytes()
    for rel,sha in manifest['sources'].items():
        assert hashlib.sha256((ROOT/rel.replace('\\','/')).read_bytes()).hexdigest()==sha,rel
    manifest.update({'output':str(OUTPUT.relative_to(ROOT)),'output_size':canvas.size,
       'output_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),'dpi':DPI,
       'changes':['Remove overall title','Show all 16 full MACCS descriptions with wrapping',
                  'Retain full B relation labels','C1–C6 Mean |IG| -> Mean |Grad × Bit|',
                  'C1–C6 radius headings -> r≤2','A/B titles on one line; C titles on two lines',
                  'Reduce vertical gaps and enlarge molecular cards uniformly'],
       'preserved':'Original CSV values in A, original heatmap body pixels in B, original molecular structures/highlights in C with uniform display scaling'})
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'output':str(OUTPUT),'size':canvas.size,'sha256':manifest['output_sha256']},indent=2))

if __name__=='__main__':
    draw()

"""Figure 2 v2: standalone plotting only; original files are never overwritten."""
from pathlib import Path
import os
CODE_DIR=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(CODE_DIR.parent/'figure1/.mpl_cache'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
ROOT=CODE_DIR.parents[1]
OUTPUT=ROOT/'paper/figures/fig5_why_3d_distillation_fails_diagnosis_v2.png'
OUTPUT_PDF=ROOT/'paper/figures/fig5_why_3d_distillation_fails_diagnosis_v2.pdf'
OUTPUT_LEGACY=ROOT/'paper/figures/fig5_why_3d_distillation_fails_diagnosis.png'
OUTPUT_LEGACY_PDF=ROOT/'paper/figures/fig5_why_3d_distillation_fails_diagnosis.pdf'
DATA=json.loads((CODE_DIR/'plot_data.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'patch.edgecolor':'none','patch.linewidth':0})

def draw():
    fig=plt.figure(figsize=(18.5,6.0),dpi=300)
    gs=gridspec.GridSpec(1,3,figure=fig,width_ratios=[1.12,1.08,1.15],wspace=.25)
    ax=fig.add_subplot(gs[0,0])
    for task,align_color,ddi_color,style in [('1','#C0392B','#2980B9','-'),('2','#E67E22','#16A085','--')]:
        d=DATA['loss'][task]
        suffix=r' ($\lambda=1.0$)' if task=='1' else ''
        focal_suffix=r' ($\alpha=0.25,\gamma=2$)' if task=='1' else ''
        ax.plot(d['Epoch'],d['Align_Loss'],color=align_color,lw=2 if task=='1' else 1.8,ls=style,label=f'Task {task} InfoNCE '+r'$\mathcal{L}_{\mathrm{align}}$'+suffix)
        ax.plot(d['Epoch'],d['DDI_Loss'],color=ddi_color,lw=2 if task=='1' else 1.8,ls=style,label=f'Task {task} Focal '+r'$\mathcal{L}_{\mathrm{DDI}}$'+focal_suffix)
    d=DATA['loss']['1']
    ax.fill_between(d['Epoch'],d['DDI_Loss'],d['Align_Loss'],facecolor='#9B72CF',alpha=.48,zorder=1)
    ax.set_yscale('log')
    ax.set_xlabel('Training Epoch',fontsize=10.2,fontweight='bold')
    ax.set_ylabel('Loss Magnitude (Log Scale)',fontsize=10.2,fontweight='bold')
    ax.set_title('A | Alignment and DDI Loss Trajectories',fontsize=13.2,fontweight='bold',pad=10)
    handles,labels=ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=to_rgba('#9B72CF',.48),edgecolor='#6E3FA5',linewidth=1.0,label='Difference Between Alignment and DDI Losses'))
    ax.legend(handles=handles,loc='upper right',fontsize=7.6,frameon=True)

    ax=fig.add_subplot(gs[0,1])
    for key,color,label,style in [('n3d','#1E8449','No-3D Fingerprint Representation','-'),('distilled','#2874A6','3D-Distilled 1D Mamba','-'),('teacher','#C0392B','Static Uni-Mol 3D Teacher','--')]:
        d=DATA['spectra'][key]
        detail=rf'($R_{{\mathrm{{eff}}}}={d["effective_rank"]:.1f}$, DDI-Jaccard={d["jaccard"]:.3f})'
        ax.plot(d['k'],d['sigma_ratio'],color=color,lw=2.2 if key!='teacher' else 2.0,ls=style,label=label+'\n'+detail)
    ax.set_yscale('log')
    ax.set_xlabel(r'Singular Value Index $k$ (Top 128 Dimensions)',fontsize=10.2,fontweight='bold')
    ax.set_ylabel(r'Normalized Singular Value $\sigma_k/\sigma_1$ (Log)',fontsize=10.2,fontweight='bold')
    ax.set_title('B | Spectral Dimensionality & DDI Neighborhood Alignment\n(1,607 Valid 3D Drugs; 99 Failed Conformer Generation)',fontsize=12.6,fontweight='bold',pad=10)
    ax.legend(loc='lower left',fontsize=7.5,frameon=True)

    ax=fig.add_subplot(gs[0,2]);x=np.arange(3);width=.25
    for offset,mode,color,label in [(-1,'uni','#99A7B5',r'With Uni-Mol 3D ($\lambda=1.0,\tau=0.5$)'),(0,'n3d','#5C87B2',r'No-3D Ablation ($\lambda=0.0,\tau=0.5$)'),(1,'dynamic','#4E9A84',r'No-3D + Validation Dynamic $\tau^*$')]:
        vals=[DATA['f1'][str(t)][mode] for t in (1,2,3)]
        means=np.array([np.mean(a) for a in vals]);sd=np.array([np.std(a,ddof=1) for a in vals])
        bars=ax.bar(x+offset*width,means,yerr=sd,width=width*0.92,capsize=4,color=color,edgecolor='none',linewidth=0,label=label)
        for bar,err in zip(bars,sd):
            h=bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2,h+err+.015,f'{h:.3f}',ha='center',va='bottom',fontsize=7.5,fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Task 1\n(Warm K-K)','Task 2\n(Cold S1 K-U)','Task 3\n(Cold S2 U-U)'],fontsize=10.2,fontweight='bold')
    ax.set_ylabel('5-Fold Test F1 Score (Mean ± SD)',fontsize=10.2,fontweight='bold')
    ax.set_ylim(.25,1.08)
    ax.set_title('C | 5-Fold Cross-Validation Performance Gain from\nRemoving Rigid 3D Distillation & Calibrating Threshold',fontsize=12.6,fontweight='bold',pad=10)
    ax.legend(loc='upper right',fontsize=7.8,frameon=True)
    # No global suptitle: the manuscript supplies the figure caption.
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUTPUT,dpi=300,bbox_inches='tight',pad_inches=.1,facecolor='white')
    fig.savefig(OUTPUT_PDF,bbox_inches='tight',pad_inches=.1,facecolor='white')
    fig.savefig(OUTPUT_LEGACY,dpi=300,bbox_inches='tight',pad_inches=.1,facecolor='white')
    fig.savefig(OUTPUT_LEGACY_PDF,bbox_inches='tight',pad_inches=.1,facecolor='white')
    plt.close(fig)
    print(f'Generated: {OUTPUT}')
    print(f'Generated: {OUTPUT_PDF}')
if __name__=='__main__':draw()

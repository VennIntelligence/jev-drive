"""Shared paper plot style: CVPR widths, STIX serif, Okabe-Ito, vector PDF/300dpi PNG."""
from pathlib import Path
import matplotlib as mpl

SINGLE_COLUMN_IN=3.25
DOUBLE_COLUMN_IN=6.875
BASELINE='#777777'
PALETTE={'orange':'#E69F00','sky_blue':'#56B4E9','green':'#009E73','yellow':'#F0E442','blue':'#0072B2','vermillion':'#D55E00','purple':'#CC79A7','black':'#000000'}
PREDICTION=PALETTE['blue']


def apply():
    mpl.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'font.size':8.5,
        'mathtext.fontset':'stix','axes.labelsize':8.5,'axes.titlesize':8.5,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'text.color':'#222222','axes.labelcolor':'#222222','axes.edgecolor':'#444444',
        'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'axes.axisbelow':True,'grid.color':'#BBBBBB','grid.alpha':.28,'grid.linewidth':.4,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.width':.5,'ytick.major.width':.5,
        'lines.linewidth':1.05,'legend.frameon':False,'legend.handlelength':2.2,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300,'figure.dpi':100,'savefig.facecolor':'white'})


def panel(ax, label):
    ax.text(0,1.025,label,transform=ax.transAxes,ha='left',va='bottom')


def zero_line(ax):
    ax.axhline(0,color='#999999',linewidth=.5,zorder=0)


def bars(ax):
    ax.grid(axis='x',visible=False)


def save(fig,stem):
    """Preserve requested physical dimensions; PNG compression is lossless."""
    stem=Path(stem)
    fig.savefig(str(stem)+'.pdf',bbox_inches=None)
    fig.savefig(str(stem)+'.png',dpi=300,bbox_inches=None,pil_kwargs={'optimize':True,'compress_level':9})
    return {'width_in':float(fig.get_figwidth()),'height_in':float(fig.get_figheight()),
            'png_dpi':300,'png_bytes':Path(str(stem)+'.png').stat().st_size,
            'pdf_bytes':Path(str(stem)+'.pdf').stat().st_size,'png_under_500KiB':Path(str(stem)+'.png').stat().st_size<500*1024}

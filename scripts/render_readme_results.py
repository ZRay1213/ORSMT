#!/usr/bin/env python3
"""Render homepage figures from checked-in paper tables and a released case trace.

Optional plotting dependencies: matplotlib and numpy.
Run from any directory: python scripts/render_readme_results.py
"""
from pathlib import Path
import json
import re
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'orsmt-mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'docs/paper-results/source_tables'
OUT = ROOT / 'assets/results'
OUT.mkdir(parents=True, exist_ok=True)
INK, TEAL, BLUE, ORANGE, GREY = '#17313A', '#007F75', '#0072B2', '#D55E00', '#98A5AE'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def clean(line):
    return re.sub(r'\\textbf\{([^{}]*)\}', r'\1', line).strip().rstrip('\\').strip()


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight')
    svg_path = OUT / (name + '.svg')
    fig.savefig(svg_path, bbox_inches='tight')
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    plt.close(fig)


def style(ax, top=100):
    ax.set_ylim(0, top)
    ax.set_ylabel('Outputs (%)')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#E3EAED', linewidth=0.7)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#C3CDD2')
    ax.tick_params(length=0)


full = []
for raw in (TABLES / 'comparison_compact.tex').read_text().splitlines():
    line = clean(raw)
    if line.split(' & ')[0] in ['Original', 'ReLoop', 'Opt-Verifier', 'ORSMT']:
        vals = re.findall(r'([\d,]+) \(([\d.]+)\)', line)
        full.append({'method': line.split(' & ')[0], 'correct_answer': float(vals[0][1]),
            'model_code_match': float(vals[1][1]), 'correct_and_matched': float(vals[2][1])})
assert len(full) == 4
fig, axs = plt.subplots(1, 2, figsize=(12, 4.1))
colors = [GREY, BLUE, '#829BB5', TEAL]
for ax, metric, title in zip(axs, ['model_code_match', 'correct_and_matched'],
        ['A  Model–code agreement', 'B  Correct answer + model–code agreement']):
    values = [r[metric] for r in full]
    bars = ax.bar(range(4), values, color=colors, width=0.62)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, val+2, f'{val:.1f}%', ha='center', fontweight='bold')
    ax.set_xticks(range(4), [r['method'] for r in full])
    style(ax)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=12, pad=16)
fig.tight_layout(w_pad=3)
save(fig, 'main-results')

controlled = []
for raw in (TABLES / 'ablation_compact.tex').read_text().splitlines():
    line = clean(raw)
    if '&' in line:
        label, val = line.split('&', 1)
        if re.fullmatch(r'\d+\.\d+', val.strip()):
            controlled.append({'method': label.strip(), 'correct_and_matched': float(val.strip())})
assert len(controlled) == 6
fig, ax = plt.subplots(figsize=(10.8, 4.2))
y = np.arange(6)
ax.barh(y, [r['correct_and_matched'] for r in controlled], color=[BLUE, '#829BB5', '#8ABDB7', '#62ABA2', '#33998E', TEAL], height=.62)
ax.set_yticks(y, [r['method'] for r in controlled]);ax.invert_yaxis();ax.set_xlim(0,60)
for yi, row in zip(y, controlled):
    ax.text(row['correct_and_matched']+0.7, yi, f"{row['correct_and_matched']:.1f}%", va='center', fontweight='bold')
ax.set_xlabel('Correct and matched outputs (%)')
ax.set_title('Same refuted cohort, same 14B reviser, one revision round', loc='left', fontweight='bold', pad=18)
ax.set_axisbelow(True);ax.xaxis.grid(True, color='#E3EAED');ax.spines['left'].set_visible(False);ax.tick_params(length=0)
fig.tight_layout();save(fig, 'controlled-repair')

hosts = [];system = None
for raw in (TABLES / 'plugin_compact.tex').read_text().splitlines():
    line = clean(raw)
    if line.startswith('OptiMUS-0.3') or line.startswith('OptiMind'):
        system = line.split('&')[0].strip()
    if system and ('Original &' in line or 'Original + ORSMT &' in line):
        values = re.findall(r'([\d,]+) \(([\d.]+)\)',line)
        hosts.append({'system':system, 'pipeline':line.split('&')[1].strip(),
          'correct_answer':float(values[0][1]), 'artifact_match':float(values[1][1]),
          'correct_and_matched':float(values[2][1])})
assert len(hosts)==4
fig, axs = plt.subplots(1,2,figsize=(11.5,4.1))
for ax, name in zip(axs,['OptiMUS-0.3','OptiMind']):
    rows=[r for r in hosts if r['system']==name];x=np.arange(2)
    for i, row in enumerate(rows):
        vals=[row['artifact_match'],row['correct_and_matched']]
        bars=ax.bar(x+(i-.5)*.34,vals,width=.31,color=[GREY,TEAL][i],label=row['pipeline'])
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}',ha='center',fontsize=10,fontweight='bold')
    ax.set_xticks(x,['Artifact match','Correct and matched']);style(ax,112)
    ax.set_yticks([0,25,50,75,100]);ax.set_title(name,loc='left',fontweight='bold',pad=15)
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc='lower center', ncol=2, bbox_to_anchor=(.5,0))
fig.tight_layout(w_pad=3, rect=(0,.08,1,1));save(fig,'host-integration')

trace=json.loads((ROOT/'data/worked_trace/nl4opt67/summary.json').read_text())
assert trace['worked_witness']=={'new_trips':15,'old_trips':10}
assert trace['before_pair']=='REFUTED' and trace['after_pair']=='CERTIFIED'
fig,axs=plt.subplots(1,3,figsize=(13.3,4.3))
x=np.linspace(0,15,500);capacity=(1000-50*x)/70
for i,(ax,title) in enumerate(zip(axs,['A  Original written model','B  Captured solver model','C  Repaired written model'])):
    bound=x+.4 if i==0 else 2*x/3
    floor=np.maximum(capacity,bound)
    ax.fill_between(x,floor,20,color=BLUE if i==0 else TEAL,alpha=.13)
    ax.plot(x,capacity,color=GREY,lw=1.4,label='Delivery capacity')
    ax.plot(x,bound,color=BLUE if i==0 else TEAL,lw=2)
    points=[(u,v) for u in range(16) for v in range(21) if 50*u+70*v>=1000 and (v>=u+.4 if i==0 else 3*v>=2*u)]
    px,py=zip(*points);ax.scatter(px,py,s=8,color=BLUE if i==0 else TEAL,alpha=.5)
    ax.scatter([2],[13],s=150,marker='*',color=INK,zorder=4)
    ax.annotate('Reported optimum\n(2, 13), value 580',(2,13),xytext=(1,6.2),fontsize=9,arrowprops={'arrowstyle':'-','color':INK})
    ax.scatter([15],[10],s=110,marker='X',color=ORANGE,zorder=5,clip_on=False)
    ax.annotate('Witness (15, 10)\n'+('excluded' if i==0 else 'admitted'),(15,10),xytext=(7.2,2.3),fontsize=9,color=ORANGE,arrowprops={'arrowstyle':'->','color':ORANGE})
    ax.text(.04,.94,('y ≥ x + 0.4' if i==0 else '3y ≥ 2x'),transform=ax.transAxes,fontsize=12,fontweight='bold',va='top')
    ax.set(xlim=(-.5,15.6),ylim=(0,20.4),xlabel='Trips by new company (x)',ylabel='Trips by old company (y)')
    ax.set_title(title,loc='left',fontweight='bold',fontsize=11,pad=15)
    ax.set_xticks([0,5,10,15]);ax.set_yticks([0,5,10,15,20]);ax.grid(alpha=.12);ax.set_axisbelow(True)
fig.tight_layout(w_pad=2);save(fig,'shipping-counterexample')

fig,axs=plt.subplots(1,3,figsize=(12.4,4))
for i,(ax,title) in enumerate(zip(axs,['A  Written: x is integer','B  Code: x is binary','C  Repaired: x is integer'])):
    polygon=[(0,0),(1,0),(1,1),(0,2)] if i==1 else [(0,0),(2,0),(0,2)]
    ax.add_patch(plt.Polygon(polygon,color=TEAL,alpha=.12))
    pts=[(x,y) for x in range(3) for y in range(3) if x+y<=2 and (x<=1 if i==1 else True)]
    ax.scatter(*zip(*pts),s=65,color=TEAL)
    ax.scatter([0],[2],s=170,marker='*',color=INK,zorder=4)
    ax.scatter([2],[0],s=120,marker='X',color=ORANGE,zorder=5)
    ax.annotate('(2, 0) '+('excluded' if i==1 else 'admitted'),(2,0),xytext=(.75,.65),fontsize=10,color=ORANGE,arrowprops={'arrowstyle':'->','color':ORANGE})
    ax.text(.06,.87,'Maximize x + y\nx + y ≤ 2',transform=ax.transAxes,fontsize=11)
    ax.set(xlim=(-.3,2.4),ylim=(-.3,2.6),xlabel='x',ylabel='y',aspect='equal')
    ax.set_title(title,loc='left',fontweight='bold',fontsize=11,pad=14)
    ax.set_xticks([0,1,2]);ax.set_yticks([0,1,2]);ax.grid(alpha=.15);ax.set_axisbelow(True)
fig.tight_layout();save(fig,'domain-counterexample')

payload={'full_population':full,'controlled_repair':controlled,'host_integration':hosts,
  'sources':{'full_population':'comparison_compact.tex','controlled_repair':'ablation_compact.tex','host_integration':'plugin_compact.tex'},
  'uncertainty':'Aggregate rates transcribed from paper tables; no uncertainty estimates or significance markers inferred.'}
(ROOT/'docs/paper-results/values.json').write_text(json.dumps(payload,indent=2)+'\n')
print('Rendered five figures in PNG and SVG from paper tables and case records.')

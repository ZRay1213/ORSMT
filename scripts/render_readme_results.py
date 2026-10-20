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
    'svg.fonttype': 'none', 'svg.hashsalt': 'orsmt-readme', 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def clean(line):
    return re.sub(r'\\textbf\{([^{}]*)\}', r'\1', line).strip().rstrip('\\').strip()


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight')
    svg_path = OUT / (name + '.svg')
    fig.savefig(svg_path, bbox_inches='tight', metadata={'Date': None})
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
ax.set_title('Repair policies on the same refuted cohort', loc='left', fontweight='bold', pad=18)
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

from fractions import Fraction
trace_dir = ROOT / 'data/worked_trace/nl4opt67'
trace = json.loads((trace_dir / 'summary.json').read_text())
written = json.loads((trace_dir / '01_input.json').read_text())['formulation']
runtime = json.loads((trace_dir / '02_native_capture.json').read_text())['ir']
repaired = json.loads((trace_dir / '07_proposed_revision.json').read_text())['formulation']

def admits(model, point):
    for var in model['variables']:
        v = point[var['name']]
        if var.get('lb') is not None and v < Fraction(str(var['lb'])): return False
        if var.get('ub') is not None and v > Fraction(str(var['ub'])): return False
        if var['type'] in ('int', 'integer', 'binary') and v.denominator != 1: return False
    for row in model['constraints']:
        lhs = sum(Fraction(str(t['coef'])) * point[t['var']] for t in row['lhs'])
        rhs = Fraction(str(row['rhs']))
        if not {'>=': lhs >= rhs, '<=': lhs <= rhs, '==': lhs == rhs}[row['op']]: return False
    return True

# The y limit is a display window, not a model constraint.
points = {(x,y): {'new_trips':Fraction(x), 'old_trips':Fraction(y)}
          for x in range(16) for y in range(21)}
sets = [{xy for xy, point in points.items() if admits(model, point)}
        for model in (written, runtime, repaired)]
assert sets[0] <= sets[1] and sets[1] == sets[2]
witness = (trace['worked_witness']['new_trips'], trace['worked_witness']['old_trips'])
assert witness in sets[1] - sets[0]
optimal = tuple(int(trace['native_returned_result']['assignment'][k]) for k in ['new_trips','old_trips'])
assert all(optimal in allowed for allowed in sets)
fig, axs = plt.subplots(1,2,figsize=(11.4,5.1))
for i, ax in enumerate(axs):
    if i==0:
        groups=[(sets[0], GREY, 'o', True), (sets[1]-sets[0], ORANGE, 'o', False)]
    else:
        groups=[(sets[2], TEAL, 'o', True)]
    for pts,color,marker,filled in groups:
        px,py=zip(*sorted(pts))
        ax.scatter(px,py,s=18,marker=marker,facecolors=color if filled else 'none',edgecolors=color,linewidths=.8)
    ax.scatter(*witness,s=100,marker='X',color=ORANGE,zorder=5)
    ax.scatter(*optimal,s=150,marker='*',color=INK,zorder=6)
    ax.annotate('Separating point (15, 10)',witness,xytext=(5.2,1.7),fontsize=9,color=ORANGE,arrowprops={'arrowstyle':'->','color':ORANGE})
    ax.annotate('Recorded optimum\n(2, 13), value 580',optimal,xytext=(.7,5.5),fontsize=9,arrowprops={'arrowstyle':'->','color':INK})
    ax.set(xlim=(-.5,15.7),ylim=(0,20.5),xlabel='New-company trips (x)',ylabel='Old-company trips (y)',aspect='equal')
    ax.set_xticks([0,5,10,15]);ax.set_yticks([0,5,10,15,20]);ax.grid(alpha=.12);ax.set_axisbelow(True)
axs[0].set_title('A  Original disagreement\nWritten: y − x ≥ 0.4; code: 3y − 2x ≥ 0',loc='left',fontsize=11,fontweight='bold',pad=14)
axs[1].set_title('B  After written-model revision\nBoth artifacts: 3y − 2x ≥ 0',loc='left',fontsize=11,fontweight='bold',pad=14)
from matplotlib.lines import Line2D
handles=[Line2D([],[],marker='o',linestyle='',color=GREY,label='Feasible in both original models'),
         Line2D([],[],marker='o',linestyle='',markerfacecolor='none',color=ORANGE,label='Feasible only in original program'),
         Line2D([],[],marker='o',linestyle='',color=TEAL,label='Feasible in both repaired models')]
fig.legend(handles=handles,loc='lower center',ncol=1,frameon=False,fontsize=9,bbox_to_anchor=(.5,0))
fig.tight_layout(rect=(0,.13,1,1),w_pad=2)
save(fig,'shipping-counterexample')

payload={'full_population':full,'controlled_repair':controlled,'host_integration':hosts,
  'sources':{'full_population':'comparison_compact.tex','controlled_repair':'ablation_compact.tex','host_integration':'plugin_compact.tex'},
  'uncertainty':'Aggregate rates transcribed from paper tables; no uncertainty estimates or significance markers inferred.'}
(ROOT/'docs/paper-results/values.json').write_text(json.dumps(payload,indent=2)+'\n')
print('Rendered homepage comparison figures and the released shipping case.')

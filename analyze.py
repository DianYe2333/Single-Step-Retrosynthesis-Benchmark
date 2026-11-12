"""Generate figures, paired bootstrap intervals and inspectable error cases."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=Path('runs/benchmark'))
p=parser.parse_args().output_dir
rows=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
methods=['NeuralSym','RetroSim'];colors=['#166678','#e78b35']
hits={m:np.array([[x['results'][m]['rank'] is not None and x['results'][m]['rank']<=k for k in [1,5,10]] for x in rows]) for m in methods}
fig,ax=plt.subplots(figsize=(7,4))
for j,m in enumerate(methods):ax.bar(np.arange(3)+(j-.5)*.32,hits[m].mean(0)*100,width=.32,color=colors[j],label=m+'-style')
ax.set(xticks=range(3),xticklabels=['Top-1','Top-5','Top-10'],ylabel='Exact precursor-set accuracy (%)',ylim=(0,100));ax.legend();fig.tight_layout();fig.savefig(p/'accuracy.png',dpi=180);plt.close(fig)
rng=np.random.default_rng(42);diff=hits[methods[0]].astype(float)-hits[methods[1]]
boot=np.array([diff[rng.integers(0,len(rows),len(rows))].mean(0) for _ in range(2000)])
summary={'n':len(rows),'paired_difference_NeuralSym_minus_RetroSim':diff.mean(0).tolist(),'bootstrap_95_percentile_ci':np.quantile(boot,[.025,.975],axis=0).tolist(),'both_top10':int(np.sum(hits[methods[0]][:,2]&hits[methods[1]][:,2])),'either_top10':int(np.sum(hits[methods[0]][:,2]|hits[methods[1]][:,2]))}
(p/'paired_analysis.json').write_text(json.dumps(summary,indent=2))
fig,ax=plt.subplots(figsize=(7,4));ax.boxplot([[r['results'][m]['seconds'] for r in rows] for m in methods],tick_labels=methods,showfliers=False);ax.set_ylabel('Seconds per product (CPU)');fig.tight_layout();fig.savefig(p/'latency.png',dpi=180);plt.close(fig)
errors=[r for r in rows if any(r['results'][m]['rank'] is None for m in methods)][:20]
with (p/'error_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['sample_index','patent_id','product','reference_precursors','NeuralSym_rank','RetroSim_rank','NeuralSym_top1','RetroSim_top1','review_note'])
 for r in errors:
  a,b=[r['results'][m] for m in methods]
  notes=[]
  def nonstereo(s):
   mol=Chem.MolFromSmiles(s);Chem.RemoveStereochemistry(mol);return '.'.join(sorted(Chem.MolToSmiles(mol).split('.')))
  for m,v in zip(methods,[a,b]):
   preds=v['predictions']
   if v['rank']:notes.append(f'{m}: reference at rank {v["rank"]}')
   elif any(nonstereo(s)==nonstereo(r['target']) for s in preds):notes.append(f'{m}: a candidate matches connectivity only; stereochemistry differs')
   elif not preds:notes.append(f'{m}: no applicable candidate')
   else:notes.append(f'{m}: reference absent; top-1 has {len(preds[0].split("."))} components versus {len(r["target"].split("."))} in reference')
  w.writerow([r['index'],r['id'],r['product'],r['target'],a['rank'],b['rank'],next(iter(a['predictions']),''),next(iter(b['predictions']),''),'; '.join(notes)])
for i,r in enumerate(errors[:10]):
 smis=[r['product'],r['target']]+[next(iter(r['results'][m]['predictions']),r['product']) for m in methods]
 img=Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in smis],molsPerRow=2,subImgSize=(500,260),legends=['Product','Reference precursors','NeuralSym-style top-1','RetroSim-style top-1'])
 img.save(str(p/f'case_{i+1:02}.png'))
print(json.dumps(summary,indent=2))

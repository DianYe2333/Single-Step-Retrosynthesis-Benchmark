"""Controlled NeuralSym-style highway vs RetroSim-style benchmark.

Modern rdchiral templates are shared between methods; this is an adapted
implementation, not a claim of exact reproduction of the 2017 papers.
"""
import json,time,random,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem
from rdchiral.main import rdchiralReaction,rdchiralReactants,rdchiralRun
from prepare_data import canonical

from neuralsym_model import TemplateNN_Highway

def load(directory, split):
    return [json.loads(line) for line in (directory/f'{split}_prepared.jsonl').read_text(encoding='utf-8').splitlines()]
def fp(s): return AllChem.GetMorganFingerprint(Chem.MolFromSmiles(s),2,useFeatures=True)
def vec(s):
    f=AllChem.GetHashedMorganFingerprint(Chem.MolFromSmiles(s),2,nBits=4096,useChirality=True)
    a=np.zeros(4096,dtype=np.float32); DataStructs.ConvertToNumpyArray(f,a)
    return np.log1p(a)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=500);ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--prepared-dir', type=Path, default=Path('data/prepared'))
    ap.add_argument('--output-dir', type=Path, default=Path('runs/benchmark'))
    args=ap.parse_args()
    if args.samples < 1 or args.epochs < 1:
        ap.error('--samples and --epochs must be positive')
    OUT=args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog('rdApp.*');random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    train=load(args.prepared_dir, 'train')
    test=load(args.prepared_dir, 'test')
    if not test or not any(row.get('template') for row in train):
        ap.error('Prepared data must include test rows and at least one training template')
    templates=sorted(set(x['template'] for x in train if x.get('template'))); ids={t:i for i,t in enumerate(templates)}
    usable=[x for x in train if x.get('template')]
    X=torch.tensor(np.stack([vec(x['product']) for x in usable])); y=torch.tensor([ids[x['template']] for x in usable])
    model=TemplateNN_Highway(len(templates),size=256,num_layers_body=5,input_size=4096)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    checkpoint=OUT/'neuralsym.pt'; history=[]
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for batch in torch.randperm(len(X)).split(128):
            opt.zero_grad();loss=nn.functional.cross_entropy(model(X[batch]),y[batch]);loss.backward();opt.step();losses.append(loss.item())
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses))});print(history[-1],flush=True)
    torch.save(model.state_dict(),checkpoint)
    (OUT/'training_history.json').write_text(json.dumps(history,indent=2))
    model.eval(); fps=[fp(x['product']) for x in train]; rfps=[fp(x['target']) for x in train]; cache={}
    def apply(t,rct):
        try:
            if t not in cache:cache[t]=rdchiralReaction('('+t.replace('>>',')>>'))
            return rdchiralRun(cache[t],rct,combine_enantiomers=False)
        except Exception:return []
    indices=np.random.default_rng(42).choice(len(test),min(args.samples,len(test)),replace=False).tolist()
    (OUT/'sample_indices.json').write_text(json.dumps(indices))
    with (OUT/'predictions.jsonl').open('w') as f:
        for n,i in enumerate(indices):
            ex=test[i];rct=rdchiralReactants(ex['product']); results={}
            for method in ['NeuralSym','RetroSim']:
                start=time.perf_counter();scores={}
                if method=='NeuralSym':
                    with torch.no_grad():rank=model(torch.tensor(vec(ex['product']))[None]).flatten().argsort(descending=True)[:100].tolist()
                    for k,j in enumerate(rank):
                        for s in apply(templates[j],rct):
                            try:scores.setdefault(canonical(s),-k)
                            except Exception:pass
                else:
                    sims=np.array(DataStructs.BulkTanimotoSimilarity(fp(ex['product']),fps))
                    for j in np.argsort(sims)[::-1][:100]:
                        t=train[j].get('template')
                        if not t:continue
                        for s in apply(t,rct):
                            try:
                                score=sims[j]*DataStructs.TanimotoSimilarity(fp(s),rfps[j]); s=canonical(s);scores[s]=max(scores.get(s,0),score)
                            except Exception:pass
                preds=sorted(scores,key=scores.get,reverse=True)[:10]
                results[method]={'predictions':preds,'seconds':time.perf_counter()-start,'rank':preds.index(ex['target'])+1 if ex['target'] in preds else None,'candidate_count':len(scores)}
            f.write(json.dumps(dict(**ex,results=results))+'\n');f.flush();print('test',n+1,len(indices),flush=True)
    records=[json.loads(x) for x in (OUT/'predictions.jsonl').read_text().splitlines()]
    metrics=[]
    training_pairs={(x['product'],x['target']) for x in train}
    training_products={x['product'] for x in train}
    for subset in ['all','no_exact_overlap','novel_product']:
      selected=[x for x in records if subset=='all' or (subset=='no_exact_overlap' and (x['product'],x['target']) not in training_pairs) or (subset=='novel_product' and x['product'] not in training_products)]
      for method in ['NeuralSym','RetroSim']:
        r=[x['results'][method] for x in selected];times=[x['seconds'] for x in r]
        if not r:
            metrics.append(dict(method=method,subset=subset,n=0,top1=None,top5=None,top10=None,median_seconds=None,mean_seconds=None))
            continue
        metrics.append(dict(method=method,subset=subset,n=len(r),**{f'top{k}':float(np.mean([v['rank'] is not None and v['rank']<=k for v in r])) for k in [1,5,10]},median_seconds=float(np.median(times)),mean_seconds=float(np.mean(times))))
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));print(metrics)
if __name__=='__main__':main()

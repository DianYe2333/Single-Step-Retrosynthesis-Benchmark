"""Extract train-only templates and canonical molecules from OpenRetro USPTO-50K.

Run with --workers 4. Outputs are checkpointed and never derived from test labels.
"""
import argparse, csv, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem, RDLogger
from rdchiral.template_extractor import extract_from_reaction

def canonical(s):
    m=Chem.MolFromSmiles(s)
    if m is None: raise ValueError('Invalid SMILES')
    for a in m.GetAtoms(): a.SetAtomMapNum(0)
    return '.'.join(sorted(Chem.MolToSmiles(m,isomericSmiles=True).split('.')))

def extract(item):
    i,row=item
    RDLogger.DisableLog('rdApp.*')
    r,_,p=row['rxn_smiles'].split('>')
    out=dict(index=i,id=row['id'],product=canonical(p),target=canonical(r))
    try:
        t=extract_from_reaction(dict(_id=str(i),reactants=r,products=p))
        out['template']=t.get('reaction_smarts')
    except Exception as e: out.update(template=None,error=str(e))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--raw-dir',type=Path,default=Path('data/raw'))
    ap.add_argument('--prepared-dir',type=Path,default=Path('data/prepared'))
    args=ap.parse_args()
    if args.workers < 1 or args.limit < 0:
        ap.error('--workers must be positive and --limit must be nonnegative')
    source=args.raw_dir
    dest=args.prepared_dir
    dest.mkdir(parents=True,exist_ok=True)
    for split in ['train','val','test']:
        rows=list(csv.DictReader((source/f'raw_{split}.csv').open(encoding='utf-8')))
        if args.limit: rows=rows[:args.limit]
        path=dest/f'{split}_prepared.jsonl'
        partial=path.with_suffix('.partial')
        with partial.open('w',encoding='utf-8') as f:
            if split=='train':
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    for n,out in enumerate(pool.map(extract,enumerate(rows),chunksize=20)):
                        f.write(json.dumps(out)+'\n')
                        if n%500==0: f.flush(); print(split,n,flush=True)
            else:
                for i,row in enumerate(rows):
                    r,_,p=row['rxn_smiles'].split('>')
                    f.write(json.dumps(dict(index=i,id=row['id'],product=canonical(p),target=canonical(r)))+'\n')
        partial.replace(path)
        print('Completed',split,len(rows),flush=True)
if __name__=='__main__': main()

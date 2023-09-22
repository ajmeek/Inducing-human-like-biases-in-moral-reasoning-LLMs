from datasets import load_dataset

ds=load_dataset('data/ds000212/ds000212_lfb', name='LFB-LAST')

print(list(ds.filter(lambda e: 'sub-07' in  e['file'])['train']))

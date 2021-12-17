from test import test
from train import train
root = '/content/drive/MyDrive/CS492I/project'
test(root,  f"{root}/final_models/seq2seqAttn_best.ckpt", use_pointer_gen=False, use_coverage=False)
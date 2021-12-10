from easydict import EasyDict

args = EasyDict()

args.vocab_size = 50000
args.embed_dim = 128
args.hidden_dim = 256
args.batch_size = 8
args.trg_max_len = 50
args.learning_rate = 0.15
args.accum_init = 0.15
args.pad_id = 0
args.seed = 123
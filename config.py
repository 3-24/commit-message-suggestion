from easydict import EasyDict

args = EasyDict()

args.vocab_size = 50000
args.embed_dim = 128
args.hidden_dim = 256
args.batch_size = 8
args.src_max_len = 10000
args.trg_max_len = 32
args.learning_rate = 0.01
args.accum_init = 0
args.pad_id = 0
args.seed = 123
args.epochs = 10
args.max_grad_norm = 2.0
# TODO: ADD beam search
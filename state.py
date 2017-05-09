def prototype_state():
    state = {}
    state['workdir']= ''
    state['train_data_file'] = '/home/tangmin/mycode/reproduction.code/NLP/machine-translation/seq2seq-attention/data_net.pkl'
    state['seed'] = 1234
    state['debug_level'] = 'DEBUG'
    state['level'] = 'DEBUG'
    state['vocab_size_enc'] = 19966
    state['vocab_size_dec'] = 19963
    state['emb_dim'] = 100
    state['mlp_dim'] = 200
    state['hidden_dim'] = 200

    state['loop_iters'] = 100000
    state['lr'] = 0.0001
    state['max_grad_norm'] = 10
    state['updater'] = 'adam'
    state['bs'] = 64
    state['seqlen'] = -1
    return state
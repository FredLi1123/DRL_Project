import os


def query_gpu():
    '''
    qargs: query arguments
    return: a list of dict
    Querying GPUs infos
    '''
    cmd = 'nvidia-smi --query-gpu=\'memory.used\' --format=csv,noheader'
    results = os.popen(cmd).readlines()
    print(results)
    return


def annealing(optimizer, decay_rate=5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / decay_rate

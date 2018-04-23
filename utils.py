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

class Recorder(object):
    
    def __init__(self, output_path):
        self.file = open(output_path, mode='w')
    
    def record(self, epoch, alpha, val_loss):
        self.file.write(str(alpha)+','+str(epoch)+','+str('{:3f}'.format(val_loss))+'\n')
        self.file.flush()
    
    def close(self):
        self.file.close()

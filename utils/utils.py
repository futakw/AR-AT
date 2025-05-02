import sys
from collections import OrderedDict


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    state_dict = new_state_dict

    # for wrapper bug
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            new_state_dict[k.replace("net.", "")] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    return state_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossMeter(object):
    """
    stores loss values for each epoch
    """

    def __init__(self):
        # loss_history :
        #   {loss_name: [loss_value]}
        self.loss_history = {}

        self.loss_raw_dict = {}
        self.loss_dict = {}
        self.loss_history = {}

    def reset(self):
        self.loss_raw_dict = {}
        self.loss_dict = {}
        self.loss_history = {}

    def update(self, loss_name, loss, loss_weight):
        self.loss_dict

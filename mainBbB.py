import torch

from mmodel import get_module
from mtrain.watcher import watcher

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    # name = input('model name:')
    name = "BY"
    try:
        param, A = get_module(name)

        if param.make_record:
            watcher.prepare_notes(name, param.tag)
        A.train_module()

    finally:
        watcher.to_json()

from mtrain.mloger import get_colored_logger, TRAIN, VALID
from tabulate import tabulate
from collections import defaultdict


loss_logger = get_colored_logger('losses_logger')

losses_history = defaultdict(list)

def get_changing_str(number):
    arrow = '⮥' if number > 0 else '⮧'
    return arrow + '  ' + '%.5f' % (abs(number))

def tabulate_log_losses(losses, trace, mode = 'train'):
    assert mode in ['train', 'valid']  

    historys = losses_history[trace]
    items = [
        (c[0], c[1], get_changing_str(c[1]-h[1])) for (h, c) in zip(historys[-1], losses)
    ] if len(historys) > 0 else [
        (c[0], c[1], 'NONE') for c in losses
    ]
    historys.append(losses)

    table = tabulate(
        items,
        # headers=['Loss', 'Value'],
        tablefmt="grid",
    )
    lines = table.split('\n')
    log_mode =  TRAIN if mode is 'train' else VALID
    for l in lines:
        loss_logger.log(log_mode, l)
from mtrain.mloger import get_colored_logger, BUILD

import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.utils as utils

logger = get_colored_logger('gpu')

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda:0")

def anpai(tom, use_gpu, need_logging=True):
    ''' Send tensor or module(tom) to correspond devices(cpu or gpu) based on 'use_gpu'.
    
    Arguments:
        tom {list} -- A list or single target tensor or module.
        use_gpu {bool} -- wheather to use gpu
    
    Keyword Arguments:
        need_logging {bool} -- wheather to make logging (default: {True})
    
    Returns:
        [list] -- handled module or tensor
    

    Example
    '
        a = nn.res50()
        b = input_tensor

        use_gpu = param.use_gpu

        a, b = anpai((a,b), use_gpu)
    '
    '''

    ## init a list to store result
    handle = list()
    
    if not isinstance(tom, (list, tuple)):
        l = list()
        l.append(tom)
        tom = l
    
    def __handle_module(module):
        # use data parallel
        dpm =  nn.DataParallel(module)
        
        handle.append(dpm)
        # info gpu used
        name = module.__class__.__name__
        info = "A >%s< object sent to Multi-GPU wits ids:" % name
        for i in dpm.device_ids:
            info += str(i)
        info += '.'
        if need_logging:
            logger.log(BUILD,info)

    def __handle_tensor(tensor, d):
        handle.append(tensor.to(d))
        name = tensor.__class__.__name__
        if need_logging :
            if str(d.type) == 'cpu':
                logger.log(BUILD,"Has no gpu or not use, %s will sent to CPU." % name)
            else:
                logger.log(BUILD,"A >%s< object sent to GPU." % name)

    device = device_cpu
    # When use_gpu and has gpu to use
    if use_gpu and cuda.is_available():

        # init a default device when has gpu
        device = device_gpu
 
        for i in tom:
            i.to(device)
            # When usable gpu more then one
            if cuda.device_count() > 1:
                # for Module object use parallel
                if isinstance(i, nn.Module):
                    __handle_module(i)
                # for tnesor object just send to default gpu
                elif isinstance(i, torch.Tensor):
                    __handle_tensor(i,device)
                else:
                    if need_logging:
                        logger.log(BUILD,i.__class__.__name__ + 'not spuuort')

            # When only one gpu can be used
            else:
                __handle_tensor(i,device)
    # use CPU
    else:
        for i in tom:
            target = i
            if isinstance(i, nn.DataParallel):
                for j in i.children():
                    target = j
                    __handle_tensor(target,device)
            else:
                __handle_tensor(target, device)
    

    return handle[0] if len(handle) == 1 else handle

# import gc
# def memReport():
#     for obj in gc.get_objects():
#         if torch.is_tensor(obj):
#             print(type(obj), obj.size())

# import sys, os   
# import psutil
# def cpuStats():
#         print(sys.version)
#         print(psutil.cpu_percent())
#         print(psutil.virtual_memory())  # physical memory usage
#         pid = os.getpid()
#         py = psutil.Process(pid)
#         memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
#         print('memory GB:', memoryUse)

def current_gpu_usage():
    print(torch.cuda.memory_allocated() / (1024**3))
    print(torch.cuda.max_memory_allocated() / (1024**3))
    print(torch.cuda.max_memory_cached() / (1024**3))
    print('===================================')

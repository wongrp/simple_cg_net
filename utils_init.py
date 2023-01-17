import torch
import logging
logger = logging.getLogger(__name__)
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from math import inf, log, log2, exp, ceil

def init_cuda(args):
    if args.cuda:
        assert(torch.cuda.is_available()), "No CUDA device available!"
        logger.info('Beginning training on CUDA/GPU! Device: {}'.format(torch.cuda.current_device()))
        torch.cuda.init()
        device = torch.device('cuda')
    else:
        logger.info('Beginning training on CPU!')
        device = torch.device('cpu')

    if args.dtype == 'double':
        dtype = torch.double
    elif args.dtype == 'float':
        dtype = torch.float
    else:
        raise ValueError('Incorrect data type chosen!')

    return device, dtype


#### Initialize optimizer ####

def init_optimizer(args, model):

    params = {'params': model.parameters(), 'lr': args.lr_init, 'weight_decay': args.weight_decay}
    params = [params]

    optim_type = args.optim.lower()

    if optim_type == 'adam':
        optimizer = optim.Adam(params, amsgrad=False)
    elif optim_type == 'amsgrad':
        optimizer = optim.Adam(params, amsgrad=True)
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(params)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(params)
    else:
        raise ValueError('Incorrect choice of optimizer')

    return optimizer

def init_scheduler(args, optimizer):
    lr_init, lr_final = args.lr_init, args.lr_final
    lr_decay = min(args.lr_decay, args.num_epoch)

    minibatch_per_epoch = ceil(args.num_train / args.batch_size)
    if args.lr_minibatch:
        lr_decay = lr_decay*minibatch_per_epoch

    lr_ratio = lr_final/lr_init

    lr_bounds = lambda lr, lr_min: min(1, max(lr_min, lr))

    if args.sgd_restart > 0:
        restart_epochs = [(2**k-1) for k in range(1, ceil(log2(args.num_epoch))+1)]
        lr_hold = restart_epochs[0]
        if args.lr_minibatch:
            lr_hold *= minibatch_per_epoch
        logger.info('SGD Restart epochs: {}'.format(restart_epochs))
    else:
        restart_epochs = []
        lr_hold = args.num_epoch
        if args.lr_minibatch:
            lr_hold *= minibatch_per_epoch

    if args.lr_decay_type.startswith('cos'):
        scheduler = sched.CosineAnnealingLR(optimizer, lr_hold, eta_min=lr_final)
    elif args.lr_decay_type.startswith('exp'):
        lr_lambda = lambda epoch: lr_bounds(exp(epoch / lr_decay) * log(lr_ratio), lr_ratio)
        scheduler = sched.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError('Incorrect choice for lr_decay_type!')

    return scheduler, restart_epochs

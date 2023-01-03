import torch 
import logging
logger = logging.getLogger(__name__)

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

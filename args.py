import argparse


def init_args():
    parser = argparse.ArgumentParser(description = 'data processing options')

    parser.add_argument('--batch-size', '-bs', type=int, default=25, metavar='N',
                        help='Mini-batch size (default: 25)')
    parser.add_argument('--datadir', type = str, default = 'datasets')

    # Dataset options
    parser.add_argument('--subtract-thermo', action=BoolArg, default=False,
                        help='Subtract thermochemical energy from relvant learning targets in QM9 dataset.')
    parser.add_argument('--force-download', action=BoolArg, default=False,
                        help='Force download and processing of dataset.')
    parser.add_argument('--num-train', type=int, default=-1, metavar='N',
                        help='Number of samples to train on. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--num-valid', type=int, default=-1, metavar='N',
                        help='Number of validation samples to use. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--num-test', type=int, default=-1, metavar='N',
                        help='Number of test samples to use. Set to -1 to use entire dataset. (default: -1)')

    # Dataloader and randomness options
    parser.add_argument('--shuffle', type = bool, default=True,
                        help='Shuffle minibatches.')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='Set random number seed. Set to -1 to set based upon clock.')

    # Computation options
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Set number of workers in dataloader. (Default: 1)')


    args = parser.parse_args()

    return args


class BoolArg(argparse.Action):
    """
    Take an argparse argument that is either a boolean or a string and return a boolean.
    """
    def __init__(self, default=None, nargs=None, *args, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")

        # Set default
        if default is None:
            raise ValueError("Default must be set!")

        default = _arg_to_bool(default)

        super().__init__(*args, default=default, nargs='?', **kwargs)

    def __call__(self, parser, namespace, argstring, option_string):

        if argstring is not None:
            # If called with an argument, convert to bool
            argval = _arg_to_bool(argstring)
        else:
            # BoolArg will invert default option
            argval = True

        setattr(namespace, self.dest, argval)

def _arg_to_bool(arg):
    # Convert argument to boolean

    if type(arg) is bool:
        # If argument is bool, just return it
        return arg

    elif type(arg) is str:
        # If string, convert to true/false
        arg = arg.lower()
        if arg in ['true', 't', '1']:
            return True
        elif arg in ['false', 'f', '0']:
            return False
        else:
            return ValueError('Could not parse a True/False boolean')
    else:
        raise ValueError('Input must be boolean or string! {}'.format(type(arg)))

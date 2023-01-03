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
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Use CUDA')
    parser.add_argument('--no-cuda', '--cpu', dest='cuda', action='store_false',
                        help='Use CPU (default)')
    parser.set_defaults(cuda=False)
    parser.add_argument('--float', dest='dtype', action='store_const', const='float',
                        help='Use floats.')
    parser.add_argument('--double', dest='dtype', action='store_const', const='double',
                        help='Use doubles.')
    parser.set_defaults(dtype='float')


    # Model options
    parser.add_argument('--num-cg-layers', type=int, default=4, metavar='N',
                        help='Number of CG levels (default: 4)')

    parser.add_argument('--max-l', nargs='*', type=int, default=[3], metavar='N',
                        help='Cutoff in CG operations (default: [3])')
    parser.add_argument('--max-sh', nargs='*', type=int, default=[3], metavar='N',
                        help='Number of spherical harmonic powers to use (default: [3])')
    parser.add_argument('--num-channels', nargs='*', type=int, default=[10], metavar='N',
                        help='Number of channels to allow after mixing (default: [10])')
    parser.add_argument('--level-gain', nargs='*', type=float, default=[10.], metavar='N',
                        help='Gain at each level (default: [10.])')

    parser.add_argument('--charge-power', type=int, default=2, metavar='N',
                        help='Maximum power to take in one-hot (default: 2)')

    parser.add_argument('--hard-cutoff', dest='hard_cut_rad',
                        type=float, default=1.73, nargs='*', metavar='N',
                        help='Radius of HARD cutoff in Angstroms (default: 1.73)')
    parser.add_argument('--soft-cutoff', dest='soft_cut_rad', type=float,
                        default=1.73, nargs='*', metavar='N',
                        help='Radius of SOFT cutoff in Angstroms (default: 1.73)')
    parser.add_argument('--soft-width', dest='soft_cut_width',
                        type=float, default=0.2, nargs='*', metavar='N',
                        help='Width of SOFT cutoff in Angstroms (default: 0.2)')
    parser.add_argument('--cutoff-type', '--cutoff', type=str, default=['learn'], nargs='*', metavar='str',
                        help='Types of cutoffs to include')

    parser.add_argument('--basis-set', '--krange', type=int, default=[3, 3], nargs=2, metavar='N',
                        help='Radial function basis set (m, n) size (default: [3, 3])')


    # More Model options
    parser.add_argument('--weight-init', type=str, default='rand', metavar='str',
                        help='Weight initialization function to use (default: rand)')

    parser.add_argument('--input', type=str, default='linear',
                        help='Function to apply to process l0 input (linear | MPNN) default: linear')
    parser.add_argument('--num-mpnn-layers', type=int, default=1,
                        help='Number levels to use in input featurization MPNN. (default: 1)')
    parser.add_argument('--top', '--output', type=str, default='linear',
                        help='Top function to use (linear | PMLP) default: linear')

    parser.add_argument('--gaussian-mask', action='store_true',
                        help='Use gaussian mask instead of sigmoid mask.')


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

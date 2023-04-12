import urllib.request
from os.path import join as join
import logging
import tarfile
import torch
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np

charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}


def load_dataset():
    """
    Returns:
    ------
    dictionary with dataset and keys "train", "val", "test"
    """

def prepare_dataset(directory, dataset, subset, force_download=False):
    """
    Prepares dataset directory.
    Currently just serves QM9, can also serve MD17 with some tweaks.

    Parameters
    __________
    directory : str
        - folder
    dataset: str
        - 'qm9' or 'md17' or something else.
    subset: str
        - (qm9 doesn't have subsets but md17 does.)
    force_download: bool, optional
        - file exists but download -- overwrite -- and process anyway
    """
    os.makedirs(directory, exist_ok = True)
    if dataset == 'qm9':
        datadir_dict = prepare_qm9(directory)

    return datadir_dict

def prepare_qm9(directory):
    """
    downloads and processes the QM9 dataset (or if processed files exist, checks that they do).
    return a dictionary {key = train, val or test : value = npz file directory}.

    """
    # False if already downloaded. True if not.
    download = False

    # To access files, create a dictionary of split datafile directories
    split_names = ['train', 'val', 'test']
    datadir_dict = {split: os.path.join(*([directory] + [split + '.npz'])) for split in split_names}

    logging.info('Checking the files {}'.format(datadir_dict))

    # Check if prepared dataset npz files exist. If not, download.
    datadir_checkfile = [os.path.exists(datadir) for datadir in datadir_dict.values()]

    if all(datadir_checkfile):
        logging.info('Dataset exists and is processed')
    elif all([not x for x in datadir_checkfile]):
            # If checks are failed.
            download = True
            logging.info('Downloading and processing dataset')
    else:
        raise ValueError(
            'Dataset only partially processed. Try deleting {} and running again to download/process.'.format(os.path.join(dataset_dir)))

    # if True, then download, process, and save processed file.
    if download == True:
        download_and_process_qm9(directory)

    return datadir_dict

def download_and_process_qm9(directory):
    """
    downloads the qm9 dataset and returns a dictionary of directories for dataset splits.
    """
    # directory names
    qm9_filename_data = join(directory,'dsgdb9nsd.xyz.tar.bz2')
    qm9_filename_excl =  join(directory,'uncharacterized.txt')

    # grab data from url
    qm9_url_data = 'https://figshare.com/ndownloader/files/3195389'
    urllib.request.urlretrieve(qm9_url_data, filename = qm9_filename_data)

    # there are 3054 molecules that failed the geometric consistency test. Remove those.
    qm9_url_excl = 'https://springernature.figshare.com/ndownloader/files/3195404'
    urllib.request.urlretrieve(qm9_url_excl, filename = qm9_filename_excl)

    logging.info('Download complete')

    # exclude failed molecules
    excluded_strings = []
    with open(qm9_filename_excl) as f:
        lines = f.readlines()
        # split each line and take column 0, which is the molecule index
        excluded_strings = [line.split()[0] for line in lines if len(line.split()) > 0]
    # then convert to integer indices (based 1 --> 0)
    excluded_idxs = [int(idx)-1 for idx in excluded_strings if str_is_int(idx)]
    assert len(excluded_idxs) == 3054, 'There should be 3054 excluded atoms. Found {}'.format(len(excluded_idxs))

    num_qm9 = 133885
    num_excluded = 3054
    num_molecules = num_qm9 - num_excluded
    included_idxs = np.array(sorted(list(set(range(num_qm9))-set(excluded_idxs))))

    # generate random permutations of data. 100k training, 10% validate, rest test.
    num_train = 100000
    num_val = int(0.1*num_molecules)
    num_test = num_molecules - (num_train + num_val)

    np.random.seed(0)
    permutation = np.random.permutation(num_molecules)

    # split the permutation
    train, val, test, extra = np.split(permutation, [num_train, num_train + num_val, num_train + num_val + num_test])
    assert(len(extra)==0), 'split was inexact {} {} {} {}'.format(len(train),len(valid), len(test),len(extra))

    # generate a dictionary of split indices
    split_idxs  = {'train': included_idxs[train], 'val': included_idxs[val], 'test': included_idxs[test]}

    # from split indices, generate a dict of split data
    data_dict = {}
    logging.info('Converting from .xyz files to dict')
    for split, split_idx in split_idxs.items():
        data_dict[split] = xyz_files_to_dict(qm9_filename_data ,xyz_to_dict,file_idx_list = split_idx, stack = True)

    # Subtract thermochemical energy if desired.
    # I need to add this back in - ryan
    # if calculate_thermo:
    #     # Download thermochemical energy from GDB9 dataset, and then process it into a dictionary
    #     therm_energy = get_thermo_dict(directory, cleanup)

    #     # For each of train/validation/test split, add the thermochemical energy
    #     for split_idx, split_data in data_dict.items():
    #         data_dict[split_idx] = add_thermo_targets(split_data, therm_energy)

    # take the dictionary and save split data in npz format.
    logging.info('Saving processed data:')
    for split, data in data_dict.items():
        savedir = join(directory, split+'.npz')
        np.savez_compressed(savedir,**data)

    # This concludes download + processing.
    logging.info('Processing/saving complete')


def get_thermo_dict(gdb9dir, cleanup=True):
    """
    Get dictionary of thermochemical energy to subtract off from
    properties of molecules.

    Probably would be easier just to just precompute this and enter it explicitly.
    """
    # Download thermochemical energy
    logging.info('Downloading thermochemical energy.')
    gdb9_url_thermo = 'https://springernature.figshare.com/ndownloader/files/3195395'
    gdb9_txt_thermo = join(gdb9dir, 'atomref.txt')

    urllib.request.urlretrieve(gdb9_url_thermo, filename=gdb9_txt_thermo)

    # Loop over file of thermochemical energies
    therm_targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']

    # Dictionary that
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(gdb9_txt_thermo) as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()

            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue

            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]
                                           ] = float(split_therm)

    # Cleanup file when finished.
    cleanup_file(gdb9_txt_thermo, cleanup)

    return therm_energy


def add_thermo_targets(data, therm_energy_dict):
    """
    Adds a new molecular property, which is the thermochemical energy.

    Parameters
    ----------
    data : ?????
        QM9 dataset split.
    therm_energy : dict
        Dictionary of thermochemical energies for relevant properties found using :get_thermo_dict:
    """
    # Get the charge and number of charges
    charge_counts = get_unique_charges(data['charges'])

    # Now, loop over the targets with defined thermochemical energy
    for target, target_therm in therm_energy_dict.items():
        thermo = np.zeros(len(data[target]))

        # Loop over each charge, and multiplicity of the charge
        for z, num_z in charge_counts.items():
            if z == 0:
                continue
            # Now add the thermochemical energy per atomic charge * the number of atoms of that type
            thermo += target_therm[z] * num_z

        # Now add the thermochemical energy as a property
        data[target + '_thermo'] = thermo

    return data

def xyz_files_to_dict(data, process_file_fn, file_ext=None, file_idx_list=None, stack=True):
    """
    Take a set of datafiles and apply a predefined data processing script to each
    one. Data can be stored in a directory, tarfile, or zipfile. An optional
    file extension can be added.

    Parameters
    ----------
    data : str
        Complete path to datafiles. Files must be in a directory, tarball, or zip archive.
    process_file_fn : callable
        Function to process files. Can be defined externally.
        Must input a file, and output a dictionary of properties, each of which
        is a torch.tensor. Dictionary must contain at least three properties:
        {'num_elements', 'charges', 'positions'}
    file_ext : str, optional
        Optionally add a file extension if multiple types of files exist.
    file_idx_list : ?????, optional
        Optionally add a file filter to check a file index is in a
        predefined list, for example, when constructing a train/valid/test split.
    stack : bool, optional
        ?????
    """
    logging.info('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)

    elif os.is_dir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        readfile = lambda data_pt: open(data_pt, 'r')

    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # Use only files that end with specified extension.
    if file_ext is not None:
        files = [file for file in files if file.endswith(file_ext)]

    # Use only files that match desired filter.
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    # Now loop over files using readfile function defined above
    # Process each file accordingly using process_file_fn

    molecules = []

    for file in files:
        with readfile(file) as openfile:
            molecules.append(process_file_fn(openfile))

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

    # If stacking is desireable, pad and then stack.
    if stack:
        molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

    return molecules



def xyz_to_dict(filename, file_idx_list = None, ):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    xyz_lines = [line.decode('UTF-8') for line in filename.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms+2]
    mol_freq = xyz_lines[num_atoms+2]

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
    molecule.update(mol_props)
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule


def str_is_int(str):
    try:
        int(str)
        return True
    except:
        return False

# Cleanup. Use try-except to avoid race condition.
def cleanup_file(file, cleanup=True):
    if cleanup:
        try:
            os.remove(file)
        except OSError:
            pass

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl 

from utils_initialize_dataset import *


class DataModule(pl.LightningDataModule):
    def __init__(self, args, collate_fn):
        super().__init__()
        self.args = args 
        self.collate_fn = collate_fn
    
    def prepare_dataset(self, ):
        ...

    def setup(self, stage = None):
        self.args, datasets, num_species, charge_scale, stats = load_processed_qm9_dataset(self.args)
   
        self.train = datasets['train']
        self.val = datasets['val']
        self.test = datasets['test']
        
        return self.args, datasets, num_species, charge_scale, stats 
    
    def train_dataloader(self):
        return DataLoader(self.train,
                                     batch_size=self.args.batch_size,
                                     shuffle=self.args.shuffle,
                                     num_workers=self.args.num_workers,
                                     collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val,
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     num_workers=self.args.num_workers,
                                     collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test,
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     num_workers=self.args.num_workers,
                                     collate_fn=self.collate_fn)
    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     ...


def load_processed_qm9_dataset(args): 
    """ 
    Loads processed QM9 dataset for DataModule class. 
    """

    # datasets is a ProcessedDataset object.
    args, datasets, num_species, charge_scale = initialize_datasets(args, args.datadir, 'qm9', subtract_thermo=args.subtract_thermo,
                                                                    force_download=args.force_download
                                                                    )

    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)

    # potential bug
    stats = datasets["train"].stats  

    return args, datasets, num_species, charge_scale, stats 


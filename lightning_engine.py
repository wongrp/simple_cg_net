import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

# The nn.module we're using.
from network_model import FullModel 
from utils_init import *


class Engine(pl.LightningModule):
    """
    Engine class. Contains hyperparameters and training information. 
    """

    def __init__(self, args, num_species, charge_scale, stats, device, dtype):
        super().__init__() 
        self.args = args
        self.stats = stats 
        
        # Initialize model
        self.model = FullModel(args.max_l, args.max_sh, args.num_cg_layers, args.num_channels, num_species,
                 args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                 args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                 charge_scale, args.gaussian_mask,
                 args.top, args.input, args.num_mpnn_layers,
                 device=device, dtype=dtype)

        # cuda
        if device == torch.device('cuda'):
            self.model.cuda()
            print("sending model to cuda!")
        
        # Initialize scheduler, optimizer, define loss function. 
        self.optimizer = init_optimizer(args, self.model)
        self.scheduler, restart_epochs = init_scheduler(args, self.optimizer)
        self.loss_fn = torch.nn.functional.mse_loss
        
        # Add logged hyperparameters
        # for  
        # add_log_hparams(args)
        self.save_hyperparameters()

    def forward(self,x): 
        return self.model(x) 

    def training_step(self, batch, batch_idx):  
        """
        Trains one batch. 
        """
        targets = self._get_target(batch, self.stats)
        predict = self.forward(batch)
        loss = self.loss_fn(predict, targets)
        self.log("train/loss", loss, logger = True)
        return loss

    def validation_step(self, batch, batch_idx): 
        """
        Validates one batch
        """
        targets = self._get_target(batch, self.stats)
        predict = self.forward(batch)
        loss = self.loss_fn(predict, targets)
        self.log("val_loss", loss, logger = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Tests one batch. 
        """
        targets = self._get_target(batch, self.stats)
        predict = self.forward(batch)
        loss = self.loss_fn(predict, targets)
        self.log("test_loss", loss, logger = True)
        return loss

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler} 

    def _get_target(self, data, stats=None):
        """
        Get the learning target.
        If a stats dictionary is included, return a normalized learning target.
        """

        targets = data[self.args.target].to(self.device, self.dtype)

        if stats is not None:
            mu, sigma = stats[self.args.target]
            targets = (targets - mu) / sigma

        return targets




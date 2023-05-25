import timm
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from dataloader_cloudsen12 import training_data, validation_data, testing_data

class lit_dataloader(pl.LightningDataModule):
    def __init__(self, batch_size):
        """
        LightningDataModule for loading and preparing data.

        Args:
            batch_size (int): The batch size for the data loaders.

        """
        super().__init__()
        self.batch_size = batch_size
        
    def train_dataloader(self):
        """
        Get the training data loader.

        Returns:
            torch.utils.data.DataLoader: The training data loader.

        """
        return torch.utils.data.DataLoader(training_data, 
                             batch_size=self.batch_size, 
                             shuffle=True,
                             pin_memory=False)

    def val_dataloader(self):
        """
        Get the validation data loader.

        Returns:
            torch.utils.data.DataLoader: The validation data loader.

        """
        return torch.utils.data.DataLoader(validation_data, 
                             batch_size=self.batch_size, 
                             shuffle=False,
                             pin_memory=False)

class reg_model(pl.LightningModule):
    def __init__(self):
        """
        LightningModule for the hardness index prediction.

        """
        super().__init__()
        self.model = timm.create_model('resnet10t', pretrained=True, num_classes=1, in_chans=13)
        self.loss = torch.nn.BCEWithLogitsLoss()
                
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        X, y = batch
        yhat = self(X).squeeze()
        loss = self.loss(yhat, y.float())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        yhat = self(X).squeeze()
        loss = self.loss(yhat, y.float())
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        yhat = self(X).squeeze()
        loss = self.loss(yhat, y.float())
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer, learning rate scheduler, and monitor.

        """
        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

if __name__ == '__main__':

    # Define training parameters
    batch_size = 64
    nepochs = 250
    
    # Logging
    logging = WandbLogger(project='IGARS2023', entity='csaybar')

    # Define callbacks
    callback1 = ModelCheckpoint(
        monitor='val_loss', save_top_k=1, mode='min', 
        filename='{epoch}-{val_loss:.2f}', dirpath='weights/',
        save_weights_only=True
    )
    callback2 = EarlyStopping(monitor='val_loss', patience=20)
    callbacks = [callback1, callback2]
    
    # Define trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logging,
        max_epochs=nepochs,
        accelerator="gpu",
        devices=[0]
    )
    
    # Train model
    lit_model = reg_model()
    lit_dataset = lit_dataloader(batch_size)
    trainer.fit(lit_model, lit_dataset)
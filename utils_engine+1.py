import torch
import numpy as np
import logging

class Engine:
    def __init__(self, args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, device, dtype):
        self.args = args
        self.dataloaders = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.restart_epochs = restart_epochs


    def train_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.a

        training_loader = self.dataloaders['train']

        for i, data in enumerate(training_loader):
            # Split data into input-label pair
            inputs, labels = data

            # Zero gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss


    def train(self):
        # timestamp and epoch number
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0
        best_val_loss = 1000000

        # validation dataloader
        validation_loader = self.dataloaders["val"]

        # iterate over epochs
        for epoch in range(epochs):
            # train one epoch and update average training loss
            avg_train_loss = train_epoch(epoch_index, tb_writer)

            # validate and update average validation loss
            current_val_loss = 0
            for index, data in enumerate(validation_loader)
                input_val, label_val = data
                pred_val = self.model(input_val)
                current_val_loss += self.loss_fn(pred_val,label_val)
            avg_val_loss = current_val_loss / (i+1)

            # track best performance and save model state
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # save model state
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(),model_path)

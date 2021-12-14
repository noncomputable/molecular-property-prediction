from pathlib import Path
import torch
from ray import tune

class Trainer:
    def __init__(self):
        self.training_losses = []
        self.validation_losses = []
        self.epoch_i = 0

    def train(self, training_dataloader, validation_dataloader, model,
              loss_func, optimizer, epochs, val_interval,
              batch_log_interval, epoch_log_interval,
              save_interval = None, checkpoint_dir = None, checkpoint_name = None,
              lr_scheduler = None, tuning = False):

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.epochs = epochs
        self.val_interval = val_interval
        self.batch_log_interval = batch_log_interval
        self.epoch_log_interval = epoch_log_interval
        self.save_interval = save_interval
        self.lr_scheduler = lr_scheduler
        self.tuning = tuning

        if checkpoint_dir and tuning:
            self.load(Path(checkpoint_dir)/"checkpoint")
        elif checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            
            if checkpoint_name is not None:
              self.load(self.checkpoint_dir/checkpoint_name)

        for epoch in range(self.epoch_i, epochs):
            self.epoch_i = epoch #+ 1 #So it won't log when epoch == 0

            self.training_loss = 0.0
            self.validation_loss = 0.0 if self.validation_dataloader else None
            
            if self.training_dataloader: 
                self.train_epoch()
            
            if self.validation_dataloader and self.epoch_i % self.val_interval == 0:
                self.validate_epoch()
            
            if self.epoch_i % self.epoch_log_interval == 0:
                self.log_epoch()
            
            if self.epoch_i % self.save_interval == 0:
                self.save()

            if self.tuning:
                tune.report(val_loss = self.validation_loss, train_loss = self.training_loss)
     
        return self.training_losses, self.validation_losses

    def train_epoch(self):
        self.model.train()
        for j, batch in enumerate(self.training_dataloader):
            batch_j = j + 1
            
            props, targets = batch
            self.optimizer.zero_grad()
            outputs = self.model(props)
            batch_loss = self.loss_func(outputs, targets)
            batch_loss.backward()
            self.optimizer.step()
           
            self.training_loss += batch_loss.detach().cpu().item()

            if batch_j % self.batch_log_interval == 0: # or self.epoch_i % self.epoch_log_interval == 0:
                #print(f"[(Output, Target),]: {list(zip(outputs.detach().tolist(), targets.tolist()))}")
                print(f"Batch loss: {batch_loss}")

        self.training_loss /= self.num_batches("training")
        self.training_losses.append(self.training_loss)

        self.lr_scheduler.step(self.training_loss)

    def validate_epoch(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.validation_dataloader:
                props, targets = batch
                outputs = self.model(props)
                batch_loss = self.loss_func(outputs, targets)
                self.validation_loss += batch_loss.detach().cpu().item()

        self.validation_loss /= self.num_batches("validation")
        self.validation_losses.append(self.validation_loss)

    def log_epoch(self):
        print(f"Epoch: { self.epoch_i } \t Training loss: {self.training_loss} \t Validation loss: {self.validation_loss}", end = "\n\n")

    def num_batches(self, which_dataloader):
        """
        Get the number of batches in the dataloader.
        
        Args:
        which_dataloader = "training" or "validation"
        """

        if which_dataloader == "training":
            return len(self.training_dataloader)
        elif which_dataloader == "validation":
            if self.validation_dataloader:
                return len(self.validation_dataloader)
            else:
                return None
        else:
            raise ValueError(f"{which_dataloader} doesn't refer to an available dataloader.")

    def save(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "epochs": self.epoch_i 
        }

        if self.tuning:
            with tune.checkpoint_dir(step = self.epoch_i) as checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir)/"checkpoint"
                torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint_path = Path(self.checkpoint_dir/f"epoch_{self.epoch_i}.chkp")
            torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_losses = checkpoint["training_losses"]
        self.validation_losses = checkpoint["validation_losses"]
        self.epoch_i = checkpoint["epochs"] + 1

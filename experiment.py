import functools
import torch
import json, os
from pathlib import Path

from ray.tune.schedulers import ASHAScheduler
from ray import tune

from core import dataset as data
from core import train
from core.model import PermPredictor

def search(num_samples, max_num_epochs, gpus_per_trial, name):
    """
    Search for good hyperparameters.
    """

    dim_seqs = [
            (500, 250, 100, 50),
            (1024, 512, 256, 128, 64),
            (256, 128, 64),
            (256, 64),
            (256,),
    ]

    dropout_pairs = [
            (0, 0),
            (.8, .5),
            (0, .5),
            (.8, .2)
    ]

    config = {
      "init_learning_rate": .1,
      "lr_step_interval": 5000,
      "n_epochs": 500,
      "epoch_log_interval": 1,
      "batch_log_interval": float('inf'),
      "val_interval": 1,
      "save_interval": 100,
      "batch_size": 32,
      "fp_type": "trnsfm",
      "dim_seq": tune.grid_search(dim_seqs),
      "dropout_pair": tune.grid_search(dropout_pairs)
    }

    scheduler = ASHAScheduler(
        max_t = max_num_epochs,
        grace_period = 35,
        reduction_factor = 2
    )

    data_dir = Path("./data").resolve()
    experiment_ = functools.partial(experiment,
                                    data_dir = data_dir, tuning = True)

    result = tune.run(
        tune.with_parameters(experiment_),
        resources_per_trial = {"cpu": 2, "gpu": gpus_per_trial},
        config = config,
        metric = "val_loss",
        mode = "min",   
        num_samples = num_samples,
        scheduler = scheduler,
        name = name,
        local_dir = "ray_log"
    )

def experiment(config, checkpoint_dir = None, checkpoint_name = None, data_dir = None, tuning = False):
    """
    Set up and train a model with the given configuration.

    Args: 
    config - Dict specifying embed_size, hidden_size, n_rnn_layers, rnn_dropout,
        n_predictor_blocks, predictor_dropout, and batch_size.
    checkpoint_dir - Path to dir where checkpoints will be saved.
    checkpoint_name - Name of checkpoint file in checkpoint_dir to load.
    tuning - Whether this experiment is part of a hyperparameter tuning run.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print("Device:", device)

    if checkpoint_dir is not None:
        with open(Path(checkpoint_dir)/"config.json", "w") as config_file:
          json.dump(config, config_file)

    data_dir = Path(data_dir)/config["fp_type"]
    train_set = data.MolData(data_dir/"train.dat")
    val_set = data.MolData(data_dir/"val.dat")

    """
    train_frac = .8
    train_num = round(len(train_set) * train_frac)
    val_num = len(train_set) - train_num
    train_set, val_set = torch.utils.data.random_split(train_set, [train_num, val_num])
    """

    model = PermPredictor(
        prop_vec_size = train_set.props.shape[1],
        dim_seq = config["dim_seq"],
        dropout_pair = config["dropout_pair"]
    ).to(device)
 
    init_learning_rate = config["init_learning_rate"]
    optimizer = torch.optim.SGD(model.parameters(), lr = init_learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)
  
    loss_func = torch.nn.MSELoss()
    epochs = 5000
    val_interval = 1
    batch_log_interval = float('inf')
    epoch_log_interval = 5
    save_interval = 1

    batch_size = config["batch_size"]
    training_dataloader = torch.utils.data.DataLoader(train_set, shuffle = True, batch_size = batch_size, 
                                                      )
    validation_dataloader = torch.utils.data.DataLoader(val_set, shuffle = True, batch_size = batch_size,
                                                      )

    trainer = train.Trainer()
    training_losses, validation_losses = trainer.train(
        training_dataloader = training_dataloader,
        validation_dataloader = validation_dataloader,
        model = model,
        loss_func = loss_func,
        optimizer = optimizer,
        epochs = epochs,
        val_interval = val_interval,
        batch_log_interval = batch_log_interval,
        epoch_log_interval = epoch_log_interval,
        save_interval = save_interval,
        checkpoint_dir = checkpoint_dir,
        checkpoint_name = checkpoint_name,
        lr_scheduler = lr_scheduler,
        tuning = tuning
    )

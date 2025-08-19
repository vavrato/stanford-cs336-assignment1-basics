import random
import numpy as np
import torch
from tqdm import tqdm
import wandb
import click, yaml

from cs336_basics.nn import AdamW, Transformer, cross_entropy, data_loader, gradient_clipping  # type: ignore


def read_config(ctx, param, value):
    if value is None:
        return
    with open(value, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # Make YAML values become defaults for all options (CLI flags still override)
    ctx.default_map = {**(ctx.default_map or {}), **cfg}


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    callback=read_config,
    is_eager=True,
    expose_value=False,
    help="YAML with defaults.",
)
@click.option("--train_dataset", type=click.Path(exists=True), help="Path to (tokenized) train dataset.")
@click.option("--valid_dataset", type=click.Path(exists=True), help="Path to (tokenized) valid dataset.")
@click.option("--tokenizer", type=click.Path(exists=True), help="Path to tokenizer config.")
@click.option("--n_steps", type=click.INT, default=1, show_default=True)
@click.option("--batch_size", default=1, show_default=True)
@click.option("--context_len", default=100, show_default=True)
@click.option("--n_layers", default=8, show_default=True)
@click.option("--n_heads", default=4, show_default=True)
@click.option("--d_model", default=64, show_default=True)
@click.option("--d_ff", default=128, show_default=True)
@click.option("--lr", default=3e-4, show_default=True)
@click.option("--logging_path", default="", show_default=True)
@click.option("--logging_frequency", default=10)
@click.option("--validation_frequency", default=None)
@click.option("--wandb_project", default=None, show_default=True)
def train(**kwargs):
    print("Final config:", kwargs)
    vocab_size = len(torch.load(kwargs["tokenizer"])["vocab_dict"])
    train_dataset = np.load(kwargs["train_dataset"], mmap_mode='r')
    valid_dataset = np.load(kwargs["valid_dataset"], mmap_mode='r')
    model = Transformer(
        vocab_size,
        kwargs["context_len"],
        kwargs["n_layers"],
        kwargs["d_model"],
        kwargs["n_heads"],
        kwargs["d_ff"],
        10000,
    )
    
    optimizer = AdamW(model.parameters(), lr = kwargs['lr'])

    with wandb.init(project=kwargs['wandb_project'], config=kwargs) as run:
        inputs, targets = data_loader(train_dataset, kwargs['batch_size'], kwargs['context_len'])
        for step in tqdm(range(1, kwargs['n_steps']+1)):
            model.train()
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)

            loss = cross_entropy(outputs.reshape(-1, vocab_size), targets.flatten())

            loss.backward()
            gradient_clipping(model.parameters(), M=1.)
            optimizer.step()
        

            # Log metrics to wandb.
            

            if (kwargs['validation_frequency'] is not None) and (step % kwargs['validation_frequency'] == 0):
                model.eval()
                valid_losses = []
                with torch.no_grad():
                    for _ in range(50):
                        inputs, targets = data_loader(valid_dataset, kwargs['batch_size'], kwargs['context_len'])
                        outputs = model.forward(inputs)
                        valid_loss = cross_entropy(outputs.reshape(-1, vocab_size), targets.flatten())
                        valid_losses.append(valid_loss)
                valid_loss = sum(valid_losses)/len(valid_losses)
            
                run.log({"train_loss": loss, "valid_loss": valid_loss})
            else:
                run.log({"train_loss": loss})


if __name__ == "__main__":
    train()

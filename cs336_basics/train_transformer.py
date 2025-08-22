import os
from time import time
import numpy as np
import torch
from tqdm import tqdm
import wandb
import click, yaml

from cs336_basics.nn import AdamW, Transformer, cosine_annealing, cross_entropy, data_loader, gradient_clipping, load_checkpoint, save_checkpoint
from cs336_basics.tokenizer import Tokenizer  # type: ignore


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
@click.option("--batch_size", default=1, type=click.INT,show_default=True)
@click.option("--context_len", default=100, type=click.INT,show_default=True)
@click.option("--n_layers", default=8, type=click.INT,show_default=True)
@click.option("--n_heads", default=4, type=click.INT,show_default=True)
@click.option("--d_model", default=64, type=click.INT,show_default=True)
@click.option("--d_ff", default=128, type=click.INT, show_default=True)
@click.option("--lr", default=3e-4, show_default=True)
@click.option("--logging_dir", default="", show_default=True)
@click.option("--logging_frequency", default=0, type=click.INT)
@click.option("--validation_frequency", type=click.INT, default=None)
@click.option("--wandb_project", default=None, show_default=True)
@click.option("--device", default='cpu')
@click.option("--use_scheduler", type=click.BOOL, default=False)
@click.option("--checkpoint_frequency", type=click.INT, default=0)
@click.option("--wandb_run_id", type=click.STRING, default=None)
@click.option("--resume_from", type=click.Path(exists=True), default=None)
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
    ).to(kwargs['device'])

    optimizer = AdamW(model.parameters(), lr = kwargs['lr'])

    start_step = 0

    if kwargs['resume_from']:
        start_step = load_checkpoint(kwargs['resume_from'], model, optimizer)


    if kwargs['logging_dir'] == "":
        kwargs['logging_dir'] = f"model_{int(time())}/"
    
    if kwargs['checkpoint_frequency'] > 0:
        os.makedirs(kwargs['logging_dir'], exist_ok=True)

    tokenizer = Tokenizer.from_trainer(kwargs['tokenizer'])
    eot_token_id = tokenizer.encode("<|endoftext|>")[0]

    

    if kwargs['use_scheduler']:
        if 'lr_max' not in kwargs:
            kwargs['lr_max'] = kwargs['lr']
        if 'lr_min' not in kwargs:
            kwargs['lr_min'] = kwargs['lr']*0.02
        if 'T_w' not in kwargs:
            kwargs['T_w'] = int(kwargs['n_steps']*0.02)
        if 'T_c' not in kwargs:
            kwargs['T_c'] = kwargs['n_steps']

    # compiling not really good with mps, it is slower :-(
    # model = torch.compile(model, backend="aot_eager")  


    train_losses = []

    with wandb.init(project=kwargs['wandb_project'], id=kwargs['wandb_run_id'], resume="allow", config=kwargs) as run:
        for step in tqdm(range(start_step+1, kwargs['n_steps']+1)):
            
            if kwargs['use_scheduler']:
                lr = cosine_annealing(step, kwargs['lr_max'], kwargs['lr_min'], kwargs['T_w'], kwargs['T_c'])

                for pg in optimizer.param_groups:
                    pg['lr'] = lr


            model.train()

            inputs, targets = data_loader(train_dataset, kwargs['batch_size'], kwargs['context_len'], device=kwargs['device'])
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)

            loss = cross_entropy(outputs.reshape(-1, vocab_size), targets.flatten())

            loss.backward()
            gradient_clipping(model.parameters(), M=1.)
            optimizer.step()
        

            if (kwargs['validation_frequency'] > 0) and (step % kwargs['validation_frequency'] == 0):
                model.eval()
                prefix = "Once upon a time"
                print(tokenizer.decode(model.generate(tokenizer.encode(prefix), max_len = 256, eot_token_id=eot_token_id, top_p = 0.7)))
                valid_losses = []
                with torch.no_grad():
                    inputs, targets = data_loader(valid_dataset, 100, kwargs['context_len'])
                    outputs = model.forward(inputs)
                    valid_loss = cross_entropy(outputs.reshape(-1, vocab_size), targets.flatten())
                    valid_losses.append(valid_loss)
                valid_loss = sum(valid_losses)/len(valid_losses)
            
                run.log({"train_loss": loss, "valid_loss": valid_loss}, step=step)
            else:
                run.log({"train_loss": loss}, step=step)

            if (kwargs['checkpoint_frequency'] > 0) and (step % kwargs['checkpoint_frequency'] == 0):
                save_checkpoint(model, optimizer, step, os.path.join(kwargs['logging_dir'], f"step_{step}.py"))

            train_losses.append(loss)

            # stopping if 10 consecutive steps are above the random baseline
            if sum(train_losses[-10:])/10 > 9:
                break


if __name__ == "__main__":
    train()

import os
import time
from tqdm import tqdm
import torch
import math
from torch.utils.data import DataLoader
import config
from dataset import CVRPDataset

def validate(model, val_dataset):
    valdataloader = DataLoader(val_dataset, batch_size=128, num_workers=1)
    def eval_batch(x):
        x = move_to(x, device="cuda:0")
        cost, _, _ = model(x)
        return cost
    return (torch.cat([eval_batch(bat) for bat in valdataloader])).mean()

def get_routes(model, dataset):
    loader = DataLoader(dataset, batch_size=1, num_workers=1)
    def getpath_coords_depot(x):
        x = move_to(x, device="cuda:0")
        _, _, pi = model(x)
        pi = pi[0].cpu().detach().numpy()
        return pi, x['depot'][0].cpu().numpy(), x['coordinates'][0].cpu().numpy()
    return [getpath_coords_depot(bat) for bat in loader]

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem):
    print(f"Start train epoch {epoch}")
    step = epoch * (config.EPOCH_SIZE // config.BATCH_SIZE)
    start_time = time.time()

    training_dataset = CVRPDataset(samples=config.EPOCH_SIZE)
    training_dataloader = DataLoader(training_dataset, batch_size=config.BATCH_SIZE, num_workers=1)

    model.train()

    for batch_id, batch in enumerate(tqdm(training_dataloader)):
        train_batch(model, optimizer, baseline, epoch, batch_id, step, batch)
        step += 1

    epoch_duration = time.time() - start_time
    print(f"Finished epoch {epoch}, took {round(epoch_duration, 2)} s")

    if (epoch % 15 == 0 and epoch > 0) or epoch == config.N_EPOCHS - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': model.state_dict(),
                'baseline': baseline.state_dict()
            },
            os.path.join(config.SAVE_DIR, 'epoch-{}.pt'.format(epoch))
        )
    
    avg_distance = validate(model, val_dataset)
    print(f"Validation epoch {epoch}: avg_reward: {avg_distance}")

    lr_scheduler.step()

def train_batch(model, optimizer, baseline, epoch, batch_id, step, batch):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, device="cuda:0")
    bl_val = move_to(bl_val, "cuda:0") if bl_val is not None else None

    cost, log_likelihood, pi = model(x)

    bl_val, bl_loss = baseline.eval(x, cost)

    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
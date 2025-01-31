import os
import config
import torch
import torch.optim as optim
from train import train_epoch, validate, get_routes
from model import AttentionModel, ExpBaseline
from dataset import CVRPDataset

def run(eval):
    torch.manual_seed(11)

    device = torch.device("cuda:0")

    problem = CVRPDataset()

    model = AttentionModel(
        config.EMBEDDING_DIM,
        config.HIDDEN_DIM,
        problem,
        n_encode_layers=config.N_ENCODE_LAYERS,
        mask_inner=True,
        mask_logits=True,
    ).to(device)
    
    baseline = ExpBaseline(config.EXP_BETA)

    optimizer = optim.Adam([{'params': model.parameters(), 'lr': config.LEARNING_RATE}])

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: config.LR_DECAY ** epoch)

    val_dataset = CVRPDataset(samples=1024, n=50)

    if eval:
        val_dataset = CVRPDataset(samples=10, n=50)
        checkpoint = torch.load(os.path.join(config.SAVE_DIR, config.MODEL_NAME), map_location=device)
        model.load_state_dict(checkpoint['model'])
        baseline.load_state_dict(checkpoint['baseline'])
        avg_score = validate(model, val_dataset).item()
        routes = get_routes(model, val_dataset)
        print(avg_score)
    else:
        for epoch in range(config.N_EPOCHS):
            train_epoch(model, optimizer, baseline, lr_scheduler,epoch, val_dataset, problem)


if __name__ == "__main__":
    run(eval=False)
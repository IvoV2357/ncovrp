import os
import config
import torch
import torch.optim as optim
from train import train_epoch, validate, get_routes
from model import AttentionModel, ExpBaseline
from dataset import CVRPDataset
import sys

def splitZeros(lst):
    result = []
    sublst = []
    for el in lst:
        if el != 0:
            sublst.append(el)
        else:
            result.append(sublst)
            sublst = []
    return result

def printRoutes(instances):
    for instance in instances:
        routes, depot, coordinates = instance
        separated_routes = splitZeros(routes.tolist())
        print("==================================================================")
        i = 1
        for route in separated_routes:
            print(f"Route {i}: 0->{'->'.join([str(el) for el in route])}->0")
            i += 1
        print(f"Depot: {' '.join([str(el) for el in depot.tolist()])}")
        node = 1
        for row in coordinates.tolist():
            print(f"Node {node}: {' '.join([str(el) for el in row])}")
            node += 1
        print("==================================================================")  

def run(eval):
    torch.manual_seed(11)

    device = torch.device(config.DEVICE)

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
        no_samples = 10
        val_dataset = CVRPDataset(samples=no_samples, n=50)
        checkpoint = torch.load(os.path.join(config.SAVE_DIR, config.MODEL_NAME), map_location=device)
        model.load_state_dict(checkpoint['model'])
        baseline.load_state_dict(checkpoint['baseline'])
        avg_score = validate(model, val_dataset).item()
        routes = get_routes(model, val_dataset)
        printRoutes(routes)
        print(f"Average score on all {no_samples} instances is {round(avg_score, 4)}")
    else:
        for epoch in range(config.N_EPOCHS):
            train_epoch(model, optimizer, baseline, lr_scheduler,epoch, val_dataset, problem)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py [train/eval]")
    if sys.argv[1] == 'eval':
        run(eval=True)
    elif sys.argv[1] == 'train':
        run(eval=False)
    else:
        print("Usage: python run.py [train/eval]")
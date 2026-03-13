import argparse
import torch
from torch_geometric import seed_everything
from tqdm import tqdm
import sys
from pathlib import Path
import random
import os
import numpy as np
import os.path as osp
import logging

import mlflow
import mlflow.pytorch

from src.dataset import load_dataset
from src.history import History
from src.loader import EventLoader
from src.measure import Measure
from src.loss import contrastive_loss, cosine_similarity

from src.models.model_arch import EnhancedTemporalGNN


############################################################
# Seed
############################################################

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


############################################################
# Training
############################################################

def train(loader, model, history, optimizer, device, data, args):
    model.train()
    history.train()

    pbar = tqdm(loader)
    total_loss = 0

    for batch in pbar:

        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)

        src = batch.src[: batch.batch_size]
        out = out[src]

        idx = batch.input_id.cpu()

        num_neg = batch.batch_size

        neg_mem = history.get_history(
            torch.randint(0, history.num_nodes, size=(num_neg,))
        )

        raw_msg = torch.zeros(idx.size(0), 1).to(device)

        cur_mem, prev_mem = history(raw_msg, data.src[idx])

        if args.con == 'his':

            loss = contrastive_loss(prev_mem, cur_mem, args.tau)

            loss += torch.exp(
                cosine_similarity(cur_mem, neg_mem)
            ).sum(dim=1).log().mean()

        elif args.con == 'stru':

            loss = contrastive_loss(out, cur_mem, args.tau)

            loss += torch.exp(
                cosine_similarity(out, neg_mem)
            ).sum(dim=1).log().mean()

        elif args.con == 'all':

            loss = args.alpha * contrastive_loss(out, cur_mem, args.tau) + \
                   contrastive_loss(prev_mem, cur_mem, args.tau)

            loss += torch.exp(
                cosine_similarity(cur_mem, neg_mem)
            ).sum(dim=1).log().mean()

            loss += args.alpha * torch.exp(
                cosine_similarity(out, neg_mem)
            ).sum(dim=1).log().mean()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        pbar.set_description(f"Train Loss = {loss.item():.4f}")

    return total_loss / len(loader)


############################################################
# Testing
############################################################

@torch.no_grad()
def test(loader, model, history, measure, device, data):

    preds = []
    labels = []

    model.eval()
    history.eval()

    for batch in tqdm(loader):

        torch.cuda.empty_cache()

        batch = batch.to(device)

        out = model(batch)

        src = batch.src[: batch.batch_size]
        out = out[src]

        label = batch.y[: batch.batch_size]

        idx = batch.input_id.cpu()

        raw_msg = torch.zeros(idx.size(0), 1).to(device)

        cur_mem, prev_mem = history(
            raw_msg,
            data.src[idx],
            update=True
        )

        if args.con == 'his':

            pred = 1 - torch.diag(
                cosine_similarity(cur_mem, prev_mem)
            ).view(-1)

        elif args.con == 'stru':

            pred = 1 - torch.diag(
                cosine_similarity(out, cur_mem)
            ).view(-1)

        elif args.con == 'all':

            p1 = torch.diag(
                cosine_similarity(out, cur_mem)
            ).view(-1)

            p2 = torch.diag(
                cosine_similarity(cur_mem, prev_mem)
            ).view(-1)

            pred = (1 - p1) + (1 - p2)

        pred = pred.sigmoid()

        preds.append(pred.detach().cpu())
        labels.append(label.detach().cpu())

    preds = torch.cat(preds)
    preds[torch.isnan(preds)] = 0.0

    labels = torch.cat(labels)

    return measure(labels[labels != 2], preds[labels != 2])


############################################################
# Argument Parser
############################################################

def read_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='wikipedia')
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--con", type=str, default='all')
    parser.add_argument("--num_neighbors", nargs="+", type=int)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_timeslots", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--nums_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--history_retrieve", type=str, default='mean')
    parser.add_argument("--recurrent", type=str, default='gru')
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--zero_edge", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    print(args)

    return args


############################################################
# Main
############################################################

args = read_parser()

if not os.path.exists('log'):
    os.makedirs('log')

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filename=f'log/{osp.basename(__file__)[:-3]}_{args.dataset}.txt',
    force=True
)

log = logging.getLogger()

log.info('########################## START #########################')
log.info(f'\n{args.dataset}-param: {args.__dict__}')


############################################################
# MLflow Setup
############################################################

mlflow.set_experiment(f"{args.dataset}_TemporalGNN")

with mlflow.start_run(run_name="EnhancedTemporalGNN"):

    mlflow.log_params(vars(args))

    mlflow.set_tag("model", "EnhancedTemporalGNN")

    mlflow.set_tag(
        "experiment_description",
        "Temporal GNN with memory history and contrastive loss"
    )


############################################################
# Data
############################################################

    data = load_dataset(args.dataset, zero_edge=args.zero_edge)

    device = torch.device(f"cuda:{args.gpu}")

    print(f"Using {device}")

    set_seed(42)

    sample_args = {
        "num_neighbors": args.num_neighbors,
        "num_workers": args.num_workers
    }

    train_loader = EventLoader(
        data,
        input_events=data.train_mask,
        shuffle=True,
        batch_size=args.batch_size,
        **sample_args
    )

    valid_loader = EventLoader(
        data,
        input_events=data.val_mask,
        shuffle=False,
        batch_size=args.batch_size,
        **sample_args
    )

    test_loader = EventLoader(
        data,
        input_events=data.test_mask,
        shuffle=False,
        batch_size=args.batch_size,
        **sample_args
    )


############################################################
# History
############################################################

    history = History(
        data.num_nodes,
        args.num_timeslots,
        args.hidden_channels,
        device=device,
        history_retrieve=args.history_retrieve,
        recurrent=args.recurrent,
    ).to(device)


############################################################
# Model
############################################################

    model = EnhancedTemporalGNN(
        num_nodes=data.num_nodes,
        in_dim=data.x.size(1),
        edge_dim=data.msg.size(1),
        hidden_dim=args.hidden_channels,
        num_layers=args.nums_layers,
        heads=args.heads,
        dropout=args.dropout,
        window=8,
        memory_momentum=0.9,
        t2v_dim=32
    ).to(device)

    print(model)

    mlflow.log_text(str(model), "model_architecture.txt")


############################################################
# Optimizer
############################################################

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(history.parameters()),
        lr=args.lr
    )

    measure = Measure("auc")

    best = 0
    best_val = 0


############################################################
# Training Loop
############################################################

    for epoch in range(1, 1 + args.epochs):

        loss = train(train_loader, model, history,
                     optimizer, device, data, args)

        val_auc = test(valid_loader, model, history,
                       measure, device, data)

        test_auc = test(test_loader, model, history,
                        measure, device, data)

        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("val_auc", val_auc, step=epoch)
        mlflow.log_metric("test_auc", test_auc, step=epoch)

        if val_auc > best_val:

            best_val = val_auc
            best = test_auc

            torch.save(model.state_dict(), "best_model.pt")

            mlflow.log_artifact("best_model.pt")

            mlflow.log_metric("best_val_auc", best_val)
            mlflow.log_metric("best_test_auc", best)

        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
            f"Val AUC: {val_auc:.2%}, "
            f"Test AUC: {test_auc:.2%}, "
            f"Best AUC: {best:.2%}"
        )

        log.info(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
            f"Val AUC: {val_auc:.2%}, "
            f"Test AUC: {test_auc:.2%}, "
            f"Best AUC: {best:.2%}"
        )


############################################################
# Final Model
############################################################

    mlflow.pytorch.log_model(model, "final_model")
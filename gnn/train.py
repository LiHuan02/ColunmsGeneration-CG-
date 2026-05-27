#!/usr/bin/env python3
"""Train the bipartite GNN for VCSP column selection.

Training configuration from EBSCO paper Table 2:
  - K=1 message-passing iteration
  - Learning rate 1e-3
  - 1000 epochs, epoch size 32 batches, batch size 16
  - Adam optimizer
  - Weighted BCE loss (10:1 positive:negative)
  - φ, ψ: 2×32 MLPs with ReLU
  - out: 3-layer (32×32×1) with sigmoid

Usage:
    python -m gnn.train --data data_generation/training_data/test --epochs 200
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from gnn.bipartite_gnn import BipartiteGNN
from gnn.dataset import VCSPBipartiteDataset


def compute_metrics(logits_list, labels_list, mask_list, threshold=0.0):
    """Compute classification metrics on the full dataset.

    Args:
        logits_list: list of logit tensors from each sample
        labels_list: list of label tensors (only for new columns)
        mask_list: list of bool masks (true for new columns)
        threshold: logit threshold for positive prediction (0 = sigmoid(0) = 0.5)

    Returns:
        dict with recall, tnr, precision, balanced_accuracy
    """
    all_preds = []
    all_labels = []

    for logits, labels, mask in zip(logits_list, labels_list, mask_list):
        new_logits = logits[mask]  # only predict for new columns
        probs = torch.sigmoid(new_logits)
        preds = (probs > 0.5).long()
        all_preds.append(preds)
        all_labels.append(labels)

    if len(all_preds) == 0:
        return {'recall': 0, 'tnr': 0, 'precision': 0, 'balanced_accuracy': 0}

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().float()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().float()

    recall = tp / (tp + fn + 1e-10)
    tnr = tn / (tn + fp + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    balanced_acc = (recall + tnr) / 2.0

    return {
        'recall': recall.item(),
        'tnr': tnr.item(),
        'precision': precision.item(),
        'balanced_accuracy': balanced_acc.item(),
    }


def train_epoch(model, dataloader, optimizer, criterion, device, pos_weight):
    """Train for one epoch.

    Since graphs have different sizes, we can't batch them in the traditional sense.
    We accumulate gradients over batch_size graphs before stepping.
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    batch_size = 16  # Paper's batch size
    accumulated_logits = []
    accumulated_labels = []
    accumulated_masks = []

    for i, batch_data in enumerate(dataloader):
        # Handle both DataLoader and list input
        if isinstance(batch_data, list):
            samples = batch_data
        else:
            samples = [batch_data]

        sample_losses = []

        for sample in samples:
            col_feat = sample['column_features'].to(device)
            constr_feat = sample['constraint_features'].to(device)
            edge_index = sample['edge_index'].to(device)
            labels = sample['labels'].to(device)
            new_mask = sample['new_col_mask'].to(device)

            if new_mask.sum() == 0:
                continue

            # Forward pass
            logits = model(col_feat, constr_feat, edge_index)

            # Loss only on new columns
            new_logits = logits[new_mask]
            # Weighted BCE: apply per-sample weights
            loss = criterion(new_logits, labels.float())

            sample_losses.append(loss)

            # For metrics
            accumulated_logits.append(logits.detach().cpu())
            accumulated_labels.append(labels.detach().cpu())
            accumulated_masks.append(new_mask.detach().cpu())

        if len(sample_losses) == 0:
            continue

        # Average loss over samples in this batch, then backprop
        batch_loss = torch.stack(sample_losses).mean()
        batch_loss.backward()

        # Step optimizer every batch_size samples
        if (i + 1) % batch_size == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += batch_loss.item() * len(sample_losses)
        n_samples += len(sample_losses)

    metrics = compute_metrics(accumulated_logits, accumulated_labels, accumulated_masks)

    return total_loss / max(n_samples, 1), metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_logits = []
    all_labels = []
    all_masks = []

    for batch_data in dataloader:
        if isinstance(batch_data, list):
            samples = batch_data
        else:
            samples = [batch_data]

        for sample in samples:
            col_feat = sample['column_features'].to(device)
            constr_feat = sample['constraint_features'].to(device)
            edge_index = sample['edge_index'].to(device)
            labels = sample['labels'].to(device)
            new_mask = sample['new_col_mask'].to(device)

            if new_mask.sum() == 0:
                continue

            logits = model(col_feat, constr_feat, edge_index)
            new_logits = logits[new_mask]
            loss = criterion(new_logits, labels.float())

            total_loss += loss.item()
            n_samples += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(new_mask.cpu())

    metrics = compute_metrics(all_logits, all_labels, all_masks)
    return total_loss / max(n_samples, 1), metrics


def main():
    parser = argparse.ArgumentParser(description='Train GNN for VCSP column selection')
    parser.add_argument('--data', type=str, default='data_generation/training_data/test',
                        help='Training data directory')
    parser.add_argument('--output', type=str, default='gnn/models',
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--pos-weight', type=float, default=10.0,
                        help='Weight for positive class in BCE loss (default: 10)')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='Hidden dimension for GNN (default: 32)')
    parser.add_argument('--num-iterations', type=int, default=1,
                        help='GNN message-passing iterations K (default: 1)')
    parser.add_argument('--val-split', type=float, default=0.25,
                        help='Validation split ratio (default: 0.25)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable feature normalization')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    full_dataset = VCSPBipartiteDataset(args.data, normalize=not args.no_normalize)
    print(f"Total samples: {len(full_dataset)}")

    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=VCSPBipartiteDataset.collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=VCSPBipartiteDataset.collate_fn, num_workers=0
    )

    # Create model
    model = BipartiteGNN(
        col_feat_dim=12, constr_feat_dim=2,
        hidden_dim=args.hidden_dim, num_iterations=args.num_iterations,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Weighted BCE loss
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_bal_acc = 0.0
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Training GNN for {args.epochs} epochs")
    print(f"  pos_weight={args.pos_weight}, lr={args.lr}")
    print(f"  hidden_dim={args.hidden_dim}, K={args.num_iterations}")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        start_t = time.time()

        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, args.pos_weight
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_t

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:4d}/{args.epochs} | "
                  f"T Loss: {train_loss:.4f} | V Loss: {val_loss:.4f} | "
                  f"Bal Acc: {val_metrics['balanced_accuracy']:.3f} | "
                  f"Recall: {val_metrics['recall']:.3f} | "
                  f"Prec: {val_metrics['precision']:.3f} | "
                  f"Time: {elapsed:.1f}s")

        # Save best model
        if val_metrics['balanced_accuracy'] > best_val_bal_acc:
            best_val_bal_acc = val_metrics['balanced_accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'feature_stats': full_dataset._feature_stats,
            }, os.path.join(args.output, 'best_model.pt'))
            print(f"  -> Saved best model (bal_acc={best_val_bal_acc:.4f})")

    # Final evaluation
    print(f"\n{'=' * 60}")
    print(f"Training complete")
    print(f"Best validation balanced accuracy: {best_val_bal_acc:.4f}")

    # Load best model for final metrics
    checkpoint = torch.load(os.path.join(args.output, 'best_model.pt'),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final metrics on train and val
    _, final_train_metrics = evaluate(model, train_loader, criterion, device)
    _, final_val_metrics = evaluate(model, val_loader, criterion, device)

    print(f"\nFinal Metrics (Best Model):")
    print(f"{'Metric':<20} {'Train':>10} {'Val':>10}")
    print(f"{'-' * 40}")
    for metric_name in ['recall', 'tnr', 'precision', 'balanced_accuracy']:
        print(f"{metric_name:<20} {final_train_metrics[metric_name]:>10.4f} "
              f"{final_val_metrics[metric_name]:>10.4f}")

    # Paper reference values (Table 3):
    print(f"\nPaper Reference (Table 3, VCSP):")
    print(f"  Recall: 86.2%   TNR: 66.6%   Precision: 23.7%   Bal Acc: 76.5%")

    # Save normalization stats for inference
    if full_dataset._feature_stats is not None:
        full_dataset.save_normalization_stats(
            os.path.join(args.output, 'norm_stats.npz')
        )


if __name__ == '__main__':
    main()

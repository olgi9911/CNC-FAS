# CNC-FAS: Contrastive Normality Constraint for One-Class Face Anti-Spoofing
This is a PyTorch implementation of the paper [Cross-modal Normality Constraint for Unsupervised Multi-class Anomaly Detection](https://ojs.aaai.org/index.php/AAAI/article/view/32856/35011). It treats FAS as a one-class anomaly detection problem, training only on "live" samples to learn a normality boundary using a Contrastive Normality Constraint (CNC).

Note: The `spoof_score` calculation in this implementation diverges from standard maximum-pixel methods. Instead, it computes the **mean** of the anomaly map for each layer and selects the maximum response across layers. This strategy was found to yield superior empirical performance for the FAS task.

## Project Structure
```text
CNC-FAS/
├── clip/                  # CLIP library
├── configs/               # YAML configuration files
│   └── cnc_fas.yaml       # Default training config
├── models/                # Model architecture
│   └── model.py           # Main CNC_FAS model
├── output/                # Training logs and checkpoints
├── scripts/               # Shell scripts for running experiments
│   └── OCI_to_M.sh        # Example: Train on OCI, Test on M
├── tools/                 # Utility scripts
│   └── evaluate.py        # Standalone evaluation script
├── trainers/              # Training logic
│   └── cnc.py             # CNCTrainer (Loss, Optimizer, Loop)
├── utils/                 # Helper functions
│   ├── config.py          # Config loading & parsing
│   └── logger.py          # Logging utility
├── dataloader.py          # FASDataset class (OULU, CASIA, etc.)
└── train.py               # Main entry point
```

## How to Run
To train the model using the standard O-C-I to M protocol:
```console
python train.py --config configs/cnc_fas.yaml --source OCI --target M
```
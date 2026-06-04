# GATsig — Graph Attention Network for Shear Jamming Classification

Translation of `GATsig_layer1head1.nb` (Mathematica 14.0) into PyTorch, with a full data pipeline and vast.ai deployment workflow.

## What the model does

Classifies 2D particle packings as **DST** (discontinuous shear thickening, φ=0.764) or **non-DST** (φ=0.752) using a single-head graph attention network with sigmoid-gated attention:

1. **Contact graph** — edges between particles whose centres are closer than the sum of their radii (no self-loops)
2. **Linear projection** — node features (5-dim) → embedding (10-dim) via weight matrix W
3. **Attention logits** — outer sum of per-node attention scores, passed through LeakyReLU(α=0.2)
4. **Sigma-gated attention** — non-contact edges masked with −10, then sigmoid (not softmax — this is the key architectural choice over vanilla GAT)
5. **Aggregation** — attention-weighted sum of projected features
6. **MLP readout** — two linear layers + softmax → scalar DST probability

## Project layout

```
gat_shearjamming/
├── src/
│   ├── model.py        # GATsig nn.Module
│   ├── data.py         # Dataset, DataLoader, .dat file parser
│   └── train.py        # Training loop with checkpointing
├── notebooks/
│   └── analyse.py      # Evaluation + plots after training
├── scripts/
│   ├── vast_deploy.sh  # Rent GPU, upload, launch on vast.ai
│   └── vast_pull.sh    # Download outputs when done
├── Dockerfile
├── requirements.txt
└── README.md
```

## Mathematica → Python translation map

| Mathematica | Python / PyTorch |
|---|---|
| `NetGraph[...]` | `nn.Module.forward()` |
| `NetArrayLayer[Output->{5,10}]` | `nn.Linear(5, 10, bias=False)` |
| `ThreadingLayer[If[dist<rsum, 1, 0]]` | `(dist < r_sum).float()` |
| `ElementwiseLayer[If[#>0, #, 0.2#]]` | `F.leaky_relu(x, 0.2)` |
| `ThreadingLayer[#1*#2 + (1-#1)*mconst]` | `A*e + (1-A)*mconst` |
| `ElementwiseLayer[1/(1+Exp[-#])]` | `torch.sigmoid(x)` |
| `DotLayer[]` | `@` (matmul) |
| `SoftmaxLayer[]` | `F.softmax(x, dim=0)` |
| `LinearLayer[n, Input->{n,k}]` | `nn.Linear(k, n)` |
| `NetTrain[..., LearningRate->0.0001]` | `Adam(lr=1e-4)` |
| `ValidationSet->Scaled[0.2]` | `random_split(..., [0.8, 0.2])` |

## Local quick-start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (local CPU/GPU)
python src/train.py \
    --data_dir /path/to/dat/files \
    --epochs 100 \
    --output_dir outputs/

# Analyse results
python notebooks/analyse.py \
    --checkpoint outputs/best.pt \
    --data_dir /path/to/dat/files
```

## Scaling up on vast.ai

### One-time setup
```bash
pip install vastai
vastai set api-key YOUR_API_KEY_FROM_VAST_AI_CONSOLE
```

### Deploy
```bash
chmod +x scripts/*.sh
./scripts/vast_deploy.sh /path/to/local/data
```

The script will:
1. Search for a cheap GPU (≥16GB VRAM, ≤$0.50/hr)
2. Show you the top options and ask for confirmation
3. Create the instance, wait for it to be ready
4. Rsync your code + data
5. Launch training in a `tmux` session
6. Print commands to monitor + pull results

### Monitor
```bash
# Live log
ssh -p PORT root@HOST -t 'tmux attach -t train'

# Or tail the log file
ssh -p PORT root@HOST 'tail -f /workspace/gat_shearjamming/outputs/training.log'
```

### Download results
```bash
./scripts/vast_pull.sh          # reads .vast_instance automatically
```

### Destroy instance (stop billing!)
```bash
vastai destroy instance INSTANCE_ID
```

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `--epochs` | 100 | Mathematica default not specified; 100-200 typical |
| `--lr` | 1e-4 | Matches Mathematica `LearningRate->0.0001` |
| `--newfdim` | 10 | Embedding dimension; try 16, 32 for scaling |
| `--n_nodes` | 2000 | Particles per packing |
| `--fdim` | 5 | Input feature dimension |
| `--mconst` | -10 | Non-contact masking constant |
| `--alpha` | 0.2 | LeakyReLU slope |

## Extending the model

**Multi-head attention** — wrap `GATsig` attention block in a loop over `n_heads`, concatenate or average outputs.

**Multi-layer GAT** — stack `GATsig` blocks, passing `H_agg` as `features` to the next layer.

**More data files** — add entries to the `files` list in `train.py` or pass `--non_dst_file` / `--dst_file` flags.

**Larger graphs** — the main memory bottleneck is the (N×N) contact/attention matrix. For N=2000 this is 16MB per sample in float32 — fine on any GPU. For N>10000 consider sparse attention.

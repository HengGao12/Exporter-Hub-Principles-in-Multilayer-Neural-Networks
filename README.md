## Installation

Base Environment
- Python == 3.9
- PyTorch == 1.11.0
- CUDA == 11.3

Now we install the [SpikingJelly](https://github.com/fangwei123456/spikingjelly?tab=readme-ov-file#build-snn-in-an-unprecedented-simple-way).

```bash
pip install spikingjelly
```

## Usage

### ANN training

```bash
python train_ann.py
```

The pretrained four-layer ANN is saved in './model_output', the training log file is saved in './log_file'.

### ANN Prunning

With retrain
```bash
python pruning.py --net ann --neuron_type weak --retrain
python pruning.py --net ann --neuron_type mid --retrain
python pruning.py --net ann --neuron_type hub --retrain
```

Without retrain
```bash
python pruning.py --net ann --neuron_type weak
python pruning.py --net ann --neuron_type mid
python pruning.py --net ann --neuron_type hub
```

### SNN training

```bash
python train_snn.py
```

The pretrained four-layer SNN is saved in './model_output'

### SNN Prunning

With retrain
```bash
python pruning.py --net snn --neuron_type weak --retrain --exp_round 1
python pruning.py --net snn --neuron_type mid --retrain --exp_round 1
python pruning.py --net snn --neuron_type hub --retrain --exp_round 1
```

Without retrain
```bash
python pruning.py --net snn --neuron_type weak --exp_round 1
python pruning.py --net snn --neuron_type mid --exp_round 1
python pruning.py --net snn --neuron_type hub --exp_round 1
```
The pruning results will be saved in './pruning_results/'.

### Weight Distribution Visualization

```bash
# ANN
python weight_distribution_visualization.py --net ann
# SNN
python weight_distribution_visualization.py --net snn
```

### TSNE Plot

```bash
# ANN
python tsne.py --net ann --delete_type control --layer_num 1
# SNN
python tsne.py --net snn --delete_type control --layer_num 1
```

### Direct Pruning

Pruning without grouping neurons.

Without Retrain
```bash
# ANN
python direct_pruning.py --net ann
# SNN
python direct_pruning.py --net snn --exp_round 1
```

With Retrain
```bash
# ANN
python direct_pruning.py --net ann --retrain
# SNN
python direct_pruning.py --net snn --retrain --exp_round 1
```

### Remark 

All visualization results are saved in './figures/'

In '../../previous_results/', we saved the training log files and visualization results in our previous study.

## Acknowledgment

Our code is developed based on [SpikingJelly](https://github.com/fangwei123456/spikingjelly?tab=readme-ov-file#build-snn-in-an-unprecedented-simple-way). Thanks for their great work.
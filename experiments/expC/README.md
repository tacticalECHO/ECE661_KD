# Experiment C: Decoupled Knowledge Distillation on CIFAR-100

This directory contains the code and raw results for Ziheng Pan's Experiment C of our
ECE 661 Project 6. It compares five ways to train the same ResNet8x4 student:

- cross-entropy training on the ground-truth labels (CE);
- classical knowledge distillation (KD; Hinton et al., 2015);
- target-class knowledge distillation only (TCKD);
- non-target-class knowledge distillation only (NCKD); and
- full decoupled knowledge distillation (DKD; Zhao et al., 2022).

All five methods use the same CIFAR-100 split, student architecture, optimizer,
schedule, model-selection rule and run seeds. Every distillation run uses one frozen
ResNet32x4 teacher.

Shared code lives at the repository root and is not duplicated here:

| Path (repo root) | Contents |
|---|---|
| `models/cifar_resnet.py` | CIFAR ResNet basic block, ResNet8x4 student and ResNet32x4 teacher |
| `losses.py` | Classical KD, TCKD, NCKD and DKD losses |
| `utils.py` | Summary-table writing and shared evaluation utilities |
| `data/` | CIFAR-100, downloaded by torchvision on first use |

## Layout

```text
experiments/expC/
|-- prepare_data.py       # CIFAR-100 transforms, deterministic train/validation split
|-- train_teacher.py      # ResNet32x4 teacher training
|-- evaluate_teacher.py   # independent evaluation of a teacher checkpoint
|-- train_student.py      # unified CE, KD, TCKD, NCKD and DKD training
|-- FLOPs.py              # parameter and FLOP counts -> results/model_stats.csv
|-- plot_results.py       # aggregate tables and report/poster figures from CSV files
|-- results/              # per-epoch logs and summary tables
|-- figures/              # generated PDF and 300-dpi PNG figures
`-- checkpoints/          # local model checkpoints; *.pt files are not committed
```

Each training script adds the repository root to `sys.path` and changes the working
directory to `experiments/expC`. Relative `results/`, `figures/` and `checkpoints/`
paths are therefore resolved inside this experiment directory. Development checks and
planning notes are retained locally under the ignored `local_only/` directory and are
not required for training, evaluation or result regeneration.

## Environment

- Conda environment: `my-neuro`
- Python 3.11.13
- PyTorch 2.9.0 development build with CUDA 12.9
- torchvision 0.24.0 development build
- GPU used for the reported runs: NVIDIA GeForce RTX 5090

Activate the environment before running the commands below:

```bash
conda activate my-neuro
```

## Models

Both networks are basic-block CIFAR ResNets for 32 x 32 images. They use a 3 x 3 stem,
three residual stages with widths 64, 128 and 256, global 8 x 8 average pooling, and a
100-class linear classifier. ResNet8x4 has one residual block per stage; ResNet32x4 has
five. The definitions follow the model family used by the official DKD implementation.

| Role | Architecture | Parameters | MACs | FLOPs |
|---|---|---:|---:|---:|
| Teacher | ResNet32x4 | 7,433,860 | 1,083.04 M | 2,166.08 M |
| Student | ResNet8x4 | 1,233,540 | 177.07 M | 354.14 M |

FLOPs are counted as 2 x MACs. Only Conv2d and Linear operations are counted;
activations, normalization, pooling and biases are omitted. Run `FLOPs.py` to reproduce
the values stored in `results/model_stats.csv`.

## Data and evaluation protocol

- **Splits.** The 50,000 CIFAR-100 training images are deterministically divided into
  45,000 training images and 5,000 model-selection validation images with
  `split_seed=42`. The official 10,000-image test set is kept separate.
- **Preprocessing.** Images are normalized with mean `(0.5071, 0.4867, 0.4408)` and
  standard deviation `(0.2675, 0.2565, 0.2761)`.
- **Augmentation.** Training uses a random 32 x 32 crop with four pixels of padding and
  random horizontal flipping. Validation and test images are only normalized.
- **Model selection.** Each run saves the checkpoint with the highest validation
  accuracy. The test set is evaluated once after training with that checkpoint.
- **Seeds.** Every student method is run with seeds 42, 123 and 456. The teacher is
  trained once with seed 42. A seed controls model initialization and data order; the
  train/validation indices remain fixed.
- **Fair comparison.** For a given seed, all student methods start from the same
  initialization and use the same data order, optimizer, schedule and validation set.

## Training settings

| Setting | Teacher | Students |
|---|---:|---:|
| Optimizer | SGD | SGD |
| Initial learning rate | 0.05 | 0.05 |
| Momentum | 0.9 | 0.9 |
| Weight decay | 5e-4 | 5e-4 |
| Batch size | 64 | 64 |
| Epochs | 240 | 240 |
| LR milestones | 150, 180, 210 | 150, 180, 210 |
| LR decay at each milestone | 0.1 | 0.1 |

Classical KD uses temperature `T=4` and `alpha=0.9`:

```text
L_KD = (1 - alpha) * CE(student, y)
       + alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T))
```

TCKD, NCKD and DKD use `T=4` and a 20-epoch linear warmup. Their training objective is

```text
L = CE(student, y) + min(epoch / 20, 1) * (alpha * TCKD + beta * NCKD)
```

with the following component weights:

| Method | alpha (TCKD) | beta (NCKD) |
|---|---:|---:|
| TCKD | 1 | 0 |
| NCKD | 0 | 8 |
| DKD | 1 | 8 |

All KL terms include the `T^2` factor and use batch-mean scaling. The teacher remains in
evaluation mode, its parameters are frozen, and its logits are detached.

## Reproducing the results

Run these commands from the repository root. Add `--download` to the first training
command if CIFAR-100 is not already present under `data/`.

```bash
# 1. Model sizes
python experiments/expC/FLOPs.py

# 2. Train and independently evaluate the teacher
python experiments/expC/train_teacher.py --device cuda --download
python experiments/expC/evaluate_teacher.py --device cuda

# 3. Run all five student methods with all three seeds
for s in 42 123 456; do
  python experiments/expC/train_student.py --method ce   --seed $s --device cuda
  python experiments/expC/train_student.py --method kd   --seed $s --device cuda
  python experiments/expC/train_student.py --method tckd --seed $s --device cuda
  python experiments/expC/train_student.py --method nckd --seed $s --device cuda
  python experiments/expC/train_student.py --method dkd  --seed $s --device cuda
done

# 4. Validate the complete result matrix and regenerate tables and figures
python experiments/expC/plot_results.py --check-only
python experiments/expC/plot_results.py
```

The student training script also supports `--smoke`, which runs one epoch with two
training, validation and test batches. Existing formal outputs are protected unless
`--overwrite` is supplied.

## Output files

| File | Contents |
|---|---|
| `results/teacher_resnet32x4_best.csv` | Per-epoch teacher training and validation log |
| `results/student_resnet8x4_<method>_..._seed<seed>_best.csv` | 240-epoch student log for one method and seed |
| `results/summary.csv` | One row per teacher/student run with configuration, selected epoch and test result |
| `results/method_summary.csv` | Three-seed means, sample standard deviations and improvements over CE/KD |
| `results/paper_comparison.csv` | Project values, paper values and protocol note |
| `results/model_stats.csv` | Architecture, parameter, MAC and FLOP counts |

Student epoch logs share the following columns:

| Column | Meaning |
|---|---|
| `epoch`, `lr` | Epoch and active learning rate |
| `train_loss` | Total optimized loss |
| `train_ce_loss` | Cross-entropy component |
| `train_kd_loss` | Classical KD soft-target component; empty for other methods |
| `train_tckd_loss`, `train_nckd_loss` | Unweighted DKD components; empty for CE/KD |
| `distill_weight` | Active linear-warmup multiplier |
| `train_acc`, `val_acc` | Training and validation accuracy |
| `val_loss` | Validation cross-entropy loss |

`plot_results.py` requires exactly 15 student runs, checks all 3,600 student epochs,
and verifies that every `best_epoch` in `summary.csv` matches the corresponding maximum
validation accuracy in the epoch log.

## Figures

`python experiments/expC/plot_results.py` reads the CSV files in `results/` and writes
each figure as a PDF for the report and a 300-dpi PNG for the poster.

| Figure | Contents | Source |
|---|---|---|
| `test_error_comparison` | Mean test errors, sample SD and individual seeds for all five methods | `summary.csv` |
| `validation_curves` | Mean validation accuracy and loss; thin lines show individual seeds | Per-epoch student logs |
| `dkd_components` | Unweighted TCKD and NCKD trajectories, with warmup shaded | TCKD, NCKD and DKD logs |

## Results summary

Test accuracy is reported as mean +/- sample standard deviation over seeds 42, 123 and
456. Test errors are out of 10,000 images.

| Method | Seed 42 | Seed 123 | Seed 456 | Mean +/- SD | Test errors, mean +/- SD |
|---|---:|---:|---:|---:|---:|
| CE | 71.80% | 71.82% | 71.72% | 71.78% +/- 0.05 | 2,822.0 +/- 5.3 |
| KD | 73.00% | 73.30% | 72.42% | 72.91% +/- 0.45 | 2,709.3 +/- 44.7 |
| TCKD | 68.17% | 68.72% | 68.03% | 68.31% +/- 0.36 | 3,169.3 +/- 36.5 |
| NCKD | 74.37% | 74.31% | 74.46% | 74.38% +/- 0.08 | 2,562.0 +/- 7.5 |
| DKD | **75.51%** | **75.15%** | **74.70%** | **75.12% +/- 0.41** | **2,488.0 +/- 40.6** |

The frozen ResNet32x4 teacher achieves 78.65% test accuracy (2,135 errors). Its best
validation checkpoint is epoch 223, with 78.24% validation accuracy.

The main comparisons are:

- classical KD improves on CE by 1.13 percentage points;
- NCKD improves on CE by 2.60 points and on classical KD by 1.47 points;
- TCKD alone is 3.47 points below CE;
- full DKD improves on CE by 3.34 points and on classical KD by 2.21 points; and
- adding TCKD to NCKD improves mean accuracy by a further 0.74 points.

Under this protocol, NCKD is the main source of the logit-distillation gain. TCKD alone
is not sufficient, but it provides an additional benefit when combined with NCKD in the
full DKD objective.

## Comparison with the DKD paper

The paper reports the same ResNet32x4 teacher and ResNet8x4 student pair. Its values are
averages over five trials; this project uses three student seeds and one teacher run.

| Model/method | This project | Zhao et al. (2022) | Project - paper |
|---|---:|---:|---:|
| Teacher | 78.65% | 79.42% | -0.77 pp |
| Student (CE) | 71.78% +/- 0.05 | 72.50% | -0.72 pp |
| KD | 72.91% +/- 0.45 | 73.33% | -0.42 pp |
| DKD | 75.12% +/- 0.41 | 76.32% | -1.20 pp |

The ranking and the advantage of DKD over KD agree with the paper. Exact numerical
agreement is not expected because this project reserves 5,000 training images for model
selection, uses a separately trained teacher that is 0.77 points weaker, and averages
three rather than five student trials.

## Notes and limitations

- Only three student seeds are used. NCKD is unusually stable here, but KD and DKD have
  larger seed-to-seed variation.
- The teacher is trained once. Teacher-run variance is therefore not measured.
- The fixed 45k/5k split leaves fewer images for optimization than protocols that train
  on all 50,000 CIFAR-100 training images.
- The test set is used only after validation-based model selection, so no test-time
  tuning is performed.
- TCKD and NCKD ablations retain the same `alpha=1`, `beta=8` scales used by full DKD.
  They answer the component-ablation question under this fixed protocol rather than
  independently tuning each component for its best possible result.
- Checkpoints are intentionally excluded from version control. The committed epoch
  logs, run-level summary, derived tables and figures preserve the reported results.

## References

- Hinton, Vinyals and Dean, "Distilling the Knowledge in a Neural Network," 2015.
- Zhao et al., "Decoupled Knowledge Distillation," CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html
- Official DKD implementation:
  https://github.com/megvii-research/mdistiller

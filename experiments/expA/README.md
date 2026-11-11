# Experiment A: Classical Knowledge Distillation on MNIST

This directory contains the code and raw results for Experiment A of our ECE 661 Project 6
(knowledge distillation). It covers the following experiments:

- a CNN teacher and CNN students, with parameter counts and FLOPs;
- the same student trained on ground-truth labels (CE) and with classical KD (Hinton et al., 2015);
- a sweep over the distillation temperature T;
- the missing-digit experiment, where one digit is removed from the transfer set;
- student capacity: three student sizes, each trained with CE and with KD;
- dropout: a dropout-trained teacher for KD, and dropout added to the CE-trained student.

Shared code lives at the repository root and is not duplicated here:

| Path (repo root) | Contents |
|---|---|
| `models/mnist.py` | `MNISTModel` and the teacher/student CNNs, including the dropout variants |
| `losses.py` | `kd_loss`, the classical KD loss |
| `utils.py` | `evaluate_per_class` and `append_summary` (writes a summary row and checks its header) |
| `data/` | MNIST, downloaded by torchvision on first use |

## Layout

```
experiments/expA/
├── prepare_data.py              # MNIST loaders, train/val split, class exclusion
├── train_teacher.py             # teacher (CE, with input jitter)
├── train_teacher_dropout.py     # teacher with dropout in the classifier
├── train_student_ce.py          # student trained on ground-truth labels
├── train_student_ce_dropout.py  # same, with dropout in the student
├── train_student_kd.py          # student trained with classical KD
├── train_student_kd_dropout.py  # KD from a dropout-trained teacher
├── evaluate_teacher.py          # test-set evaluation of a teacher checkpoint
├── check_kd.py                  # sanity check of kd_loss at several temperatures
├── check_gpu.py                 # forward/backward smoke test on the GPU
├── FLOPs.py                     # parameter and FLOP counts -> results/model_stats.csv
└── results/                     # per-run training logs and summary tables (CSV)
```

Each script adds the repository root to `sys.path` and changes the working directory to
`experiments/expA`. The scripts can therefore be launched from any directory, and every relative
path (`results/`, `checkpoints/`, and the `--teacher` and `--output` arguments) is resolved
relative to `experiments/expA`. Checkpoints (`checkpoints/*.pt`) are not committed.

## Environment

- Python 3.12, PyTorch 2.11.0 (CUDA 12.8), torchvision 0.26.0
- A CUDA GPU is required, because the device is hard-coded to `cuda`. Our runs used a single
  RTX 5090 Laptop GPU.

## Models

All models use the same template: conv3x3(c1) → ReLU → maxpool → conv3x3(c2) → ReLU → maxpool →
Linear(c2·7·7, h) → ReLU → Linear(h, 10).

| Model | Class | (c1, c2) / h | Params | MACs | FLOPs |
|---|---|---|---|---|---|
| Teacher | `TeacherCNN` | (64, 128) / 256 | 1,682,954 | 16.51 M | 33.02 M |
| Student S | `StudentCNN` | (8, 16) / 64 | 52,138 | 0.333 M | 0.666 M |
| Student XS | `StudentCNNExs` | (4, 8) / 32 | 13,242 | 0.098 M | 0.195 M |
| Student XXS | `StudentCNNExs2` | (2, 4) / 16 | 3,418 | 0.032 M | 0.063 M |

FLOPs are counted as 2 × MACs, and only Conv2d and Linear layers are counted. Activations,
pooling and biases are ignored. The numbers are produced by `FLOPs.py` and stored in
`results/model_stats.csv`.

The dropout variants (`TeacherCNNDropout`, `StudentCNNDropout`, ...) insert `nn.Dropout(p)` after
the first ReLU of the classifier. Apart from that, they are identical to the base models.

## Data and evaluation protocol

- **Splits.** The 60,000 MNIST training images are split into 55,000 for training and 5,000 for
  validation with a fixed `split_seed=42`. The split is the same for every run. The test set is
  always the full 10,000 images.
- **Preprocessing.** Normalization uses mean 0.1307 and std 0.3081. Only the teacher is trained
  with augmentation: a random translation of up to ±2 px (`RandomAffine`, translate 2/28).
  Students are trained without augmentation.
- **Model selection.** Every run keeps the checkpoint with the best validation accuracy, and the
  test set is evaluated once on that checkpoint.
- **Seeds.** Every student configuration is trained with seeds 42, 123 and 456. The seed controls
  the initialization and the shuffling. The teacher is trained once, with seed 42.
- **Missing digit.** Digit 1 was chosen at random with `random.Random(42).randint(0, 9)`. With the
  `--exclude-classes 1` option, digit 1 is removed from both the training split and the
  model-selection validation split, so model selection never sees the omitted digit. After
  training, the selected checkpoint is additionally evaluated on the complete 5,000-image
  validation split, with all ten digits (`*_full_val.csv`), and on the full test set. The
  teacher is always trained on all ten digits.

## Training settings

| | Teacher | Students (CE and KD) |
|---|---|---|
| Optimizer | Adam, lr 1e-3 | Adam, lr 5e-3 |
| Epochs | 20 | 60 |
| Batch size | 128 | 128 |
| Augmentation | ±2 px translation | none |

The KD loss (`losses.py`) is

    L = (1 − α) · CE(student, y) + α · T² · KL(softmax(t/T) ‖ softmax(s/T))

with α = 0.9 and `batchmean` reduction. The teacher's logits are detached, and the teacher is
frozen in eval mode.

## Reproducing the results

Run the following commands from `experiments/expA`. The scripts change into this directory
themselves, so this only matters for how you write relative paths on the command line.

```bash
# 1. Teacher, then freeze the checkpoint used by all KD runs
python train_teacher.py                      # -> checkpoints/teacher_best.pt
cp checkpoints/teacher_best.pt checkpoints/teacher_final.pt
python evaluate_teacher.py --teacher checkpoints/teacher_final.pt

# 2. Model size and FLOPs
python FLOPs.py                              # -> results/model_stats.csv

# 3. CE baseline vs. KD, and the temperature sweep (StudentCNN)
for s in 42 123 456; do
  python train_student_ce.py --seed $s
  for T in 1 2 4 8 20; do
    python train_student_kd.py --temperature $T --seed $s
  done
done

# 4. Missing digit (digit 1 removed from the transfer set)
for s in 42 123 456; do
  python train_student_ce.py --exclude-classes 1 --seed $s
  for T in 4 20; do
    python train_student_kd.py --temperature $T --exclude-classes 1 --seed $s
  done
done

# 5. Student capacity (T = 4)
for st in StudentCNNExs StudentCNNExs2; do
  for s in 42 123 456; do
    python train_student_ce.py --student $st --seed $s
    python train_student_kd.py --student $st --temperature 4 --seed $s
  done
done

# 6. Dropout (p = 0.3 and 0.5)
for p in 0.3 0.5; do
  # teacher with dropout; the file name is set from --dropout
  python train_teacher_dropout.py --dropout $p --output checkpoints/teacher_dropout${p}_best.pt
  cp checkpoints/teacher_dropout${p}_best.pt checkpoints/teacher_dropout${p}_final.pt
  for s in 42 123 456; do
    # KD from the dropout teacher (the student has no dropout; p is read from the checkpoint)
    python train_student_kd_dropout.py --teacher checkpoints/teacher_dropout${p}_final.pt \
        --temperature 4 --seed $s
    # CE student with dropout
    python train_student_ce_dropout.py --dropout $p --seed $s
  done
done
```

Each student run appends one row to `results/summary.csv`, or to `results/summary_dropout.csv`
for the dropout scripts. `append_summary` refuses to append a row whose columns do not match
the existing header. The teacher rows in both summary tables were added after training, from
the test evaluation of the frozen teacher checkpoints.

## Output files

| File | Contents |
|---|---|
| `results/<run_name>.csv` | Per-epoch log: `epoch, train_loss, [train_ce_loss, train_kd_loss,] train_acc, val_loss, val_acc` (the CE and KD loss terms are logged for KD runs only) |
| `results/<run_name>_full_val.csv` | Missing-digit runs only: overall and per-digit accuracy and errors of the selected checkpoint on the complete validation split |
| `results/summary.csv` | One row per run of the main experiments (teacher, CE, KD, temperature, missing digit, capacity) |
| `results/summary_dropout.csv` | Same columns plus `dropout`, for the dropout experiments |
| `results/model_stats.csv` | `model, student, params, macs, flops` |

Columns of the summary tables:

| Column | Meaning |
|---|---|
| `run_name` | Unique run identifier (see the naming note below) |
| `method` | `teacher`, `ce` or `kd` |
| `T`, `alpha` | KD temperature and weight; empty for CE and teacher rows |
| `optimizer`, `lr`, `batch_size`, `epochs`, `seed` | Training settings |
| `exclude_classes` | Digits removed from training and model selection; empty if none |
| `best_epoch`, `best_val_acc` | Epoch and accuracy of the selected checkpoint on the model-selection validation split |
| `test_acc`, `test_errors`, `test_loss` | Full 10,000-image test set |
| `params` | Number of trainable parameters |
| `student` | Model class |
| `acc_0` … `acc_9` | Per-digit test accuracy |
| `teacher` | Teacher checkpoint used for KD; empty otherwise |
| `dropout` | (`summary_dropout.csv` only) dropout rate of the trained model itself. It is `0.0` for KD students, whose teacher is named in `teacher`, and equals the teacher's rate for teacher rows |

Naming note: the earliest runs were logged before the `--student` option existed. Their names
have no `_StudentCNN` suffix (for example `student_kd_T4_alpha0.9_seed42` and
`student_ce_seed42_exclude_1`). The `student` column is filled for every row.

## Results summary

Test errors out of 10,000 are reported as mean ± sample standard deviation over seeds 42, 123
and 456. All numbers come from the summary tables above.

**Teacher and baseline.** The teacher makes 68 test errors (99.32%; validation accuracy
99.44%). The StudentCNN trained with CE makes 123.0 ± 14.7 errors.

**Temperature sweep (StudentCNN, α = 0.9).**

| | CE | T = 1 | T = 2 | T = 4 | T = 8 | T = 20 |
|---|---|---|---|---|---|---|
| Test errors | 123.0 ± 14.7 | 105.7 ± 14.7 | 86.3 ± 7.0 | 84.0 ± 6.6 | 88.7 ± 2.5 | 91.3 ± 6.8 |

**Missing digit 1** (1,135 test images of digit 1).

| | Digit-1 test accuracy | Full-validation digit-1 accuracy |
|---|---|---|
| CE | 0.0% (all seeds) | 0.0% |
| KD, T = 4 | 69.8 ± 12.3% | 72.6 ± 11.6% |
| KD, T = 20 | 65.5 ± 3.9% | 64.6 ± 6.0% |

**Student capacity (T = 4).**

| Student | Params | CE | KD |
|---|---|---|---|
| S | 52,138 | 123.0 ± 14.7 | 84.0 ± 6.6 |
| XS | 13,242 | 160.0 ± 7.0 | 120.0 ± 4.4 |
| XXS | 3,418 | 233.7 ± 17.7 | 208.7 ± 21.8 |

**Dropout (StudentCNN, T = 4 for KD).**

| | Test errors |
|---|---|
| Teacher, no dropout / p = 0.3 / p = 0.5 (single seed) | 68 / 60 / 54 |
| KD from the teacher without dropout | 84.0 ± 6.6 |
| KD from the p = 0.3 teacher | 79.0 ± 9.2 |
| KD from the p = 0.5 teacher | 72.3 ± 13.5 |
| CE, student with p = 0.3 | 110.3 ± 19.5 |
| CE, student with p = 0.5 | 110.7 ± 6.5 |

## Notes and limitations

- Only three seeds are used per configuration. The differences between T = 2, 4, 8 and 20 are
  smaller than the seed-to-seed spread, and so are the dropout effects on test error.
- The teacher is trained with a single seed. The main experiments use a teacher without dropout.
  The dropout teachers are also better teachers (54 and 60 vs. 68 errors), so their effect on the
  student cannot be separated from teacher quality.
- Both teachers are trained for 20 epochs, and the selected epoch is the last one for the
  teacher without dropout and for the p = 0.5 teacher. Longer training might improve them
  further.
- Students are trained with Adam, unlike the SGD setup of Hinton et al. With Adam, KD at T = 20
  trains stably without rescaling the learning rate.
- In the missing-digit experiment, model selection cannot see the omitted digit. This is
  intended, but it makes the digit-1 accuracy depend on which epoch happens to be selected,
  which explains the larger spread at T = 4.
- We did not apply the bias correction for the omitted class described by Hinton et al.

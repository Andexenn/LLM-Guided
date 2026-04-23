# MMIST & SYNAPSE Multimodal MIL Pipelines (LLM-Guided)

This documentation provides details on how to set up, configure, and run the LLM-guided multimodal Multiple Instance Learning (MIL) pipelines for the **MMIST** and **SYNAPSE** datasets.

---

## 1. Installation

### Environment Requirements
- **Python**: Recommended 3.9.12
- **CUDA**: Required for GPU acceleration (The code is tested with `CUDA_VISIBLE_DEVICES`).

### Required Libraries
Install the necessary packages using `pip`:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install numpy pandas scikit-learn scipy
pip install torcheval tensorboard
```

---

## 2. Directory Structure

The project is organized into dataset-specific source directories and centralized storage:

- `mmist-src/`: Source code, models, and configurations for the MMIST dataset.
- `synapse-src/`: Source code, models, and configurations for the SYNAPSE dataset.
- `feat/`: Contains extracted features (`feat-MMIST`, `feat-SYNAPSE`).
- `logs/`: Directory for execution logs and evaluation results.
- `checkpoint/`: Directory where `.pt` model checkpoints are saved.
- `run/`: TensorBoard event logs.
- `scripts/`: Utility scripts.

---

## 3. Path Configuration

Path management is handled in `path.py` within each source directory (`mmist-src/path.py` and `synapse-src/path.py`).

### Key Variables in `path.py`:
- `BASE_DIR`: The root directory of the project.
- `CHECKPOINT_DIR`: Base directory for saving model weights.
- `TENSORBOARD_DIR`: Base directory for TensorBoard logs.
- `LOG_DIR`: Base directory for execution logs.

### How to Configure:
To adapt the code to a new server environment, update the `BASE_DIR` at the top of `path.py`:

```python
# Example from mmist-src/path.py
BASE_DIR = "/your/custom/path/LLM-guided-Multimodal-MIL/"
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoint/mmist/v1.2')
TENSORBOARD_DIR = os.path.join(BASE_DIR, 'run/mmist/v1.2')
LOG_DIR = os.path.join(BASE_DIR, 'logs/mmist/v1.2')
```

---

## 4. Running the Code

### MMIST Pipeline
To run the full ablation study (multi-modal and single-modal combinations) for MMIST:

```bash
# Run the robust ablation pipeline
bash run_ablation_mmist.sh

# Or run a specific combination (e.g., CT_WSI_Clinical with MAM_G method)
bash run_ablation_mmist.sh CT_WSI_Clinical_MAM_G
```

### SYNAPSE Pipeline
To run the full ablation study for SYNAPSE:

```bash
# Run the robust ablation pipeline
bash run_ablation_synapse.sh
```

### Manual Execution (Advanced)
You can run the training or testing scripts directly with specific arguments:

```bash
cd mmist-src
python train_ddp.py \
    --ablation_mode CT_WSI_Clinical \
    --integration_method MAM_G \
    --gpu 0 \
    --batch_size 1 \
    --n_epochs 50 \
    --save_best
```

---

## 5. Server Resources & Paths

### Log Results Paths
- **MMIST Logs**: `/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/LLM-guided-Multimodal-MIL/logs/mmist/v1.2`
- **SYNAPSE Logs**: `/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/LLM-guided-Multimodal-MIL/logs/synapse/v1.2`

### Checkpoint Paths
- **MMIST Checkpoints**: `/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/LLM-guided-Multimodal-MIL/checkpoint/mmist/v1.2`
- **SYNAPSE Checkpoints**: `/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/LLM-guided-Multimodal-MIL/checkpoint/synapse/v1.2`

### Modality Feature Paths
- **MMIST Features**: `/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/LLM-guided-Multimodal-MIL/feat/feat-MMIST`
- **SYNAPSE Features**: `/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/LLM-guided-Multimodal-MIL/feat/feat-SYNAPSE`

---

## 6. Result Collection

After running the ablation studies, you can collect and summarize the results using the `collect_test_results.py` script:

```bash
python collect_test_results.py
```

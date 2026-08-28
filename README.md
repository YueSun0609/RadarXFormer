# RXF (RadarXFormer)

RXF is a multimodal 3D object detection project for the K-Radar dataset. It
combines monocular camera features with sparse 4D radar features and supports
single-GPU and distributed training/evaluation.

The default online sparse configuration performs Range-MAD sparsification while
loading each radar tensor, so sparse radar files do not need to be generated in
advance.

## Requirements

- Linux
- Python 3.8+
- CUDA-compatible PyTorch and torchvision
- spconv
- MMCV
- PyTorch3D

Install the Python dependencies and RXF in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Build the local 2D and 3D deformable-attention extensions if compatible
versions are not already installed:

```bash
cd Deformable-DETR-2D/models/ops && sh make.sh
cd ../../../Deformable-DETR-3D/models/ops && sh make.sh
cd ../../..
```

PyTorch3D source is included in `pytorch3d-stable/` for environments where a
matching prebuilt package is unavailable.

## Dataset layout

RXF expects a processed K-Radar directory similar to:

```text
processed/
├── train/
│   └── <sequence>/<sample>/
│       ├── mono.jpg
│       ├── rea.npy
│       ├── labels_rev1.npy
│       └── description.npy
└── test/
    └── <sequence>/<sample>/...
```

## Training

```bash
python -m rxf.train \
  --src /path/to/kradar/processed \
  --cfg config/kradar_our_c2_rev1_sparse_online.json \
  --dst ./logs
```

Resume from a state-dict checkpoint with `--checkpoint`:

```bash
python -m rxf.train \
  --src /path/to/kradar/processed \
  --cfg config/kradar_our_c2_rev1_sparse_online.json \
  --dst ./logs \
  --checkpoint /path/to/checkpoint.pt
```

For distributed training, launch the same module with `torchrun`:

```bash
torchrun --nproc_per_node=NUM_GPUS -m rxf.train \
  --src /path/to/kradar/processed \
  --cfg config/kradar_our_c2_rev1_sparse_online.json \
  --dst ./logs
```

## Evaluation

```bash
python -m rxf.evaluate \
  --src /path/to/kradar/processed \
  --cfg config/kradar_our_c2_rev1_sparse_online.json \
  --checkpoint /path/to/checkpoint.pt \
  --dst ./results
```

The evaluation configuration supports prediction export, online 3D metrics,
inference benchmarking, and model-complexity profiling.

## Project structure

```text
config/                   Experiment configurations
src/rxf/datasets/         K-Radar dataset and sparse radar loading
src/rxf/models/           RXF model components
src/rxf/training/         Training, matching, losses, and optimization
src/rxf/evaluation/       Metrics and K-Radar result export
Deformable-DETR-2D/       Local 2D deformable-attention extension
Deformable-DETR-3D/       Local 3D deformable-attention extension
```

## License

This project is licensed under the Apache License 2.0. Third-party components
retain their respective licenses.

# level- 1 대회 :마스크 착용 상태 분류대회

> ### File Tree
---
```bash
.
├── checkpoints
├── configs
│   ├── multi_label_train.yaml
│   ├── multi_label_test.yaml
│   ├── train.yaml
│   └── test.yaml
├── data
│   ├── train
│   └── eval
├── docs
├── notebooks
├── results
├── src
│   ├── datasets
│   │   └── mask_dataset.py
│   ├── models
│   │   └── mask_model.py
│   ├── ops
│   │   ├── losses.py
│   │   ├── optim.py
│   │   └── scheduler.py
│   ├── utils
│   │   ├── ensemble.py
│   │   ├── transform.py
│   │   └── utils.py
│   ├── visualization
│   ├── multi_label_train.py
│   ├── multi_label_test.py
│   ├── train.py - 수정중
│   └── eval.py - 수정중
├── wandb
├── train.sh
└── eval.sh

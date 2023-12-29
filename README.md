# level- 1 대회 :마스크 착용 상태 분류대회

## 팀 소개
| 이름 | 역할 |
| [박상언](https://github.com/PSangEon) | CutMix 실험, 모형 실험, Git branch 관리 |
| [지현동](https://github.com/tolfromj) | 마스크, 성별, 나이 따로 실험 후 못 맞히는 문제 판별, 모형 실험, Mixup 실험 |
| [오왕택](https://github.com/ohkingtaek) | Baseline code 작성, YOLOv8 Ultralytics로 사진 탐지, 모형 실험, Mixup 실험 |
| [이동호](https://github.com/as9786) | GradCAM, 나이 회귀 모형 실험, 사전 학습된 U-Net으로 사람 분할, Wrap-up Report 작성, 모형 실험 |
| [송지민](https://github.com/Remiing) | Data augmentation 기법 정리, Mixup 실험 |

## 프로젝트 소개
<p align="center">
<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-07/assets/49676680/f97949b9-ee29-4884-acd5-dd6e9e52b8b1">
</p>

우리는 사람 사진을 바탕으로 해당 사람이 마스크를 제대로 착용했는지, 성별 그리고 나이대를 분류하는 작업을 하였다. 마스크 라벨의 경우에는 착용하지 않는 Normal, 착용은 했지만 제대로 착용하지 않은 Incorrect 그리고 정상적으로 마스크를 착용한 Mask 라벨로 이루어져 있다. 성별은 남과 여로 나뉘어져 있고, 나이대는 30대 미만은 young, 30대 이상 그리고 60대 미만은 middle, 60대 이상은 old로 라벨링 되어 있다. 이번 프로젝트는 작게 보게 되면 총 3가지 작업에 대해서 분류를 진행하면 되고, 최종 라벨은 각 3개의 분류 라벨의 조합으로 총 18개의 라벨을 분류하는 문제로 볼 수 있다. 

## 프로젝트 일정
프로젝트 전체 일정
- 12/13 10:00 ~ 12/21 19:00

프로젝트 세부 일정
- 12/11 ~ 12/12 강의 수강, 제공 데이터 및 코드 확인
- 12/13 ~ 12/14 BaseLine Code 작성, Git Branch 생성
- 12/15 ~ 12/17 데이터 살펴보기 + EDA, 잘 맞추지 못하는 부분 확인, 나이 회귀 모형, 여러가지 증강 기법 실험
- 12/18 ~ 12/19 분할 및 탐지 기법 적용, 여러 모형 실험
- 12/20 ~ 12/21 CutMix, MixUp 적용 및 여러 가지 손실 함수 적용, 모형 실험

## 프로젝트 수행
- 데이터 살펴보기 : 데이터 정답 정보 틀린 것을 확인, old 라벨의 기준이 모호, 인물 나이 판별 위해 해상도 유지 및 증강 기법 사용해야 한다는 것을 확인
- EDA : 라벨 불균형 확인 및 회귀 방법 사용, GradCAM으로 나이 예측 시 어느 부분이 중점인지 확인
- Multi Class vs Multi Task : Multi Task가 실험적으로 우수
- 여러 증강 기법 적용 : Albumentations의 증강 기법 모두를 시각화 + CutMix & MixUp 적용
- 탐지 및 분할 기법 적용 : YOLOv8로 사람 탐지 및 U-Net으로 사람 탐지하여 데이터 전처리
- 모형 및 성능 평가 : ConvNext, EfficientNet-B6 사용

## 프로젝트 결과
- 프로젝트 결과는 Public 3등, Private 4등이라는 결과를 얻었습니다.
    - Public

    ![Untitled](https://github.com/boostcampaitech6/level1-imageclassification-cv-07/assets/49676680/96f1a617-7b87-424a-b836-87826343dcb4)

    - Private

    ![Untitled](https://github.com/boostcampaitech6/level1-imageclassification-cv-07/assets/49676680/0faabd4a-bd8f-43fb-b530-60cded8c1ca5)

## Wrap-Up Report

[Wrap-Up-Report](docs/CV기초대회 분류_CV_팀 리포트(07조).pdf) 

## File Tree

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
│   └── CV기초대회 분류_CV_팀 리포트(07조).pdf
├── notebooks
├── results
├── scripts
│   ├── unet.py
│   ├── unet_test.py
│   ├── yolo.py
│   └── yolo_test.py
├── src
│   ├── datasets
│   │   └── mask_datasets.py
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
│   ├── multi_label_train.py
│   ├── multi_label_eval.py
│   ├── train.py
│   └── eval.py
├── wandb
├── train.sh
└── eval.sh
```

| File(.py) | Description |
| --- | --- |
| unet.py | unet을 사용하여 train data를 분할 |
| unet_test.py | unet을 사용하여 eval data를 분할 |
| yolo.py | ultralytics를 사용하여 train data를 탐지 |
| yolo_test.py | ultralytics를 사용하여 eval data를 탐지 |
| mask_datasets.py | multi class와 multi task 등 여러가지 커스텀 데이터셋 코드 |
| mask_model.py | model 모아놓은 코드 |
| losses.py | loss들을 모아놓은 코드 |
| optim.py | optimizer들을 모아놓은 코드 |
| scheduler.py | scheduler들을 모아놓은 코드 |
| ensemble.py | ensemble하는 코드 |
| transform.py | transform들을 모아놓은 코드 |
| utils.py | 이외에 필요한 다양한 코드 |
| multi_label_train.py | multi task train 코드 |
| multi_label_eval.py | multi task eval 코드 |
| train.py | train 코드 |
| eval.py | eval 코드 |

## License
네이버 부스트캠프 AI Tech 교육용 데이터로 대회용 데이터임을 알려드립니다.
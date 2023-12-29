# level- 1 대회 :마스크 착용 상태 분류대회

## 팀 소개

## 프로젝트 소개
<p align="center">
<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-07/assets/49676680/f97949b9-ee29-4884-acd5-dd6e9e52b8b1">
</p>

우리는  사람  사진을  바탕으로  해당  사람이  마스크를  제대로  착용했는지,  성별  그리고  나이대를 
분류하는  작업을  하였다.  마스크  라벨의  경우에는  착용하지  않는  Normal,  착용은  했지만  제대로 
착용하지  않은  Incorrect  그리고  정상적으로  마스크를  착용한  Mask  라벨로  이루어져  있다.  성별은 
남과  여로  나뉘어져  있고,  나이대는  30대  미만은  young,  30대  이상  그리고  60대  미만은  middle, 
60대  이상은  old로  라벨링  되어  있다.  이번  프로젝트는  작게  보게  되면  총  3  가지  작업에  대해서 
분류를  진행하면  되고,  최종  라벨은  각  3개의  분류  라벨의  조합으로  총  18개의  라벨을  분류하는 
문제로  볼  수  있다. 
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
- 데이터 살펴보기
- EDA
- Multi Class vs Multi Task
- 여러 증강 기법 적용
- 탐지 및 분할 기법 적용
- 모형 및 성능 평가
## 프로젝트 결과
- 프로젝트 결과는 Public 3등, Private 4등이라는 결과를 얻었습니다.
    - Public

    ![Untitled](https://github.com/boostcampaitech6/level1-imageclassification-cv-07/assets/49676680/96f1a617-7b87-424a-b836-87826343dcb4)

    - Private

    ![Untitled](https://github.com/boostcampaitech6/level1-imageclassification-cv-07/assets/49676680/0faabd4a-bd8f-43fb-b530-60cded8c1ca5)

## Wrap-Up Report


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
│   ├── multi_label_test.py
│   ├── train.py
│   └── eval.py
├── wandb
├── train.sh
└── eval.sh
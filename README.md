# Washing Machine Door Status Detection

> **A program that checks the operation of a washing machine by detecting the door status using deep learning**

세탁기 문 열림/닫힘 상태를 **이미지 기반 딥러닝 모델(VGG16)** 로 예측하여
세탁기 동작 여부를 판단하는 프로젝트입니다.

---

## Course Information

* **Course**: Computing Thinking
* **Subject**: Cultivating Computational Thinking
* **Topic**: 문제 해결 과정 5단계 적용 프로젝트

---

## Problem-Solving Process (Computational Thinking)

본 프로젝트는 **컴퓨팅 사고력 문제 해결 5단계**에 따라 진행되었습니다.

1. **요구사항 분석**
2. **설계**
3. **구현**
4. **평가 및 검증**
5. **개선 및 확장**

---

## Project Goal

* 세탁기 문 이미지 데이터를 이용하여
  **문이 열려 있는지 / 닫혀 있는지**를 자동으로 분류
* 실제 세탁기 동작 여부 판단에 활용 가능성 검증

---

## 🛠 Technology Stack

* **Language**: Python
* **Deep Learning**: TensorFlow / Keras
* **Model**: VGG16 (Transfer Learning)
* **Data Handling**: ImageDataGenerator
* **Visualization**: Matplotlib

---

## Project Progress

### Project 1 – Initial Version

**VGG16 기반 세탁기 문 열림 예측 (초기 버전)**

* Epochs: **10**
* Dataset: 소규모 이미지 데이터
* Result: **4 / 5 정확한 예측**
* 한계:

  * 데이터 수 부족
  * 학습 안정성 낮음

---

### Project 2 – Improved Training Pipeline

**Project 1 업데이트 버전**

* ImageDataGenerator 적용

  * 학습/검증 데이터셋 분리
  * 데이터 증강(Augmentation)
* Epochs: **100**
* Early Stopping 적용
* 학습 결과 시각화

  * Loss / Accuracy 그래프
  * 데이터 증강 시각화
* Result: **6 / 7 정확한 예측**

✔ 성능과 일반화 능력 개선

---

### Project 3 – Dataset Expansion

**Project 2 업데이트 버전**

* 더 많은 이미지 데이터 확보
* Predict 데이터와 학습/검증 데이터 분리
* 데이터 전처리는 **미적용**
* 학습 및 검증 구조 개선

✔ 실제 사용 환경을 고려한 구조로 발전

---

### Project 4 – (Planned)

**이미지 전처리 적용 예정**

* Noise 제거
* 밝기 / 대비 보정
* ROI(관심 영역) 고려
* 모델 성능 추가 개선 목표

---

## Results Summary

| Project   | Epochs | Dataset                 | Accuracy |
| --------- | ------ | ----------------------- | -------- |
| Project 1 | 10     | Small                   | 4 / 5    |
| Project 2 | 100    | Expanded + Augmentation | 6 / 7    |
| Project 3 | 100    | Larger Dataset          | Improved |

---

## Key Takeaways

* Transfer Learning(VGG16)을 활용하여
  적은 데이터로도 의미 있는 성능을 달성
* 데이터 수와 전처리의 중요성 확인
* 컴퓨팅 사고력 기반 문제 해결 과정의 실효성 검증

---

## Future Improvements

* 이미지 전처리 파이프라인 추가
* 다른 CNN 모델(ResNet, EfficientNet) 비교
* 실시간 카메라 입력 적용
* 실제 세탁기 시스템과 연동 가능성 검토

---

## Repository Structure (Optional)

```text
Detection-washing-machine-door/
├── dataset/
├── model/
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Author

* **Taemin Kim**
* GitHub: [https://github.com/ktm0110](https://github.com/ktm0110)


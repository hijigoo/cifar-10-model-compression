# CIFAR-10 ResNet18 모델 압축 프로젝트

**과목**: GEV6152 Model Compression  
**프로젝트**: 중간 프로젝트 - Pruning을 통한 CIFAR-10 모델 압축

## 📋 프로젝트 개요

본 프로젝트는 CIFAR-10 이미지 분류 작업에 대해 ResNet18 아키텍처를 사용하여 세 가지 neural network pruning 기법을 구현하고 비교합니다. 목표는 희소성(sparsity)-정확도(accuracy) 간의 trade-off를 분석하고 모델 압축을 통한 효율성 향상을 측정하는 것입니다.

### 구현된 Pruning 방법

1. **Magnitude-based Pruning**: 절대값이 작은 가중치를 제거
2. **Structured Pruning**: 중요도가 낮은 필터/채널 단위로 제거
3. **Lottery Ticket Hypothesis**: Pruning 후 초기 가중치로 되돌려 재학습

## 🚀 빠른 시작

### 필수 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장)
- 10GB 이상의 여유 디스크 공간

### 설치 방법

```bash
# 프로젝트 폴더로 이동
cd project/

# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 패키지 설치
pip install -r requirements.txt
```

### 실험 실행

#### 전체 실험 (모든 방법, 모든 시드)
```bash
./run.sh
```

실행 내용:
1. 3개의 dense baseline 모델 학습 (시드: 42, 123, 456)
2. 3가지 pruning 방법을 5개 희소성 수준에 적용
3. 모든 pruned 모델 fine-tuning
4. 모든 모델 평가
5. 그래프 및 표 생성

**예상 소요 시간**: GPU 1개 기준 24-48시간

#### 빠른 테스트 실행
```bash
./run.sh --quick
```

에폭 수를 줄이고 일부 구성만 실행 (1-2시간 소요)

#### Dense 학습 건너뛰기
이미 학습된 dense 모델이 있는 경우:
```bash
./run.sh --skip-dense
```

### 개별 스크립트 실행

#### Dense Baseline 학습
```bash
python train_dense.py --seed 42 --epochs 200
```

#### Pruned 모델 학습
```bash
python train_pruned.py \
    --method magnitude \
    --sparsity 0.9 \
    --seed 42 \
    --finetune-epochs 100
```

#### 모든 모델 평가
```bash
python evaluate_all.py
```

#### 시각화 생성
```bash
python plot_results.py
```

## 📁 프로젝트 구조

```
project/
├── models/                     # 모델 아키텍처
│   ├── __init__.py
│   └── resnet.py              # ResNet18 구현
├── pruning/                    # Pruning 방법들
│   ├── __init__.py
│   ├── magnitude_pruning.py   # Magnitude-based pruning
│   ├── structured_pruning.py  # Structured pruning
│   └── lottery_ticket.py      # Lottery Ticket Hypothesis
├── utils/                      # 유틸리티 함수
│   ├── __init__.py
│   ├── train.py               # 학습 유틸리티
│   ├── evaluate.py            # 평가 유틸리티
│   └── metrics.py             # 메트릭 계산
├── experiments/                # 실험 결과
│   ├── checkpoints/           # 모델 체크포인트
│   └── results/               # 결과, 그래프, 표
├── data/                       # CIFAR-10 데이터셋 (자동 다운로드)
├── train_dense.py             # Dense baseline 학습
├── train_pruned.py            # Pruned 모델 학습
├── evaluate_all.py            # 모든 모델 평가
├── plot_results.py            # 시각화 생성
├── run.sh                     # 메인 실험 스크립트
├── requirements.txt           # Python 의존성
├── REPORT_KR.md               # 한글 보고서
└── README.md                  # 본 파일
```

## 📊 결과

실험 실행 후 결과는 `experiments/results/`에 저장됩니다:

### 주요 출력물

1. **Tradeoff Curve** (`tradeoff_curve.png`)
   - 모든 방법의 희소성 vs 정확도 그래프
   - 95% 신뢰구간 포함

2. **효율성 비교 표** (`efficiency_table.md`)
   - 모델 크기, 파라미터, 정확도 비교
   - Markdown, LaTeX, CSV 형식 제공

3. **원시 데이터** (`all_results.json`)
   - 완전한 실험 데이터
   - 모든 구성에 대한 모든 메트릭 포함

### 결과 데이터 구조 예시

```json
{
    "method": "magnitude",
    "sparsity": 0.90,
    "seed": 42,
    "test_accuracy": 92.7,
    "params_millions": 1.12,
    "model_size_mb": 6.19,
    "inference_latency_ms": 1.9
}
```

## 🔬 실험 설정

### 하이퍼파라미터

#### Dense 모델 학습
- **아키텍처**: ResNet18 (11.17M 파라미터)
- **에폭**: 200
- **배치 크기**: 128
- **옵티마이저**: SGD with momentum (0.9)
- **학습률**: 0.1 (cosine annealing)
- **가중치 감쇠**: 5e-4
- **데이터 증강**: Random crop, horizontal flip

#### Pruned 모델 Fine-tuning
- **에폭**: 100
- **학습률**: 0.01 (cosine annealing)
- **기타 파라미터**: Dense 학습과 동일

### 테스트한 희소성 수준
- 0.0 (dense baseline)
- 0.3 (30% sparse)
- 0.5 (50% sparse)
- 0.7 (70% sparse)
- 0.9 (90% sparse)
- 0.95 (95% sparse)

### Random Seeds
- 42, 123, 456 (통계적 유의성 확보)

## 📈 주요 발견

### 실험 결과

1. **Magnitude Pruning**
   - 90% 희소성까지 우수한 정확도 유지 (92.7%)
   - 구현 복잡도 낮음
   - 특수 하드웨어 없이는 실제 속도 향상 없음

2. **Structured Pruning**
   - 높은 희소성에서 정확도 다소 하락 (90.3%)
   - 표준 하드웨어에서 실제 속도 향상 (2.1배)
   - 모델 크기 감소

3. **Lottery Ticket**
   - Magnitude pruning과 유사한 성능 (92.5%)
   - 초기 가중치로부터 재학습 필요
   - 최적 부분 네트워크 찾기에 효과적

### 측정된 메트릭

- **Test Accuracy**: CIFAR-10 테스트셋에서의 Top-1 정확도
- **Sparsity**: 0인 가중치의 비율
- **Parameters**: 0이 아닌 파라미터 개수
- **Model Size**: 디스크 저장 크기 (MB)
- **Latency**: 이미지당 추론 시간 (ms)

## 🛠️ 구현 세부사항

### Pruning 알고리즘

#### Magnitude Pruning
```python
# 전역 magnitude-based pruning
pruned_model = magnitude_prune_global(model, sparsity=0.9)
```

#### Structured Pruning
```python
# 필터 단위 structured pruning
pruned_model = structured_prune_filters(model, sparsity=0.9)
```

#### Lottery Ticket
```python
# Prune 후 초기값으로 리셋
pruned_model = lottery_ticket_prune(model, initial_weights, sparsity=0.9)
```

### 메트릭 계산

모든 메트릭은 `utils/metrics.py`의 유틸리티를 사용하여 계산됩니다:

```python
from utils import get_model_info

info = get_model_info(model, device='cuda', verbose=True)
# 반환값: params, sparsity, size, latency
```

##  문제 해결

### CUDA 메모리 부족
```bash
# 배치 크기 줄이기
python train_dense.py --batch-size 64
```

### 느린 학습 속도
```bash
# CPU가 병목이면 worker 수 줄이기
python train_dense.py --num-workers 0
```

### 의존성 누락
```bash
pip install --upgrade -r requirements.txt
```

## 📚 참고문헌

1. **Magnitude Pruning**
   - Han et al., "Learning both Weights and Connections for Efficient Neural Networks" (NIPS 2015)

2. **Structured Pruning**
   - Li et al., "Pruning Filters for Efficient ConvNets" (ICLR 2017)

3. **Lottery Ticket Hypothesis**
   - Frankle & Carbin, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (ICLR 2019)

4. **PyTorch Pruning**
   - https://pytorch.org/tutorials/intermediate/pruning_tutorial.html

5. **ResNet**
   - He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)

# TGCN-StockPred

## 📌 Overview
이 프로젝트는 **시계열 그래프 신경망(GNN + LSTM)** 을 이용하여 **Cholesky 분해 기반 인접 행렬 예측**을 수행합니다.  
S&P 500 주가 데이터를 활용하여 그래프를 생성하고, Temporal Graph Convolutional Network(T-GCN)과 LSTM을 조합하여 예측을 수행합니다.

## 🛠 Features
- **시계열 그래프 데이터**를 생성하여 주가 데이터를 노드와 엣지로 표현
- **Graph Convolutional Network (GCN)** 을 이용한 노드 특성 추출
- **LSTM 기반 시계열 학습**을 통해 미래 그래프 구조 예측
- **Cholesky 분해를 활용**하여 인접 행렬 복원
  
## 🏗️ Project Structure
```plaintext
project_root/
│── engine.py           # TGCNCholeskyModel 및 학습 엔진
│── main.py             # 데이터 로드 및 학습 실행
│── preprocess.py       # 데이터 전처리 및 그래프 생성
│── requirements.txt    # 필요한 패키지 목록
│── README.md           # 프로젝트 설명


 🔧 Installation
### 1️⃣ 환경 설정
```bash
conda create -n myenv python=3.8
conda activate myenv

### 2️⃣ 패키지 설치
pip install -r requirements.txt

'''

🚀 Usage

1️⃣ 데이터 다운로드

프로젝트에서 S&P 500 데이터를 사용하므로 데이터베이스 연결이 필요합니다.연결 정보를 수정하려면 preprocess.py에서 _GOGH_URL을 설정하세요.

2️⃣ 학습 실행

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

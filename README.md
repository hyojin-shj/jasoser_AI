# 🧠 AI 기반 자소서 평가 및 XAI 채용 시스템
**"LLM 정성 평가와 로컬 Transformer 유사도 분석을 결합한 설명 가능한(Explainable) 채용 평가 서비스"**

---

## 📅 프로젝트 개요
- **프로젝트 기간**: 2026.03  
- **개발자**: 심효진 (@hyojin-shj)  

- **한 줄 소개**:  
  > 단순 키워드 매칭이 아닌, 문맥 이해 + 유사도 근거를 함께 제공하는 하이브리드 AI 자소서 평가 시스템

---

## 📊 평가 결과 예시 (AI Resume Evaluation)

| **JD-자소서 유사도 분석** | **Top-K 유사 문장 (XAI)** | **LLM 정성 평가 결과** |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/dbfdac45-742d-4f63-8f87-4135c2a4cb72" width="300"/> | <img src="https://github.com/user-attachments/assets/75ab1f7a-07eb-4935-82a8-e44e8181f580" width="300"/> | <img src="https://github.com/user-attachments/assets/02b5513b-0821-4001-b7dc-4aa8cd982f74" width="300"/> |
| 코사인 유사도 기반 정량 점수 | JD와 가장 유사한 문장 추출 | 논리성, 구체성 기반 정성 평가 |
---

## 💡 프로젝트 배경
- **기존 채용 시스템 한계**: 키워드 매칭 기반으로 문맥 이해 부족  
- **정성 vs 정량 괴리**: 사람의 평가 기준과 AI 점수 간 불일치 발생  
- **설명 가능성 필요 (XAI)**: 왜 이 점수가 나왔는지 근거 부족  

→ 이를 해결하기 위해  
**LLM + SBERT 하이브리드 구조 + XAI 로직**을 설계

---

## 📊 데이터 파이프라인 및 분석 구조

### 1. 입력 데이터 구성
- **채용 공고 (JD)**  
- **자기소개서 (Resume)**  

---

### 2. 분석 파이프라인
- JD / Resume 입력
- SBERT 기반 임베딩 벡터 변환
- Cosine Similarity 계산 (정량 평가)
- LLM 기반 정성 평가 수행
- 문장 단위 유사도 분석 (Top-K 추출)
- 통합 점수 및 리포트 생성

---

### 3. 출력 결과
- 유사도 점수 (정량)
- LLM 평가 피드백 (정성)
- Top-K 근거 문장 (XAI)
- Streamlit 대시보드 시각화

---

## 🛠️ 기술 스택 (Tech Stack)

| 구분 | 기술 | 활용 |
|------|------|------|
| AI/NLP | SBERT (KR-SBERT-V1), GPT-4o-mini | 유사도 분석 및 정성 평가 |
| Framework | PyTorch, LangChain | 모델 서빙 및 LLM orchestration |
| Library | Sentence-Transformers, Scikit-learn | 임베딩 및 코사인 유사도 계산 |
| Backend | Python 3.10+ | 전체 시스템 로직 |
| UI | Streamlit | 실시간 평가 대시보드 |
| Infra | Local CPU/GPU | 로컬 Transformer 추론 환경 |

---

## 💡 주요 기능 (Key Features)

### 1️⃣ 하이브리드 평가 엔진

- SBERT 기반 코사인 유사도 계산 (정량)
- GPT 기반 문맥 분석 평가 (정성)
- 가중치 기반 점수 통합 로직 적용

---

### 2️⃣ 설명 가능한 AI (XAI)

- JD ↔ 자소서 문장 단위 비교
- Top-K 유사 문장 추출 및 하이라이트
- 점수에 대한 근거 제공

---

### 3️⃣ 로컬 Transformer 서빙

- Hugging Face SBERT 모델 로컬 실행
- 외부 API 의존도 감소
- 보안성 및 비용 효율 개선

---

### 4️⃣ Streamlit 기반 대시보드

- 실시간 분석 결과 시각화
- 점수 + 피드백 + 근거 통합 제공
- 사용자 친화적 UI 구성

---

## 📂 프로젝트 구조 (Directory Structure)
₩₩₩
ai-hr-evaluator/
├── app.py # Streamlit 메인 앱
├── models/ # 모델 관련 코드
├── utils/ # 유사도 및 전처리 로직
├── prompts/ # LLM 프롬프트 정의
├── data/ # 샘플 데이터
├── requirements.txt # 의존성 패키지
└── README.md
₩₩₩

---

## 🚀 트러블 슈팅 (Troubleshooting)

### 🔧 이슈 1: 정성 평가 vs 유사도 점수 불일치

문제:  
- LLM은 높은 점수를 주지만 SBERT는 낮은 점수를 반환

원인:  
- LLM → 문맥 기반 평가  
- SBERT → 기술 키워드 기반 벡터 거리 계산  

해결:
- Hybrid Scoring 로직 도입 (가중치 적용)
- XAI 기반 유사 문장 제공으로 점수 근거 보완

---

### 🔧 이슈 2: Transformer 모델 로딩 지연 (Cold Start)

문제:  
- Streamlit 실행 시 모델 로딩 속도 지연 발생

원인:  
- SBERT 모델 (수백 MB) 반복 로딩

해결:
- `@st.cache_resource` 적용
- 모델 인스턴스 재사용 (Singleton Pattern)

결과:
- 초기 로딩 속도 약 80% 개선

---

### 🔧 이슈 3: 문장 단위 유사도 이상치 발생

문제:  
- 짧은 문장에서 유사도 점수 왜곡 발생

원인:  
- 의미 없는 문장이 특정 키워드와 우연히 매칭

해결:
- 최소 길이 필터링 (Preprocessing)
- Max Pooling + 평균 결합 방식 적용

---


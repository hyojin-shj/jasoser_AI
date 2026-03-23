# app.py 전체 코드 (수정본)

import streamlit as st
import pandas as pd
from engine import HREvaluator

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="Hybrid AI HR Evaluator | 자소서 정밀 분석기",
    page_icon="🤖🎯",
    layout="wide"
)

# 2. 스타일 커스텀 (CSS)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .highlight-match {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 0 5px 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 사이드바 - 채용 공고(JD) 입력 영역
with st.sidebar:
    st.title("📋 채용 정보 입력")
    st.info("지원하려는 회사의 채용 공고(JD)를 복사해서 아래에 붙여넣으세요.")
    
    jd_content = st.text_area(
        "채용 공고 (Job Description)", 
        placeholder="주요 업무, 자격 요건, 우대 사항 등을 입력하세요...",
        height=500
    )
    
    st.caption("Tip: 공고 내용이 상세할수록 더 정확한 직무 매칭 분석이 가능합니다.")

# 4. 메인 화면 - 자소서 입력 및 분석
st.title("🚀 하이브리드 AI 기반 자기소개서 정밀 분석 시스템")
st.markdown("---")

col_input, col_info = st.columns([2, 1])

with col_input:
    st.subheader("📝 나의 자기소개서")
    resume_content = st.text_area(
        "자기소개서 본문을 입력하세요",
        placeholder="분석하고 싶은 자소서 내용을 여기에 붙여넣으세요.",
        height=400
    )

with col_info:
    st.subheader("💡 하이브리드 분석")
    st.write("**외부 LLM(GPT)의 정성적 분석**과 **로컬 Transformer(SBERT)의 정량적 분석**을 결합하여 자소서를 평가합니다.")
    st.write("**XAI(설명 가능한 AI):** 로컬 모델이 JD와 가장 유사하다고 판단한 자소서 내 문장을 시각화합니다.")
    st.warning("로컬 모델 초기 로딩 및 분석에는 약 10~20초 정도 소요될 수 있습니다.")

# 5. 분석 실행 버튼
if st.button("🔍 AI 서류 평가 및 피드백 시작", use_container_width=True):
    if not resume_content:
        st.error("자기소개서 내용을 입력해 주세요!")
    else:
        with st.spinner("하이브리드 AI 모델들이 협업하여 서류를 검토 중입니다. 잠시만 기다려주세요..."):
            try:
                # engine.py의 HREvaluator 호출
                evaluator = HREvaluator()
                result = evaluator.analyze(resume_content, jd_content)
                
                st.success("✅ 분석이 완료되었습니다!")
                st.markdown("### 📊 분석 결과 리포트")
                
                # 결과 지표 출력 (Metric)
                st.markdown("#### 🎯 직무 적합성 비교 (설명 가능한 AI)")
                # app.py 내의 metric 출력 부분
                col1, col2 = st.columns(2)
                # result['relevance']로 값을 가져오는지 확인 (N/A가 나왔다면 이 키가 없었던 것)
                col1.metric("🌐 외부 LLM (GPT) 평가", f"{result.get('relevance', 0)}점") 
                col2.metric("🏠 로컬 Transformer (SBERT) 평가", f"{result['transformer_score']}점")
                                
                                # 로컬 모델 기반의 설명(Evidence)
                with st.expander("🤔 왜 로컬 Transformer 점수가 이렇게 나왔나요?", expanded=True):
                    st.markdown("자소서 내에서 채용 공고(JD)의 직무 맥락과 가장 유사한 **Top-3 문장**입니다.")
                    for i, (sentence, score) in enumerate(result['top_k_matches']):
                        st.markdown(f"""
                            <div class="highlight-match">
                                <strong>Match {i+1} (유사도: {score:.2f})</strong><br>
                                {sentence}.
                            </div>
                        """, unsafe_allow_html=True)
                
                st.divider()
                st.markdown("#### ⚖️ 다면 평가 지표 (LLM 정성 분석)")
                m1, m2, m3 = st.columns(3)
                m1.metric("📊 경험 구체성", f"{result['specificity']}점")
                m2.metric("🧠 논리적 완결성", f"{result['logic']}점")
                m3.metric("🤖 AI 의심도", f"{result['ai_score']}%")
                
                st.divider()
                
                # 상세 분석 결과 (강점/보완점)
                tab1, tab2 = st.columns(2)
                
                with tab1:
                    st.subheader("✅ 강점 및 핵심 역량")
                    for strength in result['strengths']:
                        st.success(strength)
                        
                with tab2:
                    st.subheader("⚠️ 개선 및 보완 필요 사항")
                    for weakness in result['weaknesses']:
                        st.warning(weakness)
                
                st.divider()
                st.subheader("💡 종합 피드백")
                st.info(result['overall_feedback'])
                
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")

# 하단 푸터
st.markdown("---")
st.caption("© 2026 AI HR Evaluator Project - Powered by Hybrid AI (LLM & Transformer)")
# engine.py 전체 코드

import os
import torch
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# .env 파일에서 OpenAI 키 로드
load_dotenv()

# 1. Pydantic으로 LLM 출력 구조 정의
# engine.py 내의 EvaluationResult 클래스를 아래와 같이 수정하세요.
class EvaluationResult(BaseModel):
    relevance: int = Field(description="직무 적합성 점수 (0-100)") # 이 줄이 추가되어야 합니다!
    specificity: int = Field(description="경험의 구체성 (0-100)")
    logic: int = Field(description="논리 및 가독성 (0-100)")
    ai_score: int = Field(description="AI 작성 의심도 (0-100)")
    strengths: list = Field(description="잘된 점 3가지")
    weaknesses: list = Field(description="보완할 점 3가지")
    overall_feedback: str = Field(description="전체 총평")

# 2. 로컬 Transformer 기반 시맨틱 분석 모듈
import torch
from sentence_transformers import SentenceTransformer, util

class LocalTransformerAnalyzer:
    def __init__(self):
        # 모델 로드 (Hugging Face에서 자동으로 다운로드 및 캐싱)
        # 만약 네트워크 에러가 지속되면 'paraphrase-multilingual-MiniLM-L12-v2' 등으로 시도 가능
        try:
            self.model = SentenceTransformer('snunlp/KR-SBERT-V1')
        except Exception:
            # 대안 모델 (한국어 지원 다국어 모델)
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    import torch
from sentence_transformers import SentenceTransformer, util

class LocalTransformerAnalyzer:
    def __init__(self):
        try:
            # 한국어 전용 모델 로드
            self.model = SentenceTransformer('snunlp/KR-SBERT-V1')
        except Exception:
            # 대안 모델 (네트워크 오류 대비)
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def get_top_matches(self, jd_text, resume_text, top_k=3):
        if not jd_text or not resume_text:
            return 0, []

        # 1. 문장 단위 분리
        resume_sentences = [s.strip() for s in resume_text.split('.') if s.strip()]
        if not resume_sentences:
            return 0, []

        # 2. 임베딩 계산
        jd_embedding = self.model.encode(jd_text, convert_to_tensor=True)
        resume_embeddings = self.model.encode(resume_sentences, convert_to_tensor=True)
        
        # 3. 코사인 유사도 계산 (정확한 함수명: pytorch_cos_sim)
        # 결과는 [1, N] 형태의 텐서로 나옵니다.
        cos_scores = util.pytorch_cos_sim(jd_embedding, resume_embeddings)[0]
        
        # 4. 전체 유사도 (평균값 산출)
        total_similarity = float(torch.mean(cos_scores))
        
        # 5. Top-K 유사 문장 탐색
        top_results = torch.topk(cos_scores, k=min(top_k, len(resume_sentences)))
        
        top_matches = []
        for score, idx in zip(top_results[0], top_results[1]):
            top_matches.append((resume_sentences[int(idx)], float(score)))

        return int(total_similarity * 100), top_matches
# 3. 통합 HREvaluator 클래스
class HREvaluator:
    def __init__(self):
        # LLM (GPT) 설정
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.parser = JsonOutputParser(pydantic_object=EvaluationResult)
        
        # 로컬 Transformer 설정
        self.local_analyzer = LocalTransformerAnalyzer()
        
    def analyze(self, resume_text, jd_text=""):
        # A. 로컬 Transformer 정량 분석 및 Top-K 매칭
        local_score, top_k_matches = self.local_analyzer.get_top_matches(jd_text, resume_text)
        
        # B. LLM (GPT) 정성 분석
        # JD가 있을 경우 프롬프트에 반영
        jd_context = f"다음 채용 공고의 요구사항을 중점적으로 반영하여 평가하세요: {jd_text}" if jd_text else "일반적인 신입 사원의 역량을 기준으로 평가하세요."
        
        prompt = ChatPromptTemplate.from_template(
            "당신은 대기업 HR 채용 담당자입니다. {jd_context}\n"
            "제시된 자기소개서를 바탕으로 역량을 정밀 분석하고 피드백을 주세요.\n"
            "결과는 반드시 지정된 JSON 형식으로만 답변하세요.\n\n"
            "자소서 내용: {resume}\n\n"
            "{format_instructions}"
        )
        
        chain = prompt | self.llm | self.parser
        llm_result = chain.invoke({
            "jd_context": jd_context,
            "resume": resume_text,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        # C. 두 결과 통합
        final_result = {
            **llm_result,  # LLM 결과
            "transformer_score": local_score,  # 로컬 Transformer 점수
            "top_k_matches": top_k_matches  # 유사 문장 하이라이트 데이터
        }
        
        return final_result
import streamlit as st
from transformers import BartForConditionalGeneration, AutoTokenizer

# KoBART 모델 로드 (공개된 한국어 모델)
model_name = "hyunwoongko/kobart"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 맞춤법 교정 함수
def correct_spelling_with_kobart(text):
    input_text = f"correct the spelling: {text}"  # 맞춤법 교정 태스크

    # 입력 텍스트 토큰화
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # 모델을 사용해 교정된 텍스트 생성
    outputs = model.generate(inputs['input_ids'], max_length=128, num_beams=1, early_stopping=True)

    # 생성된 텍스트 디코딩
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Streamlit 애플리케이션 설정
st.title("한글 맞춤법 교정기")
st.write("아래에 문장을 입력하고, 맞춤법 교정을 확인하세요.")

# 사용자로부터 문장 입력 받기
text_input = st.text_input("검사할 문장을 입력하세요:")

if text_input:
    # KoBART 모델을 사용한 맞춤법 교정 실행
    corrected_text = correct_spelling_with_kobart(text_input)

    # 결과 출력
    st.write(f"교정된 문장: {corrected_text}")

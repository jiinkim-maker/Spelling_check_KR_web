import streamlit as st
from transformers import BartForConditionalGeneration, AutoTokenizer
from PIL import Image


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

# 사이드바 설정
st.sidebar.title("김지후님, 반갑습니다.")
# 4. 사용자 이미지 업로드 (이미지 파일 경로)
image = Image.open("talk.gif")  # 백슬래시 두 번 사용
st.image(image, width=100)  # 이미지 크기 설정

# 1. 아카이브 버튼 (헷갈리는 맞춤법 저장)
if st.sidebar.button("헷갈리는 맞춤법 아카이브"):
    st.sidebar.write("저장된 맞춤법을 확인하세요!")  # 아카이브 버튼 클릭 시 메시지 표시

# 2. 맞춤법 교정 오류 신고 버튼
if st.sidebar.button("맞춤법 교정 오류 신고"):
    st.sidebar.write("교정 오류를 신고해주세요.")  # 오류 신고 버튼 클릭 시 메시지 표시

# 3. 한국어 맞춤법 교정 AI 모델 변경 버튼
model_option = st.sidebar.selectbox("사용할 맞춤법 교정 AI 모델 선택", ("KoBART", "GPT", "Hugging Face"))  # 모델 변경 옵션 추가

# 사용자 인사
st.header(f"한글 맞춤법 검사기")  # 사용자 이름으로 인사

# 4. 사용자 이미지 업로드 (이미지 파일 경로)
#image = Image.open("talk.gif")  # 백슬래시 두 번 사용
#st.image(image, width=100)  # 이미지 크기 설정

# 설명
st.write("아래에 문장을 입력하고, 맞춤법 교정을 확인하세요.")

# 사용자로부터 문장 입력 받기
text_input = st.text_input("검사할 문장을 입력하세요:")

if text_input:
    # KoBART 모델을 사용한 맞춤법 교정 실행
    corrected_text = correct_spelling_with_kobart(text_input)

    # 결과 출력
    st.write(f"교정된 문장: {corrected_text}")

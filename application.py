import streamlit as st
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModelForMaskedLM
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
image = Image.open("293392_218585_2828.jpg")  # 백슬래시 두 번 사용
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

# 버튼 생성
st.sidebar.markdown("""
<button>문서 전체 맞춤법 검사 요청</button>
""", unsafe_allow_html=True)

# 문서 업로드
uploaded_file = st.sidebar.file_uploader("문서를 업로드하세요", type=["docx", "pdf", "txt"])

# 버튼 스타일 적용
button_style = """
<style>
.stButton > button {
    font-family: 'Arial', sans-serif;
    background-color: #B2D7EE; /* 파스텔톤 하늘색 */
    color: black;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 4px;
}
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# 버튼 클릭 시 맞춤법 검사
if uploaded_file is not None:
    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_path = os.path.join("temp", uploaded_file.name)
    if st.sidebar.button("문서 전체 맞춤법 검사"):
        check_spelling(file_path)

if text_input:
    # KoBART 모델을 사용한 맞춤법 교정 실행
    corrected_text = correct_spelling_with_kobart(text_input)

    # 결과 출력
    st.write(f"교정된 문장: {corrected_text}")


# 모델 로드 (Hugging Face 허브에서 제공하는 모델 이름으로 변경 가능)
model_name = "monologg/kobert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_korean(text):
    inputs = tokenizer.encode_plus(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_tokens = tokenizer.convert_ids_to_tokens(torch.argmax(outputs.logits, dim=-1)[0])
    return predicted_tokens

# Streamlit 앱 시작
st.title("HanBert 형태소 분석기")

# 사용자 입력 받기
text_input = st.text_area("분석할 문장을 입력하세요")

# 분석 버튼 클릭 시
if st.button("분석"):
    if text_input:
        try:
            tokens = tokenize_korean(text_input)
            st.write("분석 결과:", tokens)
        except Exception as e:
            st.error(f"오류 발생: {e}")
    else:
        st.warning("분석할 문장을 입력해주세요.")


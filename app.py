import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import random
import re

# ------------------------------------------------------------------
# 1. 모델 준비
# ------------------------------------------------------------------
REPO_ID = "Jay1121/qwen1.5b_3rd"
FILENAME = "qwen1.5b_3rd.Q4_K_M.gguf"

print(f"📥 모델 다운로드 확인: {FILENAME}")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

print("🧠 엔진 시동 중...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=True
)
print("✅ 로딩 완료!")

# ------------------------------------------------------------------
# 2. 시스템 프롬프트 (외국어/이모지 절대 금지 강화)
# ------------------------------------------------------------------
SYSTEM_PROMPT = """ 작성
"""

# ------------------------------------------------------------------
# 2-1. 출력 후처리 필터
#  - 한글/숫자/기본 문장부호/공백만 허용
#  - 금지 문자는 그냥 버리고, 남은 한국어 텍스트만 사용
# ------------------------------------------------------------------
def sanitize_output_korean_only(text: str) -> str:
    allowed_chars = []

    for ch in text:
        code = ord(ch)

        # 한글(완성형 + 자모)
        is_hangul = (
            0xAC00 <= code <= 0xD7A3 or  # 가~힣
            0x3130 <= code <= 0x318F or  # ㄱ~ㆎ
            0x1100 <= code <= 0x11FF     # 옛 자모
        )

        # 숫자
        is_digit = ch.isdigit()

        # 공백
        is_space = ch.isspace()

        # 기본적인 문장부호
        is_punct = ch in ".,!?…~-_()[]{}'\"/:;@#%&*+=|\\"

        if is_hangul or is_digit or is_space or is_punct:
            allowed_chars.append(ch)
        else:
            # 허용되지 않는 문자(영어, 한자, 가나, 이모지 등)는 그냥 버림
            continue

    filtered = "".join(allowed_chars).strip()

    # 다 지워지고 아무것도 안 남았을 때 대비
    if not filtered:
        return "말은 했는데 남는 말이 없네."

    return filtered

# ------------------------------------------------------------------
# 3. 채팅 로직
# ------------------------------------------------------------------
def chat_response(user_input, history_pairs):
    history_pairs = history_pairs or []
    clean_input = (user_input or "").replace(" ", "")

    greeting_words = ["안녕", "ㅎㅇ", "하이", "반가", "접속"]
    is_greeting = any(word in clean_input for word in greeting_words)
    is_balance_game = ("밸런스게임" in clean_input) or ("밸런스질문" in clean_input)

    if is_balance_game:
        topics = ["음식", "연애", "고통", "돈", "초능력", "직장", "친구"]
        topic = random.choice(topics)
        final_instruction = (
            f"(사용자가 밸런스 게임을 하자고 한다. 주제는 '{topic}'이다. "
            "아주 고르기 곤란하고 짜증나는 두 가지 선택지(A vs B)를 제시해라. "
            "말투는 자 어디 한 번 골라보라는 듯이 시니컬하게 해라.) "
            "자, 질문해."
        )
    elif is_greeting:
        final_instruction = (
            f"(친한 친구가 PC통신 채팅방에 접속했다. 반갑게 맞아줘라. "
            "ㅋㅋ나 ㅎㅎ를 섞어서 자연스럽게 인사해라.) "
            f"{user_input}"
        )
    else:
        final_instruction = user_input

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, b in history_pairs:
        if u is None or b is None:
            continue
        messages.append({"role": "user", "content": str(u)})
        messages.append({"role": "assistant", "content": str(b)})
    messages.append({"role": "user", "content": final_instruction})

    r = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        stop=["<|end_of_text|>", "###", "User:"],
        temperature=0.7 if is_balance_game else 0.5,
        top_p=0.9,
        repeat_penalty=1.2
    )

    raw = r["choices"][0]["message"]["content"].strip()
    safe = sanitize_output_korean_only(raw)
    return safe

# ------------------------------------------------------------------
# 4. CSS (스크롤바 중복 해결 + 메시지 간격 축소)
# ------------------------------------------------------------------
PC_COM_CSS = r"""
@import url('https://cdn.jsdelivr.net/gh/neodgm/neodgm-webfont@latest/neodgm/neodgm.css');
:root {
  --pc-blue: #0000AA;
  --pc-white: #EFEFEF;
  --pc-yellow: #FFFF55;
  --pc-amber: #FFB000;
  --pc-cyan: #00AAAA;
  --pc-grey: #AAAAAA;
}
body, .gradio-container {
  background-color: var(--pc-blue) !important;
  font-family: 'NeoDunggeunmo', monospace !important;
  color: var(--pc-white) !important;
}
/* 타이틀바 */
h1 {
  font-family: 'NeoDunggeunmo', monospace !important;
  color: var(--pc-yellow) !important;
  background-color: #000084 !important;
  border-bottom: 2px double var(--pc-white) !important;
  padding-bottom: 10px !important;
  margin-bottom: 20px !important;
  text-align: center;
  font-size: 32px !important;
  letter-spacing: 2px;
}
h1::before { content: "☎ "; }
h1::after { content: " ☎"; }
/* 설명 텍스트 */
.gradio-container p {
  color: var(--pc-cyan) !important;
  font-size: 18px !important;
  border-bottom: 1px dashed var(--pc-grey);
  padding-bottom: 5px;
}
/* 챗봇 컨테이너 - 스크롤바 중복 해결 */
.chatbot {
  background-color: var(--pc-blue) !important;
  border: 2px solid var(--pc-white) !important;
  height: 60vh !important;
  overflow: hidden !important; /* 겉 스크롤바 제거 */
}
/* 내부 스크롤 강제 활성화 */
.chatbot > div {
    height: 100% !important;
    overflow-y: auto !important; /* 속 스크롤바만 남김 */
}
/* =================================================================
   [강제 스타일 적용 구간]
   ================================================================= */
/* 1. 기본 메시지 초기화 */
.chatbot .message, 
.chatbot .message-wrap,
.chatbot .message-row,
div[data-testid="user"],
div[data-testid="bot"] {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}
/* 메시지 행 간격 줄이기 */
.chatbot .message-row,
.chatbot .row {
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
}
/* 2. 유저 메시지 (우측 정렬) */
.chatbot .user-row, 
.chatbot .user,
div[data-testid="user"] {
  display: flex !important;
  width: 100% !important;
  justify-content: flex-end !important;
  margin-left: auto !important;
  background: transparent !important;
  padding: 2px 0 !important;
  margin-bottom: 0 !important;
}
.chatbot .user-row .message, 
.chatbot .user .message,
div[data-testid="user"] .message {
  text-align: right !important;
  color: #FFFFFF !important;
  background: transparent !important;
  padding: 5px 10px !important;
  border: none !important;
  width: auto !important;
  max-width: 80% !important;
}
.chatbot .user-row p, 
.chatbot .user p,
div[data-testid="user"] p {
  color: #FFFFFF !important;
  text-align: right !important;
  margin: 0 !important;
}
.chatbot .user-row .message::after,
.chatbot .user .message::after {
  content: " < 나";
  color: var(--pc-grey);
  margin-left: 5px;
  font-size: 16px;
  display: inline-block;
}
/* 3. 봇 메시지 (좌측 정렬) */
.chatbot .bot-row, 
.chatbot .bot,
div[data-testid="bot"] {
  display: flex !important;
  width: 100% !important;
  justify-content: flex-start !important;
  background: transparent !important;
  padding: 2px 0 !important;
  margin-bottom: 0 !important;
}
.chatbot .bot-row .message, 
.chatbot .bot .message,
div[data-testid="bot"] .message {
  text-align: left !important;
  color: var(--pc-amber) !important;
  background: transparent !important;
  padding: 5px 10px !important;
  border: none !important;
  width: auto !important;
}
.chatbot .bot-row p, 
.chatbot .bot p,
div[data-testid="bot"] p {
  color: var(--pc-amber) !important;
  margin: 0 !important;
}
.chatbot .bot-row .message::before,
.chatbot .bot .message::before {
  content: "똘배 > ";
  color: var(--pc-cyan);
  margin-right: 5px;
  font-size: 16px;
  display: inline-block;
}
/* 4. 로딩(초시계) 스타일 */
.chatbot .pending,
.chatbot .generating,
.chatbot .message.pending,
.chatbot .message.generating,
.chatbot .wrap.default.full {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.chatbot .pending table, 
.chatbot .pending tr, 
.chatbot .pending td,
.chatbot .generating table, 
.chatbot .generating tr, 
chatbot .generating td {
    background: transparent !important;
    border: none !important;
}
.chatbot .pending span, 
.chatbot .generating span,
span.progress-text {
    color: #FFFFFF !important;
    background: transparent !important;
    font-family: 'NeoDunggeunmo', monospace !important;
    font-size: 16px !important;
}
.chatbot .load-wrap,
.chatbot .loading-indicator,
.chatbot .meta-text {
    display: none !important;
}
.avatar { display: none !important; }
/* ================================================================= */
.input-container {
  background-color: var(--pc-blue) !important;
  border-top: 2px double var(--pc-white) !important;
  margin-top: 10px !important;
  gap: 10px !important;
}
textarea, input {
  background-color: var(--pc-blue) !important;
  color: var(--pc-white) !important;
  border: 1px solid var(--pc-grey) !important;
  border-radius: 0 !important;
  font-family: 'NeoDunggeunmo', monospace !important;
  font-size: 20px !important;
  outline: none !important;
  box-shadow: none !important;
}
button.primary {
  background: var(--pc-grey) !important;
  color: #000 !important;
  border: 1px solid var(--pc-white) !important;
  border-radius: 0 !important;
  font-family: 'NeoDunggeunmo', monospace !important;
  box-shadow: 2px 2px 0px #000 !important;
}
button.primary:hover { background: var(--pc-white) !important; }
#clear-btn {
  background: transparent !important;
  color: var(--pc-grey) !important;
  border: 1px solid var(--pc-grey) !important;
  font-size: 14px !important;
  padding: 2px 10px !important;
  margin-top: 5px !important;
  width: auto !important;
}
#clear-btn:hover { color: var(--pc-white) !important; border-color: var(--pc-white) !important; }
.example-btn {
  background: transparent !important;
  color: var(--pc-cyan) !important;
  border: 1px solid var(--pc-cyan) !important;
  border-radius: 0 !important;
  padding: 5px 15px !important;
  font-size: 16px !important;
  font-family: 'NeoDunggeunmo', monospace !important;
  margin-right: 8px !important;
  margin-bottom: 8px !important;
}
.example-btn:hover {
  background: var(--pc-cyan) !important;
  color: #000 !important;
  cursor: pointer !important;
}
footer { display: none !important; }
"""

# ------------------------------------------------------------------
# 5. App
# ------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Base(), css=PC_COM_CSS, title="CHOLLIAN 98") as demo:
    gr.Markdown("# ≪ 어솨요~ ≫")
    gr.Markdown(">> 01410 접속 성공... [대화실]에 입장하셨습니다.")

    history_state = gr.State([])

    chatbot = gr.Chatbot(show_label=False, elem_classes="chatbot")

    with gr.Row(elem_classes="input-container"):
        msg = gr.Textbox(
            scale=8, show_label=False, container=False,
            placeholder="명령어를 입력하세요..."
        )
        submit_btn = gr.Button("[ 전송 ]", scale=1, variant="primary")

    clear = gr.Button("[ 화면 지우기 ]", elem_id="clear-btn")

    gr.Markdown(">> 빠른 명령어 입력 (클릭)", elem_id="example-label")
    with gr.Row():
        btn1 = gr.Button("하이 방가방가", elem_classes="example-btn")
        btn2 = gr.Button("밸런스게임 ㄱㄱ", elem_classes="example-btn")
        btn3 = gr.Button("오늘 기분 꿀꿀하네..", elem_classes="example-btn")
        btn4 = gr.Button("야 밥 뭐먹지 추천좀", elem_classes="example-btn")

    def user(user_input, history):
        history = history or []
        new_history = history + [[user_input, None]]
        return "", new_history, new_history

    def bot(history):
        if not history:
            return history, history
        user_input = history[-1][0]
        hist_pairs = []
        for u, b in history[:-1]:
            if u is None or b is None:
                continue
            hist_pairs.append((u, b))

        bot_out = chat_response(user_input, hist_pairs)
        history[-1][1] = bot_out
        return history, history

    msg.submit(
        user, [msg, history_state], [msg, history_state, chatbot],
        queue=False, api_name=False
    ).then(
        bot, [history_state], [history_state, chatbot],
        queue=False, api_name=False
    )

    submit_btn.click(
        user, [msg, history_state], [msg, history_state, chatbot],
        queue=False, api_name=False
    ).then(
        bot, [history_state], [history_state, chatbot],
        queue=False, api_name=False
    )
    
    clear.click(
        lambda: ([], []), None, [history_state, chatbot],
        queue=False, api_name=False
    )

    for btn, text in [
        (btn1, "하이 방가방가"), 
        (btn2, "밸런스게임 ㄱㄱ"), 
        (btn3, "오늘 기분 거지같누"), 
        (btn4, "야 밥 뭐먹지 추천좀")
    ]:
        btn.click(
            lambda t=text: t, None, msg,
            queue=False, api_name=False
        ).then(
            user, [msg, history_state],
            [msg, history_state, chatbot],
            queue=False, api_name=False
        ).then(
            bot, [history_state],
            [history_state, chatbot],
            queue=False, api_name=False
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)


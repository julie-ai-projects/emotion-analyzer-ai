import gradio as gr
from transformers import pipeline

# Модель анализа тональности (многоязычная)
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Цвета и эмодзи для отображения
EMOTION_STYLES = {
    "positive": {"emoji": "😊", "color": "#A7E9AF"},
    "neutral": {"emoji": "😐", "color": "#F7E9A0"},
    "negative": {"emoji": "😞", "color": "#F4A9A8"}
}

def analyze_emotion(text):
    result = classifier(text)[0]
    label = result["label"].lower()
    score = result["score"]

    style = EMOTION_STYLES.get(label, {"emoji": "❓", "color": "#DDDDDD"})
    emotion_text = {
        "positive": "Позитив / Positive / 긍정적",
        "neutral": "Нейтрально / Neutral / 중립적",
        "negative": "Негатив / Negative / 부정적"
    }.get(label, label)

    html_output = f"""
    <div style="background-color:{style['color']}; 
                border-radius:20px; 
                padding:20px; 
                text-align:center;
                font-size:20px;
                box-shadow:0 2px 10px rgba(0,0,0,0.1);
                transition: all 0.3s ease;">
        <div style="font-size:50px;">{style['emoji']}</div>
        <div style="font-weight:bold;">{emotion_text}</div>
        <div style="margin-top:10px;">Уверенность: {score*100:.1f}%</div>
    </div>
    """
    return html_output

# Интерфейс
demo = gr.Interface(
    fn=analyze_emotion,
    inputs=gr.Textbox(lines=2, label="Введите текст на русском, английском или корейском"),
    outputs=gr.HTML(label="Результат"),
    title="🌍 Многоязычный AI-анализатор эмоций",
    description="Определяет эмоциональную окраску текста (позитив, нейтрально, негатив) на русском, английском и корейском языках.",
    theme="default"
)

demo.launch()


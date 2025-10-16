import gradio as gr
from transformers import pipeline

# Многоязычная модель эмоций (поддерживает русский, английский, корейский и др.)
classifier = pipeline("text-classification", model="j-hartmann/emotion-multilingual-roberta-base")

# Анализ эмоций
def analyze_emotion(text):
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]
    
    # Перевод меток эмоций (чтобы выводить по-русски)
    translations = {
        "joy": "радость",
        "anger": "злость",
        "sadness": "грусть",
        "fear": "страх",
        "disgust": "отвращение",
        "surprise": "удивление",
        "neutral": "нейтрально"
    }
    emotion = translations.get(label.lower(), label)
    
    return f"🧠 Эмоция: {emotion} (уверенность {score:.2f})"

# Интерфейс
demo = gr.Interface(
    fn=analyze_emotion,
    inputs=gr.Textbox(lines=2, label="Введите текст на русском, английском или корейском"),
    outputs="text",
    title="🌍 Многоязычный AI-анализатор эмоций",
    description="Определяет эмоцию текста на русском, английском или корейском языке.",
    theme="default"
)

demo.launch()

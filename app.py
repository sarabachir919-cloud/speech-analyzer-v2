import streamlit as st
import os
import torch
from transformers import pipeline as hf_pipeline
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile

# تحميل متغيرات البيئة (مفاتيح API)
load_dotenv()

# إعداد الصفحة
st.set_page_config(page_title="محلل الخطابات", layout="centered")
st.title("🎙️ وكيل تحليل الخطابات بالذكاء الاصطناعي")
st.write("ارفع ملفاً صوتياً لخطاب أو إلقاء، واحصل على تقرير مفصل ونصائح لتحسين الأداء.")

# تحميل النماذج مرة واحدة في بداية التشغيل لتسريع العمليات اللاحقة
@st.cache_resource
def load_models():
    """تحميل النماذج الثقيلة مرة واحدة وتخزينها في الذاكرة المؤقتة"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("مفتاح Groq API غير موجود! يرجى إعداده في متغيرات البيئة.")
        st.stop()
    
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
    
    device = 0 if torch.cuda.is_available() else -1
    asr_pipeline = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=device
    )
    return llm, asr_pipeline

llm, asr_pipeline = load_models()

# الوظائف الأساسية (نفسها من Colab مع تعديلات بسيطة)
def transcribe_audio(audio_file_path):
    """تحويل الصوت إلى نص"""
    result = asr_pipeline(audio_file_path, return_timestamps=True, chunk_length_s=30, stride_length_s=5)
    transcript = "".join(chunk['text'] for chunk in result['chunks'])
    return transcript.strip()

def analyze_and_generate_report(transcript):
    """تحليل النص وتوليد التقرير"""
    prompt_template = """
    أنت خبير متخصص في فن الخطابة. قم بتحليل النص التالي وتقديم تقرير مفصل.
    **نص الخطاب:** {transcript}
    **التقرير (باللغة العربية):**
    1.  **ملخص عام:**
    2.  **نقاط القوة:**
    3.  **مجالات التحسين:**
    4.  **الكلمات الملءئة (Filler Words):**
    5.  **هيكل الخطاب:**
    6.  **نصائح عملية قابلة للتطبيق:**
    """
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    report = chain.invoke({"transcript": transcript})
    return report.content

# --- الواجهة الرئيسية للتطبيق ---

# 1. مكون رفع الملفات
uploaded_file = st.file_uploader("اختر ملفاً صوتياً (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    st.success(f"تم رفع الملف: {uploaded_file.name}")

    # 2. زر بدء التحليل
    if st.button("🚀 حلل الخطاب الآن"):
        with st.spinner("جاري تحويل الصوت إلى نص... قد يستغرق هذا بعض الوقت."):
            # حفظ الملف المرفوع مؤقتاً على القرص
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                tmpfile_path = tmpfile.name
            
            # تحويل الصوت إلى نص
            transcript = transcribe_audio(tmpfile_path)
            st.subheader("📝 النص المستخرج من الخطاب:")
            st.write(transcript)

        with st.spinner("جاري تحليل النص وتوليد التقرير..."):
            # تحليل النص
            final_report = analyze_and_generate_report(transcript)
            
            # عرض التقرير النهائي
            st.subheader("📊 التقرير النهائي والنصائح:")
            st.markdown(final_report)
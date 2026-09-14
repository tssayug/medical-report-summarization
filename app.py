import os
import time
import torch
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (Glassmorphism + Dark Animations)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MedSummarize AI | Clinical NLP",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Dark Theme Base & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #c9d1d9;
    }
    
    /* Hero Header Banner */
    .hero-container {
        background: linear-gradient(90deg, rgba(31,111,235,0.15) 0%, rgba(137,87,229,0.15) 100%);
        border: 1px solid rgba(56,139,253,0.3);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 25px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        animation: fadeIn 1s ease-in-out;
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #8b949e;
        margin: 0;
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.65);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }

    /* Stat Badges */
    .stat-badge {
        display: inline-block;
        background: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 8px;
    }

    /* Pulse Indicator */
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #3fb950;
        box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.7);
        animation: pulse 1.8s infinite;
        margin-right: 8px;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(63, 185, 80, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Styled Text Area */
    .stTextArea textarea {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #238636 0%, #2ea043 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 4px 15px rgba(46, 160, 67, 0.4) !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Model Loading & Caching Pipeline
# -----------------------------------------------------------------------------
BASE_MODEL = "google/flan-t5-base"
ADAPTER_PATH = os.path.join("outputs", "flan_t5_lora_results", "best_lora_model")

@st.cache_resource(show_spinner=False)
def load_clinical_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    
    if os.path.exists(ADAPTER_PATH):
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        # Fallback if running on remote without local folder path
        model = base_model
        
    model.eval()
    return tokenizer, model

# -----------------------------------------------------------------------------
# 3. Sidebar Configuration & Architecture Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Inference Settings")
    max_tokens = st.slider("Max Output Length (Tokens)", 64, 256, 128, step=16)
    beam_size = st.slider("Beam Search Width", 1, 4, 2)
    
    st.markdown("---")
    st.markdown("### 📊 System Metadata")
    st.markdown("""
    <div style="font-size: 0.85rem; color: #8b949e;">
        <b>Base Architecture:</b> FLAN-T5-Base<br>
        <b>Fine-Tuning:</b> PEFT / LoRA (r=8, α=32)<br>
        <b>Trainable Parameters:</b> 884,736 (0.35%)<br>
        <b>Adapter Footprint:</b> ~3.7 MB<br>
        <b>Training Device:</b> NVIDIA Tesla T4
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div style="text-align:center;"><span class="pulse-dot"></span><b>Model Status: Operational</b></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Hero Header Section
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🩺 MedSummarize AI</div>
    <div class="hero-subtitle">Parameter-Efficient Abstractive Summarization of MIMIC-IV Discharge Reports using LoRA FLAN-T5</div>
    <div style="margin-top: 15px;">
        <span class="stat-badge">⚡ 3.7 MB Lightweight Adapter</span>
        <span class="stat-badge">🎯 ROUGE-1: 26.76%</span>
        <span class="stat-badge">🧬 0.35% Trainable Weights</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Main Application Tabs
# -----------------------------------------------------------------------------
tab_app, tab_metrics, tab_architecture = st.tabs(["🚀 Live Summarizer", "📈 Performance Analytics", "🏗 System Architecture"])

# Preset Clinical Cases for 1-Click Testing
PRESET_CASES = {
    "Select a preset clinical case...": "",
    "Case 1: Lithium Toxicity & Acute Renal Failure": (
        "Chief Complaint: Leg weakness; transferred for ARF and lithium toxicity\n"
        "History of Present Illness: This is a 68 year old man with bipolar disorder as well as PMH "
        "significant for DM, obesity, hypertension, and elevated baseline creatinine as well as recent "
        "increase in lithium dosing. He presents with complaints of weakness and difficulty walking, "
        "accompanied by coarse hand tremors. Serum lithium level on arrival was markedly elevated at 2.4 mEq/L. "
        "Urine output has decreased significantly over the past 24 hours. Nephrology consulted for acute renal management."
    ),
    "Case 2: Acute Cardiac Failure & Dyspnea": (
        "Chief Complaint: Shortness of breath, bilateral lower extremity edema\n"
        "History of Present Illness: A 72-year-old female with known history of congestive heart failure (EF 30%), "
        "coronary artery disease s/p CABG, and atrial fibrillation presents with 4 days of worsening exertional dyspnea, "
        "orthopnea requiring 3 pillows, and paroxysmal nocturnal dyspnea. Physical exam reveals bilateral bibasilar crackles, "
        "JVD at 10 cm, and 3+ pitting edema to mid-calf. Chest X-ray confirms acute pulmonary congestion."
    )
}

with tab_app:
    col_input, col_output = st.columns([1, 1], gap="medium")
    
    with col_input:
        st.markdown("#### 📥 Input Clinical Discharge Report")
        
        # Preset selection dropdown
        selected_preset = st.selectbox("Quick Test Presets:", list(PRESET_CASES.keys()))
        default_text = PRESET_CASES[selected_preset] if selected_preset != "Select a preset clinical case..." else ""
        
        user_input = st.text_area(
            "Paste raw clinical narrative text:",
            value=default_text,
            height=300,
            placeholder="Type or paste medical discharge summary here..."
        )
        
        # Word and token stats
        word_count = len(user_input.split())
        st.markdown(f"<div style='font-size:0.8rem; color:#8b949e;'>Character Count: {len(user_input)} | Word Count: {word_count}</div>", unsafe_allow_html=True)
        
        generate_btn = st.button("✨ Generate Clinical Summary", use_container_width=True)

    with col_output:
        st.markdown("#### 📋 Model Abstractive Output")
        
        if generate_btn:
            if not user_input.strip():
                st.warning("⚠️ Please provide input clinical text or select a preset case.")
            else:
                with st.spinner("⚡ Loading adapter weights & synthesizing clinical prose..."):
                    start_time = time.time()
                    tokenizer, model = load_clinical_model()
                    
                    prompt = "summarize: " + user_input.strip()
                    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, 
                            max_new_tokens=max_tokens, 
                            num_beams=beam_size
                        )
                    
                    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    latency = time.time() - start_time
                
                # Render summary inside styled card
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #3fb950;">
                    <div style="font-weight:600; color:#58a6ff; margin-bottom:8px;">Generated Narrative Summary:</div>
                    <div style="font-size: 0.98rem; line-height: 1.6; color:#e6edf3;">{summary}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Performance metrics bar below card
                st.markdown(f"""
                <div style="display:flex; justify-between; font-size:0.8rem; color:#8b949e; background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:6px;">
                    <span>⏱️ Latency: {latency:.2f}s</span> &nbsp;|&nbsp; 
                    <span>📝 Output Words: {len(summary.split())}</span> &nbsp;|&nbsp; 
                    <span>📊 Compression Ratio: {((1 - len(summary)/max(1, len(user_input)))*100):.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    label="💾 Export Summary (.txt)",
                    data=summary,
                    file_name="clinical_summary.txt",
                    mime="text/plain"
                )
        else:
            st.info("👈 Enter clinical text on the left and click **Generate** to run inference.")

# -----------------------------------------------------------------------------
# 6. Tab 2: Interactive Analytics & Visualizations
# -----------------------------------------------------------------------------
with tab_metrics:
    st.markdown("### 📊 Empirical Evaluation & Parameter Footprint")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Interactive ROUGE Bar Chart
        df_rouge = pd.DataFrame({
            'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum'],
            'Score (%)': [26.76, 9.81, 19.13, 19.12]
        })
        fig_rouge = px.bar(
            df_rouge, x='Metric', y='Score (%)', text='Score (%)',
            color='Metric',
            color_discrete_sequence=['#1f6feb', '#2ea043', '#d95f02', '#8957e5'],
            title="A. ROUGE Metrics on Unseen Test Set (500 Samples)"
        )
        fig_rouge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'), showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig_rouge.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_rouge.update_yaxes(range=[0, 35], gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig_rouge, use_container_width=True)
        
    with col_chart2:
        # Interactive Parameter Pie/Donut Chart
        df_params = pd.DataFrame({
            'Component': ['Frozen Base Parameters', 'LoRA Trainable Parameters'],
            'Count': [247577856, 884736]
        })
        fig_params = px.pie(
            df_params, values='Count', names='Component', hole=0.6,
            color_discrete_sequence=['#30363d', '#2ea043'],
            title="B. Parameter Footprint Allocation"
        )
        fig_params.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig_params.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_params, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. Tab 3: System Architecture
# -----------------------------------------------------------------------------
with tab_architecture:
    st.markdown("### 🏗 Pipeline Architecture & Design Specifications")
    st.markdown("""
    <div class="glass-card">
        <h4>System Pipeline Flow</h4>
        <p>1. <b>Data Pipeline:</b> MIMIC-IV Clinical discharge reports filtered via compression ratios and keyword density (2,406 training / 500 test records).</p>
        <p>2. <b>LoRA Fine-Tuning:</b> Low-Rank Adaptation injected into Query ($q$) and Value ($v$) attention projection matrices of <code>google/flan-t5-base</code>.</p>
        <p>3. <b>FP32 Precision Safeguard:</b> Trained under FP32 precision to prevent numerical gradient underflow bug inherent to T5 architectures.</p>
        <p>4. <b>Export & Local Evaluation:</b> Lightweight ~3.7 MB adapter weights loaded dynamically for low-latency CPU/GPU inference.</p>
    </div>
    """, unsafe_allow_html=True)
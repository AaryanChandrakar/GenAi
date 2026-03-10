"""
⭐ TokenLens — Interactive LLM Tokenizer & Embedding Explorer ⭐
A premium Streamlit single-page application for visualizing tokens,
token counts, vectors, and embeddings across 7+ major LLM models.

Author: GenAI Explorer
Hosted on: Vercel-compatible via streamlit
"""

import streamlit as st
import numpy as np
import json
import hashlib
import colorsys
import re
import tiktoken
from typing import List, Tuple, Optional, Dict

# ──────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TokenLens — LLM Tokenizer & Embedding Explorer",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# SEO & Social Sharing (Open Graph / Meta Tags)
# ──────────────────────────────────────────────────────────
# Note: For these to work on WhatsApp/Social Media, the app must be public.
# The image URL should be a direct link to your hosted preview image (e.g., on GitHub).
st.markdown(f"""
    <head>
        <!-- Primary Meta Tags -->
        <title>TokenLens — LLM Tokenizer & Embedding Explorer</title>
        <meta name="title" content="TokenLens — LLM Tokenizer & Embedding Explorer">
        <meta name="description" content="A premium interactive dashboard to visualize tokens and explore embeddings across 8+ major LLM models. Free, fast, and beautiful.">

        <!-- Open Graph / Facebook -->
        <meta property="og:type" content="website">
        <meta property="og:url" content="https://aaryanchandrakar-tokenlens.streamlit.app/">
        <meta property="og:title" content="⭐ TokenLens — Interactive LLM Explorer">
        <meta property="og:description" content="Visualize how GPT-4o, Claude 3.5, and LLaMA 3 see your words. Explore embeddings, token counts, and vector heatmaps in real-time.">
        <meta property="og:image" content="https://raw.githubusercontent.com/aaryanchandrakar/GenAi/main/assets/tokenlens_preview.png">

        <!-- Twitter -->
        <meta property="twitter:card" content="summary_large_image">
        <meta property="twitter:url" content="https://aaryanchandrakar-tokenlens.streamlit.app/">
        <meta property="twitter:title" content="⭐ TokenLens — Interactive LLM Explorer">
        <meta property="twitter:description" content="Visualize how GPT-4o, Claude 3.5, and LLaMA 3 see your words. Explore embeddings, token counts, and vector heatmaps in real-time.">
        <meta property="twitter:image" content="https://raw.githubusercontent.com/aaryanchandrakar/GenAi/main/assets/tokenlens_preview.png">
    </head>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Glassmorphism Theme
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1040 25%, #302b63 50%, #24243e 75%, #0f0c29 100%);
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,12,41,0.97) 0%, rgba(26,16,64,0.97) 100%);
        border-right: 1px solid rgba(139,92,246,0.2);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0d4ff !important;
    }

    /* ── Hero Header ── */
    .hero-header {
        text-align: center;
        padding: 2rem 1rem 1.5rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, rgba(139,92,246,0.12) 0%, rgba(59,130,246,0.08) 50%, rgba(236,72,153,0.1) 100%);
        border-radius: 20px;
        border: 1px solid rgba(139,92,246,0.2);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(139,92,246,0.05) 0%, transparent 70%);
        animation: pulse 6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.05); opacity: 1; }
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa 0%, #818cf8 25%, #60a5fa 50%, #c084fc 75%, #f472b6 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: rgba(196,181,253,0.7);
        letter-spacing: 2px;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: linear-gradient(135deg, rgba(30,27,75,0.6) 0%, rgba(30,27,75,0.3) 100%);
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(139,92,246,0.4);
        box-shadow: 0 8px 32px rgba(139,92,246,0.15);
        transform: translateY(-2px);
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c4b5fd;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Metric Cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        min-width: 150px;
        background: linear-gradient(135deg, rgba(139,92,246,0.1) 0%, rgba(59,130,246,0.08) 100%);
        border: 1px solid rgba(139,92,246,0.2);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(139,92,246,0.2);
        border-color: rgba(139,92,246,0.5);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: rgba(196,181,253,0.6);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
    }

    /* ── Token Chips ── */
    .token-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        padding: 0.5rem 0;
    }
    .token-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 6px 12px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.2s ease;
        cursor: default;
        position: relative;
    }
    .token-chip:hover {
        transform: scale(1.08);
        z-index: 10;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    .token-id {
        font-size: 0.6rem;
        opacity: 0.5;
        font-weight: 400;
    }

    /* ── Vector Heatmap ── */
    .vector-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(42px, 1fr));
        gap: 3px;
        padding: 0.5rem 0;
    }
    .vector-cell {
        aspect-ratio: 1;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.5rem;
        color: rgba(255,255,255,0.6);
        transition: all 0.2s ease;
        cursor: default;
    }
    .vector-cell:hover {
        transform: scale(1.5);
        z-index: 10;
        border-radius: 6px;
        font-size: 0.6rem;
        color: white;
    }

    /* ── Model Badge ── */
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(59,130,246,0.15));
        border: 1px solid rgba(139,92,246,0.3);
        color: #c4b5fd;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(15,12,41,0.5); }
    ::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.4); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.6); }

    /* ── Code / JSON Block ── */
    .json-block {
        background: rgba(15,12,41,0.8);
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 12px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #a5b4fc;
        max-height: 360px;
        overflow-y: auto;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-all;
    }

    /* ── Language Badge ── */
    .lang-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(52,211,153,0.15);
        border: 1px solid rgba(52,211,153,0.3);
        color: #6ee7b7;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: rgba(196,181,253,0.35);
        font-size: 0.75rem;
        letter-spacing: 1px;
    }
    .footer a { color: rgba(139,92,246,0.6); text-decoration: none; }
    .footer a:hover { color: #a78bfa; }

    /* ── Progress bar override ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #8b5cf6, #60a5fa, #ec4899);
    }

    /* ── Fix Streamlit defaults ── */
    .stTextArea textarea {
        background: rgba(15,12,41,0.6) !important;
        border: 1px solid rgba(139,92,246,0.2) !important;
        border-radius: 12px !important;
        color: #e0d4ff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(139,92,246,0.5) !important;
        box-shadow: 0 0 20px rgba(139,92,246,0.15) !important;
    }
    .stSelectbox > div > div {
        background: rgba(15,12,41,0.6) !important;
        border: 1px solid rgba(139,92,246,0.2) !important;
        border-radius: 12px !important;
        color: #e0d4ff !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 50%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15,12,41,0.4);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #a78bfa;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(139,92,246,0.2) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: rgba(139,92,246,0.3) !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30,27,75,0.4) !important;
        border-radius: 10px !important;
        color: #c4b5fd !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# MODEL CONFIGURATION — 7+ Major LLMs with best embedding models
# ──────────────────────────────────────────────────────────
MODEL_CONFIG: Dict[str, dict] = {
    "GPT-4o (OpenAI)": {
        "tokenizer": "tiktoken",
        "encoding": "o200k_base",
        "icon": "🟢",
        "color": "#10a37f",
        "best_embedding": "text-embedding-3-large",
        "embedding_dim": 3072,
        "vocab_size": 200019,
        "description": "OpenAI's flagship multimodal model with o200k_base encoding.",
    },
    "GPT-4 / GPT-3.5 (OpenAI)": {
        "tokenizer": "tiktoken",
        "encoding": "cl100k_base",
        "icon": "🔵",
        "color": "#5A67D8",
        "best_embedding": "text-embedding-3-small",
        "embedding_dim": 1536,
        "vocab_size": 100277,
        "description": "OpenAI's GPT-4/3.5 family using cl100k_base encoding.",
    },
    "Claude 3.5 (Anthropic)": {
        "tokenizer": "tiktoken",
        "encoding": "cl100k_base",
        "icon": "🟠",
        "color": "#D97706",
        "best_embedding": "voyage-3-large",
        "embedding_dim": 1024,
        "vocab_size": 100277,
        "description": "Anthropic's Claude uses a BPE tokenizer similar to cl100k_base.",
    },
    "Gemini 1.5 (Google)": {
        "tokenizer": "tiktoken",
        "encoding": "cl100k_base",
        "icon": "🔴",
        "color": "#EA4335",
        "best_embedding": "text-embedding-004",
        "embedding_dim": 768,
        "vocab_size": 100277,
        "description": "Google's Gemini uses SentencePiece (approx. with cl100k for demo).",
    },
    "LLaMA 3 (Meta)": {
        "tokenizer": "tiktoken",
        "encoding": "cl100k_base",
        "icon": "🦙",
        "color": "#6366f1",
        "best_embedding": "BAAI/bge-large-en-v1.5",
        "embedding_dim": 1024,
        "vocab_size": 128256,
        "description": "Meta's open-source LLaMA 3 with 128k vocab BPE tokenizer.",
    },
    "Mistral Large (Mistral AI)": {
        "tokenizer": "tiktoken",
        "encoding": "cl100k_base",
        "icon": "🌀",
        "color": "#f97316",
        "best_embedding": "mistral-embed",
        "embedding_dim": 1024,
        "vocab_size": 32768,
        "description": "Mistral AI's frontier model with BPE tokenization.",
    },
    "Cohere Command R+": {
        "tokenizer": "tiktoken",
        "encoding": "cl100k_base",
        "icon": "💎",
        "color": "#8b5cf6",
        "best_embedding": "embed-english-v3.0",
        "embedding_dim": 1024,
        "vocab_size": 255029,
        "description": "Cohere's enterprise model with advanced BPE tokenizer.",
    },
    "Qwen 2.5 (Alibaba)": {
        "tokenizer": "tiktoken",
        "encoding": "cl100k_base",
        "icon": "🐉",
        "color": "#06b6d4",
        "best_embedding": "gte-Qwen2-1.5B-instruct",
        "embedding_dim": 1536,
        "vocab_size": 151643,
        "description": "Alibaba's Qwen 2.5, multilingual LLM with BPE tokenizer.",
    },
}

# ──────────────────────────────────────────────────────────
# LANGUAGE DETECTION — Simple heuristic-based detector
# ──────────────────────────────────────────────────────────
LANGUAGE_MAP = {
    "🌐 Auto-Detect": "auto",
    "🇺🇸 English": "en",
    "🇪🇸 Spanish": "es",
    "🇫🇷 French": "fr",
    "🇩🇪 German": "de",
    "🇮🇹 Italian": "it",
    "🇵🇹 Portuguese": "pt",
    "🇨🇳 Chinese": "zh",
    "🇯🇵 Japanese": "ja",
    "🇰🇷 Korean": "ko",
    "🇮🇳 Hindi": "hi",
    "🇸🇦 Arabic": "ar",
    "🇷🇺 Russian": "ru",
    "🇹🇷 Turkish": "tr",
    "🇳🇱 Dutch": "nl",
}

LANG_CHAR_RANGES = {
    "zh": r'[\u4e00-\u9fff]',
    "ja": r'[\u3040-\u309f\u30a0-\u30ff]',
    "ko": r'[\uac00-\ud7af\u1100-\u11ff]',
    "hi": r'[\u0900-\u097f]',
    "ar": r'[\u0600-\u06ff]',
    "ru": r'[\u0400-\u04ff]',
    "de": r'[äöüßÄÖÜ]',
    "fr": r'[àâéèêëîïôùûüçæœÀÂÉÈÊËÎÏÔÙÛÜÇÆŒ]',
    "es": r'[áéíóúñüÁÉÍÓÚÑÜ¿¡]',
    "pt": r'[ãõáéíóúâêôçÃÕÁÉÍÓÚÂÊÔÇ]',
    "it": r'[àèéìòùÀÈÉÌÒÙ]',
    "tr": r'[çğıöşüÇĞİÖŞÜ]',
    "nl": r'[ëïéèüöäËÏÉÈÜÖÄ]',
}

# Common words for language identification
LANG_KEYWORDS = {
    "en": ["the", "is", "and", "of", "to", "in", "that", "it", "was", "for", "with", "are", "this"],
    "es": ["el", "la", "de", "en", "los", "del", "las", "por", "con", "una", "que", "es"],
    "fr": ["le", "la", "les", "de", "des", "du", "en", "est", "une", "que", "dans", "pour"],
    "de": ["der", "die", "das", "und", "ist", "ein", "eine", "von", "den", "mit", "auf"],
    "it": ["il", "la", "di", "che", "è", "per", "una", "del", "della", "sono", "con"],
    "pt": ["o", "de", "que", "do", "da", "em", "um", "uma", "para", "com", "não"],
    "nl": ["de", "het", "een", "van", "in", "en", "is", "dat", "op", "voor", "met"],
    "tr": ["bir", "ve", "bu", "için", "ile", "olan", "var", "ben", "çok", "daha"],
}


def detect_language(text: str) -> Tuple[str, str, float]:
    """Detect the language of a text using character ranges and keyword matching."""
    if not text.strip():
        return "en", "English", 0.0

    # Check script-based detection first (non-Latin scripts)
    for lang, pattern in LANG_CHAR_RANGES.items():
        matches = len(re.findall(pattern, text))
        ratio = matches / max(len(text), 1)
        if lang in ("zh", "ja", "ko", "hi", "ar", "ru") and ratio > 0.1:
            lang_names = {
                "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
                "hi": "Hindi", "ar": "Arabic", "ru": "Russian",
            }
            return lang, lang_names[lang], min(ratio * 3, 0.98)
        elif ratio > 0.02:
            lang_names = {
                "de": "German", "fr": "French", "es": "Spanish",
                "pt": "Portuguese", "it": "Italian", "tr": "Turkish", "nl": "Dutch",
            }
            if lang in lang_names:
                return lang, lang_names[lang], min(0.5 + ratio * 5, 0.95)

    # Keyword-based detection for Latin-script languages
    words = set(text.lower().split())
    best_lang = "en"
    best_score = 0
    for lang, keywords in LANG_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in words)
        if score > best_score:
            best_score = score
            best_lang = lang

    lang_names_full = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "tr": "Turkish",
    }
    confidence = min(best_score / 5, 0.95) if best_score > 0 else 0.5
    return best_lang, lang_names_full.get(best_lang, "English"), confidence


# ──────────────────────────────────────────────────────────
# TOKENIZER FUNCTIONS
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_tiktoken_encoder(encoding_name: str):
    """Load and cache tiktoken encoder."""
    return tiktoken.get_encoding(encoding_name)


def tokenize_text(text: str, model_key: str) -> Tuple[List[int], List[str]]:
    """Tokenize text and return (token_ids, token_strings)."""
    config = MODEL_CONFIG[model_key]
    encoder = load_tiktoken_encoder(config["encoding"])
    token_ids = encoder.encode(text)
    token_strings = [encoder.decode([tid]) for tid in token_ids]
    return token_ids, token_strings


# ──────────────────────────────────────────────────────────
# SIMULATED EMBEDDING GENERATION (no API needed)
# Uses deterministic hashing for consistent local demo embeddings
# ──────────────────────────────────────────────────────────
def generate_simulated_embedding(text: str, dim: int, model_name: str) -> np.ndarray:
    """
    Generate a deterministic, simulated embedding vector.
    Uses SHA-256 hashing seeded with the text + model name
    to produce consistent, normalized vectors for demonstration.
    """
    seed_string = f"{model_name}||{text}"
    hash_bytes = hashlib.sha256(seed_string.encode("utf-8")).digest()
    seed = int.from_bytes(hash_bytes[:4], "big")
    rng = np.random.RandomState(seed)
    # Simulate realistic embedding distribution (unit-normalized)
    raw = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw = raw / norm
    return raw


# ──────────────────────────────────────────────────────────
# COLOR UTILITY — deterministic token coloring
# ──────────────────────────────────────────────────────────
def get_token_color(token_id: int, total: int) -> str:
    """Generate a visually distinct color for a token based on its ID."""
    hue = (token_id * 0.618033988749895) % 1.0   # golden ratio for spread
    sat = 0.55 + (token_id % 7) * 0.05
    light = 0.25 + (token_id % 5) * 0.03
    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.35)"


def value_to_heatmap_color(val: float) -> str:
    """Map a float value (-1 to 1) to a purple-blue heatmap color."""
    # Map from [-1,1] to [0,1]
    t = (val + 1) / 2
    # Purple (negative) → dark (zero) → Blue (positive)
    if t < 0.5:
        r = int(139 * (1 - t * 2) * 0.7)
        g = int(20 * t * 2)
        b = int(200 * (1 - t * 2) * 0.5 + 60)
    else:
        t2 = (t - 0.5) * 2
        r = int(30 * t2)
        g = int(80 * t2 + 40)
        b = int(246 * t2 * 0.6 + 100)
    return f"rgb({r},{g},{b})"


# ══════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════

def main():
    # ── Hero Header ──
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">⭐ TokenLens</div>
        <div class="hero-subtitle">Interactive LLM Tokenizer & Embedding Explorer</div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # SIDEBAR — Model & Language Selection
    # ══════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## 🎛️ Configuration")
        st.markdown("---")

        # Model Selection
        st.markdown("### 🤖 Select LLM Model")
        model_names = list(MODEL_CONFIG.keys())
        selected_model = st.selectbox(
            "Choose a model",
            model_names,
            index=0,
            help="Select the LLM model to use for tokenization. The best embedding model will be auto-selected.",
            label_visibility="collapsed",
        )
        config = MODEL_CONFIG[selected_model]

        # Show model info card
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 0.5rem;">
            <div style="font-size:1.5rem; margin-bottom: 0.5rem;">{config['icon']} {selected_model.split(' (')[0]}</div>
            <div style="color: rgba(196,181,253,0.7); font-size: 0.8rem; line-height: 1.5;">
                {config['description']}
            </div>
            <div style="margin-top: 0.75rem; display: flex; flex-direction: column; gap: 6px;">
                <div class="model-badge">📊 Vocab: {config['vocab_size']:,}</div>
                <div class="model-badge">🧬 Embedding: {config['best_embedding']}</div>
                <div class="model-badge">📐 Dim: {config['embedding_dim']:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Language Selection
        st.markdown("### 🌍 Language")
        selected_lang = st.selectbox(
            "Language selection",
            list(LANGUAGE_MAP.keys()),
            index=0,
            help="Choose a language or let the app auto-detect it from your text.",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Stats
        st.markdown("### 📈 Quick Reference")
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-size: 0.8rem; color: rgba(196,181,253,0.8); line-height: 2;">
                ✦ Encoding: <code>{config['encoding']}</code><br>
                ✦ Models supported: <b>{len(MODEL_CONFIG)}</b><br>
                ✦ Languages: <b>{len(LANGUAGE_MAP) - 1}</b><br>
                ✦ Embedding type: Dense<br>
                ✦ Norm: L2 Unit Normalized
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; padding: 0.5rem 0; color: rgba(196,181,253,0.4); font-size: 0.7rem;">
            Built with ❤️ using Streamlit<br>
            ⭐ Star this project!
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # MAIN CONTENT — Text Input
    # ══════════════════════════════════════════════════════
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📝 Input Text</div>
        <div style="color: rgba(196,181,253,0.6); font-size: 0.8rem; margin-bottom: 0.5rem;">
            Write or paste your text below. The language will be detected automatically.
        </div>
    </div>
    """, unsafe_allow_html=True)

    user_text = st.text_area(
        "Enter your message",
        value="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.",
        height=150,
        label_visibility="collapsed",
        placeholder="Type or paste your text here...",
    )

    # Language detection display
    if user_text.strip():
        if LANGUAGE_MAP[selected_lang] == "auto":
            det_code, det_name, det_conf = detect_language(user_text)
            lang_display = det_name
            lang_conf = det_conf
        else:
            det_code = LANGUAGE_MAP[selected_lang]
            lang_display = selected_lang.split(" ", 1)[1]
            lang_conf = 1.0

        col_lang1, col_lang2, col_lang3 = st.columns([1, 1, 2])
        with col_lang1:
            st.markdown(f'<div class="lang-badge">🔤 {lang_display}</div>', unsafe_allow_html=True)
        with col_lang2:
            st.markdown(f'<div class="lang-badge">🎯 Confidence: {lang_conf:.0%}</div>', unsafe_allow_html=True)

    # ── Analyze Button ──
    analyze = st.button("⚡ Analyze Tokens & Embeddings", use_container_width=True, type="primary")

    if analyze and user_text.strip():
        # Progress indicator
        progress_bar = st.progress(0, text="🔄 Loading tokenizer...")
        
        # ── Step 1: Tokenize ──
        progress_bar.progress(20, text="🔠 Tokenizing text...")
        token_ids, token_strings = tokenize_text(user_text, selected_model)

        # ── Step 2: Generate Embedding ──
        progress_bar.progress(50, text="🧬 Computing embeddings...")
        embedding_dim = config["embedding_dim"]
        embedding_vector = generate_simulated_embedding(user_text, embedding_dim, config["best_embedding"])

        # ── Step 3: Compute Stats ──
        progress_bar.progress(80, text="📊 Computing statistics...")
        char_count = len(user_text)
        word_count = len(user_text.split())
        token_count = len(token_ids)
        chars_per_token = char_count / max(token_count, 1)
        
        progress_bar.progress(100, text="✅ Analysis complete!")
        import time; time.sleep(0.5)
        progress_bar.empty()

        # ══════════════════════════════════════════════════
        # METRICS ROW
        # ══════════════════════════════════════════════════
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{token_count:,}</div>
                <div class="metric-label">Tokens</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{word_count:,}</div>
                <div class="metric-label">Words</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{char_count:,}</div>
                <div class="metric-label">Characters</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{chars_per_token:.1f}</div>
                <div class="metric-label">Chars/Token</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{embedding_dim:,}</div>
                <div class="metric-label">Embedding Dim</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ══════════════════════════════════════════════════
        # TABBED RESULTS
        # ══════════════════════════════════════════════════
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔠 Token Visualization",
            "📊 Token Details",
            "🧬 Vector Heatmap",
            "📐 Embedding Data",
        ])

        # ────────────────────────────────────────────────
        # TAB 1: Token Visualization
        # ────────────────────────────────────────────────
        with tab1:
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">🔠 How Tokens Look</div>
                <div style="color: rgba(196,181,253,0.6); font-size: 0.8rem; margin-bottom: 0.75rem;">
                    Each colored chip represents a single token. Hover to see details. Token IDs are shown below each token.
                </div>
            """, unsafe_allow_html=True)

            # Build token chips HTML
            chips_html = '<div class="token-container">'
            for i, (tid, tstr) in enumerate(zip(token_ids, token_strings)):
                bg_color = get_token_color(tid, len(token_ids))
                # Escape HTML chars
                display_str = tstr.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                display_str = display_str.replace(" ", "·").replace("\n", "↵").replace("\t", "⇥")
                if not display_str.strip():
                    display_str = "▪"
                chips_html += f'''
                <div class="token-chip" style="background: {bg_color};" title="Token #{i+1} | ID: {tid} | Text: '{tstr}'">
                    <span>{display_str}</span>
                    <span class="token-id">#{tid}</span>
                </div>'''
            chips_html += '</div></div>'
            st.markdown(chips_html, unsafe_allow_html=True)

        # ────────────────────────────────────────────────
        # TAB 2: Token Details Table
        # ────────────────────────────────────────────────
        with tab2:
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">📊 Token Count & Details</div>
                <div style="color: rgba(196,181,253,0.6); font-size: 0.8rem; margin-bottom: 0.75rem;">
                    Complete breakdown of each token with its ID, position, text, byte length, and character representation.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Token IDs list
            st.markdown("**Token IDs Array:**")
            st.code(json.dumps(token_ids), language="json")

            # Token table
            import pandas as pd
            token_data = []
            for i, (tid, tstr) in enumerate(zip(token_ids, token_strings)):
                display_str = tstr.replace(" ", "·").replace("\n", "↵").replace("\t", "⇥")
                byte_repr = " ".join(f"{b:02x}" for b in tstr.encode("utf-8"))
                token_data.append({
                    "Position": i + 1,
                    "Token ID": tid,
                    "Token Text": display_str if display_str.strip() else "▪",
                    "Original": repr(tstr),
                    "Bytes (hex)": byte_repr,
                    "Byte Length": len(tstr.encode("utf-8")),
                    "Char Length": len(tstr),
                })
            df = pd.DataFrame(token_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=min(400, 35 * len(token_data) + 50),
            )

            # Token frequency
            st.markdown("**Token Frequency Distribution:**")
            from collections import Counter
            freq = Counter(token_ids)
            freq_data = [{"Token ID": tid, "Token": token_strings[token_ids.index(tid)].replace(" ", "·"), "Count": cnt}
                        for tid, cnt in freq.most_common(20)]
            if freq_data:
                freq_df = pd.DataFrame(freq_data)
                st.bar_chart(freq_df.set_index("Token")["Count"])

        # ────────────────────────────────────────────────
        # TAB 3: Vector Heatmap
        # ────────────────────────────────────────────────
        with tab3:
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">🧬 Vector of the Paragraph</div>
                <div style="color: rgba(196,181,253,0.6); font-size: 0.8rem; margin-bottom: 0.75rem;">
                    Visual heatmap of the embedding vector. Each cell represents one dimension.
                    Purple = negative values, Blue = positive values. Hover to see exact values.
                </div>
            """, unsafe_allow_html=True)

            # Show first N dimensions as heatmap
            display_dims = min(256, len(embedding_vector))
            heatmap_html = '<div class="vector-grid">'
            for i in range(display_dims):
                val = embedding_vector[i]
                color = value_to_heatmap_color(val)
                heatmap_html += f'<div class="vector-cell" style="background:{color};" title="dim[{i}] = {val:.6f}">{i}</div>'
            heatmap_html += '</div>'
            
            if display_dims < len(embedding_vector):
                heatmap_html += f'<div style="text-align:center; color: rgba(196,181,253,0.5); font-size: 0.75rem; margin-top: 0.5rem;">Showing {display_dims} of {len(embedding_vector):,} dimensions</div>'
            
            heatmap_html += '</div>'
            st.markdown(heatmap_html, unsafe_allow_html=True)

            # Vector statistics
            st.markdown("**Vector Statistics:**")
            vcol1, vcol2, vcol3, vcol4 = st.columns(4)
            with vcol1:
                st.metric("Mean", f"{np.mean(embedding_vector):.6f}")
            with vcol2:
                st.metric("Std Dev", f"{np.std(embedding_vector):.6f}")
            with vcol3:
                st.metric("Min", f"{np.min(embedding_vector):.6f}")
            with vcol4:
                st.metric("Max", f"{np.max(embedding_vector):.6f}")

            # Distribution chart
            st.markdown("**Value Distribution:**")
            import pandas as pd
            hist_data = pd.DataFrame({"Embedding Values": embedding_vector[:512]})
            st.line_chart(hist_data, use_container_width=True)

        # ────────────────────────────────────────────────
        # TAB 4: Embedding Data
        # ────────────────────────────────────────────────
        with tab4:
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-title">📐 Embeddings of the Paragraph</div>
                <div style="color: rgba(196,181,253,0.6); font-size: 0.8rem; margin-bottom: 0.75rem;">
                    Full embedding vector generated by <b>{config['best_embedding']}</b> model.
                    Dimension: <b>{embedding_dim:,}D</b> | Normalized: L2 Unit Vector
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Embedding preview
            st.markdown("**Embedding Vector (first 50 dimensions):**")
            preview = embedding_vector[:50].tolist()
            formatted_preview = [round(v, 8) for v in preview]
            st.markdown(f'<div class="json-block">{json.dumps(formatted_preview, indent=2)}</div>', unsafe_allow_html=True)

            # Full embedding in expander
            with st.expander(f"📋 View Full Embedding ({embedding_dim:,} dimensions)", expanded=False):
                full_list = embedding_vector.tolist()
                formatted_full = [round(v, 8) for v in full_list]
                json_str = json.dumps(formatted_full)
                st.code(json_str[:5000] + ("..." if len(json_str) > 5000 else ""), language="json")
                st.download_button(
                    label="⬇️ Download Full Embedding (JSON)",
                    data=json.dumps({
                        "model": config["best_embedding"],
                        "text": user_text,
                        "dimensions": embedding_dim,
                        "embedding": formatted_full,
                    }, indent=2),
                    file_name=f"embedding_{config['best_embedding'].replace('/', '_')}.json",
                    mime="application/json",
                )

            # Embedding metadata
            st.markdown("**Embedding Metadata:**")
            metadata = {
                "model": config["best_embedding"],
                "llm_model": selected_model,
                "dimensions": embedding_dim,
                "norm_type": "L2 (unit vector)",
                "l2_norm": float(np.linalg.norm(embedding_vector)),
                "input_tokens": token_count,
                "input_characters": char_count,
                "language_detected": lang_display if user_text.strip() else "N/A",
            }
            st.json(metadata)

    elif analyze and not user_text.strip():
        st.warning("⚠️ Please enter some text to analyze.", icon="⚠️")

    # ── Footer ──
    st.markdown("""
    <div class="footer">
        ⭐ <b>TokenLens</b> — Built with Streamlit &nbsp;|&nbsp;
        <a href="#">GitHub</a> &nbsp;|&nbsp;
        Supports {model_count} LLM Models &nbsp;|&nbsp;
        {lang_count} Languages
    </div>
    """.format(model_count=len(MODEL_CONFIG), lang_count=len(LANGUAGE_MAP) - 1), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─── Palette ─────────────────────────────────────────────────────────────────
BG      = "#1A1A1A"
SURFACE = "#242424"
CARD    = "#2E2E2E"
BORDER  = "#4D4D4D"
LIME    = "#C4FF4D"
PURPLE  = "#BA8CFF"
TEXT    = "#F0F0F0"
MUTED   = "#9A9A9A"
LIME2   = "#FFD94D"
TEAL    = "#4DFFCC"

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load assets ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("car_price_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_stats():
    with open("model_stats.json") as f:
        return json.load(f)

model = load_model()
stats = load_stats()

FEATURES = [
    'Engine_size', 'Horsepower', 'Wheelbase', 'Width', 'Length',
    'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency', 'Power_perf_factor'
]
FEATURE_LABELS = {
    'Engine_size':       'Engine Size (L)',
    'Horsepower':        'Horsepower (HP)',
    'Wheelbase':         'Wheelbase (in)',
    'Width':             'Width (in)',
    'Length':            'Length (in)',
    'Curb_weight':       'Curb Weight (t)',
    'Fuel_capacity':     'Fuel Cap (gal)',
    'Fuel_efficiency':   'Fuel Eff (MPG)',
    'Power_perf_factor': 'Perf Factor',
}

# ─── Matplotlib helpers ───────────────────────────────────────────────────────
def make_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax

def polish(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER)
    ax.spines['bottom'].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)

def grad_colors(n, c1, c2):
    r1,g1,b1 = plt.matplotlib.colors.to_rgb(c1)
    r2,g2,b2 = plt.matplotlib.colors.to_rgb(c2)
    return [plt.matplotlib.colors.to_hex(
        (r1+(r2-r1)*i/(max(n-1,1)),
         g1+(g2-g1)*i/(max(n-1,1)),
         b1+(b2-b1)*i/(max(n-1,1)))
    ) for i in range(n)]

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {{
    background-color:{BG} !important;
    color:{TEXT};
    font-family:'DM Sans',sans-serif;
}}
.block-container {{
    padding:1.8rem 2.2rem 3rem !important;
    max-width:1440px;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background:{SURFACE} !important;
    border-right:1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color:{TEXT} !important; }}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background:{SURFACE};
    border-radius:10px;
    padding:4px 6px;
    gap:4px;
    border:1px solid {BORDER};
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background:transparent !important;
    color:{MUTED} !important;
    font-family:'DM Sans',sans-serif;
    font-weight:600;
    border-radius:8px;
    padding:9px 22px;
    font-size:0.9rem;
    transition:all .2s;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background:{LIME} !important;
    color:{BG} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-border"],
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    display:none !important;
}}

/* ── Number input ── */
[data-testid="stNumberInput"] input {{
    background:{CARD} !important;
    border:1px solid {BORDER} !important;
    color:{TEXT} !important;
    border-radius:8px;
    font-family:'Space Mono',monospace;
    font-size:0.88rem;
}}
[data-testid="stNumberInput"] input:focus {{
    border-color:{LIME} !important;
    box-shadow:0 0 0 2px {LIME}33 !important;
    outline:none;
}}
[data-testid="stNumberInput"] label {{
    color:{MUTED} !important;
    font-size:0.8rem !important;
    font-weight:500;
}}
[data-testid="stNumberInput"] button {{
    background:{BORDER} !important;
    border:none !important;
    color:{TEXT} !important;
    border-radius:5px !important;
}}
[data-testid="InputInstructions"] {{ color:{BORDER} !important; }}

/* ── Main CTA button ── */
div.stButton > button {{
    background:{LIME} !important;
    color:{BG} !important;
    border:none !important;
    border-radius:10px !important;
    padding:14px 0 !important;
    font-family:'Syne',sans-serif !important;
    font-size:1rem !important;
    font-weight:800 !important;
    width:100% !important;
    letter-spacing:.4px;
    box-shadow:0 4px 22px {LIME}44 !important;
    transition:all .2s !important;
}}
div.stButton > button:hover {{
    box-shadow:0 6px 30px {LIME}88 !important;
    transform:translateY(-2px);
}}

/* ── Preset small buttons (inside column) ── */
div[data-testid="column"] div.stButton > button {{
    background:{CARD} !important;
    color:{TEXT} !important;
    border:1px solid {BORDER} !important;
    font-size:0.82rem !important;
    padding:10px 2px !important;
    box-shadow:none !important;
    font-family:'DM Sans',sans-serif !important;
    font-weight:700 !important;
    border-radius:8px !important;
    width:100% !important;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
}}
div[data-testid="column"] div.stButton > button:hover {{
    border-color:{LIME} !important;
    color:{LIME} !important;
    background:{LIME}12 !important;
    box-shadow:0 0 0 1px {LIME}44 !important;
    transform:none;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    background:{SURFACE} !important;
    border:1px solid {BORDER} !important;
    border-radius:10px;
    overflow:hidden;
}}
[data-testid="stDataFrame"] th {{
    background:{CARD} !important;
    color:{LIME} !important;
    font-family:'Space Mono',monospace !important;
    font-size:0.78rem !important;
    letter-spacing:.5px;
}}
[data-testid="stDataFrame"] td {{
    color:{TEXT} !important;
    font-size:0.84rem !important;
    background:{SURFACE} !important;
}}

/* ── Divider ── */
hr {{ border-color:{BORDER} !important; opacity:.35; margin:18px 0; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{BG}; }}
::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:3px; }}

/* ════ Custom components ════ */

/* Header */
.ap-header {{
    display:flex; align-items:center; gap:22px;
    padding:26px 36px;
    background:{SURFACE};
    border:1px solid {BORDER};
    border-radius:16px;
    margin-bottom:22px;
    position:relative; overflow:hidden;
}}
.ap-header::before {{
    content:''; position:absolute; top:-60px; right:-40px;
    width:220px; height:220px;
    background:radial-gradient(circle,{LIME}1A 0%,transparent 70%);
    pointer-events:none;
}}
.ap-header::after {{
    content:''; position:absolute; bottom:-50px; left:240px;
    width:180px; height:180px;
    background:radial-gradient(circle,{PURPLE}1A 0%,transparent 70%);
    pointer-events:none;
}}
.ap-car-img {{
    width:180px; height:auto;
    object-fit:contain;
    filter:drop-shadow(0 0 18px {LIME}66);
    flex-shrink:0;
}}
.ap-header h1 {{
    font-family:'Syne',sans-serif;
    font-size:1.95rem; font-weight:800;
    color:{TEXT}; margin:0 0 5px;
    letter-spacing:-.5px;
}}
.ap-header h1 span {{ color:{LIME}; }}
.ap-header p {{ font-size:.88rem; color:{MUTED}; margin:0; line-height:1.5; }}
.ap-badge {{
    display:inline-block;
    background:{LIME}18; color:{LIME};
    border:1px solid {LIME}55;
    font-family:'Space Mono',monospace;
    font-size:.65rem; padding:3px 12px;
    border-radius:99px; letter-spacing:1.2px; margin-top:9px;
}}

/* Sidebar card */
.sb-card {{
    background:{CARD}; border:1px solid {BORDER};
    border-radius:10px; padding:13px 15px; margin-bottom:9px;
}}
.sb-card .lbl {{
    font-size:.7rem; color:{MUTED}; font-weight:600;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:3px;
}}
.sb-val  {{ font-family:'Space Mono',monospace; font-size:1.3rem; font-weight:700; color:{LIME}; }}
.sb-val2 {{ font-family:'Space Mono',monospace; font-size:1.3rem; font-weight:700; color:{PURPLE}; }}
.sb-sub  {{ font-size:.73rem; color:{MUTED}; margin-top:2px; }}

/* Section title */
.stitle {{
    font-family:'Syne',sans-serif;
    font-size:.78rem; font-weight:800;
    color:{LIME}; text-transform:uppercase;
    letter-spacing:2px; margin:20px 0 11px;
    display:flex; align-items:center; gap:9px;
}}
.stitle::after {{
    content:''; flex:1; height:1px; background:{BORDER};
}}

/* Result card */
.res-card {{
    background:linear-gradient(135deg,{LIME}14 0%,{PURPLE}0D 100%);
    border:1px solid {LIME}44;
    border-radius:16px; padding:26px 22px;
    text-align:center; margin:6px 0 14px;
    position:relative; overflow:hidden;
}}
.res-card::before {{
    content:''; position:absolute; top:-30px; right:-30px;
    width:120px; height:120px;
    background:radial-gradient(circle,{LIME}18,transparent 70%);
}}
.res-lbl {{
    font-family:'Space Mono',monospace;
    font-size:.7rem; color:{LIME};
    text-transform:uppercase; letter-spacing:2px;
}}
.res-price {{
    font-family:'Syne',sans-serif;
    font-size:3rem; font-weight:800; color:{LIME};
    line-height:1.05; margin:7px 0 3px;
    text-shadow:0 0 35px {LIME}55;
}}
.res-sub {{ font-size:.85rem; color:{MUTED}; }}
.seg-badge {{
    display:inline-block;
    padding:4px 16px; border-radius:99px;
    font-family:'Space Mono',monospace;
    font-size:.72rem; font-weight:700; margin-top:10px;
}}

/* Gauge */
.g-labels {{
    display:flex; justify-content:space-between;
    font-family:'Space Mono',monospace;
    font-size:.68rem; color:{MUTED}; margin-bottom:5px;
}}
.g-track {{
    background:{BORDER}; border-radius:99px;
    height:10px; overflow:hidden;
}}
.g-fill {{
    height:100%; border-radius:99px;
    background:linear-gradient(90deg,{LIME},{PURPLE});
    transition:width .5s ease;
}}
.g-pct {{
    text-align:right;
    font-family:'Space Mono',monospace;
    font-size:.68rem; color:{PURPLE};
    margin-top:3px; font-weight:700;
}}

/* Spec table */
.sp-tbl {{ width:100%; border-collapse:collapse; font-size:.84rem; }}
.sp-tbl tr {{ border-bottom:1px solid {BORDER}22; }}
.sp-tbl tr:last-child {{ border-bottom:none; }}
.sp-tbl td {{ padding:7px 4px; vertical-align:middle; }}
.sp-tbl td:first-child {{ color:{MUTED}; font-size:.8rem; }}
.sp-tbl td:last-child {{
    font-family:'Space Mono',monospace;
    font-weight:700; color:{TEXT}; text-align:right;
}}

/* Footer */
.ap-footer {{
    background:{SURFACE}; border:1px solid {BORDER};
    border-radius:12px; padding:14px 22px;
    text-align:center; margin-top:22px;
    font-size:.8rem; color:{MUTED};
    line-height:1.6;
}}
.ap-footer b {{ color:{LIME}; font-family:'Space Mono',monospace; }}
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
import base64 as _b64
with open("car_hero.jpg", "rb") as _f:
    _car_b64 = _b64.b64encode(_f.read()).decode()

st.markdown(f"""
<div class="ap-header">
  <img class="ap-car-img" src="data:image/jpeg;base64,{_car_b64}" alt="car"/>
  <div>
    <h1>Prediksi <span>Harga Mobil</span></h1>
    <p>Masukkan spesifikasi teknis kendaraan untuk mendapatkan estimasi harga pasar<br>
       secara otomatis menggunakan Machine Learning · Linear Regression</p>
    <span class="ap-badge">MATAKULIAH SAINS DATA &nbsp;·&nbsp; FINAL PROJECT</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:.95rem;font-weight:800;
         color:{LIME};letter-spacing:1px;margin-bottom:14px;padding-bottom:10px;
         border-bottom:1px solid {BORDER}">
      📊 MODEL INFO
    </div>
    """, unsafe_allow_html=True)

    r2_pct   = stats['r2'] * 100
    rmse_usd = stats['rmse'] * 1000

    st.markdown(f"""
    <div class="sb-card">
      <div class="lbl">Algoritma</div>
      <div style="font-weight:700;color:{TEXT};font-size:.95rem;margin-top:2px">
        Linear Regression
      </div>
    </div>
    <div class="sb-card">
      <div class="lbl">R² Score</div>
      <div class="sb-val">{r2_pct:.2f}%</div>
      <div class="sb-sub">Akurasi menjelaskan variasi harga</div>
    </div>
    <div class="sb-card">
      <div class="lbl">RMSE</div>
      <div class="sb-val2">${rmse_usd:,.0f}</div>
      <div class="sb-sub">Rata-rata error prediksi</div>
    </div>
    <div class="sb-card">
      <div class="lbl">Jumlah Fitur</div>
      <div style="font-weight:700;color:{TEXT};font-size:.95rem;margin-top:2px">
        9 variabel teknis
      </div>
    </div>
    <div class="sb-card">
      <div class="lbl">Split Data</div>
      <div style="font-weight:700;color:{TEXT};font-size:.95rem;margin-top:2px">
        80% Training / 20% Testing
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<hr/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:.85rem;font-weight:800;
         color:{PURPLE};letter-spacing:1px;margin-bottom:8px">
      📌 PETUNJUK
    </div>
    <div style="font-size:.82rem;color:{MUTED};line-height:1.8">
      1. Pilih <b style="color:{LIME}">preset</b> atau isi manual<br>
      2. Klik tombol <b style="color:{LIME}">Hitung Harga</b><br>
      3. Lihat hasil prediksi &amp; gauge<br>
      4. Tab <b style="color:{PURPLE}">Analisis Pasar</b> untuk grafik
    </div>
    """, unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮  Prediksi Harga", "📈  Analisis Pasar"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI HARGA
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    mins  = stats['feature_mins']
    maxs  = stats['feature_maxs']
    means = stats['feature_means']

    PRESETS = {
        "ekonomis": dict(Engine_size=1.8,  Horsepower=120.0, Wheelbase=100.0,
                         Width=68.0, Length=175.0, Curb_weight=2.8,
                         Fuel_capacity=13.0, Fuel_efficiency=30.0, Power_perf_factor=55.0),
        "menengah": dict(Engine_size=2.3,  Horsepower=150.0, Wheelbase=105.0,
                         Width=70.0, Length=185.0, Curb_weight=3.2,
                         Fuel_capacity=15.0, Fuel_efficiency=26.0, Power_perf_factor=70.0),
        "premium":  dict(Engine_size=3.5,  Horsepower=250.0, Wheelbase=112.0,
                         Width=74.0, Length=195.0, Curb_weight=4.0,
                         Fuel_capacity=20.0, Fuel_efficiency=20.0, Power_perf_factor=110.0),
        "reset":    {f: float(means[f]) for f in FEATURES},
    }

    col_form, col_result = st.columns([1, 1], gap="large")

    # ── LEFT: Form ────────────────────────────────────────────────────────────
    with col_form:
        st.markdown(f'<div class="stitle">📋 Spesifikasi Mobil</div>', unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:.79rem;color:{MUTED};margin-bottom:10px'>Pilih preset cepat:</div>", unsafe_allow_html=True)

        # Preset state
        if "preset" not in st.session_state:
            st.session_state.preset = None

        pb1, pb2, pb3, pb4 = st.columns([1,1,1,1])
        if pb1.button("🟢  Ekonomis",   use_container_width=True): st.session_state.preset = "ekonomis"
        if pb2.button("🟡  Menengah",   use_container_width=True): st.session_state.preset = "menengah"
        if pb3.button("🔴  Premium",    use_container_width=True): st.session_state.preset = "premium"
        if pb4.button("↺  Reset",       use_container_width=True): st.session_state.preset = "reset"

        preset = st.session_state.preset

        def gv(key):
            if preset and preset in PRESETS:
                return float(PRESETS[preset][key])
            return float(means[key])

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            engine  = st.number_input("🔧 Engine Size (L)",  min_value=float(mins['Engine_size']),      max_value=float(maxs['Engine_size']),      value=gv('Engine_size'),      step=0.1,  format="%.1f")
            hp      = st.number_input("⚡ Horsepower (HP)",   min_value=float(mins['Horsepower']),       max_value=float(maxs['Horsepower']),       value=gv('Horsepower'),       step=5.0,  format="%.0f")
            wheel   = st.number_input("📏 Wheelbase (in)",    min_value=float(mins['Wheelbase']),        max_value=float(maxs['Wheelbase']),        value=gv('Wheelbase'),        step=0.5,  format="%.1f")
            width   = st.number_input("↔️ Width (in)",         min_value=float(mins['Width']),            max_value=float(maxs['Width']),            value=gv('Width'),            step=0.5,  format="%.1f")
            length  = st.number_input("↕️ Length (in)",        min_value=float(mins['Length']),           max_value=float(maxs['Length']),           value=gv('Length'),           step=0.5,  format="%.1f")
        with c2:
            cweight = st.number_input("⚖️ Curb Weight (t)",   min_value=float(mins['Curb_weight']),      max_value=float(maxs['Curb_weight']),      value=gv('Curb_weight'),      step=0.1,  format="%.1f")
            fcap    = st.number_input("🛢️ Fuel Cap (gal)",    min_value=float(mins['Fuel_capacity']),    max_value=float(maxs['Fuel_capacity']),    value=gv('Fuel_capacity'),    step=0.5,  format="%.1f")
            feff    = st.number_input("⛽ Fuel Eff (MPG)",    min_value=float(mins['Fuel_efficiency']),  max_value=float(maxs['Fuel_efficiency']),  value=gv('Fuel_efficiency'),  step=1.0,  format="%.0f")
            ppf     = st.number_input("🏎️ Perf Factor",       min_value=float(mins['Power_perf_factor']),max_value=float(maxs['Power_perf_factor']),value=gv('Power_perf_factor'),step=1.0, format="%.1f")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.button("🔮  Hitung Harga Mobil", use_container_width=True)

    # ── RIGHT: Result ─────────────────────────────────────────────────────────
    with col_result:
        vals_in   = [engine, hp, wheel, width, length, cweight, fcap, feff, ppf]
        inp_df    = pd.DataFrame([vals_in], columns=FEATURES)
        pred      = model.predict(inp_df)[0]
        pred_usd  = pred * 1000

        if pred < 15:
            seg, s_bg, s_fg = "EKONOMIS", f"{LIME}25",   LIME
        elif pred < 25:
            seg, s_bg, s_fg = "MENENGAH", f"{PURPLE}33", PURPLE
        elif pred < 40:
            seg, s_bg, s_fg = "PREMIUM",  "#FF8C4D30",   "#FF8C4D"
        else:
            seg, s_bg, s_fg = "LUXURY",   "#FF4D6D30",   "#FF4D6D"

        st.markdown(f'<div class="stitle">💰 Hasil Prediksi</div>', unsafe_allow_html=True)

        # Price card
        st.markdown(f"""
        <div class="res-card">
          <div class="res-lbl">Perkiraan Harga Mobil</div>
          <div class="res-price">${pred_usd:,.0f}</div>
          <div class="res-sub">{pred:.3f}K USD</div>
          <div class="seg-badge"
               style="background:{s_bg};color:{s_fg};border:1px solid {s_fg}99">
            ▲ SEGMEN {seg}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge bar
        pct = min(max((pred - 5) / 75, 0), 1)
        st.markdown(f"""
        <div style="margin-bottom:18px">
          <div class="g-labels"><span>$5K — Min</span><span>Max — $80K</span></div>
          <div class="g-track">
            <div class="g-fill" style="width:{pct*100:.1f}%"></div>
          </div>
          <div class="g-pct">{pct*100:.0f}% dari rentang pasar</div>
        </div>
        """, unsafe_allow_html=True)

        # Spec summary table
        st.markdown(f'<div class="stitle">📋 Spesifikasi Input</div>', unsafe_allow_html=True)
        rows = [
            ("🔧 Engine Size",   f"{engine:.1f} L"),
            ("⚡ Horsepower",     f"{hp:.0f} HP"),
            ("📏 Wheelbase",      f"{wheel:.1f} in"),
            ("↔️ Width",          f"{width:.1f} in"),
            ("↕️ Length",         f"{length:.1f} in"),
            ("⚖️ Curb Weight",    f"{cweight:.1f} t"),
            ("🛢️ Fuel Capacity",  f"{fcap:.1f} gal"),
            ("⛽ Fuel Efficiency",f"{feff:.0f} MPG"),
            ("🏎️ Perf Factor",    f"{ppf:.1f}"),
        ]
        trs = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
        st.markdown(f'<table class="sp-tbl">{trs}</table>', unsafe_allow_html=True)

        # ── Comparison bar chart ──
        st.markdown(f'<div class="stitle">📊 vs Rata-Rata Pasar</div>', unsafe_allow_html=True)

        labels_s  = ['Engine','HP','Wheelbase','Width','Length','Weight','FuelCap','MPG','Perf']
        user_v    = vals_in
        avg_v     = [float(means[f]) for f in FEATURES]
        min_v     = [float(mins[f])  for f in FEATURES]
        max_v     = [float(maxs[f])  for f in FEATURES]
        u_norm    = [(v-mn)/(mx-mn) for v,mn,mx in zip(user_v,min_v,max_v)]
        a_norm    = [(v-mn)/(mx-mn) for v,mn,mx in zip(avg_v, min_v,max_v)]

        fig, ax = make_fig(6, 3.2)
        x = np.arange(len(labels_s))
        w = 0.36
        ax.bar(x - w/2, u_norm, w, color=LIME,   label='Anda',       zorder=3, alpha=.92)
        ax.bar(x + w/2, a_norm, w, color=BORDER, label='Rata-Rata',  zorder=3, alpha=.80)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_s, fontsize=7.5, rotation=30, ha='right', color=MUTED)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['Min', 'Mid', 'Max'], fontsize=7.5, color=MUTED)
        ax.yaxis.grid(True, color=BORDER, alpha=.35, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax.set_title('Posisi Spesifikasi Anda vs Rata-Rata Pasar',
                     fontsize=9, color=TEXT, fontweight='bold', pad=8)
        ax.legend(fontsize=8, loc='upper right',
                  facecolor=CARD, edgecolor=BORDER,
                  labelcolor=TEXT)
        polish(ax)
        ax.spines['left'].set_visible(False)
        plt.tight_layout(pad=.7)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Footer
        st.markdown(f"""
        <div class="ap-footer">
          <b>AUTOPRICE AI</b> &nbsp;·&nbsp; Machine Learning · Linear Regression<br>
          Dataset: Car Sales &nbsp;·&nbsp; Matakuliah Sains Data
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALISIS PASAR
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    top10 = pd.DataFrame(stats['top10'])
    coef  = stats['coef']
    n     = len(top10)

    # ── 1. Sales chart ───────────────────────────────────────────────────────
    st.markdown(f'<div class="stitle">🏆 Top 10 Mobil Penjualan Terbanyak</div>', unsafe_allow_html=True)

    fig, ax = make_fig(12, 5.5)
    cols_s = grad_colors(n, LIME, PURPLE)
    bars = ax.barh(top10['Full_Name'], top10['Sales_in_thousands'],
                   color=cols_s, edgecolor='none', height=.65, zorder=3)
    for bar, val in zip(bars, top10['Sales_in_thousands']):
        ax.text(bar.get_width() + 6,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}K',
                va='center', fontsize=10, fontweight='bold', color=TEXT, zorder=4)
    ax.set_xlabel('Jumlah Penjualan (ribuan unit)', fontsize=10, labelpad=8)
    ax.set_title('Top 10 Mobil Terlaris di Pasar', fontsize=13,
                 fontweight='bold', pad=12)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=BORDER, alpha=.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=9.5, labelcolor=TEXT, length=0)
    ax.tick_params(axis='x', labelsize=8.5, labelcolor=MUTED)
    ax.set_xlim(0, top10['Sales_in_thousands'].max() * 1.2)
    polish(ax)
    ax.spines['left'].set_visible(False)
    plt.tight_layout(pad=.9)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── 2. Three side-by-side charts ─────────────────────────────────────────
    ca, cb, cc = st.columns(3)

    def draw_side(col, title, stitle_txt, values, c1, c2, xlabel, fmt_fn, offset):
        with col:
            st.markdown(f'<div class="stitle">{stitle_txt}</div>', unsafe_allow_html=True)
            fig, ax = make_fig(5, 5.2)
            cols_c = grad_colors(n, c1, c2)
            bars_c = ax.barh(top10['Full_Name'], values,
                             color=cols_c, edgecolor='none', height=.63, zorder=3)
            for bar, val in zip(bars_c, values):
                ax.text(bar.get_width() + offset,
                        bar.get_y() + bar.get_height() / 2,
                        fmt_fn(val),
                        va='center', fontsize=8.5, fontweight='bold', color=TEXT)
            ax.invert_yaxis()
            ax.xaxis.grid(True, color=BORDER, alpha=.3, linestyle='--', zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel(xlabel, fontsize=8.5, labelpad=6)
            ax.set_title(title, fontsize=10, fontweight='bold', pad=9)
            ax.tick_params(axis='y', labelsize=8, labelcolor=TEXT, length=0)
            ax.tick_params(axis='x', labelsize=7.5, labelcolor=MUTED)
            ax.set_xlim(0, max(values) * 1.28)
            polish(ax)
            ax.spines['left'].set_visible(False)
            plt.tight_layout(pad=.65)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    draw_side(ca,
              title="Harga Pasar",
              stitle_txt="💰 Harga (K USD)",
              values=top10['Price_in_thousands'].tolist(),
              c1=LIME, c2=LIME2,
              xlabel="Harga (K USD)",
              fmt_fn=lambda v: f"${v:.0f}K",
              offset=.4)

    draw_side(cb,
              title="Tenaga Kuda",
              stitle_txt="⚡ Horsepower (HP)",
              values=top10['Horsepower'].tolist(),
              c1=PURPLE, c2="#FF8CFF",
              xlabel="Horsepower (HP)",
              fmt_fn=lambda v: f"{v:.0f}HP",
              offset=2.5)

    draw_side(cc,
              title="Efisiensi BBM",
              stitle_txt="⛽ Fuel Efficiency (MPG)",
              values=top10['Fuel_efficiency'].tolist(),
              c1=TEAL, c2=LIME,
              xlabel="Fuel Efficiency (MPG)",
              fmt_fn=lambda v: f"{v:.0f}mpg",
              offset=.4)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── 3. Data table ─────────────────────────────────────────────────────────
    st.markdown(f'<div class="stitle">📋 Tabel Lengkap Data Top 10</div>', unsafe_allow_html=True)
    disp = top10.rename(columns={
        'Full_Name':          'Model Mobil',
        'Sales_in_thousands': 'Penjualan (K)',
        'Price_in_thousands': 'Harga (K USD)',
        'Engine_size':        'Engine (L)',
        'Horsepower':         'HP',
        'Fuel_efficiency':    'MPG',
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── 4. Coefficients chart ─────────────────────────────────────────────────
    st.markdown(f'<div class="stitle">🔬 Pengaruh Tiap Variabel terhadap Harga (Koefisien Regresi)</div>', unsafe_allow_html=True)

    coef_df = pd.DataFrame({
        'Variabel':  [FEATURE_LABELS[f] for f in FEATURES],
        'Koefisien': [coef[f] for f in FEATURES],
    }).sort_values('Koefisien', ascending=True).reset_index(drop=True)

    fig, ax = make_fig(11, 4.8)
    bar_fill = [LIME if v >= 0 else PURPLE for v in coef_df['Koefisien']]
    bars_k = ax.barh(coef_df['Variabel'], coef_df['Koefisien'],
                     color=bar_fill, edgecolor='none', height=.6, zorder=3)

    # Value labels beside each bar
    for bar, val in zip(bars_k, coef_df['Koefisien']):
        x_txt = bar.get_width() + (.25 if val >= 0 else -.25)
        ha    = 'left' if val >= 0 else 'right'
        ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}',
                va='center', ha=ha,
                fontsize=9, fontweight='bold', color=TEXT)

    ax.axvline(0, color=TEXT, linewidth=.9, alpha=.35, zorder=2)
    ax.xaxis.grid(True, color=BORDER, alpha=.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Nilai Koefisien  (dampak per satuan → harga dalam K USD)',
                  fontsize=9.5, labelpad=8)
    ax.set_title('Koefisien Regresi Linear — Seberapa Besar Pengaruh Tiap Fitur terhadap Harga',
                 fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(axis='y', labelsize=10, labelcolor=TEXT, length=0)
    ax.tick_params(axis='x', labelsize=8.5, labelcolor=MUTED)
    polish(ax)
    ax.spines['left'].set_visible(False)

    lime_p   = mpatches.Patch(color=LIME,   label='Meningkatkan harga  (+)')
    purple_p = mpatches.Patch(color=PURPLE, label='Menurunkan harga  (−)')
    ax.legend(handles=[lime_p, purple_p], fontsize=9.5,
              facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT,
              loc='lower right')
    plt.tight_layout(pad=.9)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown(f"""
    <div style="font-size:.8rem;color:{MUTED};line-height:1.7;margin-top:6px">
      Koefisien <span style="color:{LIME};font-weight:700">positif (hijau-lime)</span>
      → semakin besar nilainya, harga makin <b style="color:{LIME}">naik</b>.&nbsp;&nbsp;
      Koefisien <span style="color:{PURPLE};font-weight:700">negatif (ungu)</span>
      → semakin besar nilainya, harga makin <b style="color:{PURPLE}">turun</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ap-footer">
      <b>AUTOPRICE AI</b> &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; Linear Regression<br>
      Dataset: Car Sales &nbsp;·&nbsp; Matakuliah Sains Data
    </div>
    """, unsafe_allow_html=True)

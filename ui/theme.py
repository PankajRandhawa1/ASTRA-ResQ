"""Shared ASTRA ResQ styling."""
import streamlit as st

def apply_theme():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root{--ink:#eaf2ff;--muted:#9bb0c9;--line:rgba(148,178,211,.18);--teal:#48e0c1}
    html,body,[class*='css']{font-family:'DM Sans',sans-serif}
    .stApp{background:radial-gradient(ellipse at 12% 0%,#173a50 0%,#0a1423 45%,#07101b 100%);color:var(--ink)}
    [data-testid='stHeader']{background:transparent}[data-testid='stSidebar']{background:linear-gradient(180deg,#0d1b2c,#091321);border-right:1px solid var(--line)}
    h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:#f4f8ff!important}
    .hero{padding:1.35rem 1.55rem;border:1px solid rgba(72,224,193,.25);border-radius:22px;background:linear-gradient(115deg,rgba(24,57,78,.92),rgba(13,28,46,.85));margin-bottom:1rem}
    .eyebrow,.section-label{text-transform:uppercase;letter-spacing:.15em;font-size:.76rem;color:var(--teal);font-weight:700}
    .hero-title{font-size:2.3rem;font-weight:700;line-height:1.15;margin:.4rem 0;color:#f4f8ff}.subtle{color:#a9bdd3;font-size:.95rem}
    .status-chip{display:inline-block;border:1px solid rgba(72,224,193,.28);background:rgba(72,224,193,.09);color:#8ff2dc;padding:.28rem .62rem;border-radius:999px;font-size:.76rem;margin:.15rem .2rem .15rem 0}
    .metric-card{background:rgba(17,32,51,.78);border:1px solid var(--line);padding:1rem 1.1rem;border-radius:16px;min-height:104px}.metric-name{color:#9bb0c9;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em}.metric-value{color:#f2f7ff;font-size:1.65rem;font-weight:700;margin-top:.3rem}
    div.stButton>button[kind='primary']{background:linear-gradient(90deg,#32cdb2,#66e4cd);color:#06211f;border:0;font-weight:800;border-radius:12px;min-height:3rem}div.stButton>button{border-radius:10px}
    [data-testid='stFileUploader']{background:rgba(14,29,47,.62);border:1px dashed rgba(111,159,193,.5);border-radius:15px;padding:.5rem}.footnote{color:#91a7bf;font-size:.82rem}hr{border-color:var(--line)}
    </style>""", unsafe_allow_html=True)

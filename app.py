from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import streamlit as st
import torch
from PIL import Image
from models.architectures import ASTRA_GAN_Generator, FloodUNet
from services.inference import load_models, run_pipeline, create_overlay
from services.routing import astar_grid
from ui.theme import apply_theme

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
GAN_PATH = MODEL_DIR / "astra_gan_generator.pth"
FLOOD_PATH = MODEL_DIR / "astra_resq_flood_unet.pth"
IMAGE_SIZE = 256
st.set_page_config(page_title="ASTRA ResQ | Flood Intelligence", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")
apply_theme()

def image_bytes(arr: np.ndarray, fmt: str = "PNG") -> bytes:
    buff = io.BytesIO()
    Image.fromarray(arr).save(buff, format=fmt, quality=95 if fmt.upper() == "JPEG" else None)
    return buff.getvalue()

# --- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌊 ASTRA ResQ")
    st.caption("Flood intelligence prototype")
    st.markdown("---")
    st.markdown("**Analysis settings**")
    threshold = st.slider("Flood probability threshold", 0.10, 0.90, 0.50, 0.05,
                          help="Higher thresholds mark fewer pixels as predicted flood. This is not a calibrated real-world risk score.")
    alpha = st.slider("Flood overlay opacity", 0.15, 0.80, 0.42, 0.05)
    st.markdown("---")
    st.markdown("**Model files**")
    st.caption("Expected in the `models/` folder:")
    st.code("astra_gan_generator.pth\nastra_resq_flood_unet.pth", language="text")
    if torch.cuda.is_available():
        st.success("CUDA device available")
    else:
        st.info("Running on CPU")
    st.markdown("---")
    st.caption("Prototype only · Not for emergency decisions")

# --- Header -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-assisted disaster response · research prototype</div>
      <div class="hero-title">See the flood. Understand the risk.</div>
      <div class="subtle">Restore a cloudy image, inspect a predicted flood mask, and explore an experimental image-grid route.</div>
      <div style="margin-top:.8rem"><span class="status-chip">ASTRA-GAN</span><span class="status-chip">Flood U-Net</span><span class="status-chip">A* grid demo</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.4, 1], gap="large")
original = None  # Always define this before the analysis button is checked.
original_image = None

with left:
    st.markdown('<div class="section-label">01 / Input imagery</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
    "Upload a cloudy satellite image (TIFF only)",
    type=["tif", "tiff"],
    accept_multiple_files=False,
    key="tiff_satellite_uploader"
)
with right:
    st.markdown('<div class="section-label">02 / Run analysis</div>', unsafe_allow_html=True)
    st.write("TIFF input → image reconstruction → reconstructed JPEG → flood mapping → experimental route finding.")
    run = st.button("Analyse image  →", type="primary", use_container_width=True, disabled=uploaded is None)

if uploaded is not None:
    filename = uploaded.name.lower()

    if not filename.endswith((".tif", ".tiff")):
        st.error("Please upload a TIFF image (.tif or .tiff).")
        st.stop()

    try:
        from PIL import Image
        import io

        uploaded_bytes = uploaded.getvalue()

        original_image = Image.open(io.BytesIO(uploaded_bytes))
        original_image.load()
        # The trained flood model expects RGB input; keep a decoded RGB copy for inference.
        original = original_image.convert("RGB")

        st.success(f"TIFF uploaded successfully: {uploaded.name}")
        st.write("Image dimensions:", original_image.size)
        st.write("Image format:", original_image.format)
        st.image(original_image, caption="Uploaded satellite image", use_container_width=True)

    except Exception as e:
        original = None
        st.error(f"Unable to read this TIFF file: {e}")

if run and original is not None:
    if not GAN_PATH.is_file() or not FLOOD_PATH.is_file():
        st.error("Model checkpoint(s) not found. Put both trained .pth files in the models/ folder.")
        st.markdown(f"- ASTRA-GAN: `{GAN_PATH}` {'✅ found' if GAN_PATH.is_file() else '❌ missing'}")
        st.markdown(f"- Flood U-Net: `{FLOOD_PATH}` {'✅ found' if FLOOD_PATH.is_file() else '❌ missing'}")
    else:
        try:
            device_text = "cuda" if torch.cuda.is_available() else "cpu"
            with st.spinner("Restoring image and predicting flood regions…"):
                generator, flood_model, device, image_size = load_models(str(GAN_PATH), str(FLOOD_PATH), device_text)
                result_data = run_pipeline(original, generator, flood_model, device, image_size, threshold, alpha)
                result_data["filename"] = uploaded.name
                st.session_state["analysis_result"] = result_data
            st.success("Analysis completed. Review the experimental outputs below.")
        except Exception as exc:
            st.error(f"Model loading or inference failed: {exc}")

result = st.session_state.get("analysis_result")
if result is not None:
    st.markdown("---")
    st.markdown('<div class="section-label">03 / AI outputs</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**TIFF input preview (model size)**")
        st.image(result["original"], use_container_width=True)
        st.download_button("Download input preview", image_bytes(result["original"], "PNG"), "astra_resq_input_preview.png", "image/png")
    with col2:
        st.markdown("**Reconstructed image (JPEG)**")
        st.image(result["restored"], use_container_width=True)
        st.download_button("Download reconstructed JPEG", result["restored_jpeg_bytes"], "astra_resq_reconstructed.jpg", "image/jpeg")

    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.markdown("**Flood mask (predicted from reconstructed JPEG)**")
        st.image(result["mask_image"], use_container_width=True, clamp=True)
        st.download_button("Download flood mask", image_bytes(result["mask_image"], "PNG"), "astra_resq_flood_mask.png", "image/png")
    with col4:
        st.markdown("**Flood overlay (JPEG-based prediction)**")
        st.image(result["overlay"], use_container_width=True)
        st.download_button("Download flood overlay", image_bytes(result["overlay"], "PNG"), "astra_resq_flood_overlay.png", "image/png")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-name">Predicted image coverage</div><div class="metric-value">{result["coverage"]:.1f}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-name">Flood threshold</div><div class="metric-value">{result["threshold"]:.2f}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-name">Model input</div><div class="metric-value">{result["image_size"]}² px</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">04 / Experimental route exploration</div>', unsafe_allow_html=True)
    st.warning("This is an image-grid A* demonstration, not a real-world evacuation route. The image is not automatically georeferenced, and predicted flood pixels are not verified ground truth. Do not use this output for emergency decisions.")
    route_col, map_col = st.columns([1, 1.5], gap="large")
    with route_col:
        st.write("Choose start and destination cells on the 256 × 256 prediction grid.")
        start_row = st.number_input("Start row", min_value=0, max_value=result["image_size"] - 1, value=10, step=1)
        start_col = st.number_input("Start column", min_value=0, max_value=result["image_size"] - 1, value=10, step=1)
        goal_row = st.number_input("Destination row", min_value=0, max_value=result["image_size"] - 1, value=result["image_size"] - 11, step=1)
        goal_col = st.number_input("Destination column", min_value=0, max_value=result["image_size"] - 1, value=result["image_size"] - 11, step=1)
        calculate = st.button("Find candidate grid path", use_container_width=True)
    with map_col:
        if calculate:
            route_result = astar_grid(result["mask"], (int(start_row), int(start_col)), (int(goal_row), int(goal_col)))
            if route_result is None:
                st.error("No path found through the unblocked cells. Try different endpoints or review the predicted mask.")
            else:
                path, cost = route_result
                route_vis = result["overlay"].copy()
                # Draw the candidate path in bright cyan on a copy of the overlay.
                for r, c in path:
                    route_vis[r, c] = [36, 245, 222]
                route_vis[int(start_row), int(start_col)] = [80, 255, 120]
                route_vis[int(goal_row), int(goal_col)] = [255, 221, 80]
                st.image(route_vis, caption="Candidate A* grid path · cyan path, green start, yellow destination", use_container_width=True)
                st.metric("Path cost (grid units)", f"{cost:.1f}")
                st.caption(f"Path contains {len(path)} grid cells. Grid units are not metres, kilometres, or minutes.")
                st.download_button("Download route demonstration", image_bytes(route_vis, "PNG"), "astra_resq_grid_route_demo.png", "image/png")
        else:
            st.image(result["overlay"], caption="Flood overlay used as the experimental grid", use_container_width=True)

    st.markdown("---")
    st.markdown("**Interpretation note**")
    st.write("The uploaded TIFF is reconstructed first, the reconstruction is encoded and decoded as JPEG, and Flood U-Net runs on that JPEG image. ASTRA-GAN reconstruction and Flood U-Net segmentation are separate learned models chained for exploratory analysis. The flood model was trained on a separate labelled RGB dataset; applying it to reconstructed imagery is a cross-dataset experiment. A geographic route requires georeferenced imagery, aligned road data, verified closures and shelter information, and independent validation.")

st.markdown("---")
st.markdown('<div class="footnote">ASTRA ResQ · Academic research prototype · Outputs are experimental, may be wrong, and are not a substitute for official flood warnings or emergency services.</div>', unsafe_allow_html=True)

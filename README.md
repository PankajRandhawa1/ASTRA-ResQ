# ASTRA ResQ — Modular Streamlit Project

## Project structure
```text
ASTRA_ResQ_VSCode/
├── app.py                    # Streamlit pages and dashboard flow
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── architectures.py      # ASTRA-GAN + Flood U-Net definitions
│   ├── astra_gan_generator.pth
│   └── astra_resq_flood_unet.pth
├── services/
│   ├── inference.py          # checkpoint loading + image pipeline
│   └── routing.py            # experimental pixel-grid A*
├── ui/
│   └── theme.py              # shared dashboard styling
├── assets/
└── road_data/                # reserved for georeferenced routing data
```

## Run in VS Code (Windows)
Open this folder in VS Code, open Terminal, then run:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again.

## What works
- Upload RGB images and run ASTRA-GAN reconstruction followed by Flood U-Net segmentation.
- View and download original preview, reconstruction, predicted mask, and overlay.
- View the predicted pixel coverage and threshold.
- Explore a candidate A* path on the image grid.

## Important limitations
- The flood coverage is a percentage of image pixels, not geographic area.
- Flood predictions are experimental, especially when the flood model is applied to ASTRA-GAN reconstructed imagery.
- The A* module is an image-grid demonstration only, not a geographic road route or a verified safe evacuation route. Real routing requires georeferenced imagery, spatially aligned road/shelter data, verified closures, and validation.
- Never use this academic prototype for emergency decisions.


## Updated image workflow

1. Upload a TIFF/TIF satellite image only.
2. ASTRA-GAN reconstructs the image (the reconstruction behavior depends on the trained checkpoint).
3. The reconstructed output is exported as JPEG at quality 95.
4. Flood U-Net receives the decoded JPEG pixels for flood mapping.
5. The predicted flood mask/overlay is used by the experimental A* grid route demo.

The route remains a pixel-grid demonstration, not a geographically verified evacuation route. The reconstruction output is JPEG; mask and overlay downloads are PNG.

"""Model loading and inference utilities for ASTRA ResQ."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from models.architectures import ASTRA_GAN_Generator, FloodUNet

IMAGE_SIZE = 256

def _load_state(path: Path):
    # Only load trusted project checkpoints. weights_only avoids arbitrary pickle loading.
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected checkpoint format in {path.name}; expected a state dictionary.")
    return obj

def load_models(gan_path_text: str, flood_path_text: str, device_text: str):
    device = torch.device(device_text)
    gan_state = _load_state(Path(gan_path_text))
    flood_ckpt = _load_state(Path(flood_path_text))
    flood_state = flood_ckpt.get("model_state_dict")
    if not isinstance(flood_state, dict):
        raise KeyError("Flood checkpoint must contain a 'model_state_dict' dictionary.")
    known_prefixes = ("enc1.", "enc2.", "enc3.", "enc4.", "bridge.", "up4.", "dec4.",
                      "up3.", "dec3.", "up2.", "dec2.", "up1.", "dec1.")
    unmatched = [k for k in flood_state if not k.startswith(known_prefixes)]
    output_weight_keys = [k for k in unmatched if k.endswith(".weight") and torch.is_tensor(flood_state[k]) and flood_state[k].ndim == 4]
    if len(output_weight_keys) != 1:
        raise ValueError(f"Could not identify Flood U-Net output layer. Candidate keys: {output_weight_keys}")
    output_key = output_weight_keys[0]
    output_name = output_key.rsplit(".", 1)[0]
    output_weight = flood_state[output_key]
    if "." in output_name or int(output_weight.shape[1]) != 32:
        raise ValueError("Flood checkpoint output layer does not match the notebook architecture.")
    kernel = int(output_weight.shape[2])
    if int(output_weight.shape[3]) != kernel:
        raise ValueError("Flood checkpoint output kernel is not square.")
    generator = ASTRA_GAN_Generator()
    generator.load_state_dict(gan_state, strict=True)
    flood_model = FloodUNet(output_layer_name=output_name, output_channels=int(output_weight.shape[0]),
                            output_kernel_size=kernel, output_bias=(output_name + ".bias") in flood_state)
    flood_model.load_state_dict(flood_state, strict=True)
    generator.to(device).eval()
    flood_model.to(device).eval()
    image_size = int(flood_ckpt.get("image_size", IMAGE_SIZE))
    if image_size < 32 or image_size % 16 != 0:
        image_size = IMAGE_SIZE
    return generator, flood_model, device, image_size

def pil_to_tensor(image: Image.Image, size: int, device: torch.device):
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device), resized

def to_uint8_rgb(tensor: torch.Tensor):
    arr = tensor.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)

def run_pipeline(original: Image.Image, generator, flood_model, device, image_size: int, threshold: float, alpha: float = 0.42):
    input_tensor, resized_input = pil_to_tensor(original, image_size, device)
    gan_input = input_tensor * 2.0 - 1.0
    with torch.inference_mode():
        gan_output = generator(gan_input)
        restored_tensor = ((gan_output + 1.0) / 2.0).clamp(0.0, 1.0)

    # Explicit pipeline boundary: reconstructed image is encoded to JPEG, then
    # decoded again; Flood U-Net receives pixels from this JPEG representation.
    restored = to_uint8_rgb(restored_tensor)
    jpeg_buffer = __import__("io").BytesIO()
    Image.fromarray(restored, mode="RGB").save(jpeg_buffer, format="JPEG", quality=95, optimize=True)
    restored_jpeg_bytes = jpeg_buffer.getvalue()
    jpeg_buffer.seek(0)
    reconstructed_jpeg = Image.open(jpeg_buffer).convert("RGB")
    jpeg_array = np.asarray(reconstructed_jpeg, dtype=np.uint8)
    flood_arr = jpeg_array.astype(np.float32) / 255.0
    flood_input = torch.from_numpy(flood_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = flood_model(flood_input)
        probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    restored = jpeg_array
    probability = np.clip(probability, 0, 1)
    mask = probability >= threshold
    original_array = np.asarray(resized_input).copy()
    overlay = create_overlay(restored, mask, alpha)
    return {"original": original_array, "restored": restored, "restored_jpeg_bytes": restored_jpeg_bytes,
            "probability": probability, "mask": mask, "mask_image": mask.astype(np.uint8) * 255, "overlay": overlay,
            "coverage": float(mask.mean() * 100.0), "image_size": image_size, "threshold": threshold}

def create_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.42):
    base = np.asarray(rgb, dtype=np.uint8).copy()
    overlay = base.copy()
    overlay[mask] = np.array([255, 74, 89], dtype=np.uint8)
    return (base.astype(np.float32) * (1-alpha) + overlay.astype(np.float32) * alpha).clip(0,255).astype(np.uint8)

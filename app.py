import streamlit as st
from PIL import Image
import requests
import numpy as np
import io
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline

@st.cache_resource
def load_models():
    seg_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small", torch_dtype=torch.float32)
    processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32
    )
    return seg_model.eval(), processor, inpaint

seg_model, processor, inpaint_pipeline = load_models()

def get_clothes_mask(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = seg_model(**inputs)
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    clothing_classes = [5, 6, 7]
    mask = np.isin(seg, clothing_classes).astype(np.uint8) * 255
    return Image.fromarray(mask).convert("L")

def recolor_clothes(image, color):
    mask = get_clothes_mask(image)
    prompt = f"A person wearing {color} clothes"
    image = image.convert("RGB").resize((512, 512))
    mask = mask.resize((512, 512))

    result = inpaint_pipeline(prompt=prompt, image=image, mask_image=mask).images[0]
    return result

st.set_page_config(page_title="Kleidung umfärben", layout="centered")
st.title("👕 Kleidung umfärben mit KI")

uploaded_image = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])
color_input = st.text_input("Neue Farbe (z. B. 'rot', 'blue', '#00ff00')", "blau")

if uploaded_image and color_input:
    image = Image.open(uploaded_image)
    st.image(image, caption="Originalbild", use_column_width=True)

    with st.spinner("Färbe Kleidung um..."):
        result = recolor_clothes(image, color_input)
    st.image(result, caption="Umgefärbte Kleidung", use_column_width=True)
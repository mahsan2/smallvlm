# app.py  ────────────────────────────────────────────────────────────
# Streamlit front-end for the Small-VLM multimodal classifier
# ────────────────────────────────────────────────────────────────────
import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, numpy as np
from torchvision import models, transforms
from PIL import Image

# ⬇️ 1. ----------------------------------------------------------------
# load model & tabular data  – cached so it happens only once
# the StandardScaler class is allow-listed *inside* this function
# so that torch.load can safely un-pickle it on Streamlit Cloud.
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_and_data():
    # allow-list sklearn's StandardScaler for safe unpickling
    from sklearn.preprocessing import StandardScaler
    import torch.serialization
    torch.serialization.add_safe_globals([StandardScaler])

    ckpt   = torch.load("small_vlm_defect.pt", map_location="cpu")
    scaler = ckpt["scaler"]
    df     = pd.read_csv("param_df_cleaned.csv")

    # ── model definition (must match training) ───────────────────────
    class SmallVLM(nn.Module):
        def __init__(self, n_params: int):
            super().__init__()
            base = models.resnet18(weights="IMAGENET1K_V1")
            base.fc = nn.Identity()      # 512-D feature vector
            self.cnn = base
            self.mlp = nn.Sequential(
                nn.Linear(n_params, 64), nn.ReLU(),
                nn.Linear(64, 32)
            )
            self.classifier = nn.Sequential(
                nn.Linear(512 + 32, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )

        def forward(self, img, vec):
            return self.classifier(
                torch.cat((self.cnn(img), self.mlp(vec)), dim=1)
            )

    model = SmallVLM(n_params=scaler.mean_.shape[0])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, scaler, df


# ⬇️ 2. ----------------------------------------------------------------
# initialisation (runs once thanks to cache_resource)
# ---------------------------------------------------------------------
model, scaler, param_df = load_model_and_data()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

label_map = {0: "Powder (empty bed)",
             1: "Printed region (healthy)"}

# ⬇️ 3. ----------------------------------------------------------------
# Streamlit user interface
# ---------------------------------------------------------------------
st.title("🖼️ Small-VLM • Multimodal Defect Classifier")

uploaded = st.file_uploader(
    "Upload an image (filename **must** start with layer number, e.g. `5_slice_0.png`)",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    # show uploaded picture
    img_pil = Image.open(uploaded).convert("RGB")
    st.image(img_pil, caption="Uploaded image", width=240)

    # extract layer number from filename
    layer_str = uploaded.name.split("_")[0]
    if not layer_str.isdigit():
        st.error("❌  Filename invalid – it should start with a layer number.")
        st.stop()

    layer_idx = int(layer_str)
    if layer_idx >= len(param_df):
        st.error(f"❌  No tabular row exists for layer {layer_idx}.")
        st.stop()

    # tabular vector for this layer (scaled)
    vec_raw = param_df.loc[layer_idx]
    vec = torch.tensor(
        scaler.transform(vec_raw.values.reshape(1, -1)),
        dtype=torch.float32
    )

    # image tensor
    x_img = transform(img_pil).unsqueeze(0)

    # model prediction
    with torch.no_grad():
        probs = F.softmax(model(x_img, vec), dim=1)
        conf, pred = torch.max(probs, 1)

    pred_label = label_map[pred.item()]
    st.markdown(f"### ✅ Prediction: **{pred_label}** "
                f"({conf.item()*100:.1f}% confidence)")

    # contextual parameters
    st.write(f"""
    **Process context**

    • Top-chamber T = **{vec_raw['top_chamber_temperature']:.0f} °C**  
    • Bottom flow = **{vec_raw['bottom_flow_rate']:.1f}%**  
    • Ventilator = **{vec_raw['ventilator_speed']:.0f} rpm**  
    • O₂ in gas loop = **{vec_raw['gas_loop_oxygen']:.1f} ppm**
    """)

    # simple rule-based suggestions
    tips = []
    if vec_raw['bottom_flow_rate'] < 45:
        tips.append("🔧 Increase bottom-flow > **45 %**.")
    if vec_raw['ventilator_speed'] < 40:
        tips.append("🔧 Boost ventilator > **40 rpm**.")
    if vec_raw['gas_loop_oxygen'] > 10:
        tips.append("🔧 Purge chamber (keep O₂ < **10 ppm**).")

    if tips:
        st.warning(" ".join(tips))
    else:
        st.success("All key parameters within nominal range ✅")

    # ⬇️ Grad-CAM (optional) -----------------------------------------
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        target_layer = model.cnn.layer4[-1]
        cam = GradCAM(model, [target_layer], device="cpu")
        heat = cam(x_img, extra_forward_args=(vec,))[0]

        rgb  = np.transpose(x_img.squeeze().numpy(), (1, 2, 0))
        cam_img = show_cam_on_image(rgb, heat, use_rgb=True)

        st.image(cam_img, caption="Grad-CAM", width=240)
    except Exception as e:
        st.info(f"Grad-CAM unavailable ({e})")

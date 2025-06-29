import streamlit as st
import torch, pandas as pd, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------------
# Load model + data
# ------------------------------
@st.cache_resource
def load_model_and_data():
    ckpt = torch.load("small_vlm_defect.pt", map_location="cpu")
    scaler = ckpt["scaler"]
    df = pd.read_csv("param_df_cleaned.csv")
    
    class SmallVLM(nn.Module):
        def __init__(self, n_params):
            super().__init__()
            base = models.resnet18(weights="IMAGENET1K_V1")
            base.fc = nn.Identity()
            self.cnn = base
            self.mlp = nn.Sequential(nn.Linear(n_params,64), nn.ReLU(), nn.Linear(64,32))
            self.classifier = nn.Sequential(nn.Linear(512+32,64), nn.ReLU(), nn.Linear(64,2))
        def forward(self, img, vec):
            return self.classifier(torch.cat((self.cnn(img), self.mlp(vec)), dim=1))

    model = SmallVLM(n_params=scaler.mean_.shape[0])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, scaler, df

model, scaler, param_df = load_model_and_data()

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
label_map = {0: "Powder (empty bed)", 1: "Printed region (healthy)"}

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Multimodal Defect Classifier (Small-VLM)")
uploaded_img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Uploaded image", width=256)

    match = uploaded_img.name.split("_")[0]
    if not match.isdigit():
        st.error("Filename must start with layer number (e.g., '5_slice_0.png')")
    else:
        layer_idx = int(match)
        if layer_idx >= len(param_df):
            st.error(f"No tabular data found for layer {layer_idx}")
        else:
            vec = torch.tensor(scaler.transform(param_df.loc[layer_idx].values.reshape(1,-1)), dtype=torch.float32)
            x_img = transform(img).unsqueeze(0)
            with torch.no_grad():
                probs = F.softmax(model(x_img, vec), dim=1)
                conf, pred = torch.max(probs, 1)

            label = label_map[pred.item()]
            st.markdown(f"### Prediction: **{label}** ({conf.item()*100:.1f}% confidence)")

            row = param_df.loc[layer_idx]
            st.write(f"""
                Top-chamber T = {row['top_chamber_temperature']:.0f} °C  
                Bottom flow   = {row['bottom_flow_rate']:.1f} %  
                Ventilator    = {row['ventilator_speed']:.0f} rpm  
                O₂            = {row['gas_loop_oxygen']:.1f} ppm  
            """)

            tips = []
            if row['bottom_flow_rate'] < 45: tips.append("🔧 Increase bottom-flow > 45 %.")
            if row['ventilator_speed'] < 40: tips.append("🔧 Boost ventilator > 40 rpm.")
            if row['gas_loop_oxygen'] > 10: tips.append("🔧 Purge chamber (O₂ < 10 ppm).")
            if tips:
                st.warning(" ".join(tips))
            else:
                st.success("✅ Parameters within nominal range.")

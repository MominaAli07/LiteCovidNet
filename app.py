import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# Load Model
# -------------------------
from custom import ImprovedSGLDModel, compute_entropy

# Load model checkpoint
model = ImprovedSGLDModel.load_from_checkpoint('epoch=99-step=46900.ckpt')
model.eval()
model.freeze()

# Correct class names based on folder order
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Pneumonia', 'TB']

# Prediction function
def predict_image(model, image, class_names, n_passes=30):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    x = transform(image).unsqueeze(0).to(model.device)

    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds.append(probs.cpu())
    preds = torch.cat(preds, dim=0)

    mean_probs = preds.mean(dim=0).squeeze()
    predicted_idx = mean_probs.argmax().item()
    predicted_class = class_names[predicted_idx]

    entropy = compute_entropy(mean_probs.unsqueeze(0)).item()

    # Compute variation ratio
    mode_preds = preds.argmax(dim=-1)
    majority_vote = mode_preds.mode(dim=0)[0]
    variation_ratio = 1.0 - (mode_preds.eq(majority_vote).sum().item() / n_passes)

    return predicted_class, mean_probs.numpy(), entropy, variation_ratio

# -------------------------
# Streamlit Frontend
# -------------------------
st.set_page_config(page_title="Chest X-Ray Diagnosis", layout="wide")

st.title("AI-based Chest X-Ray Diagnosis System")
st.subheader("Upload a chest X-ray image and get fast, reliable diagnosis!")

st.markdown("---")

uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([2, 3])

    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)

    with col2:
        if st.button('Predict Diagnosis'):
            predicted_class, confidences, entropy, variation_ratio = predict_image(model, image, class_names)

            st.success(f"Predicted Class: **{predicted_class}**")


st.markdown("---")
st.caption("Developed by Momina L. Ali using PyTorch Lightning, and Streamlit.")

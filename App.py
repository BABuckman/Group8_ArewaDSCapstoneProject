
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, vit_b_16, EfficientNet_B0_Weights, ViT_B_16_Weights
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Healthy", "Infected"]

# === ViT Base Model ===
def ViTBase(n_classes=2, freeze_backbone=True):
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    if freeze_backbone:
        for param in model.encoder.parameters():
            param.requires_grad = False
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.LayerNorm(num_ftrs),
        nn.Linear(num_ftrs, num_ftrs * 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(num_ftrs * 2, n_classes)
    )
    return model

# === EfficientNet Model ===
def EfficientNetB0(n_classes=2, freeze_backbone=True, dropout=0.5):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Linear(num_ftrs, num_ftrs * 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_ftrs * 2, n_classes)
    )
    return model

# === Load Models ===
vit_model = ViTBase(n_classes=2).to(DEVICE)
vit_model.load_state_dict(torch.load("/content/drive/MyDrive/Arewa_capstone_project/test_save_models/best_model_vit_b_16.pth", map_location=DEVICE))
vit_model.eval()

eff_model = EfficientNetB0(n_classes=2).to(DEVICE)
eff_model.load_state_dict(torch.load("/content/drive/MyDrive/Arewa_capstone_project/test_save_models/best_model_efficientnet_b0.pth", map_location=DEVICE))
eff_model.eval()

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

# === Grad-CAM reshape for ViT ===
def vit_reshape_transform(tensor):
    tensor = tensor[:, 1:, :]
    h = w = int(tensor.size(1) ** 0.5)
    return tensor.reshape(tensor.size(0), h, w, tensor.size(2)).permute(0, 3, 1, 2)

# === Prediction Function ===
def predict_with_gradcam(image, model_choice):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    if model_choice == "ViT":
        model = vit_model
        target_layers = [model.encoder.layers[-1].ln_1]
        reshape_transform = vit_reshape_transform
    else:
        model = eff_model
        target_layers = [model.features[-1]]
        reshape_transform = None

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        prediction = class_names[pred_idx]
        confidence = probs[pred_idx]

    # Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
    rgb_img = inv_normalize(img_tensor[0].cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Format probabilities
    prob_text = f"Confidence: {confidence:.2f}\n"
    for i, cls in enumerate(class_names):
        prob_text += f"{cls}: {probs[i]:.2f}\n"

    return prediction, cam_image, prob_text

# === Gradio App ===
def run_gradio_app():
    with gr.Blocks(css="""
        body { background-color: #fffbe6; }
        .centered { text-align: center; }
        #logo { display: block; margin-left: auto; margin-right: auto; width: 200px; }
        .label-icons { text-align: center; font-size: 14px; color: #555; }
    """) as demo:

        gr.Markdown("<img id='logo' src='file=/content/drive/MyDrive/Arewa_capstone_project/ArewaDS_logo.PNG'>")
        gr.Markdown("<h2 class='centered' style='color:#e0b528;'>Deep Learning Fellow Cohort 2</h2>")
        gr.Markdown("<h3 class='centered' style='color:#e0b528;'>Group 8 - ArewaDS Capstone Project</h3>")

        gr.Markdown("""
        <p class='centered'><b>Team Members:</b> Abubakar Abubakar Al-amin, Bernard Adjei Buckman, Halimat Musa, Kaloma Usman Majikumna<br>
        <b>Mentor:</b> Engr. Bala Abduljalil<br>
        <b>Project Title:</b> Improving Binary Classification of Tomato Leaf Health in Northern Nigeria Using Transfer Learning on TomatoEbola Dataset</p>
        """)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Tomato Leaf Image")
                model_selector = gr.Radio(choices=["ViT", "EfficientNet"], value="ViT", label="Select Model")
                predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                label_output = gr.Label(label="Predicted Class")
                cam_output = gr.Image(label="Grad-CAM Visualization")
                prob_output = gr.Textbox(label="Model Confidence & Probabilities")

        predict_btn.click(fn=predict_with_gradcam, inputs=[image_input, model_selector], outputs=[label_output, cam_output, prob_output])

        gr.Markdown("""
        <div class='label-icons'>
            <p>📤 Upload &nbsp;&nbsp;&nbsp; 📸 Camera &nbsp;&nbsp;&nbsp; 🧪 Examples</p>
        </div>
        """)

    demo.launch(share=True)

if __name__ == "__main__":
    run_gradio_app()

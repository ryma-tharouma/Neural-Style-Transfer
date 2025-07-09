import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import copy
import io

# --- Device and transforms ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 256
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])
unloader = transforms.ToPILImage()

def image_loader(image_file):
    image = Image.open(image_file).convert("RGB")
    image = loader(image).unsqueeze(0).to(device, torch.float)
    return image

# --- Loss Modules ---
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)
    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    return optim.LBFGS([input_img.requires_grad_()])

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img,
                       num_steps=300, style_weight=1e4, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            return loss
        optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    return input_img

# --- Streamlit App ---
st.title("🎨 Neural Style Transfer Web App")
st.write("Upload a **content** image and a **style** image, and get a stylized result.")

# Uploaders
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

# Side-by-side sliders
col_steps, col_style = st.columns(2)
with col_steps:
    steps = st.slider("🌀 Optimization steps", 50, 1000, 300, step=50)
with col_style:
    style_weight = st.slider("🎚️ Style weight", 1e3, 1e7, 1e4, step=1e3, format="%.0f")

# Button
if st.button("Run Style Transfer") and content_file and style_file:
    with st.spinner("Processing..."):

        # Load and preprocess
        content_img = image_loader(content_file)
        style_img = image_loader(style_file)
        input_img = content_img.clone()

        # Convert back to PIL for display
        content_display = unloader(content_img.cpu().squeeze(0))
        style_display = unloader(style_img.cpu().squeeze(0))

        # Show input images side-by-side
        st.markdown("### Uploaded Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(content_display, caption="🖼️ Content Image")
        with col2:
            st.image(style_display, caption="🎨 Style Image")

        # Load model
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        # Run NST
        output = run_style_transfer(
            cnn, cnn_normalization_mean, cnn_normalization_std,
            content_img, style_img, input_img,
            num_steps=steps, style_weight=style_weight
        )

        # Convert tensor → image
        output = output.cpu().clone().squeeze(0)
        result_image = unloader(output)

        # Show result
        st.markdown("### 🖌️ Stylized Output")
        st.image(result_image, caption="Stylized Output")

        # Download button
        img_bytes = io.BytesIO()
        result_image.save(img_bytes, format="PNG")
        st.download_button("📥 Download Output", img_bytes.getvalue(), file_name="stylized_output.png", mime="image/png")

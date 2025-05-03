import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from Util.Dataset import test_transform
from model.Mae import MaskedAutoEncoderViT
import mediapipe as mp


def main():
    image_path = "real-vs-fake/test/real/00001.jpg"
    #torch.Size([1, 3, 256, 256])
    #9451.72265625

    #image_path = "real-vs-fake/train/real/00006.jpg"
    #torch.Size([1, 3, 256, 256])
    #10665.046875

    #image_path = "real-vs-fake/train/fake/0A0IAK9X2W.jpg"
    #torch.Size([1, 3, 256, 256])
    #8410.689453125

    #image_path = "real-vs-fake/test/fake/0BI6V9VNPK.jpg"

    # Load image with OpenCV (in BGR), then convert to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    image_pil = Image.fromarray(image)

    image_pil.show()

    image_tensor = test_transform(image_pil).unsqueeze(0)  # Add batch dimension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming your VAE class is defined elsewhere
    vae = MaskedAutoEncoderViT(256).to(device)
    vae.load_state_dict(torch.load('MaeCheckPoint3.pth', map_location=device))
    vae.eval()

    image_tensor = image_tensor.to(device)

    print(image_tensor.shape)

    with torch.no_grad():
        output, loss = vae(image_tensor)

    # Convert tensor to image
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1 / s for s in [0.229, 0.224, 0.225]]
    )
    output = inv_normalize(output.squeeze(0).cpu())
    output_pil = transforms.ToPILImage()(output)
    output_pil.show()

    print(loss.item())


if __name__ == "__main__":
    main()
import os

import cv2
import torch

from codeFor490.Dataset import test_transform
from codeFor490.GenContVit.Mae1 import MaskedAutoEncoderViT


def main():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #if os.path.exists('MaeCheckPoint.pth'):
    #    model_mae = MaskedAutoEncoderViT(256)
    #    #model_mae.to(device)
    #    model_mae.load_state_dict(torch.load('MaeCheckPoint.pth', weights_only=True,  map_location=torch.device('cpu')))
    model_mae = MaskedAutoEncoderViT(256)
    model_mae.eval()
    image = cv2.imread("./real-vs-fake/test/real/00007.jpg", cv2.IMREAD_COLOR_RGB)

    x = test_transform(image)

    np_img1 = x.permute(1, 2, 0).detach().numpy()
    cv2.imshow("image1", np_img1)
    cv2.waitKey(0)

    x = x.unsqueeze(0)
    x, _ = model_mae(x)
    x = x.squeeze(0)
    np_img = x.permute(1, 2, 0).detach().numpy()
    cv2.imshow("image", np_img)
    cv2.waitKey(0)




if __name__ == "__main__":
    main()
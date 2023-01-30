import os

from tqdm import tqdm

import torch
from torchvision import transforms

from imageutil import imgconv
from model import s3fd_util
from dataset.simple import DirectoryImageDataset


def main():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    weight_path = "weights/s3fd.pth"
    model = s3fd_util.load_model(weight_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    model.eval()

    image_set_path = "datasets/casiagait_b_video90/test/"
    image_set = DirectoryImageDataset(
        image_set_path, transform=transform)

    image_loader = torch.utils.data.DataLoader(image_set)

    for (image_list, image_path) in tqdm(image_loader):

        pil_image_list = imgconv.tensor2pil(image_list)
        encoded_tuple = s3fd_util.image_list_encode(
            pil_image_list)

        s3fd_image_list = encoded_tuple[0].to(
            device=device, dtype=torch.float)
        scale_list = encoded_tuple[1].to(
            device=device, dtype=torch.float)

        with torch.no_grad():
            output = model(s3fd_image_list)

        detections_list = s3fd_util.make_detections_list(
            output, scale_list, s3fd_util.detections_s3fd, 0.6)

        for i, detections in enumerate(detections_list):
            if (detections is None) or (detections.total_det == 0):
                os.remove(image_path[i])


if __name__ == "__main__":
    main()

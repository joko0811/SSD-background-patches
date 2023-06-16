import os

import torch
from torchvision import transforms

from tqdm import tqdm

from model.S3FD import s3fd
from model.s3fd_util import S3fdResize
from dataset.simple import DirectoryImageDataset


def generate_data(path):
    def _load_model(path):
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Select device for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = s3fd.build_s3fd("test", 2)
        model.load_state_dict(torch.load(path))
        return model.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4

    out_path = os.path.join(path, "detection/")
    image_dir_path = os.path.join(path, "face/")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    S3FD_TRANSFORMS = transforms.Compose(
        [
            S3fdResize(),
            transforms.PILToTensor(),
            # transforms.ConvertImageDtype(torch.float),
        ]
    )

    image_set = DirectoryImageDataset(image_dir_path, transform=S3FD_TRANSFORMS)
    image_loader = torch.utils.data.DataLoader(image_set, batch_size=BATCH_SIZE)

    model = _load_model("weights/s3fd.pth")
    model.eval()

    thresh = 0.6
    for (image, image_path) in tqdm(image_loader):
        image = image.to(device=device, dtype=torch.float)

        output = model(image)

        for img_idx, data in enumerate(output):
            extract_output = data[data[..., 0] >= thresh]
            det_str = ""
            for det in extract_output:
                if det.nelement() == 0:
                    continue

                conf = det[0]
                box = det[1:]

                det_str += (
                    str(conf.item())
                    + " "
                    + str(box[0].item())
                    + " "
                    + str(box[1].item())
                    + " "
                    + str(box[2].item())
                    + " "
                    + str(box[3].item())
                    + "\n"
                )

            # ファイルまでのパス、拡張子を除いたファイル名を取得
            image_name = os.path.basename(image_path[img_idx]).split(".")[0]
            det_file_path = os.path.join(out_path, image_name + ".txt")

            with open(det_file_path, "w") as f:
                f.write(det_str)
    print("complete!")


def main():
    # path = sys.argv[1]
    path = "datasets/casia/train/"

    with torch.no_grad():
        generate_data(path)
    # tbx_monitor(path)


if __name__ == "__main__":
    main()

import os
from tqdm import tqdm
import pandas as pd

import torch
from torchvision import transforms


from model.S3FD import s3fd
from model.s3fd_util import S3fdResize
from dataset.simple import DirectoryImageDataset


def generate_data(model, image_set, path):
    out_path = os.path.join(path, "detection/")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    image_loader = torch.utils.data.DataLoader(image_set, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1
    thresh = 0.6
    for image, image_path in tqdm(image_loader):
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


def calculate_face_area(model, image_set):
    """
    全部の顔画像について顔領域の面積計算を行い、保存する
    例えばタイルと顔領域の顔領域比を考慮するために用いる
    """
    BATCH_SIZE = 1
    image_loader = torch.utils.data.DataLoader(image_set, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresh = 0.6

    area_df_columns = ["area"]
    area_df = pd.DataFrame([], columns=area_df_columns)
    for image_list, image_path_list in tqdm(image_loader):
        image_list = image_list.to(device=device, dtype=torch.float)
        output = model(image_list)

        for image_idx, det in enumerate(output):
            extract_output = det[det[..., 0] >= thresh]
            for det in extract_output:
                if det.nelement() == 0:
                    continue
                image_height, image_width = image_list[image_idx].shape[1:]
                det_height = det[..., 3] - det[..., 1]
                det_width = det[..., 2] - det[..., 0]

                # 出力された座標は画像サイズで正規化されているため乗算
                area = (
                    ((image_height * det_height) * (image_width * det_width))
                    .detach()
                    .cpu()
                    .resolve_conj()
                    .resolve_neg()
                    .numpy()
                )
                area_df.loc[image_path_list[image_idx]] = area

    area_df.to_csv("./all_face_area.csv")


def main():
    # データセットの画像全部に対して処理をする
    path = "datasets/faceforensics_pp/train"
    image_dir_path = os.path.join(path, "face/")

    def _load_model(path):
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Select device for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = s3fd.build_s3fd("test", 2)
        model.load_state_dict(torch.load(path))
        return model.to(device)

    S3FD_TRANSFORMS = transforms.Compose(
        [
            S3fdResize(),
            transforms.PILToTensor(),
            # transforms.ConvertImageDtype(torch.float),
        ]
    )

    image_set = DirectoryImageDataset(image_dir_path, transform=S3FD_TRANSFORMS)

    model = _load_model("weights/s3fd.pth")
    model.eval()

    with torch.no_grad():
        # 画像を読み込んで画像と同名の検出ファイルを作成する(yolo形式)
        # generate_data(model, image_set, path)

        # データセットの画像を全探索して全部の面積を算出する
        # 分布を見たい時にやる
        calculate_face_area(model, image_set)


if __name__ == "__main__":
    main()

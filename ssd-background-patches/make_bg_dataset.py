"""
convert dataset
video only dataset
→ dataset.train_background.TrainBackgroundDataset
"""
import os
import sys
import glob
import argparse
import logging
from tqdm import tqdm

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model.S3FD import s3fd
from model import s3fd_util
from dataset.simple import DirectoryImageDataset
from imageutil.imgseg import generate_mask_image
from util.videoutil import video2image_list


def step1_face(in_path, out_path):
    """convert dataset(video to image)
    ```in
    in_path/
      vid1.mp4
      ...
    ```

    ```out
    out_path/
      vid1_frame1.png
      vid1_frame5.png
      ...
    ```

    """
    image_per_frame = 5
    video_path_list = sorted(glob.glob("%s/*.*" % in_path))
    for video_path in video_path_list:
        video_filename = os.path.basename(video_path).split(".")[0]
        image_list = video2image_list(video_path, image_per_frame)

        for i, image in enumerate(image_list):
            image_file_path = os.path.join(
                out_path,
                video_filename + "_frame" + str((i) * image_per_frame) + ".png",
            )
            cv2.imwrite(image_file_path, image)


def step2_rm_non_detection(input_path):
    """rm non detection image"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            s3fd_util.S3fdResize(),
            transforms.PILToTensor(),
        ]
    )
    image_set = DirectoryImageDataset(input_path, transform=transform)
    image_loader = DataLoader(image_set, batch_size=1)

    def _load_model(path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        model = s3fd.build_s3fd("test", 2)
        model.load_state_dict(torch.load(path))
        return model.to(device)

    model = _load_model("weights/s3fd.pth")
    model.eval()

    thresh = 0.6

    with torch.no_grad():
        for image, image_size, path in image_loader:
            # 変換前の画像の座標に変換するための配列
            image = image.to(device=device, dtype=torch.float)
            output = model(image)
            extract_output = output[output[..., 0] >= thresh]
            if extract_output.nelement() == 0:
                os.remove(path[0])


def step3_split_train_test_val(
    input_path, output_path, data_split_name, data_split_rate, output_dir
):
    """
    ```in
    {input_path}/
        image1.png
        ...
    ```
    ```out
    {output_path}/
        {data_split_name[0]}/
            image1.png
            ...
        {dada_s...
    ```
    """

    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    image_set = DirectoryImageDataset(input_path, transform=transform)

    # データセットの分割後の画像数を決める処理
    remaining_image_num = len(image_set)
    split_num = [0] * len(data_split_rate)
    for i in range(len(split_num) - 1):
        split_num[i] = int(data_split_rate[i] * len(image_set))
        remaining_image_num -= split_num[i]
    # 少数点以下の誤差を埋めるため最後の分割はデータセットの未割り当て数とする
    split_num[-1] = remaining_image_num

    set_list = torch.utils.data.random_split(
        dataset=image_set,
        lengths=split_num,
        generator=torch.Generator(device=device).manual_seed(42),
    )

    def _save_dataset(dataset, save_path):
        image_loader = DataLoader(dataset, batch_size=1)
        for image, image_size, path in image_loader:
            image = transforms.functional.to_pil_image(image.squeeze())

            image_name = os.path.basename(path[0])
            image.save(os.path.join(save_path, image_name))

    for i, dataset in enumerate(set_list):
        save_path = os.path.join(
            os.path.join(output_path, data_split_name[i]), output_dir
        )
        _save_dataset(dataset, save_path)


def step4_mask(base_path, data_split_name, input_file_dir, output_file_dir):
    """
    ```
    {base_path}/
        {data_split_name[0]}/
            {input_file_dir}/
            {output_file_dir}/
        {data_split_name[1]}/
            {input_file_dir}/
            ...
    ```
    """
    transform = transforms.Compose([transforms.ToTensor()])
    for dsn in data_split_name:
        dsn_path = os.path.join(base_path, dsn)
        input_path = os.path.join(dsn_path, input_file_dir)
        output_path = os.path.join(dsn_path, output_file_dir)

        image_set = DirectoryImageDataset(input_path, transform=transform)
        image_loader = DataLoader(image_set, batch_size=1)

        for image, image_size, image_path in tqdm(image_loader):
            image_name = os.path.basename(image_path[0])
            save_path = os.path.join(output_path, image_name)
            image = transforms.functional.to_pil_image(image.squeeze())
            mask_image = generate_mask_image(image)
            mask_image.save(save_path)


def step5_detection(base_path, data_split_name, input_file_dir, output_file_dir):
    """
    ```
    {base_path}/
        {data_split_name[0]}/
            {input_file_dir}/
            {output_file_dir}/
        {data_split_name[1]}/
            {input_file_dir}/
            ...
    ```
    """

    def _load_model(path):
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Select device for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = s3fd.build_s3fd("test", 2)
        model.load_state_dict(torch.load(path))
        return model.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4

    transform = transforms.Compose(
        [
            s3fd_util.S3fdResize(),
            transforms.PILToTensor(),
        ]
    )

    model = _load_model("weights/s3fd.pth")
    model.eval()

    thresh = 0.6

    with torch.no_grad():
        for dsn in data_split_name:
            dsn_path = os.path.join(base_path, os.path.join(dsn, input_file_dir))
            image_set = DirectoryImageDataset(dsn_path, transform=transform)
            image_loader = DataLoader(image_set, batch_size=1)

            output_path = os.path.join(base_path, os.path.join(dsn, output_file_dir))

            for image, image_size, image_path in tqdm(image_loader):
                image = image.to(device=device, dtype=torch.float)

                output = model(image).squeeze()

                extract_output = output[output[..., 0] >= thresh]

                for det in extract_output:
                    if det.nelement() == 0:
                        continue

                    conf = det[0]
                    box = det[1:]

                    det_str = ""
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
                    image_name = os.path.basename(image_path[0]).split(".")[0]
                    det_file_path = os.path.join(output_path, image_name + ".txt")

                    with open(det_file_path, "w") as f:
                        f.write(det_str)


def main():
    """
    {input_path}/
        video_1.mp4
        ...
    {output_path}/
        train/
            face/
                video_1_frame_1.png
                ...
            mask/
                video_1_frame_1.png
                ...
            detection/
                video_1_frame_1.txt
                ...
        test/
            face/
            ...
        val/
            ...
    """
    parser = argparse.ArgumentParser(
        prog="make_bg_dataset",
        description="convert TrainBackgroundDataset format",
    )
    # video dir path
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # step 1,2 path
    # step 3,4,5 path
    data_split_name = ["train", "test", "val"]
    data_split_rate = [0.5, 0.4, 0.1]
    file_type = ["face", "mask", "detection"]

    rm_non_det_path = args.input
    rm_non_det_path = os.path.join(output_path, "rm_non_det")

    os.makedirs(rm_non_det_path)
    for dpn in data_split_name:
        dpn_path = os.path.join(output_path, dpn)
        os.mkdir(dpn_path)
        for ft in file_type:
            os.mkdir(os.path.join(dpn_path, ft))

    # {output_path}/rm_non_det
    step1_face(input_path, rm_non_det_path)
    logging.info("end of video2image[1/5]")
    sys.stdout.flush()

    step2_rm_non_detection(rm_non_det_path)
    logging.info("end of rm_non_det generation[2/5]")
    sys.stdout.flush()

    # {output_path}/[data_split_name]/file_type[0]
    step3_split_train_test_val(
        rm_non_det_path, output_path, data_split_name, data_split_rate, file_type[0]
    )
    logging.info("end of " + file_type[0] + " generation[3/5]")
    sys.stdout.flush()

    # {output_path} / [data_split_name] / file_type[1]
    step4_mask(output_path, data_split_name, file_type[0], file_type[1])
    logging.info("end of " + file_type[1] + " generation[4/5]")
    sys.stdout.flush()

    # {output_path}/[data_split_name]/file_type[2]
    step5_detection(output_path, data_split_name, file_type[0], file_type[2])
    logging.info("end of " + file_type[2] + " generation[5/5]")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

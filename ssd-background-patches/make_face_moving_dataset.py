import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model.S3FD import s3fd
from model import s3fd_util
from dataset.mask import DirectoryImageWithMaskDataset
from dataset.simple import DirectoryImageDataset
from imageutil.imgarg import augmentation_from_short_form_xywh


def step1_rotate(input_path, output_path):
    """
    データセットの画像からランダムに10件抽出
    画像上のマス目上の領域に顔が存在するように画像をローテートさせ、保存
    """

    BATCH_SIZE = 1
    total_image_num = 10
    input_face_dir = os.path.join(input_path, "face")
    input_mask_dir = os.path.join(input_path, "mask")
    output_face_dir = os.path.join(output_path, "face")
    output_mask_dir = os.path.join(output_path, "mask")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            s3fd_util.S3fdResize(),
            transforms.PILToTensor(),
        ]
    )

    dataset = DirectoryImageWithMaskDataset(
        input_face_dir, input_mask_dir, transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator(device=device),
    )

    def _load_model(path):
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Select device for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = s3fd.build_s3fd("test", 2)
        model.load_state_dict(torch.load(path))
        return model.to(device)

    model = _load_model("weights/s3fd.pth")
    model.eval()

    conf_thresh = 0.6
    for image_num, (image_list, mask_image_list, image_info) in enumerate(dataloader):
        if image_num >= total_image_num:
            break

        # 拡張子を除いた画像のファイル名を取得
        filename = os.path.basename(image_info["image_path"][0]).split(".")[0]

        image_list = image_list.to(device=device, dtype=torch.float)
        mask_image_list = mask_image_list.to(device=device)

        image = image_list.squeeze(0)
        mask_image = mask_image_list.squeeze(0)

        output = model(image_list)
        extract_output = output[output[..., 0] >= conf_thresh]

        if extract_output.nelement() == 0:
            continue

        conf = extract_output[0, 0]
        box = extract_output[0, 1:]

        augmentation_shift_list = augmentation_from_short_form_xywh(image, box)

        # 保存
        for aug_num, augmentation_shift in tqdm(enumerate(augmentation_shift_list)):
            augmented_image = torch.roll(
                image, dims=(1, 2), shifts=augmentation_shift.tolist()
            )
            augmented_mask_image = torch.roll(
                mask_image, dims=(1, 2), shifts=augmentation_shift.tolist()
            )

            desize_aug_img = s3fd_util.S3fdDeTransform()(
                augmented_image, image_info["image_size"][0]
            )
            desize_aug_mask_img = s3fd_util.S3fdDeTransform()(
                augmented_mask_image, image_info["image_size"][0]
            )

            face_save_path = os.path.join(
                output_face_dir, filename + "-" + str(aug_num) + ".png"
            )
            mask_save_path = os.path.join(
                output_mask_dir, filename + "-" + str(aug_num) + ".png"
            )
            desize_aug_img.save(face_save_path)
            desize_aug_mask_img.save(mask_save_path)

        print("progress: " + str(image_num + 1) + "/" + str(total_image_num))


def step2_detection(input_path):
    BATCH_SIZE = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_face_dir = os.path.join(input_path, "face")
    output_dir = os.path.join(input_path, "detection")

    transform = transforms.Compose(
        [
            s3fd_util.S3fdResize(),
            transforms.PILToTensor(),
        ]
    )

    dataset = DirectoryImageDataset(input_face_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator(device=device),
    )

    def _load_model(path):
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Select device for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = s3fd.build_s3fd("test", 2)
        model.load_state_dict(torch.load(path))
        return model.to(device)

    model = _load_model("weights/s3fd.pth")
    model.eval()

    conf_thresh = 0.6
    for image, image_size, image_path in dataloader:
        image = image.to(device=device, dtype=torch.float)
        output = model(image).squeeze()
        extract_output = output[output[..., 0] >= conf_thresh]

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
            det_file_path = os.path.join(output_dir, image_name + ".txt")

            with open(det_file_path, "w") as f:
                f.write(det_str)


def main():
    input_path = "datasets/casia/test/"
    output_path = "datasets/casia/test_face_moving/"

    with torch.no_grad():
        # step1_rotate(input_path, output_path)
        step2_detection(output_path)


if __name__ == "__main__":
    main()

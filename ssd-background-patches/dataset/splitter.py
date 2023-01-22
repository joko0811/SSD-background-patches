import os
import glob

from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class ImagePerVideoDataset(Dataset):
    def __init__(self, dir_path, transform=None):

        # casia gait dataset all video
        video_num = 124  # 001..124

        self.image_path_list = list()
        for i in range(video_num):
            # image(001....png,002....png) list par video
            image_regex = os.path.join(dir_path, str(i+1).zfill(3)+"*.png")
            self.image_path_list.append(sorted(glob.glob(image_regex)))

        self.transform = transform

    def __getitem__(self, index):
        image_path_list = self.image_path_list[index % len(
            self.image_path_list)]
        image_list = list()
        for image_path in image_path_list:
            image = Image.open(image_path)

            if self.transform is not None:
                image = self.transform(image)
            image_list.append(image)

        return image_list, image_path_list

    def __len__(self):
        return len(self.image_path_list)


def all_saver(dataset, path):
    dataloader = DataLoader(dataset)
    if not os.path.exists(path):
        os.mkdir(path)

    for image_list, image_path_list in tqdm(dataloader):
        for image, image_path in zip(image_list, image_path_list):
            save_path = os.path.join(path, os.path.basename(image_path[0]))
            pil_image = transforms.functional.to_pil_image(image[0])
            pil_image.save(save_path)


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    path = "datasets/casiagait_b_video90/image"
    dataset = ImagePerVideoDataset(path, transform=transform)

    video_num = len(dataset)
    train_size = int(video_num*0.8)
    test_size = video_num-train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    base_path = os.path.join(*path.split("/")[:-1])

    train_path = os.path.join(base_path, "train")
    all_saver(train_set, train_path)

    test_path = os.path.join(base_path, "test")
    all_saver(test_set, test_path)


if __name__ == "__main__":
    main()

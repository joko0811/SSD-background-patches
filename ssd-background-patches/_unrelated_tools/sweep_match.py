import os
import sys
import glob

# Delete source images that do not match the mask image
# I don't use it because it's a broken algorithm.


def sweep_match(path: str):
    orig_image_dir_path = os.path.join(path, "image/")
    orig_image_path_list = sorted(glob.glob("%s/*.*" % orig_image_dir_path))

    mask_image_dir_path = os.path.join(path, "mask/")
    mask_image_path_list = sorted(glob.glob("%s/*.*" % mask_image_dir_path))

    # image
    m_img_i = 0
    for o_img_i in range(len(orig_image_path_list)):

        if m_img_i >= len(mask_image_path_list):
            os.remove(orig_image_path_list[o_img_i])
            continue

        orig_image_name = os.path.basename(orig_image_path_list[o_img_i])
        mask_image_name = os.path.basename(mask_image_path_list[m_img_i])

        if orig_image_name != mask_image_name:
            os.remove(orig_image_path_list[o_img_i])
        else:
            m_img_i += 1

    return


def is_sweeped(path: str):
    orig_image_dir_path = os.path.join(path, "image/")
    orig_image_path_list = sorted(glob.glob("%s/*.*" % orig_image_dir_path))

    mask_image_dir_path = os.path.join(path, "mask/")
    mask_image_path_list = sorted(glob.glob("%s/*.*" % mask_image_dir_path))

    # image
    m_img_i = 0
    for o_img_i in range(len(orig_image_path_list)):

        orig_image_name = os.path.basename(orig_image_path_list[o_img_i])
        mask_image_name = os.path.basename(mask_image_path_list[m_img_i])

        if orig_image_name != mask_image_name:
            return False
        else:
            m_img_i += 1

    return True


def main():
    dataset_path = sys.argv[1]
    sweep_match(dataset_path)
    if is_sweeped(dataset_path):
        print("success!")
    else:
        print("sorry, something went wrong...")


if __name__ == "__main__":
    main()

import torch

from omegaconf import DictConfig

from box import boxconv, seek, transform


def calculate_search_area(image, boxes, config: DictConfig):
    """計算量削減のため、画像からスライディングウインドウで探索する領域を抽出する"""

    # パッチと対象物体の間のoffsetを算出する際に用いる
    largest_dist_rate = config.largest_dist_rate  # default 0.2

    # グループに属する箱の最外殻を決める
    box_outermost = seek.smallest_box_containing(boxes)
    # 箱の最大辺から拡張する幅を決める
    search_offset = seek.get_max_edge(boxes) * largest_dist_rate
    # 箱の最外殻から拡張幅分増やした箱を取得する
    before_clamp_area = transform.box_expand(box_outermost, search_offset)
    # 領域を画像サイズに収める
    search_area = transform.box_clamp(
        before_clamp_area, image.shape[-2], image.shape[-1]
    )

    # 画像から探索領域を切り出す
    search_image = transform.image_crop_by_box(image, search_area)

    return search_image, search_area


def calculate_window_wh(boxes, config: DictConfig):
    """スライディングウインドウのwhを決定する
    Args:
        boxes:
            xywh形式の箱
    """

    # スライディングウインドウのサイズを決める定数
    # （論文内では明記されていなかったが、サイズ＝面積とする）
    initialize_size_rate = config.initialize_size_rate  # default 0.2
    # スライディングウインドウのアスペクト比を示す配列(1:要素)
    aspect_rate = torch.tensor([1, 0.67, 0.75, 1.5, 1.33], device=boxes.device)

    # ウインドウサイズ決定
    # ウインドウサイズ決定のため、グループ内で最大の面積を求める
    max_size = torch.max(boxes[:, 2] * boxes[:, 3])
    # アスペクト比、サイズからウインドウの高さと幅を求める
    window_w = torch.sqrt(initialize_size_rate * max_size / aspect_rate)
    window_h = window_w * aspect_rate
    return window_w, window_h


def extract_sliding_windows(
    partial_image, x1y1_partial_image, sw_w, sw_h, n, ignore_boxes
):
    """スライディングウインドウを作成、順位づけ

    画像からw*hのスライディングウインドウを作成、ウインドウ内の勾配強度の合計をとり、これを基に降順に順位付け
    ignore_boxに重ならないものを対象に、上位n件を抽出


    Args:
        img: これをもとにスライディングウインドウを作成
        img_offset: 画像の右上の座標[x,y]。これをもとに画像全体での座標を算出する
        sw_w: スライディングウインドウの幅
        sw_h: スライディングウインドウの高さ
        n: 抽出してくる数を決めるパラメータ
        ignore_box: 除外対象の領域を定める。[x1,y1,x2,y2]
    """

    # スライディングウインドウで切り出せる範囲を全部取ってくる
    # ([1,3,a,b])->([1,3,a-h+1,b-w+1,h,w])
    slide_windows = partial_image.abs().unfold(2, sw_h, 1).unfold(3, sw_w, 1)

    # ウインドウごとに範囲内の全画素の合計をとり、三原色を合計
    # ([1,3,a-h+1,b-w+1,h,w])->([1,a-h+1,b-w+1])
    windows_grad_sum = slide_windows.sum(dim=5).sum(dim=4).sum(dim=1)

    # ソートしたときに全ての要素の順位が出るようにreshape
    # ([1,a-h+1,b-w+1])->([1,(a-h+1)*(b-w+1)])
    windows_num = windows_grad_sum.shape[1] * windows_grad_sum.shape[2]
    sort_array = windows_grad_sum.reshape((windows_grad_sum.shape[0], windows_num))
    # 順位づけ配列の形状をwindows_grad_sumに合わせる
    # これでwindows_grad_sumと同じインデックスの要素に全要素中の降順番号が降られた
    rsp_windows_grad_sum_sortnum = sort_array.argsort(descending=True).reshape(
        windows_grad_sum.shape
    )

    # 画像サイズhwの二次元マップ
    point_map = create_box_map_of_original_image(
        x1y1_partial_image, list(windows_grad_sum.shape[1:]), sw_w, sw_h
    )

    # 上位n_b件を抽出する
    extract_counter = 0
    extract_iter = 0
    output_box = torch.zeros(n, 4, device=point_map.device)
    output_gradient_sum = torch.zeros(n, device=windows_grad_sum.device)
    while extract_counter < n and extract_iter < windows_num:
        # 座標と勾配の合計を取ってくる
        extract_idx = (rsp_windows_grad_sum_sortnum == extract_iter).nonzero()[0]
        extract_xyxy = point_map[extract_idx[1], extract_idx[2]]
        extract_gradient_sum = windows_grad_sum[
            extract_idx[0], extract_idx[1], extract_idx[2]
        ]

        if (not boxconv.is_overlap_list(extract_xyxy, output_box)) and (
            not boxconv.is_overlap_list(extract_xyxy, ignore_boxes)
        ):
            # 除外リストと一つも重ならない場合、返り値に含める
            output_box[extract_counter] = extract_xyxy
            output_gradient_sum[extract_counter] = extract_gradient_sum
            extract_counter += 1

        extract_iter += 1
    return output_box, output_gradient_sum


def initial_background_patches(
    ground_truthes, gradient_image: torch.tensor, config: DictConfig, scale=None
):
    if scale is None:
        scale = torch.tensor([1, 1, 1, 1]).to(device=gradient_image.device)

    # グループごとのパッチ最大数
    n_b = config.n_b  # default 3

    # 返り値
    bp_boxes = torch.zeros(
        ground_truthes.total_group, n_b, 4, device=gradient_image.device
    )  # 選択したパッチの座標(xyxy)
    bp_grad_sumes = torch.zeros(
        ground_truthes.total_group, n_b, device=gradient_image.device
    )  # 選択したパッチ内部の勾配合計

    for group_idx in range(ground_truthes.total_group):
        group_bp_boxes = torch.zeros(n_b, 4, device=bp_boxes.device)
        group_bp_grad_sum = torch.ones(n_b, device=bp_grad_sumes.device) * (
            -1 * float("inf")
        )

        # グループに属するboxの抽出
        group_xyxy = (
            ground_truthes.xyxy[ground_truthes.group_labels == group_idx]
        ) * scale
        group_xywh = boxconv.xyxy2xywh(group_xyxy)

        # 探索対象領域の決定
        search_grad_img, search_area = calculate_search_area(
            gradient_image, group_xyxy, config.calculate_search_area
        )

        # ウインドウの高さ、幅の決定
        window_list_w, window_list_h = calculate_window_wh(
            group_xywh, config.calculate_window_wh
        )

        for window_w, window_h in zip(window_list_w, window_list_h):
            x1y1_search_area = search_area[:2]

            ignore_boxes = torch.cat(
                (
                    ground_truthes.xyxy * scale,
                    group_bp_boxes,
                    bp_boxes.view(1, -1, bp_boxes.shape[-1]).squeeze(0),
                )
            )
            ex_boxes, ex_grad_sumes = extract_sliding_windows(
                search_grad_img,
                x1y1_search_area,
                int(window_w.item()),
                int(window_h.item()),
                n_b,
                ignore_boxes,
            )

            # もしex_boxesの勾配の合計値(=ex_grad_sum)が既存のパッチ領域の勾配の合計値のいずれかより大きかった場合交換
            # bp_boxes,bp_grad_sumesはbp_gd_totalesの昇順に並んでいる
            for ex_idx in range(n_b):
                for bp_idx in range(n_b):
                    if group_bp_grad_sum[bp_idx] <= ex_grad_sumes[ex_idx]:
                        group_bp_boxes[bp_idx] = ex_boxes[ex_idx]
                        group_bp_grad_sum[bp_idx] = ex_grad_sumes[ex_idx]
                        break

                # group_idx番目のグループに対して適用するパッチ領域の決定
        bp_boxes[group_idx] = group_bp_boxes
        bp_grad_sumes[group_idx] = group_bp_grad_sum

    return bp_boxes.reshape((ground_truthes.total_group * n_b, 4)) / scale


def expanded_background_patches(
    bp_boxes, ground_truthes, gradient_image, config: DictConfig, scale=None
):
    if scale is None:
        scale = torch.tensor([1, 1, 1, 1]).to(device=bp_boxes.device)

    scaled_gt_xywh = boxconv.xyxy2xywh(ground_truthes.xyxy * scale)

    stride_rate = config.stride_rate  # default 0.02
    image_h, image_w = gradient_image.shape[2:4]
    # stride = stride_rate*max(image_h, image_w)
    stride = stride_rate * scaled_gt_xywh[:, 2:].max()

    bp_area_threshold_rate = config.bp_area_threshold_rate  # default 0.02
    max_gt_area = (scaled_gt_xywh[:, 2] * scaled_gt_xywh[:, 3]).max()
    bp_area_threshold = max_gt_area * bp_area_threshold_rate

    scaled_bp_boxes = bp_boxes * scale
    new_bp_boxes = scaled_bp_boxes.clone()

    for i, bp_box in enumerate(scaled_bp_boxes):
        bp_box_wh = boxconv.xyxy2xywh(bp_box)[2:]
        bp_box_area = bp_box_wh[0] * bp_box_wh[1]
        if bp_box_area > bp_area_threshold:
            continue

        max_gradient_sum = torch.tensor([0], device=bp_box.device)

        # 4方向への拡張領域のうち、勾配総和が最大になるものを選ぶ
        for j in range(len(bp_box)):
            # 拡張したとき、差分となる領域のbox
            bp_box_diff = bp_box.clone()

            if j <= 1:
                bp_box_diff[j + 2] = bp_box_diff[j]
                bp_box_diff[j] -= stride
            else:
                bp_box_diff[j - 2] = bp_box_diff[j]
                bp_box_diff[j] += stride

            if j % 2 == 0:
                bp_box_diff = bp_box_diff.clamp(min=0, max=(image_w - 1))
            else:
                bp_box_diff = bp_box_diff.clamp(min=0, max=(image_h - 1))

            gradient_sum = (
                transform.image_crop_by_box(gradient_image, bp_box_diff).abs().sum()
            )

            if max_gradient_sum <= gradient_sum:
                # 拡張するbp_box
                expand_bp_box = bp_box.clone()
                if j <= 1:
                    expand_bp_box[j] -= stride
                else:
                    expand_bp_box[j] += stride

                if j % 2 == 0:
                    expand_bp_box = expand_bp_box.clamp(min=0, max=(image_w - 1))
                else:
                    expand_bp_box = expand_bp_box.clamp(min=0, max=(image_h - 1))

                # new_bp_boxesからj番目の要素を除いた配列
                compare_boxes = torch.cat((new_bp_boxes[:i], new_bp_boxes[i + 1 :]))
                # パッチ領域が除外対象の領域と重ならない場合更新
                ignore_boxes = torch.cat((ground_truthes.xyxy * scale, compare_boxes))
                if not boxconv.is_overlap_list(expand_bp_box, ignore_boxes):
                    max_gradient_sum = gradient_sum
                    new_bp_boxes[i] = expand_bp_box

    return new_bp_boxes / scale


def perturbation_in_background_patches(gradient_image, bp_boxes):
    # bp_boxesを元にマスクを作成
    bp_mask = transform.box2mask(gradient_image, bp_boxes)
    return bp_mask * gradient_image


def perturbation_normalization(perturbation_image, config: DictConfig):
    l2_norm_lambda = config.l2_norm_lambda  # default 0.03=1/30
    return (
        l2_norm_lambda / (torch.linalg.norm(perturbation_image) + 1e-9)
    ) * perturbation_image


def update_i_with_pixel_clipping(image, perturbated_image):
    return torch.clamp(image - perturbated_image, 0, 255)


# windows_grad_sum[1,y,x]の時、map[1,y,x]→スライディングウインドウのxyxyになる座標マップ
def create_box_map_of_original_image(x1y1_partial_image, hw_partial_image, w, h):
    """切り出してきた画像の右上座標xy、whを元に元画像のxyxy座標を返す座標マップを作成"""
    map_hw = hw_partial_image.copy()
    map_hw.append(4)

    point_map = torch.ones(map_hw, device=x1y1_partial_image.device) * torch.cat(
        (x1y1_partial_image, x1y1_partial_image)
    )
    img_h, img_w = hw_partial_image

    i_0 = (
        torch.arange(img_h, device=x1y1_partial_image.device)
        .unsqueeze(dim=1)
        .repeat(1, img_w)
    )
    i_1 = torch.arange(img_w, device=x1y1_partial_image.device)

    point_map[..., 0] += i_1
    point_map[..., 1] += i_0
    point_map[..., 2] += i_1 + w
    point_map[..., 3] += i_0 + h

    return point_map

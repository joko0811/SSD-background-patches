import torch

from box import condition, seek, transform


def calc_z(class_scores, iou_scores):
    """calc z param
    zの要素数はクラス数と等しい
    Args:
      class_score:
        GroundTruthesに対応する検出の攻撃対象クラスのクラススコア
      iou_score:
        Ground Truthesに対応する検出とGroundTruthesのIoUスコア
    """
    class_score_threshold = 0.1
    iou_score_threshold = 0.5

    z = torch.logical_and((class_scores > class_score_threshold),
                          (iou_scores > iou_score_threshold))
    return z.long()


def calc_r(iou_scores, detection_boxes, ground_truth_boxes):
    """calc r param
    rの要素数はクラス数と等しい
    Args:
      iou_score:
        検出と背景パッチのIoUスコア
      predict_boxes:
        検出全て
      target_boxes:
        ground truthのボックス全て
    """
    iou_score_threshold = 0.1
    iou_flag = iou_scores > iou_score_threshold
    overlap_flag = torch.logical_not(condition.are_overlap_list(
        detection_boxes, ground_truth_boxes))
    r = torch.logical_and(iou_flag, overlap_flag)
    return r.long()


def extract_sliding_windows(img, img_offset, sw_w, sw_h, n, ignore_box1, ignore_box2):
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
    # a,b,w,hは式は合ってるけど項の場所は怪しいかも
    slide_windows = img.abs().unfold(2, sw_h, 1).unfold(3, sw_w, 1)

    # ウインドウごとに範囲内の全画素の合計をとり、三原色を合計
    # ([1,3,a-h+1,b-w+1,h,w])->([1,a-h+1,b-w+1])
    windows_grad_sum = slide_windows.sum(dim=5).sum(
        dim=4).sum(dim=1)

    # ソートしたときに全ての要素の順位が出るようにreshape
    # ([1,a-h+1,b-w+1])->([1,(a-h+1)*(b-w+1)])
    windows_num = windows_grad_sum.shape[1]*windows_grad_sum.shape[2]
    sort_array = windows_grad_sum.reshape(
        (windows_grad_sum.shape[0], windows_num))
    # 順位づけ配列の形状をwindows_grad_sumに合わせる
    # これでwindows_grad_sumと同じインデックスの要素に全要素中の降順番号が降られた
    rsp_windows_grad_sum_sortnum = sort_array.argsort(
        descending=True).reshape(windows_grad_sum.shape)

    # windows_grad_sum[1,y,x]の時、map[1,y,x]→スライディングウインドウのxyxyになる座標マップ
    def index2xyxy(img_shape, img_offset, w, h):
        map_shape = img_shape.copy()
        map_shape.append(4)

        point_map = torch.ones(
            map_shape, device=img_offset.device)*torch.cat((img_offset, img_offset))
        _, img_h, img_w = img_shape

        i_0 = torch.arange(img_h, device=img_offset.device).unsqueeze(
            dim=1).repeat(1, img_w)
        i_1 = torch.arange(img_w, device=img_offset.device)

        point_map[..., 0] += i_1
        point_map[..., 1] += i_0
        point_map[..., 2] += i_1+w
        point_map[..., 3] += i_0+h

        return point_map
    # 画像サイズのマップ
    point_map = index2xyxy(list(windows_grad_sum.shape),
                           img_offset, sw_w, sw_h)

    # 上位n_b件を抽出する
    extract_counter = 0
    extract_iter = 0
    output_box = torch.zeros(n, 4, device=point_map.device)
    output_gradient_sum = torch.zeros(n, device=windows_grad_sum.device)
    while extract_counter < n and extract_iter < windows_num:
        # 座標と勾配の合計を取ってくる
        extract_idx = (rsp_windows_grad_sum_sortnum ==
                       extract_iter).nonzero()[0]
        extract_xyxy = point_map[extract_idx[0],
                                 extract_idx[1], extract_idx[2]]
        extract_gradient_sum = windows_grad_sum[extract_idx[0],
                                                extract_idx[1], extract_idx[2]]

        if (not condition.is_overlap_list(extract_xyxy, output_box)) and (not condition.is_overlap_list(extract_xyxy, ignore_box1)) and (not condition.is_overlap_list(extract_xyxy, ignore_box2)):
            # 除外リストと一つも重ならない場合、返り値に含める
            output_box[extract_counter] = extract_xyxy
            output_gradient_sum[extract_counter] = extract_gradient_sum
            extract_counter += 1

        extract_iter += 1
    return output_box, output_gradient_sum


def initial_background_patches(ground_truthes, gradient_image: torch.tensor):

    # パラメータの設定
    initialize_size_rate = 0.2  # 箱のサイズ（論文内では明記されていなかったが、サイズ＝面積とする）
    aspect_rate = torch.tensor(
        [1, 0.67, 0.75, 1.5, 1.33], device=ground_truthes.xyxy.device)  # 箱のアスペクト比 1:要素
    largest_dist_rate = 0.2  # パッチと対象物体の間の許容できる最大距離に関してのパラメータ。最大距離は対象物体の最大辺＊このパラメータで算出する
    n_b = 3  # グループごとのパッチ最大数

    # 返り値
    bp_boxes = torch.zeros(
        ground_truthes.total_group, n_b, 4, device=gradient_image.device)  # 選択したパッチの座標(xyxy)
    bp_grad_sumes = torch.zeros(
        ground_truthes.total_group, n_b, device=gradient_image.device)  # 選択したパッチ内部の勾配合計

    for group_idx in range(ground_truthes.total_group):

        group_bp_box = torch.zeros(n_b, 4, device=bp_boxes.device)
        group_bp_grad_sum = torch.ones(
            n_b, device=bp_grad_sumes.device)*(-1*float('inf'))

        # グループに属するboxの抽出
        group_boxes_xywh = (
            ground_truthes.xywh[ground_truthes.group_labels == group_idx])
        group_boxes_xyxy = (
            ground_truthes.xyxy[ground_truthes.group_labels == group_idx])

        # ウインドウサイズ決定
        # ウインドウサイズ決定のため、グループ内で最大の面積を求める
        max_size = torch.max(group_boxes_xywh[:, 2]*group_boxes_xywh[:, 3])
        # アスペクト比、サイズからウインドウの高さと幅を求める
        window_w = torch.sqrt(initialize_size_rate*max_size/aspect_rate)
        window_h = window_w*aspect_rate

        # 探索対象領域の決定
        for w, h in zip(window_w, window_h):

            # 探索範囲の決定（探索範囲はオブジェクトからオブジェクトボックスの最大辺の0.2倍の距離の範囲内
            # グループに属する箱の最外殻を決める
            groupbox_outermost = seek.smallest_box_containing(
                group_boxes_xyxy)
            search_offset = seek.get_max_edge(
                group_boxes_xyxy)*largest_dist_rate
            before_clamp_area = transform.box_expand(
                groupbox_outermost, search_offset)
            # 領域を画像サイズに収める
            search_area = transform.box_clamp(
                before_clamp_area, gradient_image.shape[2], gradient_image.shape[3])

            # 画像から探索領域を切り出す
            search_grad_img = transform.image_crop_by_box(
                gradient_image, search_area)
            # lgi_offset=x1y1
            lgi_offset = search_area[:2]

            ex_boxes, ex_grad_sumes = extract_sliding_windows(
                search_grad_img, lgi_offset, int(w.item()), int(h.item()), n_b, group_bp_box, bp_boxes)

            # もしex_boxesの勾配の合計値(=ex_grad_sum)が既存のパッチ領域の勾配の合計値のいずれかより大きかった場合交換
            # bp_boxes,bp_grad_sumesはbp_gd_totalesの昇順に並んでいる
            for ex_idx in range(n_b):
                for bp_idx in range(n_b):
                    if group_bp_grad_sum[bp_idx] <= ex_grad_sumes[ex_idx]:
                        group_bp_box[bp_idx] = ex_boxes[ex_idx]
                        group_bp_grad_sum[bp_idx] = ex_grad_sumes[ex_idx]
                        break

                # group_idx番目のグループに対して適用するパッチ領域の決定
        bp_boxes[group_idx] = group_bp_box
        bp_grad_sumes[group_idx] = group_bp_grad_sum

    return bp_boxes


def expanded_background_patches(bp_boxes, gradient_image):
    stride_rate = 0.02
    image_h, image_w = gradient_image.shape[2:4]
    stride = stride_rate*max(image_h, image_w)

    new_bp_boxes = bp_boxes.clone()

    for i, bp_box in enumerate(bp_boxes):
        max_gradient_sum = torch.tensor([0], device=bp_box.device)

        # 4方向への拡張領域のうち、勾配総和が最大になるものを選ぶ
        for j in range(len(bp_box)):

            # 拡張したとき、差分となる領域のbox
            bp_box_diff = bp_box.clone()

            if j <= 1:
                bp_box_diff[j+2] = bp_box_diff[j]
                bp_box_diff[j] -= stride
            else:
                bp_box_diff[j-2] = bp_box_diff[j]
                bp_box_diff[j] += stride

            if j % 2 == 0:
                bp_box_diff = bp_box_diff.clamp(min=0, max=(image_w-1))
            else:
                bp_box_diff = bp_box_diff.clamp(min=0, max=(image_h-1))

            gradient_sum = transform.image_crop_by_box(
                gradient_image, bp_box_diff).abs().sum()

            if max_gradient_sum <= gradient_sum:
                # 拡張するbp_box
                expand_bp_box = bp_box.clone()
                if j <= 1:
                    expand_bp_box[j] -= stride
                else:
                    expand_bp_box[j] += stride

                if j % 2 == 0:
                    expand_bp_box = expand_bp_box.clamp(min=0, max=(image_w-1))
                else:
                    expand_bp_box = expand_bp_box.clamp(min=0, max=(image_h-1))

                # new_bp_boxesからj番目の要素を除いた配列
                compare_boxes = torch.cat(
                    (new_bp_boxes[:i], new_bp_boxes[i+1:]))
                if (not condition.is_overlap_list(expand_bp_box, compare_boxes)):
                    max_gradient_sum = gradient_sum
                    new_bp_boxes[i] = expand_bp_box

    return new_bp_boxes


def perturbation_in_background_patches(gradient_image, bp_boxes):
    # bp_boxesを元にマスクを作成
    bp_mask = transform.box2mask(gradient_image, bp_boxes)
    return bp_mask*gradient_image


def perturbation_normalization(perturbation_image):
    l2_norm_lambda = 30
    return (l2_norm_lambda/(torch.pow(torch.linalg.norm(perturbation_image), 2))) * perturbation_image


def update_i_with_pixel_clipping(image, perturbated_image):
    return torch.clamp(image-perturbated_image, 0, 255)

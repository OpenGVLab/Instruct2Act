import cv2
import os
from segment_anything import (
    # for automatic mask generation
    build_sam,
    SamAutomaticMaskGenerator,
    build_sam_vit_b,
    build_sam_vit_l,
    build_sam_vit_h,
    # for mask generation with user input like click
    sam_model_registry,
    SamPredictor,
)
from PIL import Image, ImageDraw
import clip
import torch
import copy
import numpy as np

# used for storing the click location
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
click_point_x, click_point_y = 0, 0
click_points = []
cid, fig = None, None

import open_clip

from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading SAM...")
# mask_generator = SamAutomaticMaskGenerator(
#     build_sam(checkpoint="/data/s2/sam_ckpt/sam_vit_h_4b8939.pth", device=device)
# )

sam_pred_with_click = False
sam_path = "/data/s2/sam_ckpt/"
sam_model = ["sam_vit_b_01ec64.pth", "sam_vit_l_0b3195.pth", "sam_vit_h_4b8939.pth"]
build_sam_func = [build_sam_vit_b, build_sam_vit_l, build_sam_vit_h]
sam_idx = 2  # default to use the sam_vit_h
if not sam_pred_with_click:
    mask_generator = SamAutomaticMaskGenerator(
        build_sam_func[sam_idx](
            checkpoint=os.path.join(sam_path, sam_model[sam_idx]), device=device
        )
    )
else:
    sam = sam_model_registry["default"](
        checkpoint=os.path.join(sam_path, sam_model[sam_idx])
    )
    sam.to(device=device)
    mask_generator = SamPredictor(sam)

engine = "openclip"  # "openclip" or "clip"
if engine == "clip":
    print("Loading CLIP...")
    model, preprocess = clip.load("ViT-L/14", device=device)
elif engine == "openclip":
    print("Loading OpenCLIP CLIP...")
    # add offline model for OpenCLIP
    # model, _, preprocess = open_clip.create_model_and_transforms(
    #     "ViT-H-14", device=device, pretrained="laion2b_s32b_b79k"
    # )
    open_clip_path = "/data/open_clip/"
    model_cards = {
        "ViT-B-16": "ViT-B-16_openai.pt",
        "ViT-B-32": "ViT-B-32_openai.pt",
        "ViT-L-14": "ViT-L-14_laion2b_s32b_b82k.pt",
        "ViT-H-14": "open_clip_pytorch_model.bin",
    }
    models = list(model_cards.keys())
    # add offline model for OpenCLIP
    # model, _, preprocess = open_clip.create_model_and_transforms(
    #     "ViT-H-14", device=device, pretrained="laion2b_s32b_b79k"
    # )
    # model, _, preprocess = open_clip.create_model_and_transforms(
    #     "ViT-H-14", device=device, pretrained="/data/open_clip/open_clip_pytorch_model.bin"
    # )
    # tokenizer = open_clip.get_tokenizer("ViT-H-14")
    clip_index = 3  # default to use the VIT-H-14
    model, _, preprocess = open_clip.create_model_and_transforms(
        models[clip_index],
        device=device,
        pretrained=os.path.join(open_clip_path, model_cards[models[clip_index]]),
    )
    tokenizer = open_clip.get_tokenizer("/data/openclip_tokenizer", direct_load=True)


def convery_yaw_to_quaternion(yaw, degrees=True):
    r = R.from_euler("z", -yaw, degrees=degrees)
    return r.as_quat()


def get_indices_of_values_above_threshold(values, threshold):
    filter_values = {i: v for i, v in enumerate(values) if v > threshold}
    sorted_ids = sorted(filter_values, key=filter_values.get, reverse=True)
    return sorted_ids


@torch.no_grad()
def retriev_openclip(elements: list[Image.Image], search_text: str) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    txt = tokenizer(search_text).to(device)
    stacked_images = torch.stack(preprocessed_images)
    img_features = model.encode_image(stacked_images)
    text_features = model.encode_text(txt)
    img_features /= img_features.norm(dim=-1, keepdim=True)  # imgs * 1024
    text_features /= text_features.norm(dim=-1, keepdim=True)  # 1 * 1024
    probs = 100.0 * img_features @ text_features.T
    return probs[:, 0].softmax(dim=0)


@torch.no_grad()
def retriev_clip(elements: list[Image.Image], search_text: str) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100.0 * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)


def retriev_with_text(elements: list[Image.Image], search_text: str) -> int:
    if engine == "openai":
        return retriev_clip(elements, search_text)
    elif engine == "openclip":
        return retriev_openclip(elements, search_text)
    else:
        raise Exception("Engine not supported")


@torch.no_grad()
def retriev_with_template_image(
    elements: list[Image.Image], search_template_image
) -> int:
    preprocessed_search_template_image = (
        preprocess(search_template_image).unsqueeze(0).to(device)
    )
    search_template_image_features = model.encode_image(
        preprocessed_search_template_image
    )
    search_template_image_features /= search_template_image_features.norm(
        dim=-1, keepdim=True
    )  # 1 * 1024

    preprocessed_images = [preprocess(image).to(device) for image in elements]
    stacked_images = torch.stack(preprocessed_images)
    img_features = model.encode_image(stacked_images)
    img_features /= img_features.norm(dim=-1, keepdim=True)  # imgs * 1024

    probs = 100.0 * img_features @ search_template_image_features.T
    return probs[:, 0].softmax(dim=0)


@torch.no_grad()
def get_img_features(imgs: list[Image.Image]):
    preprocessed_imgs = [preprocess(img).to(device) for img in imgs]
    stacked_imgs = torch.stack(preprocessed_imgs)
    img_features = model.encode_image(stacked_imgs)
    img_features /= img_features.norm(dim=-1, keepdim=True)
    return img_features


@torch.no_grad()
def img_sets_similarity(targets: list[Image.Image], templates: list[Image.Image]):
    targets_features = get_img_features(targets)
    templates_features = get_img_features(templates)
    similarity = targets_features @ templates_features.T

    rotated_targets = [target.rotate(90) for target in targets]
    rotated_targets_features = get_img_features(rotated_targets)
    rotated_similarity = rotated_targets_features @ templates_features.T

    return (similarity + rotated_similarity) / 2


def get_objs_match(target_objs: list[Image.Image], obs_objs: list[Image.Image]):
    sim = img_sets_similarity(target_objs, obs_objs)
    cost = 1 - sim.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def bbox_to_center(bbox):
    return (bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)


def pixel_coords_to_action_coords(pixel_coords, pxiel_size=0.0078125):
    x = (pixel_coords[1] - 0.6) / 128 * 0.5 + 0.25
    y = (pixel_coords[0] - 128) / 128 * 0.5
    return x, y


def pixel2action_dict(
    pixel_coords_src,
    pixel_coords_target,
    yaw_angle=None,
    degrees=True,
    pxiel_size=0.0075,
):
    x0, y0 = pixel_coords_to_action_coords(pixel_coords_src, pxiel_size)
    x1, y1 = pixel_coords_to_action_coords(pixel_coords_target, pxiel_size)

    if yaw_angle is not None:
        qua = torch.from_numpy(
            convery_yaw_to_quaternion(yaw_angle, degrees=degrees)
        ).to(torch.float32)
    else:
        qua = torch.tensor([0.0, 0.0, 0.0, 0.9999])

    action = {
        "pose0_position": torch.tensor([x0, y0], dtype=torch.float32),
        "pose1_position": torch.tensor([x1, y1], dtype=torch.float32),
        "pose0_rotation": torch.tensor([0.0, 0.0, 0.0, 0.9999]),
        "pose1_rotation": qua,
    }
    return action


def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


def clamp_action(actions, action_bound):
    x_min = action_bound["low"][0]
    x_max = action_bound["high"][0]
    y_min = action_bound["low"][1]
    y_max = action_bound["high"][1]
    action = copy.deepcopy(actions)
    action["pose0_position"][0] = np.clip(actions["pose0_position"][0], x_min, x_max)
    action["pose0_position"][1] = np.clip(actions["pose0_position"][1], y_min, y_max)
    action["pose1_position"][0] = np.clip(actions["pose1_position"][0], x_min, x_max)
    action["pose1_position"][1] = np.clip(actions["pose1_position"][1], y_min, y_max)

    for ele in range(4):
        action["pose1_rotation"][ele] = np.clip(actions["pose1_rotation"][ele], -1, 1)
        action["pose0_rotation"][ele] = np.clip(actions["pose0_rotation"][ele], -1, 1)

    return action


def list_remove_element(list_, **kwargs):
    for key in kwargs:
        if "pre_obj" in key:
            try:
                list_.remove(kwargs[key])
            except:
                pass
    return list_


def remove_boundary(image, boundary_length=4):
    image[0 : int(boundary_length / 2), :, :] = 47
    image[-boundary_length:, :, :] = 47
    image[:, 0:boundary_length, :] = 47
    image[:, -boundary_length:, :] = 47
    return image


def nms(bboxes, scores, iou_thresh):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = x1 + bboxes[:, 2]
    y2 = y1 + bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    result = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        result.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]
    return result


def mask_preprocess(MASKS):
    MASKS_filtered = []
    for MASK in MASKS:
        if MASK["bbox"][2] < 10 or MASK["bbox"][3] < 10:
            continue
        if MASK["bbox"][2] > 100 or MASK["bbox"][3] > 100:
            continue
        if MASK["area"] < 100:
            continue

        mask = MASK["segmentation"]
        mask = mask.astype("uint8") * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        if np.count_nonzero(mask) < 50:
            continue  # too small, ignore to avoid empty operation
        MASK["area"] = np.count_nonzero(mask)
        ys, xs = np.nonzero(mask)
        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)
        MASK["bbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
        MASK["segmentation"] = mask.astype("bool")
        MASKS_filtered.append(MASK)

    bboxs = np.asarray([MASK["bbox"] for MASK in MASKS_filtered])
    areas = np.asarray([MASK["area"] for MASK in MASKS_filtered])

    result = nms(bboxs, areas, 0.3)
    MASKS_filtered = [MASKS_filtered[i] for i in result]
    return MASKS_filtered


def image_preprocess(image):
    # shadow remove to avoid ghost mask
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    gray = cv2.inRange(gray_img, 47, 150)

    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    image = cv2.bitwise_and(image, image, mask=gray)

    empty = np.ones(image.shape, dtype=np.uint8) * 47  # 47 is the background color
    background = cv2.bitwise_and(empty, empty, mask=cv2.bitwise_not(gray))
    image = cv2.add(image, background)
    return image

def unified_mask_representation(masks):
    """
        input: masks: [N, H, W], numpy.ndarray
        output: masks: list(dict(segmentation, bbox, area)) -> bbox XYWH
    """
    MASKS = [] 
    for mask in masks:
        MASK = {}
        MASK["segmentation"] = mask.astype("bool")
        MASK["area"] = np.count_nonzero(mask)
        ys, xs = np.nonzero(mask)
        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)
        MASK["bbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
        MASKS.append(MASK)
    return MASKS

def get_click_point(event):
    global click_points
    global cid, fig
    if event.button is MouseButton.LEFT:
        point_x = int(event.xdata)
        point_y = int(event.ydata)
        click_points.append([point_x, point_y])
    elif event.button is MouseButton.RIGHT:
        # print('disconnecting callback')
        fig.canvas.mpl_disconnect(cid)

def SAM(image, with_click=sam_pred_with_click, image_preprocess_flag=True, mask_preprocess_flag=True):
    image = remove_boundary(image)
    if image_preprocess_flag:
        image = image_preprocess(image)
    assert (
        with_click == sam_pred_with_click
    ), "the initialzaition of sam is not consistent with the current setting"

    if with_click:
        # use cursor click to guide the SAM
        mask_generator.set_image(image)
        global cid, fig, click_points
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.imshow(image)
        cid = fig.canvas.mpl_connect("button_press_event", get_click_point)
        plt.show()
        plt.waitforbuttonpress()
        ax.clear()
        plt.close('all') 

        masks_all_click = []
        for point in click_points:
            input_point = np.array([[point[0], point[1]]])
            input_label = np.array([1])
            masks, _, _ = mask_generator.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            masks_all_click.extend(masks)
        MASKS = unified_mask_representation(masks_all_click)
    else:
        MASKS = mask_generator.generate(image)
    if mask_preprocess_flag:
        MASKS = mask_preprocess(MASKS)
    return MASKS


def GetObsImage(obs):
    """Get the current image to start the system.
    Examples:
        image = GetObsImage()
    """
    return np.transpose(obs["rgb"]["top"], (1, 2, 0))


def ImageCrop(image, masks):
    image = Image.fromarray(image)
    cropped_boxes = []
    used_masks = []
    for mask in masks:
        cropped_boxes.append(
            segment_image(image, mask["segmentation"]).crop(
                convert_box_xywh_to_xyxy(mask["bbox"])
            )
        )
        used_masks.append(mask)
    return cropped_boxes, used_masks


def CLIPRetrieval(objs, query, **kwargs):
    if isinstance(query, str):
        scores = retriev_with_text(objs, query)
    else:
        scores_1 = retriev_with_template_image(objs, query)
        scores_2 = retriev_with_template_image(objs, query.rotate(90))
        scores = (scores_1 + scores_2) / 2

    obj_idx = get_indices_of_values_above_threshold(scores, 0.1)
    if len(obj_idx) > 1:
        list_remove_element(obj_idx, **kwargs)
    return obj_idx[0]


def Pixel2Loc(obj, masks):
    return bbox_to_center(masks[obj]["bbox"])


def PickPlace(pick, place, bounds, yaw_angle=None, degrees=True):
    action = pixel2action_dict(pick, place, yaw_angle, degrees)
    action = clamp_action(action, bounds)
    action = {k: np.asarray(v) for k, v in action.items()}
    if isinstance(action, tuple):
        return action[0]
    return action


def DistractorActions(MASKS_OBS, col_ind, bounds):
    distracted_id = [i for i in range(len(MASKS_OBS)) if i not in col_ind]
    DISTRACTOR_ACTION = []
    for i in range(len(distracted_id)):
        LOC_DIS = Pixel2Loc(i, MASKS_OBS)
        PLACE = [10 * (i + 1), 10 * (i + 1)]
        DISTRACTOR_ACTION.append(PickPlace(LOC_DIS, PLACE, bounds=bounds))
    return DISTRACTOR_ACTION


def RearrangeActions(place_masks, pick_masks, place_ind, pick_ind, bounds):
    REARRANGE_ACTION = []
    for i in range(len(place_ind)):
        LOC_place = Pixel2Loc(place_ind[i], place_masks)
        LOC_pick = Pixel2Loc(pick_ind[i], pick_masks)
        REARRANGE_ACTION.append(PickPlace(LOC_pick, LOC_place, bounds=bounds))
    return REARRANGE_ACTION

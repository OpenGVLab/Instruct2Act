from einops import rearrange
import numpy as np
from PIL import Image

PLACEHOLDER_TOKENS = [
    "{base_obj}",
    "{base_obj_1}",
    "{base_obj_2}",
    "{dragged_obj}",
    "{dragged_obj_1}",
    "{dragged_obj_2}",
    "{dragged_obj_3}",
    "{dragged_obj_4}",
    "{dragged_obj_5}",
    "{swept_obj}",
    "{bounds}",
    "{constraint}",
    "{scene}",
    "{demo_blicker_obj_1}",
    "{demo_less_blicker_obj_1}",
    "{demo_blicker_obj_2}",
    "{demo_less_blicker_obj_2}",
    "{demo_blicker_obj_3}",
    "{demo_less_blicker_obj_3}",
    "{start_scene}",
    "{end_scene}",
    "{before_twist_1}",
    "{after_twist_1}",
    "{before_twist_2}",
    "{after_twist_2}",
    "{before_twist_3}",
    "{after_twist_3}",
    "{frame_0}",
    "{frame_1}",
    "{frame_2}",
    "{frame_3}",
    "{frame_4}",
    "{frame_5}",
    "{frame_6}",
    "{ring}",
    "{hanoi_stand}",
    "{start_scene_1}",
    "{end_scene_1}",
    "{start_scene_2}",
    "{end_scene_2}",
    "{start_scene_3}",
    "{end_scene_3}",
]


def prepare_prompt(
    prompt: str, prompt_assets: dict, single_model: bool, task: str = "rotate"
):
    if single_model:
        return prepare_prompt_pure_language(prompt, prompt_assets, task)
    else:
        return prepare_prompt_multi_modal(prompt, prompt_assets, task)

def prepare_prompt_pure_language(
    prompt: str, prompt_assets: dict, task: str = "rotate"
):
    words = prompt.split(" ")

    prompt_words_dots = []
    for word in words:
        if "." in word:
            word = word.replace(".", "")
            prompt_words_dots.append(word)
            prompt_words_dots.append(".")
        else:
            prompt_words_dots.append(word)
    prompt_words = prompt_words_dots

    filled_prompt = []
    obj_counter = 1
    cfg = {}
    for word in prompt_words:
        if word not in PLACEHOLDER_TOKENS:
            assert "{" not in word and "}" not in word
            filled_prompt.append(word)
            if task == "rotate":
                if word.isdigit():
                    cfg["degrees"] = int(word)
        else:
            assert word.startswith("{") and word.endswith("}")
            asset_name = word[1:-1]
            assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
            asset = prompt_assets[asset_name]
            placeholder_type = asset["placeholder_type"]
            if placeholder_type == "scene":
                cfg["query"] = "scene"
                filled_prompt.append("scene")
                continue
            obj_info = asset["segm"]["obj_info"]
            obj_desription = obj_info["obj_color"] + " " + obj_info["obj_name"]
            filled_prompt.append(obj_desription)
            if task == "rotate":
                cfg["query"] = " 'the " + str(obj_desription) + "' "
            elif task == "visual_manipulation" or task == "pick_in_order_then_restore":
                query_name = "query_" + str(obj_counter)
                cfg[query_name] = " 'the " + str(obj_desription) + "' "
                obj_counter += 1
                
    if task == "scene_understanding":
        query_name = "query_" + str(obj_counter)
        cfg[query_name] =  " 'the " + prompt.split("Put the")[-1].split("in")[0].strip() + "' "
        obj_counter += 1
        query_name = "query_" + str(obj_counter)
        cfg[query_name] = " 'the" + prompt.split("into the")[-1].strip() + "' "

    full_task = ""
    for word in filled_prompt:
        if word != ".":
            full_task += word + " "
        else:
            full_task += word

    return full_task, None, cfg

def prepare_prompt_multi_modal(prompt: str, prompt_assets: dict, task: str = "rotate"):
    words = prompt.split(" ")

    prompt_words_dots = []
    for word in words:
        if "." in word:
            word = word.replace(".", "")
            prompt_words_dots.append(word)
            prompt_words_dots.append(".")
        else:
            prompt_words_dots.append(word)
    prompt_words = prompt_words_dots

    filled_prompt = []
    templates = {}
    cfg = {}
    obj_counter = 1
    for word in prompt_words:
        if word not in PLACEHOLDER_TOKENS:
            assert "{" not in word and "}" not in word
            filled_prompt.append(word)
            if task == "rotate":
                if word.isdigit():
                    cfg["degrees"] = int(word)
        else:
            assert word.startswith("{") and word.endswith("}")
            asset_name = word[1:-1]
            assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
            asset = prompt_assets[asset_name]
            obj_info = asset["segm"]["obj_info"]
            placeholder_type = asset["placeholder_type"]
            if placeholder_type == "object":
                objects = [obj_info["obj_id"]]
            elif placeholder_type == "scene":
                objects = [each_info["obj_id"] for each_info in obj_info]
                scene_img = rearrange(asset["rgb"]["top"], "c h w -> h w c")
            rgb_this_view = asset["rgb"]["top"]
            segm_this_view = asset["segm"]["top"]

            for obj_id in objects:
                if placeholder_type == "scene":
                    templates[asset_name] = scene_img
                    filled_prompt.append(asset_name)
                    break # no need to crop the scene
                ys, xs = np.nonzero(segm_this_view == obj_id)
                if len(xs) < 2 or len(ys) < 2:  # filter out small objects
                    continue
                xmin, xmax = np.min(xs), np.max(xs)
                ymin, ymax = np.min(ys), np.max(ys)
                x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]

                cropped_img = rearrange(cropped_img, "c h w -> h w c")
                cropped_img = Image.fromarray(np.asarray(cropped_img))

                # for object type, one object per word_token
                templates[asset_name] = cropped_img
                filled_prompt.append(asset_name)

            if task == "rotate":
                cfg["query"] = "templates['{}']".format(asset_name) 
            elif task == "visual_manipulation" or task == "pick_in_order_then_restore":
                query_name = "query_" + str(obj_counter)
                cfg[query_name] = "templates['{}']".format(asset_name)
                obj_counter += 1
            elif task == "rearrange" or task == "rearrange_then_restore":
                cfg["query"] = "templates['scene']"
    
    if task == "scene_understanding":
        query_name = "query_" + str(obj_counter)
        cfg[query_name] = prompt.split("Put")[-1].split("in")[0].strip()
        obj_counter += 1
        query_name = "query_" + str(obj_counter)
        cfg[query_name] = prompt.split("into")[-1].strip()
            

    full_task = ""
    for word in filled_prompt:
        if word != ".":
            full_task += word + " "
        else:
            full_task += word

    return full_task, templates, cfg


def already_executed(all_infos, task_id, task, failure_reaction=True):
    for info in all_infos:
        if info["task_id"] == task_id and info["task"] == task:
            if failure_reaction and not info["success"]:
                return False
            return True
    return False

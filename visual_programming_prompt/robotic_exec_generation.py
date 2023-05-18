import os
import openai

# Set the proxy based on your environment if needed
# os.environ["http_proxy"] = "http://127.0.0.1:58591"
# os.environ["https_proxy"] = "http://127.0.0.1:58591"

# load different types of prompt
from visual_programming_prompt.object_query_prompt import PROMPT as object_query_prompt
from visual_programming_prompt.visual_programm_prompt import (
    PROMPT as visual_programm_prompt,
)

full_prompt_file = "visual_programming_prompt/full_prompt.ini"
with open(full_prompt_file, "r") as f:
    full_prompt_i2a = f.readlines()
full_prompt_i2a = "".join(full_prompt_i2a)

openai.api_key = "YOUR API KEY"
# this api key is only for demo, please use your own api key

prompt_style = "VISPROG"
prompts = {
    "instruct2act": full_prompt_i2a,
    "VISPROG": visual_programm_prompt,
}
# You can choose different prompt here
# visual_programm_prompt: VISPROG style prompt
# full_prompt_i2a: VISPROG + ViperGPT style prompt
prompt_base = prompts[prompt_style]

folder = "visual_programming_prompt/output/" + prompt_style + "/"
if not os.path.exists(folder):
    os.makedirs(folder)

def turn_list_to_string(all_result):
    if not isinstance(all_result, list):
        return all_result
    all_in_one_str = ""
    for r in all_result:
        all_in_one_str = all_in_one_str + r + "\n"
    if all_in_one_str.endswith("\n"):
        all_in_one_str = all_in_one_str[:-1]
    if all_in_one_str.endswith("."):
        all_in_one_str = all_in_one_str[:-1]
    return all_in_one_str

def result_preprocess(results):
    """
        Only used for the result with full_prompt_i2a
    """
    codes = []
    if isinstance(results, list):
        results = results[0]
    for code in results.splitlines():
        if "main" in code or len(code) < 2:
            continue
        if  "(" not in code and ")" not in code:
            continue
        if code.startswith("#"):
            continue
        codes.append(code.strip())
    
    # codes = turn_list_to_string(codes)
    return codes

def insert_task_into_prompt(task, prompt_base, insert_index="INSERT TASK HERE"):
    full_prompt = prompt_base.replace(insert_index, task)
    return full_prompt


def exec_steps(instruction_task, task_id=None):
    save_file = instruction_task.replace(" ", "_").replace(".", "") + ".txt"
    if task_id is not None:
        save_file = str(task_id) + "_" + save_file
    save_file = os.path.join(folder, save_file)

    reuse = True
    if os.path.exists(save_file) and reuse:
        # not needed exactly, just because the cost and time of openai api
        print(f"The code file already exists, direct load {save_file}")
        with open(save_file, "r") as tfile:
            all_result = tfile.readlines()
            all_in_one_str = turn_list_to_string(all_result)
            return all_in_one_str
    else:
        trials = 0
        # the response could be imcomplete, so we need to run it multiple times
        while True and trials < 5:
            curr_prompt = insert_task_into_prompt(instruction_task, prompt_base)
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=curr_prompt,
                temperature=0.99,
                max_tokens=512,
                n=1,
                stop=".",
            )

            all_result = []
            for r in range(len(response["choices"])):
                result = response["choices"][r]["text"]
                if prompt_style == "instruct2act":
                    all_result.append(result)
                else:
                    all_result.append(result.replace("\n\n", ""))
                    all_result = all_result[0]
            
            all_result = result_preprocess(all_result)
            trials += 1
            if len(all_result) > 3: # the result should be at least 5 lines code
                break
            else:
                print("The result is too short, retry...")
                print(all_result)
                continue

        print("Save result to: ", save_file)
        with open(save_file, "w") as tfile:
            tfile.write("\n".join(all_result))
        all_result = turn_list_to_string(all_result)
        return all_result


def exec_steps_faked(
    task="rotate", query=None, degrees=None, query_1=None, query_2=None, query_3=None
):
    # this function is used when offline mode, to generate the code for the task
    if task == "rotate":
        codes =  rotate_exec_steps_faked(query, degrees)
    elif task == "visual_manipulation":
        codes =  visual_manipulation_steps_faked(query_2, query_1)
    elif task == "pick_in_order_then_restore":
        codes = pick_in_order_then_restore_faked(query_2, query_3,  query_1)
    elif task == "rearrange":
        codes = rearrange_faked()
    elif task == "rearrange_then_restore":
        codes = rearrange_then_restore_faked()
    elif task == "scene_understanding":
        codes = scene_understanding_faked(query_1, query_2)

    all_in_one_str = ""
    for r in codes:
        all_in_one_str = all_in_one_str + r + "\n"
    if all_in_one_str.endswith("\n"):
        all_in_one_str = all_in_one_str[:-1]
    if all_in_one_str.endswith("."):
        all_in_one_str = all_in_one_str[:-1]
    return all_in_one_str


def rotate_exec_steps_faked(query, degrees):
    # degrees is the rotation degress
    # query is the object query, obj description for single; obj reference for multiple modal, like templates["dragged"]
    codes = [
        "MASKS=SAM(image=IMAGE)",
        "OBJS, MASKS=ImageCrop(image=IMAGE, masks=MASKS)",
        "OBJ0=CLIPRetrieval(objs=OBJS, query={})".format(query),
        "LOC0=Pixel2Loc(obj=OBJ0, masks=MASKS)",
        "PickPlace(pick=LOC0, place=LOC0, bounds=BOUNDS, yaw_angle={}, degrees=True)".format(
            degrees
        ),
    ]
    return codes


def visual_manipulation_steps_faked(query_1, query_2):
    codes = [
        "MASKS=SAM(image=IMAGE)",
        "OBJS, MASKS=ImageCrop(image=IMAGE, masks=MASKS)",
        "OBJ0=CLIPRetrieval(objs=OBJS, query={})".format(query_1),
        "LOC0=Pixel2Loc(obj=OBJ0, masks=MASKS)",
        "OBJ1=CLIPRetrieval(objs=OBJS, query={})".format(query_2),
        "LOC1=Pixel2Loc(obj=OBJ1, masks=MASKS)",
        "PickPlace(pick=LOC1, place=LOC0, bounds=BOUNDS)",
    ]
    return codes

def pick_in_order_then_restore_faked(query_1, query_2, query_3):
    # query_1 and query_2 are the containers, query_3 is the object
    codes = [
        "MASKS=SAM(image=IMAGE)", 
        "OBJS, MASKS=ImageCrop(image=IMAGE, masks=MASKS)",
        "OBJ0=CLIPRetrieval(objs=OBJS, query={})".format(query_1),
        "LOC0=Pixel2Loc(obj=OBJ0, masks=MASKS)",
        "OBJ1=CLIPRetrieval(objs=OBJS, query={}, pre_obj1=OBJ0)".format(query_2),
        "LOC1=Pixel2Loc(obj=OBJ1, masks=MASKS)",
        "OBJ2=CLIPRetrieval(objs=OBJS, query={}, pre_obj1=OBJ0, pre_obj2=OBJ1)".format(query_3),
        "LOC2=Pixel2Loc(obj=OBJ2, masks=MASKS)",
        "PickPlace(pick=LOC2, place=LOC0, bounds=BOUNDS)",
        "PickPlace(pick=LOC0, place=LOC1, bounds=BOUNDS)",
        "PickPlace(pick=LOC1, place=LOC2, bounds=BOUNDS)",
    ]
    return codes

def rearrange_faked():
    codes = [
        "MASKS_OBS=SAM(image=IMAGE)", 
        "OBJS_OBS, MASKS_OBS=ImageCrop(image=IMAGE, masks=MASKS_OBS)", 
        "MASKS_GOAL=SAM(image=templates['scene'])",
        "OBJS_GOAL, MASKS_GOAL=ImageCrop(image=templates['scene'], masks=MASKS_GOAL)",
        "ROW, COL =get_objs_match(OBJS_GOAL, OBJS_OBS)",
        "DistractorActions=DistractorActions(MASKS_OBS, COL, bounds=BOUNDS)",
        "RerrangeActions=RerrangeActions(place_masks=MASKS_GOAL, pick_masks=MASKS_OBS, place_ind=ROW, pick_ind=COL, bounds=BOUNDS)",
    ]
    return codes

def rearrange_then_restore_faked():
    codes = [
        "MASKS_OBS=SAM(image=IMAGE)", 
        "OBJS_OBS, MASKS_OBS=ImageCrop(image=IMAGE, masks=MASKS_OBS)", 
        "MASKS_GOAL=SAM(image=templates['scene'])",
        "OBJS_GOAL, MASKS_GOAL=ImageCrop(image=templates['scene'], masks=MASKS_GOAL)",
        "ROW, COL =get_objs_match(OBJS_GOAL, OBJS_OBS)",
        "DistractorActions=DistractorActions(MASKS_OBS, COL, bounds=BOUNDS)",
        "RerrangeActions=RerrangeActions(place_masks=MASKS_GOAL, pick_masks=MASKS_OBS, place_ind=ROW, pick_ind=COL, bounds=BOUNDS)",
        "RerrangeActions=RerrangeActions(pick_masks=MASKS_GOAL, place_masks=MASKS_OBS, pick_ind=ROW, place_ind=COL, bounds=BOUNDS)",

    ]
    return codes

def scene_understanding_faked(query_1, query_2):
    # query_1 is the object, query_2 is the container
    codes = [
        "MASKS_OBS=SAM(image=IMAGE)",
        "OBJS_OBS, MASKS_OBS=ImageCrop(image=IMAGE, masks=MASKS_OBS)", 
        "MASKS_GOAL=SAM(image=templates['scene'])",
        "OBJS_GOAL, MASKS_GOAL=ImageCrop(image=templates['scene'], masks=MASKS_GOAL)",
        "GOAL=CLIPRetrieval(OBJS_GOAL, query='{}')".format(query_1),
        "TARGET=CLIPRetrieval(OBJS_OBS, query=OBJS_GOAL[GOAL])",
        "LOC0=Pixel2Loc(TARGET, MASKS_OBS)",
        "OBJ1=CLIPRetrieval(OBJS_OBS, query='{}', pre_obj=TARGET)".format(query_2),
        "LOC1=Pixel2Loc(OBJ1, MASKS_OBS)",
        "PickPlace(pick=LOC0, place=LOC1, bounds=BOUNDS)",
    ]

    return codes
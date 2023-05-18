import os
import time
import json
import pickle
import logging

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from easydict import EasyDict

from vima_bench import *
from gym.wrappers import TimeLimit as _TimeLimit
from gym import Wrapper
import argparse

from engine_robotic import *
from visual_programming_prompt.robotic_exec_generation import *
from utils.data_prepare import *
from utils.common_utils import create_logger


def exception_handler(exception, logger, **kwargs):
    logger.error("Exception: {}".format(exception))
    task_info = {
        "task_id": kwargs["task_id"],
        "task": kwargs["whole_task"],
        "exec": kwargs["exec_codes"],
        "skip": True,
        "success": False,
        "exception": str(exception),
    }
    return task_info


class ResetFaultToleranceWrapper(Wrapper):
    max_retries = 10

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        for _ in range(self.max_retries):
            try:
                return self.env.reset()
            except:
                current_seed = self.env.unwrapped.task.seed
                self.env.global_seed = current_seed + 1
        raise RuntimeError(
            "Failed to reset environment after {} retries".format(self.max_retries)
        )


class TimeLimitWrapper(_TimeLimit):
    def __init__(self, env, bonus_steps: int = 0):
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)


@torch.no_grad()
def main(cfg, logger):
    logger.info("cfg: {}".format(cfg))
    debug_flag = cfg.debug_flag
    assert cfg.partition in ALL_PARTITIONS
    assert cfg.task in PARTITION_TO_SPECS["test"][cfg.partition]

    seed = cfg.seed
    env = TimeLimitWrapper(
        ResetFaultToleranceWrapper(
            make(
                cfg.task,
                modalities=["segm", "rgb"],
                task_kwargs=PARTITION_TO_SPECS["test"][cfg.partition][cfg.task],
                seed=seed,
                render_prompt=False,
                display_debug_window=debug_flag,
                hide_arm_rgb=cfg.hide_arm,
            )
        ),
        bonus_steps=2,
    )
    single_model_flag = True if cfg.prompt_modal == "single" else False
    result_folder = (
        cfg.save_dir + "/" + cfg.partition + "/" + cfg.task + "/" + cfg.prompt_modal
    )
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    eval_res_name = cfg.partition + "_" + cfg.task + ".json"
    eval_result_file_path = os.path.join(result_folder, eval_res_name)

    task_id = 0
    all_infos = []
    if cfg.reuse and os.path.exists(eval_result_file_path):
        with open(eval_result_file_path, "r") as f:
            all_infos = json.load(f)

    while True:
        env.global_seed = seed

        obs = env.reset()
        env.render()
        meta_info = env.meta_info
        prompt = env.prompt
        prompt_assets = env.prompt_assets

        whole_task, templates, task_setting = prepare_prompt(
            prompt, prompt_assets, single_model=single_model_flag, task=cfg.task
        )
        task_id += 1
        logger.info(f"==================Task {task_id}=========================")
        logger.info(whole_task)
        if not single_model_flag:
            # get full task description for debug
            whole_task_debug, _, _ = prepare_prompt(
                prompt, prompt_assets, single_model=True, task=cfg.task
            )
            logger.info(f"The initial intention {whole_task_debug}")

        if cfg.reuse and already_executed(all_infos, task_id, whole_task):
            logger.info("Already executed, skip")
            continue
        # # Code block for saving demo
        # a = input("Press s to save, c to continue, q to quit:")
        # if a == "q":
        #     break
        # elif a == "s":
        #     # save multi-modal data with the task description
        #     for ele in templates:
        #         img = np.asarray(templates[ele])
        #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #         img = cv2.resize(img, (1024, 1024))
        #         img_name = str(task_id) + whole_task_debug + "_" + ele + ".png"
        #         img_path = os.path.join(result_folder, "single_obj")
        #         if not os.path.exists(img_path):
        #             os.makedirs(img_path)
        #         cv2.imwrite(os.path.join(img_path, img_name), img)
        # elif a == "c":
        #     continue # skip
        BOUNDS = meta_info["action_bounds"]
    
        exec_codes = exec_steps(whole_task)    
        if task_id >= 150:
            break
        done = False
        elapsed_steps = 0
        ACTIONS = []
        ACTION = None
        IMAGE_INIT_top = np.transpose(obs["rgb"]["top"], (1, 2, 0))
        IMAGE_INIT_front = np.transpose(obs["rgb"]["front"], (1, 2, 0))
        while True:
            top_img = np.transpose(obs["rgb"]["top"], (1, 2, 0))
            IMAGE = top_img
            info = None
            for code in exec_codes.splitlines():
                try:
                    if "EXE".lower() in code.lower() or len(code) < 4:
                        # the exe is done by the simulator
                        continue
                    elif "PickPlace".lower() in code.lower():
                        code = "PickPlace" + code.split("PickPlace")[-1]
                        ACTION = eval(code)
                        ACTIONS.append(ACTION)
                    elif "Actions".lower() in code.lower():
                        ACTIONS_ = eval(code.split("Actions=")[-1])
                        ACTIONS.extend(ACTIONS_)
                    else:
                        exec(code)
                except Exception as e:
                    logger.info(f"Exception: {e} for {code}")
                    task_info = exception_handler(
                        e,
                        logger,
                        task_id=task_id,
                        whole_task=whole_task,
                        exec_codes=exec_codes,
                    )
                    all_infos.append(task_info)
                    with open(eval_result_file_path, "w") as f:
                        json.dump(all_infos, f)
                    done = True
                    break

            while len(ACTIONS) > 0 and not done:
                ACTION = ACTIONS.pop(0)

                if isinstance(ACTION, tuple):
                    ACTION = ACTION[0]

                if not isinstance(ACTION, dict):
                    # this uses to skip the task, mainly due to the generated code is not correct
                    task_info = exception_handler(
                        "not a dict",
                        logger,
                        task_id=task_id,
                        whole_task=whole_task,
                        exec_codes=exec_codes,
                    )
                    all_infos.append(task_info)
                    with open(eval_result_file_path, "w") as f:
                        json.dump(all_infos, f)
                    break

                obs, _, done, info = env.step(ACTION)
            elapsed_steps += 1
            if done and info:
                task_info = {
                    "task_id": task_id,
                    "task": whole_task,
                    "exec": exec_codes,
                    "steps": elapsed_steps,
                    "success": info["success"],
                    "failure": info["failure"],
                }
            else:
                task_info = {
                    "task_id": task_id,
                    "task": whole_task,
                    "exec": exec_codes,
                    "steps": elapsed_steps,
                    "success": False,
                    "failure": False,
                }
            logger.info(
                f"task id: {task_info['task_id']} success: {task_info['success']}"
            )
            if cfg.reuse and task_id - 1 < len(all_infos):
                all_infos[task_id - 1] = task_info

            all_infos.append(task_info)
            with open(eval_result_file_path, "w") as f:
                json.dump(all_infos, f)

            if debug_flag or (info and not info["success"]):
                img_path = os.path.join(
                    result_folder, "imgs", f"{task_id}_{whole_task}_top.png"
                )
                if not os.path.exists(os.path.dirname(img_path)):
                    os.makedirs(os.path.dirname(img_path))
                IMAGE_INIT_top = cv2.cvtColor(IMAGE_INIT_top, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, IMAGE_INIT_top)
                if cfg.task == "rearrange":
                    img_path = os.path.join(
                        result_folder, "imgs", f"{task_id}_scene.png"
                    )
                    cv2.imwrite(img_path, templates["scene"])

            break

    success_rate = sum([info["success"] for info in all_infos]) / len(all_infos)
    logger.warning(msg="==================Evaluation Done=========================")
    logger.info(cfg)
    logger.info("Success rate: {}".format(success_rate))
    env.env.close()
    del env
    # time.sleep(5)


if __name__ == "__main__":
    prompt_modal = ["multi"]
    # prompt_modal = ["multi", "single"]
    tasks = [
        "visual_manipulation",
        "rotate",
        "pick_in_order_then_restore",
        "rearrange_then_restore",
        "rearrange",
        "scene_understanding",
    ]
    partitions = [
        "placement_generalization",
        "combinatorial_generalization",
        "novel_object_generalization",
    ]
    save_dir = "output"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    seed = 42
    hide_arm =False # False for demo usage, True for eval usage
    for task in tasks:
        for partition in partitions:
            for modal in prompt_modal:
                eval_cfg = {
                    "partition": partition,
                    "task": task,
                    "device": "cuda:0",
                    "prompt_modal": modal,
                    "reuse": False,
                    "save_dir": save_dir,
                    "debug_flag": True,
                    "hide_arm": hide_arm,
                    "seed": seed,
                }
                logger_file = (
                    save_dir
                    + "/eval_on_seed_{}_hide_arm_{}_{}_{}_{}_modal.log".format(
                        eval_cfg["seed"],
                        eval_cfg["hide_arm"],
                        partition,
                        task,
                        modal,
                    )
                )
                if os.path.exists(path=logger_file):
                    os.remove(logger_file)
                logger = create_logger(logger_file)
                main(EasyDict(eval_cfg), logger)
                del logger

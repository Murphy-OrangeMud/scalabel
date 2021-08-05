from typing import Dict, List

import torch
import os
import logging
from detectron2 import model_zoo
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config.config import get_cfg
import detectron2.data.transforms as T
from detectron2.modeling.meta_arch.build import build_model
import numpy as np
import redis
import json
import torch.multiprocessing as mp
from torch.multiprocessing import Pool

from io import BytesIO
from PIL import Image
import requests


class ModelServerScheduler(object):
    def __init__(self, server_config, model_config, logger):
        self.server_config = server_config
        self.model_config = model_config

        self.redis = redis.Redis(host=server_config["redis_host"], port=server_config["redis_port"])

        self.model_register_channel = "modelRegister"
        self.model_request_channel = "modelRequest_%s_%s"
        self.model_response_channel = "modelResponse_%s_%s"

        self.tasks = {}

        self.threads = {}

        self.logger = logger

    # restore when server restarts, connects to redis channels.
    def restore(self):
        pass

    # save the loaded tasks
    def save(self):
        pass

    def listen(self):
        self.logger.info("Listening...")
        model_register_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        model_register_subscriber.subscribe(**{self.model_register_channel: self.register_handler})
        thread = model_register_subscriber.run_in_thread(sleep_time=0.001)

        self.threads[self.model_register_channel] = thread

    def register_handler(self, register_message):
        self.logger.info("RegisterHandler running...")
        register_message = json.loads(register_message["data"])
        project_name = register_message["projectName"]
        task_id = register_message["taskId"]
        item_list = register_message["items"]

        self.register_task(project_name, task_id, item_list)

        self.logger.info(f"Set up model inference for {project_name}: {task_id}.")

    def request_handler(self, request_message):
        self.logger.info("RequestHandler running...")
        request_message = json.loads(request_message["data"])
        project_name = request_message["projectName"]
        task_id = request_message["taskId"]
        items = request_message["items"]
        item_indices = request_message["itemIndices"]
        action_packet_id = request_message["actionPacketId"]

        # here needs to change 
        model_name = self.tasks[f'{project_name}_{task_id}']["model"]
        image_dict = self.tasks[f'{project_name}_{task_id}']["image_dict"]
        input_data = [image_dict[item["url"]] for item in items]
        # 似乎torchserve只支持部署在本地（服务器），即只支持单机，不支持分布式调度
        results = requests.post(url="http://127.0.0.1:8080/predictions/%s" % model_name, data=input_data)

        pred_boxes: List[List[float]] = []
        for box in results[0]["instances"].pred_boxes:
            box = box.cpu().numpy()
            pred_boxes.append(box.tolist())

        model_response_channel = self.model_response_channel % (project_name, task_id)
        self.redis.publish(model_response_channel, json.dumps([pred_boxes, item_indices, action_packet_id]))

    # 假定每个register的task用的model不是同一个
    def register_task(self, project_name, task_id, item_list):
        self.logger.info("RegisterTask running...")
        model_name, image_dict = self.deploy_model(self.model_config["model_name"], item_list, project_name)

        self.tasks[f'{project_name}_{task_id}'] = {
            "project_name": project_name,
            "task_id": task_id,
            "model": model_name,
            "image_dict": image_dict
        }

        model_request_channel = self.model_request_channel % (project_name, task_id)
        model_request_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        model_request_subscriber.subscribe(**{model_request_channel: self.request_handler})
        thread = model_request_subscriber.run_in_thread(sleep_time=0.001)
        self.threads[model_request_channel] = thread

    def deploy_model(self, model_name, item_list, project_name):
        NUM_WORKERS = 8

        image_dict = {}

        def load_inputs(item_list: List, num_workers: int) -> None:
            def url_to_img(url):
                img_response = requests.get(url)
                return img_response

            urls = [item["urls"]["-1"] for item in item_list]
            if num_workers > 1:
                pool = Pool(num_workers)
                image_list = list(pool.starmap(url_to_img, urls))
            else:
                image_list = [url_to_img(url) for url in urls]

            for url, image in zip(urls, image_list):
                image_dict[url] = image

        load_inputs(item_list, NUM_WORKERS)
        
        model_id = model_name + " " + project_name
        export_path = "model_store"
        handler = "./handlers.py"
        
        torch_archive_cmd = "torch-model-archiver --model-name %s\
                                --export-path %s \
                                --handler %s" \
                                % (model_name, export_path, handler)
        ts_start_cmd = "torchserve --start --ncs --model-store %s --models %s.mar" \
                        % (export_path, model_id)

        os.system(torch_archive_cmd)
        self.logger.info("Model archived.")
        os.system(ts_start_cmd)
        self.logger.info("Torchserve started.")

        return model_id, image_dict

    def close(self):
        for thread_name, thread in self.threads.items():
            thread.stop()


def launch() -> None:
    """Launch processes."""
    log_f = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=log_f)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create scheduler
    server_config = {
        "redis_host": "127.0.0.1",
        "redis_port": 6379
    }
    model_config = {
        "model_name": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    }

    scheduler = ModelServerScheduler(server_config, model_config, logger)
    scheduler.listen()
    logger.info("Model server launched.")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    mp.set_start_method("spawn")
    launch()

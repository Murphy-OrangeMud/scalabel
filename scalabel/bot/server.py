from functools import total_ordering
import ray
from ray import serve
import logging
import numpy as np
import redis
from io import BytesIO
from PIL import Image
import os
import requests
import time
from typing import Dict, List
import json

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool

from detectron2 import model_zoo
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config.config import get_cfg
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer, build_lr_scheduler

ray.init(address="auto", namespace="hello")
serve.start(detached=True)

# Actor
# TODO: 是否先把每个label暂存起来，等找到了足够多的label之后再进行一次训练
@ray.remote
class ModelDriver:
    # gpu_id: int, 0 or 1
    # max_iter: 
    def __init__(self, cfg_path, item_list, num_workers, logger, gpu_id, max_iter):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
        # NOTE: you may customize cfg settings
        # cfg.MODEL.DEVICE="cuda" # use gpu by default
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        cfg.MODEL_DEVICE="cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # you can also give a path to you checkpoint
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

        self.cfg = cfg.clone()
        self.model = build_model(cfg)
        self.optimizer = build_optimizer(cfg, self.model)
        self.training_cfg = {
            "iter": 0,
            "start_iter": 0,
            "max_iter": max_iter
        }
        
        self.checkpointer = DetectionCheckpointer(self.model)
        self.checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.logger = logger

        self.image_dict = {}

        self.load_inputs(item_list, num_workers)

    def load_inputs(self, item_list, num_workers):
        urls = [item["urls"]["-1"] for item in item_list]
        if num_workers > 1:
            pool = Pool(num_workers)
            image_list = list(pool.starmap(ModelDriver.url_to_img,
                                           zip(urls,
                                               [self.aug] * len(urls),
                                               [self.cfg.MODEL.DEVICE] * len(urls)
                                               )))
        else:
            image_list = [ModelDriver.url_to_img(url, self.aug, self.cfg.MODEL.DEVICE) for url in urls]

        for url, image in zip(urls, image_list):
            self.image_dict[url] = image

    def predict(self, items):
        self.model.eval()
        inputs = [self.image_dict[item["url"]] for item in items]
        with torch.no_grad():
            predictions = self.model(inputs)

            pred_boxes: List[List[float]] = []
            for box in predictions[0]["instances"].pred_boxes:
                box = box.cpu().numpy()
                pred_boxes.append(box.tolist())
            return pred_boxes

    def train_step(self, items):
        self.model.train()
        start = time.perf_counter()

        # TODO: to be revised, unknown format
        data = self.process_data(items)
        
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        training_time = time.perf_counter() - start

        ModelDriver.write_metrics(loss_dict, training_time)
        self.optimizer.step()

    @staticmethod
    def write_metrics(loss_dict, training_time, prefix):
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["training_time"] = training_time
        
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            training_time = np.max([x.pop("training_time") for x in all_metrics_dict])
            storage.put_scalar("training_time", training_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration = {storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    @staticmethod
    def url_to_img(url, aug, device):
        img_response = requests.get(url)
        img = np.array(Image.open(BytesIO(img_response.content)))
        height, width = img.shape[:2]
        img = aug.get_transform(img).apply_image(img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        if device == "cuda":
            img = img.pin_memory()
            img = img.cuda(non_blocking=True)
        return {"image": img, "height": height, "width": width}

    def save_model(self):
        pass

    def resume_model(self):
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, True)

@serve.deployment(num_replicas=1)
class ModelServerScheduler(object):
    def __init__(self, server_config, model_config, logger):
        self.server_config = server_config
        self.model_config = model_config

        self.redis = redis.Redis(host=server_config["redis_host"], port=server_config["redis_port"], health_check_interval=3)

        self.model_register_channel = "modelRegister"
        self.model_request_channel = "modelRequest_%s_%s"
        self.model_response_channel = "modelResponse_%s_%s"

        self.tasks = {}

        self.threads = {}

        self.logger = logger
        
        self.listen()

    def restore(self):
        pass

    def save(self):
        pass

    def listen(self):
        model_register_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        model_register_subscriber.subscribe(**{self.model_register_channel: self.register_handler})
        thread = model_register_subscriber.run_in_thread(sleep_time=0.001)

        self.threads[self.model_register_channel] = thread

    def register_handler(self, register_message):
        register_message = json.loads(register_message["data"])
        project_name = register_message["projectName"]
        task_id = register_message["taskId"]
        item_list = register_message["items"]

        self.register_task(project_name, task_id, item_list)

        self.logger.info(f"Set up model inference for {project_name}: {task_id}.")

    def request_handler(self, request_message):
        request_message = json.loads(request_message["data"])
        project_name = request_message["projectName"]
        task_id = request_message["taskId"]
        items = request_message["items"]
        item_indices = request_message["itemIndices"]
        action_packet_id = request_message["actionPacketId"]

        model = self.tasks[f'{project_name}_{task_id}']["model"]

        start_time = time.time()
        results = model.predict.remote(items)
        results = ray.get(results, timeout=300)
        end_time = time.time()
        print("PREDICTION TIME: ", end_time - start_time)

        model_response_channel = self.model_response_channel % (project_name, task_id)
        self.redis.publish(model_response_channel, json.dumps([results, item_indices, action_packet_id]))

    def register_task(self, project_name, task_id, item_list):
        model = self.get_model(self.model_config["model_name"], item_list)
        self.put_model(model)

        self.tasks[f'{project_name}_{task_id}'] = {
            "project_name": project_name,
            "task_id": task_id,
            "model": model,
        }

        model_request_channel = self.model_request_channel % (project_name, task_id)
        model_request_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        model_request_subscriber.subscribe(**{model_request_channel: self.request_handler})
        thread = model_request_subscriber.run_in_thread(sleep_time=0.001)
        self.threads[model_request_channel] = thread

        # model_train_channel = 

    # 在171行，ray会直接把这个actor对象放到一个有gpu的node上并调用构造函数。不需要手动放。需要做的就是在构造函数里指定CUDA_VISIBLE_ENVIRONMENT
    def get_model(self, model_name, item_list):
        NUM_WORKERS = 1
        model = ModelDriver.remote(model_name, item_list, NUM_WORKERS, self.logger, "1")
        return model

    def put_model(self, model):
        pass
    
    def close(self):
        for thread_name, thread in self.threads.items():
            thread.stop()


def launch():
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
    ModelServerScheduler.deploy(server_config, model_config, logger)
    logger.info("Model server launched.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    launch()

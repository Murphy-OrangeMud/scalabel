import torch
import ray
from ray import serve
import logging
from detectron2 import model_zoo
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config.config import get_cfg
import detectron2.data.transforms as T
from detectron2.modeling import build_model
import numpy as np
import redis
from typing import Dict, List
import json
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from io import BytesIO
from PIL import Image
import os

ray.init(address="auto", namespace="hello")
serve.start(detached=True)

# Actor
@ray.remote
class Predictor:
    # gpu_id: int, 0 or 1
    def __init__(self, cfg_path, item_list, num_workers, logger, gpu_id):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
        # NOTE: you may customize cfg settings
        # cfg.MODEL.DEVICE="cuda" # use gpu by default
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # you can also give a path to you checkpoint
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

        self.cfg = cfg.clone()
        self.model = build_model(cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.logger = logger

        self.image_dict = {}

        os.environ["CUDA_VISIBLE_DEVICE"] = gpu_id
        self.load_inputs(item_list, num_workers)

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

    def load_inputs(self, item_list, num_workers):
        urls = [item["urls"]["-1"] for item in item_list]
        if num_workers > 1:
            pool = Pool(num_workers)
            image_list = list(pool.starmap(self.url_to_img,
                                           zip(urls,
                                               [self.aug] * len(urls),
                                               [self.cfg.MODEL.DEVICE] * len(urls)
                                               )))
        else:
            image_list = [self.url_to_img(url, self.aug, self.cfg.MODEL.DEVICE) for url in urls]

        for url, image in zip(urls, image_list):
            self.image_dict[url] = image

    @ray.method
    def predict(self, items):
        inputs = [self.image_dict[item["url"]] for item in items]
        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions

@serve.deployment
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

    def restore(self):
        pass

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
        request_message = json.loads(request_message["data"])
        project_name = request_message["projectName"]
        task_id = request_message["taskId"]
        items = request_message["items"]
        item_indices = request_message["itemIndices"]
        action_packet_id = request_message["actionPacketId"]

        model = self.tasks[f'{project_name}_{task_id}']["model"]

        results = model.predict.remote(items)
        results = ray.get(results)

        pred_boxes: List[List[float]] = []
        for box in results[0]["instances"].pred_boxes:
            box = box.cpu().numpy()
            pred_boxes.append(box.tolist())

        model_response_channel = self.model_response_channel % (project_name, task_id)
        self.redis.publish(model_response_channel, json.dumps([pred_boxes, item_indices, action_packet_id]))

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

    # 在171行，ray会直接把这个actor对象放到一个有gpu的node上并调用构造函数。不需要手动放。需要做的就是在构造函数里指定CUDA_VISIBLE_ENVIRONMENT
    def get_model(self, model_name, item_list):
        NUM_WORKERS = 8
        model = Predictor.remote(model_name, item_list, NUM_WORKERS, self.logger, "0")
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

    scheduler = ModelServerScheduler(server_config, model_config, logger)
    scheduler.listen()
    logger.info("Model server launched.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    launch()

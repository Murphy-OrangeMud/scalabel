from io import BytesIO
import os
import time
from PIL import Image
from detectron2 import model_zoo
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config.config import get_cfg
from detectron2.modeling.meta_arch.build import build_model
import detectron2.data.transforms as T
import numpy as np
from ts.torch_handler.vision_handler import VisionHandler
import torch
from torch.multiprocessing import Pool

class Detectron2Handler:

    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None
        self.aug = None

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

        model_name = self.manifest["model"]["modelName"]
        cfg_path = model_name.split("_")[0] + ".yaml"
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
        # NOTE: you may customize cfg settings
        # cfg.MODEL.DEVICE="cuda" # use gpu by default
        cfg.MODEL.DEVICE="cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # you can also give a path to you checkpoint
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

        self.model = build_model(cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        
        self.initialized = True

    def inference(self, data, *args, **kwargs):
        with torch.no_grad():
            predictions = self.model(data)
            return predictions

    def handle(self, data, context):
        start_time = time.time()
        
        self.context = context
        metrics = self.context.metrics

        results = self.inference(self.preprocess(data))

        stop_time = time.time()
        metrics.add_time("HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms")

        return results

    def preprocess(self, data):
        # data: List of Bytes representing pictures
        images = []
        for item in data["image"]:
            img = np.array(Image.open(BytesIO(item.content)))
            height, width = img.shape[:2]
            img = self.aug.get_transform(img).apply_image(img)
            img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
            if self.map_location == "cuda":
                img = img.pin_memory()
                img = img.cuda(non_blocking=True)
            images.append({"image": img, "height": height, "width": width})
        
        return images


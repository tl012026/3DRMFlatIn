"""
Module role:
    Detect person instances in each view (Ultralytics YOLO or equivalent).
    Responsible for bounding boxes only, not fine masks.

Inputs:
    - images: batched image tensors or ndarray list
      OR image_paths: List[str]
    - weights_path: detector checkpoint path (from config)
    - conf_thres: confidence threshold
    - device: "cuda" | "cpu"
    - person_class_id: class index for "person" (if multi-class model)

Outputs:
    # MODIFIED: dropped track_ids / det_meta (optional fields removed)
    - bboxes: per-image boxes, List[Nx4] in xyxy; list length = num images
    - scores: per-box confidence, List[N]; aligned with bboxes

Variables (contract):
    - self.model: loaded detector weights
    - self.conf_thres, self.device, self.person_class_id
    - images / image_paths
    # MODIFIED: only bboxes, scores
    - bboxes, scores
"""

from ultralytics import YOLO


class PersonDetector:
    """
    Role:
        Thin wrapper around the detection backend.

    Inputs (constructor):
        weights_path, conf_thres, device, person_class_id

    Outputs (predict):
        # MODIFIED: only bboxes, scores
        bboxes, scores
    """

    def __init__(self, weights_path, conf_thres, device, person_class_id):
        self.weights_path = weights_path
        self.conf_thres = conf_thres
        self.device = device
        self.person_class_id = person_class_id

        self.model = YOLO(self.weights_path).to(self.device)

    def predict(self, images):
        """
        Inputs:
            images: tensors / ndarrays / paths for one or more views
        Outputs:
            # MODIFIED: one dict for the whole batch (not one dict per image).
            #   bboxes[i] / scores[i] = detections for image i
            dict with keys: bboxes, scores
        Variables used:
            self.model, self.conf_thres, self.device, self.person_class_id
        """
        # MODIFIED: filter person class inside predict (classes=...); also apply conf/device here
        results = self.model(
            images,
            conf=self.conf_thres,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False,
        )

        bboxes = []
        scores = []

        # MODIFIED: aggregate all images into list-of-per-image arrays
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                bboxes.append([])
                scores.append([])
                continue
            # xyxy float arrays; conf aligned with boxes
            bboxes.append(boxes.xyxy.cpu().numpy())
            scores.append(boxes.conf.cpu().numpy())

        # MODIFIED: return only bboxes + scores (track_ids / det_meta removed)
        return {
            "bboxes": bboxes,
            "scores": scores,
        }

import os
import cv2
import yaml
import numpy as np
import tensorflow as tf

# Load config from YAML
with open('./configs/lead_segmentation.yaml', 'r') as f:
    config = yaml.safe_load(f)

CONF_THRESHOLD = config['conf_threshold']
MODEL_PATH = config['model_path']
LEFT_LABELS = config['left_labels']
RIGHT_LABELS = config['right_labels']


def filter_Detections(results, thresh = 0.5):
    """
    Filters out the detections based on the confidence threshold
    Args:
        results: Inference results of the shape (8400, 17)
        thrsh: confidence threshold
    Returns:
        considerable_detections: Filtered Detections of the size (n, 6)
    """
    A = []
    for detection in results:

        class_id = detection[4:].argmax()
        confidence_score = detection[4:].max()

        new_detection = np.append(detection[:4],[class_id,confidence_score])

        A.append(new_detection)

    A = np.array(A)

    # filter out the detections with confidence > thresh
    considerable_detections = [detection for detection in A if detection[-1] > thresh]
    considerable_detections = np.array(considerable_detections)

    return considerable_detections
    
def NMS(boxes, conf_scores, iou_thresh = 0.55):
    """
    Applies NMS to the detections
    Args:
        boxes: The detetcted boxes coordinates
        conf_scores: Conf scores for each box
        iou_thresh: Intersection over union threshold
    Returns:
        keep: The remaining boxes (x1, y1, x2, y2, cls_id) 
        keep_confidences: The confidence scores of the remaining boxes
    """

    #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices= order)
        yy1 = np.take(y1, indices= order)
        xx2 = np.take(x2, indices= order)
        yy2 = np.take(y2, indices= order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter/union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2-xx1, 0)
        h = np.maximum(yy2-yy1, 0)

        intersection = w*h

        # union = areaA + other_areas - intesection
        other_areas = np.take(areas, indices= order)
        union = areas[idx] + other_areas - intersection

        iou = intersection/union

        boleans = iou < iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]

    return keep, keep_confidences



def rescale_back(results,img_w,img_h):
    """
    Rescales the normalized coordinates and applies nms
    Args:
        results: Boxes with confidence scores
        img_w: Width to rescale to
        img_h: Height to rescale to
    Returns:
        keep: Boxes (x1,y1,x2,y2,cls_id)
        keep_confidences: Confidence scores for eachh box
    """
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes,confidence)
    return keep, keep_confidences
   


def init_model(model_path=MODEL_PATH):
    """
    Initialize the tflite model for lead segmentation.
    Args:
        model_path: Path to the tflite model file.
    Returns:
        interpreter: tflite interpreter
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found at {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter



def inference_and_label_and_crop(interpreter,
                                 input_image_path, output_dir,
                                 conf_threshold=CONF_THRESHOLD):
    """
    Perform inference on a single image using ONNX, label detected boxes, and save cropped leads.

    Args:
        interpreter: tflite model interpreter
        input_image_path: Path to input image
        output_dir: Directory to save cropped leads
        conf_threshold: Confidence threshold for detection

    Returns:
        cropped_leads: List of (cropped lead image, label)
        labeled_boxes_paths: List of saved image paths
    """
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_image_path}")
    img_h, img_w = image.shape[:2]

    # Preprocess
    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)[None]   # shape [1,3,640,640]
    img = img.astype(np.float32) / 255.0

    # Inference
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    img = np.transpose(img, (0, 2, 3, 1))  # Now [1, 640, 640, 3]
    print("Input shape to TFLite model:", img.shape)
    # Should be (1, 640, 640, 3)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # [1, C, N]
    results = output_data[0].T  # transpose to shape [N, C]

    # Postprocess
    filtered = filter_Detections(results, thresh=conf_threshold)
    keep, confidences = rescale_back(filtered, img_w, img_h)
    # only class 0
    keep = [b for b in keep if int(b[-1]) == 0]

    if len(keep) < len(LEFT_LABELS) + len(RIGHT_LABELS):
        raise RuntimeError(f"Detected only {len(keep)} boxes, expected {len(LEFT_LABELS)+len(RIGHT_LABELS)}.")


    wave_boxes = [b[:4] for b in keep]
    x_centers = [((x1+x2)/2) for x1,y1,x2,y2 in wave_boxes]
    median_x = np.median(x_centers)
    left, right = [], []
    for box in wave_boxes:
        x1, x2 = box[0], box[2]
        (left if x1<median_x else right).append(box)
    left.sort(key=lambda b: (b[1]+b[3])/2); right.sort(key=lambda b: (b[1]+b[3])/2)
    labeled = list(zip(left, LEFT_LABELS)) + list(zip(right, RIGHT_LABELS))

    cropped_leads = []
    labeled_boxes_paths = []
    base = os.path.splitext(os.path.basename(input_image_path))[0]
    for box, label in labeled:
        x1,y1,x2,y2 = map(int,box)
        crop = image[y1:y2, x1:x2]
        if crop.size==0: continue
        path = os.path.join(output_dir, f"{base}_{label}.jpg")
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY,100])
        cropped_leads.append((crop,label))
        labeled_boxes_paths.append(path)

    print(f"[DEBUG] Saved {len(labeled_boxes_paths)} crops")
    return cropped_leads, labeled_boxes_paths

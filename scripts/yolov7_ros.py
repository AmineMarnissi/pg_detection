from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from detection_msgs.msg import BoundingBox, BoundingBoxes
from torchvision.transforms import ToTensor
import torch
from typing import Tuple
import rospy
import cv2
import numpy as np
import random
import sys
import os
import sort

# add yolov7 submodule to path
FILE_ABS_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV7_ROOT = os.path.abspath(os.path.join(FILE_ABS_DIR, '../src/yolov7'))
if str(YOLOV7_ROOT) not in sys.path:
    sys.path.append(str(YOLOV7_ROOT))
from visualizer import draw_detections
from utils.general import non_max_suppression
from models.experimental import attempt_load
from utils.ros import create_detection_msg


def rescale(ori_shape, boxes, target_shape):
    '''Rescale the output to the original image shape
    copied from https://github.com/meituan/YOLOv6/blob/main/yolov6/core/inferer.py
    '''
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (
        ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0].clamp_(0, target_shape[1])  # x1
    boxes[:, 1].clamp_(0, target_shape[0])  # y1
    boxes[:, 2].clamp_(0, target_shape[1])  # x2
    boxes[:, 3].clamp_(0, target_shape[0])  # y2

    return boxes


class YoloV7_ROS:
    def __init__(self, weights, conf_thresh: float = 0.5, iou_thresh: float = 0.45,
                 img_size: int = 640, device: str = "cuda",
                 visualize: bool = True,
                 input_img_topic: str = "/image_raw",
                 pub_topic: str = "yolov7_detections",
                 output_img_topic: str = "yolov7/image_raw"):
        rospy.loginfo("Starting Yolov7_ROS node")
        self.__conf_thresh = conf_thresh
        self.__iou_thresh = iou_thresh
        self.__img_size = img_size
        self.__device = device
        self.__weights = weights
        self.__model = attempt_load(self.__weights, map_location=device)
        self.__names = self.__model.names
        self.__colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.__names]
        self.__visualize = visualize
        self.__input_img_topic = input_img_topic
        self.__output_topic = pub_topic
        self.__output_img_topic = output_img_topic

        #create instance of the SORT tracker
        self.tracker = sort.Sort(max_age=10, min_hits=3)
        self.current_person_ids = np.array([], dtype=int)
        self.previous_person_ids = np.array([], dtype=int)
        
        # ROS
        self.visualization_publisher = rospy.Publisher(
            self.__output_img_topic, Image, queue_size=10) if visualize else None
        self.img_subscriber = rospy.Subscriber(
            self.__input_img_topic, Image, self.__img_cb)
        self.detection_publisher = rospy.Publisher(
            self.__output_topic, BoundingBoxes, queue_size=10)
        self.bridge = CvBridge()
        self.model_info()

    def model_info(self, verbose=False, img_size=640):
        # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
        n_p = sum(x.numel()
                  for x in self.__model.parameters())  # number parameters
        n_g = sum(x.numel() for x in self.__model.parameters()
                  if x.requires_grad)  # number gradients
        if verbose:
            rospy.loginfo('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name',
                                                                'gradient', 'parameters', 'shape', 'mu', 'sigma'))
            for i, (name, p) in enumerate(self.__model.named_parameters()):
                name = name.replace('module_list.', '')
                rospy.loginfo('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                              (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

        try:  # FLOPS
            from thop import profile
            stride = max(int(self.__model.stride.max()), 32) if hasattr(
                self.__model, 'stride') else 32
            img = torch.zeros((1, self.__model.yaml.get('ch', 3), stride, stride), device=next(
                self.__model.parameters()).device)  # input
            flops = profile(deepcopy(self.__model), inputs=(img,), verbose=False)[
                0] / 1E9 * 2  # stride GFLOPS
            img_size = img_size if isinstance(img_size, list) else [
                img_size, img_size]  # expand if int/float
            fs = ', %.1f GFLOPS' % (
                flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
        except (ImportError, Exception):
            fs = ''
        summary = f"\N{rocket}\N{rocket}\N{rocket} Yolov7 Detector summary:\n" \
            + f"Weights: {self.__weights}\n" \
            + f"Confidence Threshold: {self.__conf_thresh}\n" \
            + f"IOU Threshold: {self.__iou_thresh}\n"\
            + f"{len(list(self.__model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}\n"\
            + f"Input topic: {self.__input_img_topic}\n"\
            + f"Output topic: {self.__output_topic}\n" \
            + f"Output image topic: {self.__output_img_topic}"
        rospy.loginfo(summary)

    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        img = img.unsqueeze(0)
        pred_results = self.__model(img)[0]
        detections = non_max_suppression(
            pred_results, conf_thres=self.__conf_thresh, iou_thres=self.__iou_thresh
        )
        if detections:
            detections = detections[0]
        return detections

    def process_img(self, img_input):
        # automatically resize the image to the next smaller possible size
        h_ori, w_ori, _ = img_input.shape
        w_scaled = self.__img_size
        h_scaled = int(w_scaled * h_ori/w_ori)

        # w_scaled = w_orig - (w_orig % 8)
        np_img_resized = cv2.resize(img_input, (w_scaled, h_scaled))
        # conversion to torch tensor (copied from original yolov7 repo)
        img = np_img_resized.transpose(
            (2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.__device)
        return img

    def __img_cb(self, img_msg):
        """ callback function for publisher """
        frame = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='bgr8'
        )
        h_ori, w_ori, c = frame.shape
        w_scaled = self.__img_size
        h_scaled = int(w_scaled * h_ori/w_ori)
        img_input = self.process_img(frame)
        detections = self.inference(img_input)
        detections[:, :4] = rescale(
            [h_scaled, w_scaled], detections[:, :4], [h_ori, w_ori])
        detections[:, :4] = detections[:, :4].round()

        ####################################################        
        # Assuming 'detections' is the provided tensor
        filtered_detections = detections[(detections[:, 5] < 8).nonzero().squeeze()]
        if filtered_detections.shape == (6,):
            filtered_detections = filtered_detections.view(1, 6)

        # Tracking Sort
        # Assuming 'detections' is your tensor
        dets_to_sort = np.empty((0, 6))

        # NOTE: We send in detected object class too
        for x1, y1, x2, y2, conf, detclass in filtered_detections.cpu().detach().numpy():
            dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

        trackers = self.tracker.update(dets_to_sort)
        ###########################################################
        # publishing
        detection_msg = create_detection_msg(img_msg, detections, self.__names)
        self.detection_publisher.publish(detection_msg)
        
        # Assuming draw_detections is a function to draw detections with additional info

        if self.__visualize:
            if detections.numel() != 0 and len(trackers) > 0:
                bboxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in trackers[:, :4]]
                classes = [int(c) for c in trackers[:, 4]]
                conf = [c for c in trackers[:, 8].tolist()]
                
                # Compute velocity magnitude and angle (theta)
                velocities = trackers[:, 5:7]  # Assuming velocities are in columns 5 and 6
                v_magnitudes = trackers[:, 5] 
                angles = trackers[:, 6] 
                
                vis_img = draw_detections(frame, bboxes, classes, self.__names, conf, self.__colors)
                
                # Draw velocity vectors on the image
                for bbox, v_mag, angle in zip(bboxes, v_magnitudes, angles):
                    center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                    end_point = (int(center[0] + v_mag * np.cos(angle)), int(center[1] + v_mag * np.sin(angle)))
                    cv2.arrowedLine(vis_img, center, end_point, (0, 255, 0), 2)
                    
                    # Calculate angle in degrees
                    angle_degrees = angle
                    if v_mag>0.99:
                        # Annotate velocity magnitude and angle
                        text = "Velocity: {:.2f}, Angle: {:.2f}°".format(v_mag, angle_degrees)
                        print(text)
                        cv2.putText(vis_img, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow("yolov7", vis_img)
                vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
                vis_msg.header.stamp = detection_msg.header.stamp
                self.visualization_publisher.publish(vis_msg)
                cv2.waitKey(1)


        # if self.__visualize:
        #     if detections.numel() != 0 and len(trackers) > 0:
        #         bboxes = [[int(x1), int(y1), int(x2), int(y2)]
        #                 for x1, y1, x2, y2 in trackers[:, :4]]
        #         classes = [int(c) for c in trackers[:, 4]]
        #         conf = [c for c in trackers[:, 8].tolist()]
        #         vis_img = draw_detections(
        #             frame, bboxes, classes, self.__names, conf, self.__colors)
        #         cv2.imshow("yolov7", vis_img)
        #         vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
        #         vis_msg.header.stamp = detection_msg.header.stamp
        #         self.visualization_publisher.publish(vis_msg)
        #         cv2.waitKey(1)

import cv2
import time
import numpy as np
import torch
import mediapipe as mp
import onnxruntime
from .eyelid_utils import get_box_from_landmarks, get_crop_from_box, pad_and_crop, decode_landmarks


class MediaPipeFaceDetector():
    def __init__(self, model_select=0) -> None:
        # model_selection: 0 for near face (< 2 meters), 1 for far face (2-5 meters)
        # --------------------------------------------------------------------------
        # key_points semantics:
        # RIGHT_EYE = 0
        # LEFT_EYE = 1
        # NOSE_TIP = 2
        # MOUTH_CENTER = 3
        # RIGHT_EAR_TRAGION = 4
        # LEFT_EAR_TRAGION = 5
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=model_select
        )
    
    def __call__(self, img, ds=1, ret_kp=False):
        H, W = img.shape[:2]
        if ds != 1:
            H_ds, W_ds = H // ds, W // ds
            img_ds = cv2.resize(img, (W_ds, H_ds))
        else:
            img_ds = img
        res = self.face_detector.process(img_ds)
        if res.detections is None:
            return None, None if ret_kp else None
        rel_bbox = res.detections[0].location_data.relative_bounding_box
        face_box = np.array([[(rel_bbox.xmin) * W, (rel_bbox.ymin) * H],
                             [(rel_bbox.xmin + rel_bbox.width) * W, (rel_bbox.ymin + rel_bbox.height) * H]])
        if ret_kp:
            key_points = np.array([[res.detections[0].location_data.relative_keypoints[i].x * W, res.detections[0].location_data.relative_keypoints[i].y * H] for i in range(6)])
            return face_box, key_points # [leye, reye, nose, mouth, lear, rear]
        
        return face_box


class LandmarkDetectorV3():
    def __init__(self, ckpt_path, use_onnx=True, use_filter=False, device='cuda:0'):
        self.face_detector = MediaPipeFaceDetector(model_select=1)

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 4
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.net = onnxruntime.InferenceSession(ckpt_path, providers=providers, sess_options=options)

        # if use_filter:
        #     # decrease min_cutoff, more stable
        #     # increase beta, follow faster
        #     self.filter_face = OneEuroFilter(min_cutoff=0.3, beta=0.01, d_cutoff=1.0)
        #     self.filter_leye = OneEuroFilter(min_cutoff=0.3, beta=0.02, d_cutoff=1.0)
        #     self.filter_reye = OneEuroFilter(min_cutoff=0.3, beta=0.02, d_cutoff=1.0)
        #     self.t_count = 0

        # warmup
        onnx_input = {
            self.net.get_inputs()[0].name: np.random.rand(1, 3, 256, 256).astype(np.float32),
            self.net.get_inputs()[1].name: np.random.rand(1, 3, 128, 128).astype(np.float32),
            self.net.get_inputs()[2].name: np.random.rand(1, 3, 128, 128).astype(np.float32)
        }
        _ = self.net.run(None, onnx_input)

        self.use_filter = use_filter
        self.use_onnx = use_onnx
        assert use_onnx, "Only Support onnx!"

    def reset(self):
        if self.use_filter:
            self.filter_face.reset()
            self.filter_leye.reset()
            self.filter_reye.reset()
            self.t_count = 0

    def crop_eye_region(self, img, lcenter, rcenter):
        lr_distance = np.linalg.norm(rcenter - lcenter)
        size = np.round(lr_distance * 0.33).astype(np.int32)

        lcenter = np.round(lcenter).astype(np.int32)
        lcrop_lt = lcenter - size
        lcrop_rb = lcenter + size
        leye_img_crop = pad_and_crop(img, lcrop_lt, lcrop_rb)
        leye_img_resize = cv2.resize(leye_img_crop, (128, 128))

        rcenter = np.round(rcenter).astype(np.int32)
        rcrop_lt = rcenter - size
        rcrop_rb = rcenter + size
        reye_img_crop = pad_and_crop(img, rcrop_lt, rcrop_rb)
        reye_img_resize = cv2.resize(reye_img_crop, (128, 128))

        return leye_img_resize, leye_img_crop, lcrop_lt, lcrop_rb, \
                reye_img_resize, reye_img_crop, rcrop_lt, rcrop_rb

    @torch.no_grad()
    def __call__(self, img, key_points):
        # Face Detection
        # assume the first face box is what we want
        st = time.time()
        # face_box, key_points = self.face_detector(img, ret_kp=True)
        # for i in range(6):
        #     x, y = int(key_points[i, 0] + 0.5), int(key_points[i, 1] + 0.5)
        #     img[y-2:y+3, x-2:x+3] = np.array([255, 0, 0])
        #     cv2.imwrite("kp%d.jpg" % i, img[..., [2, 1, 0]])
        # if face_box is None:
        #     return np.zeros([68+48+38, 2], np.float32), 0
        # key_points[3, 1] += 50  # empirically finetune
        # key_points[0, 1] -= 10  # empirically finetune
        # key_points[1, 1] -= 10  # empirically finetune
        face_box = get_box_from_landmarks(key_points[:4], img.shape[0], img.shape[1], pad_scale=0.5)
        interval_det = (time.time() - st) * 1000

        # Crop Face Region
        face_crop_lt, face_crop_rb = get_crop_from_box(face_box)
        face_img_crop = pad_and_crop(img, face_crop_lt, face_crop_rb)
        face_img_resize = cv2.resize(face_img_crop, (256, 256))

        # Crop Eye Region
        leye_img_resize, leye_img_crop, leye_crop_lt, leye_crop_rb, \
        reye_img_resize, reye_img_crop, reye_crop_lt, reye_crop_rb \
            = self.crop_eye_region(img, key_points[0], key_points[1])
        
        # for i in range(6):
        #     x = int(key_points[i, 0] + 0.5) - face_crop_lt[0]
        #     y = int(key_points[i, 1] + 0.5) - face_crop_lt[1]
        #     cv2.circle(face_img_crop, (x, y), 2, (0, 255, 0), -1)
        # cv2.imshow("face_img_crop", cv2.cvtColor(face_img_crop, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(30)
        # cv2.imwrite("face_img_crop.png", cv2.cvtColor(face_img_crop, cv2.COLOR_RGB2BGR))
        # cv2.imwrite("leye_img_crop.png", cv2.cvtColor(leye_img_crop, cv2.COLOR_RGB2BGR))
        # cv2.imwrite("reye_img_crop.png", cv2.cvtColor(reye_img_crop, cv2.COLOR_RGB2BGR))
        # exit(0)

        face_img_onnx = face_img_resize.astype(np.float32)[np.newaxis, :].transpose((0, 3, 1, 2)) / 255
        leye_img_onnx = cv2.flip(leye_img_resize, flipCode=1).astype(np.float32)[np.newaxis, :].transpose((0, 3, 1, 2)) / 255
        reye_img_onnx = reye_img_resize.astype(np.float32)[np.newaxis, :].transpose((0, 3, 1, 2)) / 255

        onnx_input = {
            self.net.get_inputs()[0].name: face_img_onnx,
            self.net.get_inputs()[1].name: leye_img_onnx,
            self.net.get_inputs()[2].name: reye_img_onnx,
        }
        face_heatmaps, eye_heatmaps = self.net.run(None, onnx_input)
        face_lmk, face_scores = decode_landmarks(torch.from_numpy(face_heatmaps), gamma=1.0, radius=0.1)
        eye_lmks, eye_scores = decode_landmarks(torch.from_numpy(eye_heatmaps), gamma=1.0, radius=0.1)

        # xxx_crop      [H, W, C]
        # xxx_heatmaps  [B, C, H, W]
        face_lmk = face_lmk[0].detach().numpy() * np.array([face_img_crop.shape[-2] / face_heatmaps.shape[-1], face_img_crop.shape[-3] / face_heatmaps.shape[-2]])
        leye_lmk = eye_lmks[0].detach().numpy() * np.array([leye_img_crop.shape[-2] / eye_heatmaps.shape[-1], leye_img_crop.shape[-3] / eye_heatmaps.shape[-2]])
        reye_lmk = eye_lmks[1].detach().numpy() * np.array([reye_img_crop.shape[-2] / eye_heatmaps.shape[-1], reye_img_crop.shape[-3] / eye_heatmaps.shape[-2]])

        # Map Back to Original Face Image
        face_lmk = face_lmk + face_crop_lt
        leye_lmk[:, 0] = leye_img_crop.shape[-2] - 1 - leye_lmk[:, 0]   # flip back
        leye_lmk = leye_lmk + leye_crop_lt
        reye_lmk = reye_lmk + reye_crop_lt

        if self.use_filter:
            norm_factor = np.array([img.shape[1], img.shape[0]])
            face_lmk = self.filter_face(self.t_count, face_lmk / norm_factor) * norm_factor
            leye_lmk = self.filter_leye(self.t_count, leye_lmk / norm_factor) * norm_factor
            reye_lmk = self.filter_reye(self.t_count, reye_lmk / norm_factor) * norm_factor
            self.t_count += 1

        lmk = np.concatenate([face_lmk, leye_lmk[:19], reye_lmk[:19], leye_lmk[19:], reye_lmk[19:]], axis=0)

        return lmk, interval_det

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中国护照 / “中国领事”App 电子证件照处理脚本（v2，YuNet 关键点版）。

处理链路
    读取(含 EXIF 转正) → rembg 抠图合成纯白底 → YuNet 人脸关键点
    → 关键点 + 抠图剪影量头廓 → 水平校正 + 缩放 + 构图（一次仿射重采样）
    → 纯白背景清理 → JPEG 体积二分 → 成品规格自检报告

人脸定位（v2 起不再使用 haarcascade_eye.xml / haarcascade_frontalface_default.xml）
    主引擎 cv2.FaceDetectorYN：OpenCV 自带的 YuNet 神经网络人脸检测器，一次前向
    同时给出人脸框 + 5 个关键点（左右眼瞳孔中心 / 鼻尖 / 左右嘴角）。模型
    face_detection_yunet_2023mar.onnx 仅 227 KB，CPU 单帧 <1 s，对倾斜 / 侧脸 /
    逆光的鲁棒性远高于 Haar 级联；瞳孔中心直接给出，不必再用“人脸上半部找眼睛框”
    这类启发式（v1 那段不仅不准，代码本身也是坏的，见 process_passport.v1-haar.py）。

    可选精修 cv2.face.createFacemarkLBF（opencv_contrib，68 点，含下颌轮廓，
    能量出真实下巴最低点）。模型 lbfmodel.yaml 约 54 MB，默认不下载：本地已有则
    自动启用（--refine auto），或显式 --download-lbf。

    兜底 YuNet 模型缺失且无法下载时，退回 Haar 正脸级联 + 人体测量比例推算眼位
    （依然不使用 haarcascade_eye.xml），并在报告里标注精度降级。

    瞳孔亚像素精修 refine_eyes()：YuNet 的关键点精度随检测尺度波动（同一张图在
    长边 640~1600 上测出的倾角能差 ±3°），直接拿来水平校正不稳。于是在每个眼位附近
    取窗口做暗度加权质心（三次幂压掉背景、减均值去白底偏置）重定瞳孔中心：实测对
    施加的 -4°~+5° 旋转响应线性，恒定偏置 +0.33°，跨尺度误差 <0.4°。嘴角线只作
    交叉校验（两者差 >3° 时提示人工确认），不参与计算。

头部尺寸怎么量的（v2 的关键改动）
    只靠人脸框推头高头宽误差很大，而且人脸在画面里占比不同时结果还会漂移
    （实测同一张图：长边 480~1280 检测 score≈0.93 且跨尺度一致，长边 4096 时
    score 掉到 0.80、瞳孔偏上百像素）。所以：
      · 检测统一缩放到长边 --detect-long（默认 1024）再反算坐标；
      · 头顶 = 抠图剪影在人脸窗口内自顶向下第一个连续成片的行（真实可见头顶）；
      · 下巴 = 眼线 + CHIN_IOD_RATIO × 瞳距（比例稳定，且与画面占比无关）；
      · 耳间头宽 = 眼线到下巴之间剪影宽度的高分位数（正好覆盖两耳）。
    源图与成品用同一套量法，自检才有意义。

 头宽怎么处理
    规范里还写着「两耳之间 177-260」，但耳朵外张 + 发量厚的头型（本目录样本实测
    宽/高比 0.78~0.83，而规范极限 260/330≈0.79）在数学上无法与头高同时满足，
    受理时看的是头部尺寸与居中，不是耳间宽。所以头宽只作为**内部软约束**参与选
    缩放（尽量别把脸撑满画面），不再出现在自检报告里——给用户看一条既满足不了、
    又不影响受理的 WARN，除了添堵没有别的用处。

 背景纯白化
    从画面边缘泛洪把近白背景压成 255，但泛洪以抠图遮罩为屏障——否则浅色衬衫上的
    高光（与白墙只差十几级灰度）会顺连通路径被一路刷白，在胸口留下白色缺口。

用法
    venv/bin/python3 process_passport.py <input_image> <output_image> [选项]
    venv/bin/python3 process_passport.py in.jpg out.jpg --debug --strict

标定/回归测试
    venv/bin/python3 test_geometry.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import unicodedata
import urllib.request
from dataclasses import dataclass

# --- 本地化缓存目录（必须在导入 rembg / matplotlib 相关库之前设置） ---
os.environ.setdefault("REMBG_HOME", os.path.abspath("./.rembg"))
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("./.matplotlib"))
for _d in (os.environ["REMBG_HOME"], os.environ["MPLCONFIGDIR"]):
    os.makedirs(_d, exist_ok=True)

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # 静音 OpenCV 的 INFO/WARN 噪声（例如 new graph engine 的 target 提示）
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:  # pragma: no cover - 老版本没有该 API
    pass

# --------------------------------------------------------------------------
# 规格常量（中国护照 / “中国领事”App 电子照硬性标准，354x472 px 画布）
# --------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 354, 472
HEAD_W_MIN, HEAD_W_MAX = 177, 260        # 头部宽度（两耳之间）
HEAD_H_MIN, HEAD_H_MAX = 330, 390        # 头部高度（下巴 → 头顶）
HEAD_H_GUARD = 3                        # 选缩放时留的余量：成品复测有 1~2% 噪声
TOP_MARGIN_MIN, TOP_MARGIN_MAX = 35, 59  # 头顶留白
BOTTOM_GAP_MIN = 82                      # 下巴距下边缘
FILE_KB_MIN, FILE_KB_MAX = 40.0, 75.0    # 文件体积
BG_RGB = (255, 255, 255)                 # 背景色（RGB，给人看的顺序）
BG_BGR = BG_RGB[::-1]                    # 同上，cv2 的通道顺序

# --------------------------------------------------------------------------
# 规格表：一个证件类型一行，这里的数值就是构图依据与自检条目
#   canvas       画布像素（宽, 高）
#   head_h       头部高度硬区间（下巴 → 头顶，px）
#   top_margin   头顶留白区间（px）
#   chin_gap     下巴距下边缘下限（px）
#   head_w       耳间头宽软区间：只参与选缩放，不出自检（见文件头说明）
#   bg           背景色 RGB
#   kb           交付 JPEG 体积区间
#   dpi          写进 JPEG 的 DPI
#   aim_head_h / aim_top_margin  构图目标值：取可行区间偏上，给成品复测噪声留余量
# 数值来源与官方出处见 docs/specs.md；apply_spec() 把选中的规格绑到上面的常量。
# --------------------------------------------------------------------------
SPECS: dict[str, dict] = {
    "passport": dict(
        name="中国护照", canvas=(354, 472), head_h=(330, 390), top_margin=(35, 59),
        chin_gap=82, head_w=(177, 260), bg=(255, 255, 255), kb=(40.0, 75.0), dpi=600,
        aim_head_h=346.0, aim_top_margin=40.0),
}
DEFAULT_SPEC = "passport"


def apply_spec(spec_id: str) -> dict:
    """把规格表绑到模块常量上：全流程只读这些常量，绑一次就整体切换规格。"""
    global CANVAS_W, CANVAS_H, HEAD_W_MIN, HEAD_W_MAX, HEAD_H_MIN, HEAD_H_MAX
    global TOP_MARGIN_MIN, TOP_MARGIN_MAX, BOTTOM_GAP_MIN, FILE_KB_MIN, FILE_KB_MAX
    global BG_RGB, BG_BGR
    s = SPECS[spec_id]
    CANVAS_W, CANVAS_H = s["canvas"]
    HEAD_H_MIN, HEAD_H_MAX = s["head_h"]
    TOP_MARGIN_MIN, TOP_MARGIN_MAX = s["top_margin"]
    BOTTOM_GAP_MIN = s["chin_gap"]
    HEAD_W_MIN, HEAD_W_MAX = s["head_w"]
    FILE_KB_MIN, FILE_KB_MAX = s["kb"]
    BG_RGB = s["bg"]
    BG_BGR = s["bg"][::-1]
    return s

# 人体测量比例（test_geometry.py 里有标定用例，可用命令行覆盖）
CHIN_IOD_RATIO = 1.77  # (下巴 - 眼线) / 瞳距；成人常见 1.70~1.85
CROWN_RATIO = 0.95     # 兜底：剪影不可用时 (头顶 - 眼线) / (下巴 - 眼线)
EAR_RATIO = 1.15       # 兜底：耳间宽 / YuNet 人脸框宽
JAW_TO_EAR = 1.06      # FacemarkLBF 下颌轮廓 0↔16 点间距 → 耳间宽

YUNET_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = [
    # opencv_zoo 用 git-lfs 存模型，必须走 media.githubusercontent.com；
    # raw.githubusercontent.com 只会拿到 131 字节的 LFS 指针文件。
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_BYTES = 232589

LBF_URLS = ["https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"]


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# 人脸关键点
# --------------------------------------------------------------------------
@dataclass
class Face:
    """一次人脸检测结果，坐标为原图像素坐标。"""

    bbox: tuple[float, float, float, float]  # x, y, w, h
    score: float
    right_eye: tuple[float, float]
    left_eye: tuple[float, float]
    nose: tuple[float, float]
    mouth_right: tuple[float, float]
    mouth_left: tuple[float, float]
    engine: str = "yunet"

    @property
    def eye_center(self) -> tuple[float, float]:
        return ((self.right_eye[0] + self.left_eye[0]) / 2.0,
                (self.right_eye[1] + self.left_eye[1]) / 2.0)

    @property
    def iod(self) -> float:
        """瞳距。"""
        return float(np.hypot(self.left_eye[0] - self.right_eye[0],
                              self.left_eye[1] - self.right_eye[1]))

    @property
    def roll_angle(self) -> float:
        """头部倾斜角（度）。正值 = 从观看者看顺时针歪头，需逆时针旋转校正。

        以瞳孔连线为准（refine_eyes 之后精度 ~0.3°）。嘴角线只作交叉校验，
        不并入——嘴角点噪声比瞳孔大，混进来反而把误差拉回 1° 以上。
        图像坐标 y 轴向下，故 atan2 得到的角度取正值即为 PIL / cv2 约定的逆时针角。
        """
        return float(np.degrees(np.arctan2(self.left_eye[1] - self.right_eye[1],
                                           self.left_eye[0] - self.right_eye[0])))

    @property
    def mouth_roll(self) -> float:
        """嘴角线倾角，用于交叉校验眼线。"""
        return float(np.degrees(np.arctan2(self.mouth_left[1] - self.mouth_right[1],
                                           self.mouth_left[0] - self.mouth_right[0])))

    def scaled(self, k: float) -> "Face":
        """整体缩放坐标（score 不变），用于把检测映射回原图/画布坐标。"""
        return Face(
            bbox=tuple(v * k for v in self.bbox), score=self.score,
            right_eye=(self.right_eye[0] * k, self.right_eye[1] * k),
            left_eye=(self.left_eye[0] * k, self.left_eye[1] * k),
            nose=(self.nose[0] * k, self.nose[1] * k),
            mouth_right=(self.mouth_right[0] * k, self.mouth_right[1] * k),
            mouth_left=(self.mouth_left[0] * k, self.mouth_left[1] * k),
            engine=self.engine)


def refine_pupil(gray: np.ndarray, p: tuple[float, float], iod: float) -> tuple[float, float]:
    """把 YuNet 的眼点精修到瞳孔质心（暗度加权质心，亚像素）。

    YuNet 关键点本身有 2~5px 抖动，且随检测尺度漂移（实测同一张图 detect_long
    640→1280 时 roll 在 +7.1°~-1.4° 之间跳），直接拿来水平校正不够稳。瞳孔是眼窗
    里最暗的一团，质心定位精度 ~1px，跨尺度几乎不变：实测施加 -4°~+5° 旋转后复测，
    误差恒定在 +0.3° 左右（裸关键点误差 -3.7°~+0.4° 乱跳）。
    """
    h, w = gray.shape
    r = max(6, int(iod * 0.16))
    x0, x1 = int(max(0, p[0] - r)), int(min(w, p[0] + r))
    y0, y1 = int(max(0, p[1] - r * 0.8)), int(min(h, p[1] + r * 0.8))
    if x1 <= x0 or y1 <= y0:
        return p
    win = cv2.GaussianBlur(gray[y0:y1, x0:x1].astype(np.float32), (5, 5), 0)
    lo, hi = float(win.min()), float(win.max())
    if hi - lo < 1.0:
        return p
    wgt = np.clip((hi - win) / (hi - lo), 0.0, 1.0) ** 3
    wgt -= wgt.mean()
    np.clip(wgt, 0.0, None, out=wgt)
    if wgt.sum() <= 0:
        return p
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return (float((xs * wgt).sum() / wgt.sum()), float((ys * wgt).sum() / wgt.sum()))


def refine_eyes(bgr: np.ndarray, face: Face) -> Face:
    """精修左右瞳孔，返回新的 Face（眼线角度与瞳距随之更准）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    re = refine_pupil(gray, face.right_eye, face.iod)
    le = refine_pupil(gray, face.left_eye, face.iod)
    out = Face(bbox=face.bbox, score=face.score, right_eye=re, left_eye=le,
               nose=face.nose, mouth_right=face.mouth_right, mouth_left=face.mouth_left,
               engine=face.engine)
    if out.iod < face.iod * 0.5 or out.iod > face.iod * 2.0:   # 精修跑飞了就用原值
        return face
    return out


@dataclass
class HeadGeom:
    crown_y: float
    eye_y: float
    chin_y: float
    head_h: float
    head_w: float
    crown_off: float
    chin_off: float
    iod: float
    roll: float
    crown_src: str = "ratio"   # silhouette | ratio
    width_src: str = "ratio"   # silhouette | ratio

    def scaled(self, k: float) -> "HeadGeom":
        """整体缩放到另一坐标系（像素量全部乘 k，roll 不变）。"""
        return HeadGeom(*[v * k for v in (self.crown_y, self.eye_y, self.chin_y,
                                          self.head_h, self.head_w, self.crown_off,
                                          self.chin_off, self.iod)],
                        roll=self.roll, crown_src=self.crown_src,
                        width_src=self.width_src)


# --------------------------------------------------------------------------
# 画图（--debug 与 make_debug.py 共用）
# --------------------------------------------------------------------------
COLOR_CROWN, COLOR_EYE, COLOR_CHIN, COLOR_BOX = (255, 0, 255), (0, 255, 255), (0, 0, 255), (0, 200, 0)


def draw_face_marks(img: np.ndarray, face: Face, k: float = 1.0, box: bool = True) -> np.ndarray:
    """画人脸框（绿）与 5 个关键点（红）。k = 原图 → 画布的缩放系数。"""
    if box:
        x, y, bw, bh = face.bbox
        cv2.rectangle(img, (int(x * k), int(y * k)),
                      (int((x + bw) * k), int((y + bh) * k)), COLOR_BOX, 2)
    for p in (face.right_eye, face.left_eye, face.nose, face.mouth_right, face.mouth_left):
        cv2.circle(img, (int(p[0] * k), int(p[1] * k)), max(2, int(round(4 * k))),
                   (0, 0, 255), -1)
    return img


def draw_head_lines(img: np.ndarray, geom: HeadGeom, k: float = 1.0,
                    crown_y: float | None = None) -> np.ndarray:
    """画头顶（品红）/ 眼线（黄）/ 下巴（红）三条水平线。

    crown_y 缺省用 geom.crown_y；成品面板要画的是“推算留白线”，由调用方传入。
    """
    for yy, color in ((geom.crown_y if crown_y is None else crown_y, COLOR_CROWN),
                      (geom.eye_y, COLOR_EYE), (geom.chin_y, COLOR_CHIN)):
        cv2.line(img, (0, int(yy * k)), (img.shape[1], int(yy * k)), color, 1)
    return img


# --------------------------------------------------------------------------
# 剪影测量（头顶 / 耳间宽）
# --------------------------------------------------------------------------
def subject_mask(bgr: np.ndarray, alpha: np.ndarray | None = None,
                 thresh: int = 245) -> np.ndarray:
    """主体掩码：优先用 rembg 的 alpha，否则用“非近白”近似（要求背景已接近纯白）。"""
    if alpha is not None:
        return alpha > 128
    return bgr.min(axis=2) < thresh


def silhouette_top(mask: np.ndarray, cx: float, half_w: float, y_top: float,
                   y_bottom: float, min_frac: float = 0.06, run: int = 16):
    """人脸窗口内自顶向下找头顶：某行主体像素数达标且向下连续 run 行。"""
    h, w = mask.shape
    x0, x1 = max(0, int(cx - half_w)), min(w, int(cx + half_w))
    win = max(1, x1 - x0)
    need = max(4, int(win * min_frac))
    y0, y1 = max(0, int(y_top)), min(h, int(y_bottom))
    if y1 - y0 <= run:
        return None
    rows = mask[y0:y1, x0:x1].sum(axis=1)
    for i in range(len(rows) - run):
        if rows[i] >= need and rows[i:i + run].min() >= need:
            if y0 + i <= 2:          # 头顶顶到画面边缘 = 被裁切，剪影不可信
                return None
            return float(y0 + i)
    return None


def silhouette_width(mask: np.ndarray, cx: float, half_w: float, y0: float, y1: float,
                     pct: float = 90.0) -> float | None:
    """[y0,y1] 各行主体左右极值跨度，取高分位数——正好覆盖两耳之间的宽度。"""
    h, w = mask.shape
    x0, x1 = max(0, int(cx - half_w)), min(w, int(cx + half_w))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    widths = []
    for row in mask[y0:y1, x0:x1]:
        idx = np.flatnonzero(row)
        if idx.size:
            widths.append(idx[-1] - idx[0] + 1)
    if not widths:
        return None
    return float(np.percentile(widths, pct))


def head_geometry(face: Face, mask: np.ndarray | None,
                  chin_iod_ratio: float = CHIN_IOD_RATIO,
                  crown_ratio: float = CROWN_RATIO,
                  ear_ratio: float = EAR_RATIO,
                  chin_override: float | None = None) -> HeadGeom:
    """由 5 关键点 + 剪影推算头顶 / 下巴 / 头高 / 耳间头宽。

    剪影量到就用剪影（头顶 = 可见发顶，耳间宽 = 耳廓带内的跨度分位数）；
    剪影不可用时退回瞳距比例，并在 crown_src / width_src 里如实标注来源。
    """
    cx, eye_y = face.eye_center
    iod = face.iod
    chin_y = float(chin_override) if chin_override is not None else eye_y + iod * chin_iod_ratio
    chin_off = chin_y - eye_y
    if chin_off < iod:  # 防退化：下巴不可能离眼线这么近
        chin_y = eye_y + iod * chin_iod_ratio
        chin_off = chin_y - eye_y

    half_w = face.bbox[2] * 1.25
    crown_y = silhouette_top(mask, cx, half_w, eye_y - chin_off * 2.2,
                             eye_y - chin_off * 0.15) if mask is not None else None
    if crown_y is None:
        crown_y, crown_src = eye_y - chin_off * crown_ratio, "ratio"
    else:
        crown_src = "silhouette"

    width = silhouette_width(mask, cx, half_w, eye_y, chin_y) if mask is not None else 0.0
    if width > 0:
        width_src = "silhouette"
    else:
        width, width_src = face.bbox[2] * ear_ratio, "ratio"

    return HeadGeom(crown_y=crown_y, eye_y=eye_y, chin_y=chin_y,
                    head_h=chin_y - crown_y, head_w=width,
                    crown_off=eye_y - crown_y, chin_off=chin_off,
                    iod=iod, roll=face.roll_angle,
                    crown_src=crown_src, width_src=width_src)


# --------------------------------------------------------------------------
# 人脸检测引擎
# --------------------------------------------------------------------------
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(path: str, urls: list[str], sha256: str | None,
                 nbytes: int | None, allow_download: bool) -> bool:
    """确保本地模型可用；必要时下载并校验 sha256。"""
    if os.path.isfile(path) and os.path.getsize(path) > 1024:
        if sha256 and sha256_of(path) != sha256:
            log(f"[!] 模型校验失败（sha256 不匹配）：{path}")
            return False
        return True
    if not allow_download:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    for url in urls:
        tmp = path + ".part"
        try:
            log(f"[-] 下载模型 {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "passport-photo/2.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception as exc:  # noqa: BLE001 - 逐个镜像重试
            log(f"[!] 下载失败：{exc}")
            if os.path.exists(tmp):
                os.remove(tmp)
            continue
        size = os.path.getsize(tmp)
        if nbytes and size != nbytes:
            log(f"[!] 大小不符（{size} != {nbytes}），可能是 LFS 指针或错误页，丢弃")
            os.remove(tmp)
            continue
        if sha256 and sha256_of(tmp) != sha256:
            log("[!] sha256 校验失败，丢弃")
            os.remove(tmp)
            continue
        os.replace(tmp, path)
        log(f"[+] 模型就绪：{path} ({size / 1024:.0f} KB)")
        return True
    return False


class YuNetDetector:
    """cv2.FaceDetectorYN 封装：统一检测尺度、补齐到 32 倍数、坐标反算。"""

    name = "yunet"

    def __init__(self, model_path: str, score_thr: float = 0.6, nms_thr: float = 0.3,
                 top_k: int = 5000, detect_long: int = 1024):
        self.net = cv2.FaceDetectorYN.create(model_path, "", (32, 32),
                                             score_thr, nms_thr, top_k)
        self.detect_long = int(detect_long)

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        s = self.detect_long / float(max(h, w))
        small = cv2.resize(bgr, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        ph = int(np.ceil(small.shape[0] / 32.0) * 32)
        pw = int(np.ceil(small.shape[1] / 32.0) * 32)
        pad = cv2.copyMakeBorder(small, 0, ph - small.shape[0], 0, pw - small.shape[1],
                                 cv2.BORDER_CONSTANT, value=BG_BGR)
        self.net.setInputSize((pw, ph))
        n, res = self.net.detect(pad)
        faces = []
        for row in (res if n else []):
            v = np.asarray(row[:14], dtype=np.float64) / s
            faces.append(Face(
                bbox=(v[0], v[1], v[2], v[3]), score=float(row[14]),
                right_eye=(v[4], v[5]), left_eye=(v[6], v[7]), nose=(v[8], v[9]),
                mouth_right=(v[10], v[11]), mouth_left=(v[12], v[13])))
        faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        return faces


class HaarFallbackDetector:
    """YuNet 不可用时的兜底：Haar 正脸框 + 人体测量比例推算眼位（不使用眼睛级联）。"""

    name = "haar-fallback"

    def __init__(self) -> None:
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                             "haarcascade_frontalface_default.xml")

    def detect(self, bgr: np.ndarray) -> list[Face]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(80, 80))
        faces = []
        for (x, y, w, h) in boxes:
            eye_y, cx = y + h * 0.45, x + w / 2.0
            faces.append(Face(bbox=(float(x), float(y), float(w), float(h)), score=0.0,
                              right_eye=(cx - w * 0.21, eye_y), left_eye=(cx + w * 0.21, eye_y),
                              nose=(cx, y + h * 0.66),
                              mouth_right=(cx - w * 0.17, y + h * 0.78),
                              mouth_left=(cx + w * 0.17, y + h * 0.78),
                              engine=self.name))
        faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        return faces


class LbfRefiner:
    """可选：FacemarkLBF 68 点，给出真实下巴最低点与下颌宽度。"""

    def __init__(self, model_path: str):
        self.fm = cv2.face.createFacemarkLBF()
        self.fm.loadModel(model_path)

    def refine(self, bgr: np.ndarray, face: Face) -> dict | None:
        try:
            ok, landmarks = self.fm.fit(bgr, np.array([list(face.bbox)], dtype=np.float32))
        except cv2.error:
            return None
        if not ok or landmarks is None or len(landmarks) == 0:
            return None
        # OpenCV 返回的是 (68, 1, 2) 的 Mat，必须 reshape 成 (68, 2) 再用
        pts = np.asarray(landmarks[0], dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 17:
            return None
        jaw_w = float(np.hypot(pts[16, 0] - pts[0, 0], pts[16, 1] - pts[0, 1]))
        return {"chin_y": float(pts[8, 1]), "head_w": jaw_w * JAW_TO_EAR}


# --------------------------------------------------------------------------
# 图像处理
# --------------------------------------------------------------------------
def load_rgb(path: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def remove_background(rgb: Image.Image, model: str, alpha_matting: bool) -> Image.Image:
    from rembg import new_session, remove
    log(f"[-] rembg 抠图（{model}, alpha_matting={alpha_matting}）…")
    return remove(rgb, session=new_session(model), alpha_matting=alpha_matting)


def composite_white(rgba: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """RGBA 合成到纯白底，返回 (BGR, alpha)。"""
    arr = np.asarray(rgba.convert("RGBA"), dtype=np.float32)
    a = arr[..., 3:4] / 255.0
    out = arr[..., :3] * a + np.full(arr[..., :3].shape, 255.0, np.float32) * (1.0 - a)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR), arr[..., 3]


def whiten_background(bgr: np.ndarray, tol: int = 14,
                      barrier: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """从画面边缘泛洪，把与背景连成一片的近白像素压成纯 255，消除灰边 / 渐变墙。

    barrier=True 的区域（抠图主体）作为泛洪屏障，绝不被涂白。没有这道屏障时，
    浅色衣服上的高光（与白墙只差十几级灰度）会顺着连通路径被一路刷白，在成品
    胸口留下白色缺口——本图第一版就踩过这个坑。
    """
    h, w = bgr.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    if barrier is not None:
        mask[1:-1, 1:-1][barrier] = 255          # floodFill 跳过非 0 的 mask 像素
    for sx, sy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
                   (w // 2, 0), (0, h // 2), (w - 1, h // 2), (w // 2, h - 1)):
        if mask[sy + 1, sx + 1]:
            continue
        cv2.floodFill(bgr, mask, (sx, sy), BG_BGR,
                      loDiff=(tol, tol, tol), upDiff=(tol, tol, tol),
                      flags=cv2.FLOODFILL_FIXED_RANGE)
    region = mask[1:-1, 1:-1] == 1               # 泛洪写入的是 1，屏障是 255
    bgr[region] = BG_BGR
    return bgr, float(region.mean())


def place_matrix(geom: HeadGeom, eye_center: tuple[float, float], angle_deg: float,
                 scale: float, top_margin: float) -> np.ndarray:
    """旋转 + 缩放 + 平移合成一个仿射矩阵，全流程只做一次重采样。

    dst = scale · R(angle) · (src − eye_center) + target
    target 让头顶落在 y=top_margin、人脸水平居中。
    R 用 cv2.getRotationMatrix2D 的约定（正角度 = 视觉逆时针，与 PIL.rotate 一致，
    已用施加 ±2°/±3.86°/±8° 的合成旋转实测确认符号）。
    """
    cx, cy = eye_center
    th = np.deg2rad(angle_deg)
    a, b = scale * np.cos(th), scale * np.sin(th)
    tx = CANVAS_W / 2.0 - (a * cx + b * cy)
    ty = (top_margin + scale * geom.crown_off) - (-b * cx + a * cy)
    return np.array([[a, b, tx], [-b, a, ty]], dtype=np.float64)


def warp(img: np.ndarray, m: np.ndarray, scale: float, nearest: bool = False,
         border=None) -> np.ndarray:
    if border is None:
        border = BG_BGR          # 画布外补的是背景色，不是写死的白
    if nearest:
        flags = cv2.INTER_NEAREST
    else:
        flags = cv2.INTER_AREA if scale < 0.85 else cv2.INTER_CUBIC
    return cv2.warpAffine(img, m, (CANVAS_W, CANVAS_H), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def solve_scale(geom: HeadGeom, top_margin: float, target_head_h: float,
                warnings: list[str], gain: float = 1.0) -> float:
    """选缩放系数：硬约束 = 头高区间 + 头顶留白 / 下巴留白；软约束 = 耳间头宽。

    头宽（含耳廓与发量）与头高可能无法同时落在规范区间内（极限比 260/330≈0.79，
    本目录样本实测 0.78~0.83）。头宽只是构图参考、不参与自检判定：软区间能守住
    就守住（避免脸撑满画面），守不住就让位给头高，不报警。

    gain 是“成品复测 ÷ 源图推算”的头高修正系数，只作用于头高这条硬约束：
    成品复测的头宽和源图推算几乎一致（实测差 <1px），跟着一起缩放会把宽度上限
    越压越低，反而把缩放系数拽下去，闭环就永远修不回来。
    """
    hh = geom.head_h * gain
    hard_lo = (HEAD_H_MIN + HEAD_H_GUARD) / hh
    hard_hi = min((HEAD_H_MAX - HEAD_H_GUARD) / hh,
                  (CANVAS_H - top_margin - BOTTOM_GAP_MIN) / hh)
    if hard_lo > hard_hi:      # 头太长，留白放不下
        warnings.append("头高与下巴留白冲突，已取留白允许范围内的最大头部")
        hard_lo = hard_hi
    lo, hi = max(hard_lo, HEAD_W_MIN / geom.head_w), min(hard_hi, HEAD_W_MAX / geom.head_w)
    span = (lo, hi) if lo <= hi else (hard_lo, hard_hi)
    return float(min(max(target_head_h / hh, span[0]), span[1]))


def encode_jpeg(pil: Image.Image, quality: int, dpi: int) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True, dpi=(dpi, dpi))
    return buf.getvalue()


def compress_to_range(pil: Image.Image, min_kb: float, max_kb: float, dpi: int,
                      warnings: list[str]) -> tuple[bytes, int]:
    """二分找“体积 ≤ max_kb 的最高画质”；不足 min_kb 时逐步锐化补熵。"""
    from PIL import ImageFilter
    data, quality = b"", 95
    for attempt in range(4):
        img = pil if attempt == 0 else pil.filter(
            ImageFilter.UnsharpMask(radius=2, percent=40 + 40 * attempt, threshold=2))
        best: tuple[bytes, int] | None = None
        low, high = 20, 95
        while low <= high:
            mid = (low + high) // 2
            blob = encode_jpeg(img, mid, dpi)
            if len(blob) / 1024.0 <= max_kb:
                best = (blob, mid)
                low = mid + 1
            else:
                high = mid - 1
        data, quality = best or (encode_jpeg(img, 20, dpi), 20)
        kb = len(data) / 1024.0
        if kb >= min_kb:
            if kb < min_kb * 1.08:
                warnings.append(f"体积 {kb:.1f} KB 贴近下限 {min_kb:.0f} KB")
            return data, quality
        warnings.append(f"q=95 仅 {kb:.1f} KB（<{min_kb:.0f} KB），第 {attempt + 1} 次锐化补熵")
    return data, quality


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def disp_width(text: str) -> int:
    """终端显示宽度：CJK 全角字符按 2 计。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - disp_width(text))


def print_report(rows: list[tuple[str, str, str, bool | str | None]]) -> bool:
    """ok: True=PASS False=FAIL "warn"=WARN（不判失败）None=INFO。"""
    w1 = max([disp_width("项目")] + [disp_width(r[0]) for r in rows]) + 2
    w2 = max([disp_width("实测")] + [disp_width(r[1]) for r in rows]) + 2
    w3 = max([disp_width("要求")] + [disp_width(r[2]) for r in rows]) + 2
    line = "  " + "-" * (w1 + w2 + w3 + 8)
    log("")
    log("  " + pad("项目", w1) + pad("实测", w2) + pad("要求", w3) + "结果")
    log(line)
    ok_all = True
    for name, value, req, ok in rows:
        if ok is None:
            mark, good = "INFO", True
        elif ok is True:
            mark, good = "PASS", True
        elif ok == "warn":
            mark, good = "WARN", True
        else:
            mark, good = "FAIL", False
        ok_all = ok_all and good
        log("  " + pad(name, w1) + pad(value, w2) + pad(req, w3) + f"[{mark}]")
    log(line)
    return ok_all


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def write_json_report(path: str, rows: list, warnings: list, metrics: dict) -> None:
    """自检结果落 JSON：第四位映射成 PASS/WARN/FAIL/INFO，服务程序可直接渲染。"""
    state = {True: "PASS", False: "FAIL", "warn": "WARN", None: "INFO"}
    payload = {
        "rows": [{"name": r[0], "value": r[1], "req": r[2], "state": state[r[3]]} for r in rows],
        "warnings": list(warnings),
        "all_pass": all(r[3] in (True, None, "warn") for r in rows),
        "metrics": metrics,
    }
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)


def process(args: argparse.Namespace) -> int:
    warnings: list[str] = []
    # 规格先行：把选中的规格绑成模块常量，再把没显式指定的构图/体积参数填上规格值。
    spec = apply_spec(args.spec)
    if args.target_head_h is None:
        args.target_head_h = spec["aim_head_h"]
    if args.top_margin is None:
        args.top_margin = spec["aim_top_margin"]
    if args.min_kb is None:
        args.min_kb = spec["kb"][0]
    if args.max_kb is None:
        args.max_kb = spec["kb"][1]
    if args.dpi is None:
        args.dpi = spec["dpi"]
    log(f"[-] 规格 {args.spec}（{spec['name']}）画布 {CANVAS_W}x{CANVAS_H} "
        f"头高 {HEAD_H_MIN}-{HEAD_H_MAX} 背景 RGB{BG_RGB}")
    log(f"[-] 读取 {args.input}")
    rgb = load_rgb(args.input)
    log(f"    原始尺寸 {rgb.size[0]}x{rgb.size[1]}")

    # 1) 抠图 + 纯白底
    alpha = None
    if args.no_bg_remove:
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        warnings.append("已跳过抠图（--no-bg-remove），背景不会变成纯白")
    else:
        bgr, alpha = composite_white(remove_background(rgb, args.rembg_model,
                                                       not args.no_matting))
    mask = subject_mask(bgr, alpha)
    if alpha is None and mask.mean() > 0.9:
        # 没抠图又不在白墙前：整幅画面都被当成“主体”，剪影量出来的头顶/头宽是假的
        warnings.append("未抠图且背景不接近白色，剪影不可信，头顶与头宽改用瞳距比例推算")
        mask = None
    barrier_u8 = (alpha > 128).astype(np.uint8) * 255 if alpha is not None else None
    if args.whiten:
        bgr, frac = whiten_background(bgr, args.whiten_tol, barrier=mask)
        log(f"[-] 背景纯白化：覆盖 {frac * 100:.1f}% 画面"
            + ("（已用抠图遮罩挡住衣物高光）" if barrier_u8 is not None else ""))

    # 2) 人脸检测
    detector = build_detector(args)
    refiner = build_refiner(args)
    faces = detector.detect(bgr)
    if not faces:
        warnings.append("未检测到人脸，改用画面中心默认构图")
        h, w = bgr.shape[:2]
        face = Face(bbox=(w * 0.3, h * 0.25, w * 0.4, h * 0.45), score=0.0,
                    right_eye=(w * 0.42, h * 0.38), left_eye=(w * 0.58, h * 0.38),
                    nose=(w * 0.5, h * 0.46), mouth_right=(w * 0.44, h * 0.53),
                    mouth_left=(w * 0.56, h * 0.53), engine="fallback-center")
    else:
        face = faces[0]
        if len(faces) > 1:
            warnings.append(f"检测到 {len(faces)} 张人脸，已取最大的一张")
        log(f"[-] 人脸（{detector.name}, score={face.score:.3f}）"
            f" bbox=({face.bbox[0]:.0f},{face.bbox[1]:.0f},{face.bbox[2]:.0f},{face.bbox[3]:.0f})"
            f" 瞳距={face.iod:.1f}px")
        log(f"    关键点 右眼({face.right_eye[0]:.0f},{face.right_eye[1]:.0f}) "
            f"左眼({face.left_eye[0]:.0f},{face.left_eye[1]:.0f}) "
            f"鼻尖({face.nose[0]:.0f},{face.nose[1]:.0f}) "
            f"嘴角({face.mouth_right[0]:.0f},{face.mouth_right[1]:.0f})/"
            f"({face.mouth_left[0]:.0f},{face.mouth_left[1]:.0f})")
        if face.bbox[1] + face.bbox[3] < face.eye_center[1] + face.iod * 1.4:
            warnings.append("人脸框下沿明显高于推算下巴，请确认下巴未被裁切")
        face = refine_eyes(bgr, face)
        log(f"    瞳孔精修 右眼({face.right_eye[0]:.1f},{face.right_eye[1]:.1f}) "
            f"左眼({face.left_eye[0]:.1f},{face.left_eye[1]:.1f}) "
            f"瞳距={face.iod:.1f}px 眼线={face.roll_angle:.2f}° 嘴角线={face.mouth_roll:.2f}°")
        if abs(face.roll_angle - face.mouth_roll) > 3.0:
            warnings.append(f"眼线 {face.roll_angle:.1f}° 与嘴角线 {face.mouth_roll:.1f}° 不一致，"
                            f"已按瞳孔连线校正，请人工确认水平")

    refine_info = refiner.refine(bgr, face) if (refiner and face.engine == "yunet") else None
    if refine_info:
        log("[-] FacemarkLBF 68 点精修：下巴 y=%.0f（下颌宽 %.0fpx，仅作参考——规范量的是两耳之间，"
            "头宽仍按剪影取）" % (refine_info["chin_y"], refine_info["head_w"]))
    elif refiner:
        warnings.append("FacemarkLBF 精修失败，改用瞳距比例推算下巴")

    def geom_of(f: Face) -> HeadGeom:
        # 只用 LBF 的下巴点：它是真实 landmark，比瞳距比例准。
        # 头宽不用 LBF 的下颌宽覆盖——规范写的是“两耳之间”，下颌宽不含耳朵，
        # 两者混用会让源图和成品的宽度量法对不上。
        extra = refine_info if (refine_info and f is face) else {}
        return head_geometry(f, mask, args.chin_iod_ratio, args.crown_ratio,
                             args.ear_ratio, chin_override=extra.get("chin_y"))

    geom = geom_of(face)
    angle = face.roll_angle if args.rotate else 0.0
    if abs(angle) > args.max_roll:
        warnings.append(f"倾斜角 {angle:.1f}° 超过阈值 {args.max_roll}°，不做旋转校正")
        angle = 0.0
    log(f"[-] 头廓  头顶y={geom.crown_y:.0f}({geom.crown_src}) "
        f"眼线y={geom.eye_y:.0f} 下巴y={geom.chin_y:.0f} "
        f"头高={geom.head_h:.0f}px 耳间宽={geom.head_w:.0f}px({geom.width_src}) "
        f"倾斜={face.roll_angle:.2f}°")

    # 3) 构图：缩放 + 旋转一次仿射；按成品残差再校正一次倾斜
    ref_frac = face.bbox[3] / float(bgr.shape[0])   # 源图里人脸占画面高度的比例
    canvas = None
    canvas_mask: np.ndarray | None = None
    geom_out: HeadGeom | None = None
    face_out: Face | None = None
    scale = 1.0
    gain = 1.0     # 源图量法 → 成品量法 的实测修正系数（检测噪声 + 重采样，约 1~2%）
    best: tuple | None = None    # (头高偏差, canvas, mask, geom_out, face_out, scale)
    for attempt in range(4):
        scale = solve_scale(geom, args.top_margin, args.target_head_h, warnings, gain)
        m = place_matrix(geom, face.eye_center, angle, scale, args.top_margin)
        warped = warp(bgr, m, scale)
        canvas_mask = warp(barrier_u8, m, scale, nearest=True) > 127 \
            if barrier_u8 is not None else None
        # 头廓必须在提亮背景之前量：whiten 会把发际边缘的浅色头发并进白背景，
        # 头顶被量低十几像素，头高就永远够不到 330 下限。几何量测用未提亮的
        # warped，交付用的仍是提亮后的 canvas——两者像素位置完全一致。
        geom_mask = canvas_mask if canvas_mask is not None else subject_mask(warped)
        canvas = warped
        if args.whiten:
            canvas, _ = whiten_background(warped, args.whiten_tol, barrier=canvas_mask)
        geom_out, face_out = measure_output(detector, warped, args, ref_frac, geom_mask)
        if geom_out is None or face_out is None:
            warnings.append("成品中未再次检测到人脸，无法自检构图")
            break
        # 永远保留头高最接近规范的一版：个别照片的成品量法会反向偏移，
        # 不锁最优就可能越修越糟。
        h_px = round(geom_out.head_h)
        err = 0.0 if HEAD_H_MIN <= h_px <= HEAD_H_MAX else \
            min(abs(h_px - HEAD_H_MIN), abs(h_px - HEAD_H_MAX))
        if best is None or err < best[0]:
            best = (err, canvas, canvas_mask, geom_out, face_out, scale)
        residual = face_out.roll_angle
        if attempt == 0 and abs(residual) > 1.5:
            log(f"[-] 成品残差倾斜 {residual:.2f}°，追加校正后重建")
            angle += residual
            continue
        # 闭环校正：bias = 成品复测 ÷ 源图推算。bias<1 说明同一张脸在成品里量得
        # 更小（人脸框随占比变化、重采样、发际被提亮吃掉），那就得把缩放**放大**——
        # 是除以 bias 不是乘，方向搞反就会越修越小（之前就是这样翻车的）。
        if err > 0 and attempt < 3:
            bias = geom_out.head_h / max(1.0, geom.head_h * scale)
            if 0.6 <= bias <= 1.6 and abs(bias - gain) > 0.002:
                log(f"[-] 成品头高 {geom_out.head_h:.0f}px 不在 {HEAD_H_MIN}-{HEAD_H_MAX}，"
                    f"实测/推算 = {bias:.3f}，按此回修缩放后重建")
                gain = bias
                continue
        log(f"[-] 缩放系数 {scale:.4f}（目标头高 {args.target_head_h:.0f}px）"
            f" 成品残差倾斜 {residual:.2f}°")
        break

    if best is not None and best[1] is not canvas:
        _, canvas, canvas_mask, geom_out, face_out, scale = best
        warnings.append("多次重建后取头高最接近规范的一版")
        log(f"[-] 取最优一版：成品头高 {geom_out.head_h:.0f}px")

    assert canvas is not None
    kb = 0.0
    quality = 0
    data = b""
    if args.write:
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        data, quality = compress_to_range(pil, args.min_kb, args.max_kb, args.dpi, warnings)
        parent = os.path.dirname(os.path.abspath(args.output))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output, "wb") as f:
            f.write(data)
        kb = len(data) / 1024.0

    # 4) 规格自检（与源图同一套量法，数字才有可比性）
    rows: list[tuple[str, str, str, bool | str | None]] = [
        ("画布尺寸", f"{CANVAS_W} x {CANVAS_H} px", f"{CANVAS_W} x {CANVAS_H} px", True)]
    if geom_out is not None and face_out is not None:
        top_margin = geom_out.eye_y - geom_out.crown_off
        bottom_gap = CANVAS_H - top_margin - geom_out.head_h
        # 判定和显示都用四舍五入后的整数：显示 260 却判 FAIL（实际 260.4）会让人
        # 完全看不懂；人工量照片读的也是整数像素。
        h_px, w_px = round(geom_out.head_h), round(geom_out.head_w)
        tm_px, bg_px = round(top_margin), round(bottom_gap)
        rows += [
            ("头部高度", f"{h_px} px", f"{HEAD_H_MIN}-{HEAD_H_MAX} px",
             HEAD_H_MIN <= h_px <= HEAD_H_MAX),
            ("头顶留白", f"{tm_px} px", f"{TOP_MARGIN_MIN}-{TOP_MARGIN_MAX} px",
             TOP_MARGIN_MIN <= tm_px <= TOP_MARGIN_MAX),
            ("下巴距下边缘", f"{bg_px} px", f"≥{BOTTOM_GAP_MIN} px",
             bg_px >= BOTTOM_GAP_MIN),
            ("人脸水平", f"{face_out.roll_angle:.2f}°", "≤1.5°",
             abs(face_out.roll_angle) <= 1.5),
        ]
    else:
        rows.append(("头部测量", "成品内未检出人脸", "—", False))

    pure = float((canvas == np.array(BG_BGR)).all(axis=2).mean())
    rows.append(("背景", f"纯色 {pure * 100:.1f}% 画面", f"RGB{BG_RGB}", pure > 0.15))
    if args.write:
        rows.append(("文件格式", f"JPEG q={quality} dpi={args.dpi}", "24bit JPEG", True))
        rows.append(("文件体积", f"{kb:.1f} KB", f"{args.min_kb:.0f}-{args.max_kb:.0f} KB",
                     args.min_kb <= kb <= args.max_kb))
    rows.append(("检测引擎", detector.name + ("+LBF" if refine_info else ""), "YuNet", None))

    ok = print_report(rows)
    if args.json_report:
        metrics: dict = {"engine": detector.name + ("+LBF" if refine_info else ""),
                         "kb": round(kb, 1), "quality": quality}
        if geom_out is not None and face_out is not None:
            metrics.update(head_h=round(geom_out.head_h, 1), head_w=round(geom_out.head_w, 1),
                           top_margin=round(top_margin, 1), bottom_gap=round(bottom_gap, 1),
                           residual_roll=round(face_out.roll_angle, 2), scale=round(scale, 5))
        write_json_report(args.json_report, rows, warnings, metrics)
    for wmsg in warnings:
        log(f"  ! {wmsg}")
    if args.write:
        log(f"\n[+] 已保存 {args.output}  ({kb:.1f} KB, {CANVAS_W}x{CANVAS_H})")

    if args.debug:
        dbg = args.debug if args.debug is not True else \
            os.path.splitext(args.output)[0] + ".debug.png"
        save_debug(bgr, face, geom, canvas, geom_out, dbg)
        log(f"[+] 调试图 {dbg}")

    return 0 if (ok or not args.strict) else 2


def measure_output(detector, canvas: np.ndarray, args: argparse.Namespace,
                   ref_frac: float | None = None, mask: np.ndarray | None = None):
    """在成品上复测头廓。

    YuNet 人脸框的形状会随“人脸占画面比例”变化，直接量成品与源图口径不一致
    （源图里人脸占 31% 画面，成品里占 72%）。先把成品四周补白到与源图相同的
    占比再检测，坐标减偏移映射回画布，两次测量才可比。
    """
    faces = detector.detect(canvas)
    if not faces:
        return None, None
    face, dx, dy = faces[0], 0.0, 0.0
    if ref_frac and ref_frac > 0:
        want_h = face.bbox[3] / ref_frac
        if want_h > canvas.shape[0] * 1.05:
            extra_h = want_h - canvas.shape[0]
            extra_w = extra_h * canvas.shape[1] / canvas.shape[0]
            dy, dx = extra_h / 2.0, extra_w / 2.0
            big = cv2.copyMakeBorder(canvas, int(dy), int(extra_h - dy), int(dx),
                                     int(extra_w - dx), cv2.BORDER_CONSTANT,
                                     value=BG_BGR)
            again = detector.detect(big)
            if again:
                face = _unpad(again[0], dx, dy)
    face = refine_eyes(canvas, face)
    geom = head_geometry(face, mask if mask is not None else subject_mask(canvas),
                         args.chin_iod_ratio, args.crown_ratio, args.ear_ratio)
    return geom, face

def _unpad(face: Face, dx: float, dy: float) -> Face:
    """把补白后检测到的坐标平移回画布坐标。"""
    def mv(p):
        return (p[0] - dx, p[1] - dy)
    return Face(bbox=(face.bbox[0] - dx, face.bbox[1] - dy, face.bbox[2], face.bbox[3]),
                score=face.score, right_eye=mv(face.right_eye), left_eye=mv(face.left_eye),
                nose=mv(face.nose), mouth_right=mv(face.mouth_right),
                mouth_left=mv(face.mouth_left), engine=face.engine)


def build_detector(args: argparse.Namespace):
    model = args.model or os.path.join("models", YUNET_NAME)
    ok = ensure_model(model, YUNET_URLS, YUNET_SHA256,
                      YUNET_BYTES if model.endswith(YUNET_NAME) else None,
                      not args.no_download)
    if ok:
        try:
            det = YuNetDetector(model, args.score_thr, args.nms_thr, 5000, args.detect_long)
            log(f"[-] 人脸引擎：YuNet ({os.path.basename(model)}, 检测长边 {args.detect_long})")
            return det
        except cv2.error as exc:
            log(f"[!] YuNet 初始化失败：{exc}")
    log("[!] 回退到 Haar 正脸级联 + 比例推算眼位（精度较低）")
    return HaarFallbackDetector()


def build_refiner(args: argparse.Namespace):
    if args.refine == "off":
        return None
    path = args.lbf_model or os.path.join("models", "lbfmodel.yaml")
    if args.download_lbf:
        ensure_model(path, LBF_URLS, None, None, True)
    if not os.path.isfile(path):
        if args.refine == "lbf":
            log(f"[!] 找不到 FacemarkLBF 模型：{path}（--download-lbf 可自动下载约 54MB）")
        return None
    try:
        log(f"[-] 关键点精修：FacemarkLBF ({os.path.basename(path)})")
        return LbfRefiner(path)
    except Exception as exc:  # noqa: BLE001
        log(f"[!] FacemarkLBF 初始化失败：{exc}")
        return None


def save_debug(src_bgr: np.ndarray, face: Face, geom: HeadGeom,
               canvas: np.ndarray, geom_out: HeadGeom | None, path: str) -> None:
    """左：原图关键点与推算头廓；右：成品上的实测构图线。"""
    h, w = src_bgr.shape[:2]
    k = CANVAS_H / h
    left = cv2.resize(src_bgr, (max(1, int(round(w * k))), CANVAS_H),
                      interpolation=cv2.INTER_AREA)
    draw_face_marks(left, face, k)
    draw_head_lines(left, geom, k)
    caption(left, f"src  head {geom.head_h:.0f}px  ear {geom.head_w:.0f}px  roll {geom.roll:.2f}deg")

    right = canvas.copy()
    if geom_out is not None:
        tm = geom_out.eye_y - geom_out.crown_off
        draw_head_lines(right, geom_out, crown_y=tm)
        cx, hw = CANVAS_W / 2.0, geom_out.head_w / 2.0
        cv2.rectangle(right, (int(cx - hw), int(tm)),
                      (int(cx + hw), int(tm + geom_out.head_h)), COLOR_BOX, 1)
        caption(right, f"out  head {geom_out.head_h:.0f}px  ear {geom_out.head_w:.0f}px"
                       f"  margin {tm:.0f}px")
    sep = np.full((CANVAS_H, 10, 3), 40, np.uint8)
    cv2.imwrite(path, np.hstack([left, sep, right]))


def caption(img: np.ndarray, text: str) -> np.ndarray:
    """左上角加一行带底色的说明文字（cv2.putText 不支持 CJK，用 ASCII）。"""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(img, (0, 0), (tw + 10, th + 10), (255, 255, 255), -1)
    cv2.putText(img, text, (5, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="中国护照 / “中国领事”App 电子证件照处理（YuNet 关键点版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="输入照片")
    p.add_argument("output", help="输出 JPEG")
    p.add_argument("--spec", default=DEFAULT_SPEC, choices=sorted(SPECS),
                   help="证件类型（决定画布/头高/留白/背景/体积；单项 flag 可覆盖）")
    p.add_argument("--model", help=f"YuNet onnx 路径（默认 models/{YUNET_NAME}，缺失时自动下载）")
    p.add_argument("--no-download", action="store_true", help="禁止联网下载模型")
    p.add_argument("--score-thr", type=float, default=0.6, help="YuNet 置信度阈值")
    p.add_argument("--nms-thr", type=float, default=0.3, help="YuNet NMS 阈值")
    p.add_argument("--detect-long", type=int, default=1024,
                   help="人脸检测时的图像长边（YuNet 在小图上更准，480~1280 为宜）")
    p.add_argument("--refine", choices=["off", "auto", "lbf"], default="auto",
                   help="FacemarkLBF 68 点精修（auto=本地已有模型才用）")
    p.add_argument("--lbf-model", help="lbfmodel.yaml 路径")
    p.add_argument("--download-lbf", action="store_true", help="允许下载约 54MB 的 lbfmodel.yaml")
    p.add_argument("--rembg-model", default="isnet-general-use", help="rembg 抠图模型")
    p.add_argument("--no-matting", action="store_true", help="关闭 alpha matting（更快，发丝边缘略硬）")
    p.add_argument("--no-bg-remove", action="store_true", help="跳过抠图（仅做几何裁切）")
    p.add_argument("--no-whiten", dest="whiten", action="store_false", help="关闭背景纯白化")
    p.add_argument("--whiten-tol", type=int, default=14, help="纯白化泛洪容差")
    p.add_argument("--no-rotate", dest="rotate", action="store_false", help="关闭水平校正")
    p.add_argument("--max-roll", type=float, default=15.0, help="超过该倾斜角就不校正（度）")
    # 规范给的 330-390 在这张画布上取不到上限：472 = 头顶留白(≥35) + 头高 +
    # 下巴留白(≥82)，头高最多 472-35-82 = 355。可行区间 [330, 355]。
    # 目标取偏上的 346：成品复测比源图量法低 1~5%（发际被提亮、人脸框随占比
    # 变化），贴着下限选就会掉到 328/329。
    p.add_argument("--target-head-h", type=float, default=None,
                   help="目标头部高度 px（默认取规格表里给复测噪声留了余量的值）")
    p.add_argument("--top-margin", type=float, default=None,
                   help="目标头顶留白 px（默认取规格表值；取小值给头高和下巴留白腾空间）")
    p.add_argument("--chin-iod-ratio", type=float, default=CHIN_IOD_RATIO,
                   help="(下巴-眼线)/瞳距")
    p.add_argument("--crown-ratio", type=float, default=CROWN_RATIO,
                   help="兜底比例：(头顶-眼线)/(下巴-眼线)")
    p.add_argument("--ear-ratio", type=float, default=EAR_RATIO, help="兜底比例：耳间宽/人脸框宽")
    p.add_argument("--min-kb", type=float, default=None, help="目标体积下限（默认取规格表）")
    p.add_argument("--max-kb", type=float, default=None, help="目标体积上限（默认取规格表）")
    p.add_argument("--dpi", type=int, default=None, help="写入 JPEG 的 DPI（默认取规格表）")
    p.add_argument("--no-write", dest="write", action="store_false", help="只分析不写文件")
    p.add_argument("--debug", nargs="?", const=True, default=False, help="输出调试叠加图（可跟路径）")
    p.add_argument("--json-report", help="把自检结果写成 JSON（供服务程序读取，不影响终端输出）")
    p.add_argument("--strict", action="store_true", help="自检不通过时以非零码退出")
    p.set_defaults(whiten=True, rotate=True, write=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(process(parse_args()))

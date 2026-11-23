#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""process_passport.py 的标定 / 回归测试。

    venv/bin/python3 test_geometry.py [样本照片] [合成图 png] [alpha npy]

纯逻辑用例（缩放约束、剪影测量、泛洪屏障）始终执行；依赖样本图 / 模型的用例在
缺文件时打印 SKIP，不会让测试挂掉。数值断言里的常量是在样本照片
IMG_20261121_161043.jpg（3072x4096，男孩，刘海，头略歪 4.2°）上实测标定的结果。
"""

from __future__ import annotations

import os
import sys
import traceback

import cv2
import numpy as np

import process_passport as pp

SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "IMG_20261121_161043.jpg"
COMPOSITED = sys.argv[2] if len(sys.argv) > 2 else "/tmp/composited.png"
ALPHA = sys.argv[3] if len(sys.argv) > 3 else "/tmp/alpha.npy"
MODEL = os.path.join("models", pp.YUNET_NAME)

RESULTS: list[tuple[str, str, str]] = []


class Skip(Exception):
    """依赖缺失（样本图 / 模型），不算失败。"""


def case(name: str):
    def deco(fn):
        try:
            msg = fn()
            RESULTS.append((name, "PASS", msg or ""))
        except Skip as exc:
            RESULTS.append((name, "SKIP", str(exc)))
        except AssertionError as exc:
            RESULTS.append((name, "FAIL", str(exc)))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, "ERROR", traceback.format_exc(limit=3)))
        return fn
    return deco


def yUNET_or_none():
    if not os.path.isfile(MODEL):
        return None
    try:
        return pp.YuNetDetector(MODEL, 0.6, 0.3, 5000, 1024)
    except Exception:  # noqa: BLE001
        return None


def rotate_about(img: np.ndarray, ang: float, c: tuple[float, float]) -> np.ndarray:
    """绕 c 旋转（正角度 = 视觉逆时针），与 place_matrix 同一套约定。"""
    a, b = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    cx, cy = c
    return cv2.warpAffine(img, np.array([[a, b, cx - a * cx - b * cy],
                                         [-b, a, cy + b * cx - a * cy]]),
                          (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC,
                          borderValue=(255, 255, 255))


def _need_face():
    """返回 (检测器, 样本 BGR)；缺依赖时抛断言，由 case 记为 SKIP 原因。"""
    det = yUNET_or_none()
    if det is None:
        raise Skip(f"缺少 YuNet 模型 {MODEL}")
    if not os.path.isfile(SAMPLE):
        raise Skip(f"缺少样本图 {SAMPLE}")
    return det, cv2.cvtColor(np.asarray(pp.load_rgb(SAMPLE)), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- 纯逻辑用例
@case("solve_scale 正常可行域")
def _t_scale_ok():
    g = pp.HeadGeom(crown_y=0, eye_y=800, chin_y=1600, head_h=1600, head_w=1000,
                    crown_off=800, chin_off=800, iod=450, roll=0.0)
    s = pp.solve_scale(g, 45, 340, [])
    assert 330 <= g.head_h * s <= 390, f"头高 {g.head_h * s:.0f} 越界"
    assert 177 <= g.head_w * s <= 260, f"头宽 {g.head_w * s:.0f} 越界"
    assert 472 - 45 - g.head_h * s >= pp.BOTTOM_GAP_MIN, "下巴留白不足"


@case("solve_scale 头宽超限时优先保头高")
def _t_scale_conflict():
    g = pp.HeadGeom(crown_y=0, eye_y=800, chin_y=1600, head_h=1600, head_w=1400,
                    crown_off=800, chin_off=800, iod=450, roll=0.0)
    warns: list[str] = []
    s = pp.solve_scale(g, 45, 340, warns)
    assert g.head_h * s >= pp.HEAD_H_MIN - 1e-6, \
        f"头宽冲突时把头缩到了 {g.head_h * s:.0f}px，低于下限"
    assert not warns, f"头宽只是软约束，守不住就该安静让位，不该报警：{warns}"


@case("solve_scale 留白夹住上限")
def _t_scale_gap():
    g = pp.HeadGeom(crown_y=0, eye_y=800, chin_y=1600, head_h=1600, head_w=1000,
                    crown_off=800, chin_off=800, iod=450, roll=0.0)
    s = pp.solve_scale(g, 45, 390, [])          # 想要 390，但留白只允许 345
    assert 472 - 45 - g.head_h * s >= pp.BOTTOM_GAP_MIN - 1e-6, "留白被突破"
    assert g.head_h * s <= 345 + 1e-6, f"缩放未被留白夹住：{g.head_h * s:.0f}"


@case("silhouette_top 连续成片才算头顶")
def _t_top_run():
    m = np.zeros((200, 200), bool)
    m[0:2, 90:110] = True          # 飘在顶上的噪声，撑不过 run 行
    m[60:200, 40:160] = True       # 真正的头
    top = pp.silhouette_top(m, 100, 80, 0, 190, min_frac=0.06, run=16)
    assert top == 60, f"把噪声当成了头顶：{top}"


@case("silhouette_top 头顶贴边判不可信")
def _t_top_edge():
    m = np.zeros((200, 200), bool)
    m[0:200, 40:160] = True        # 头被画面顶边裁掉
    assert pp.silhouette_top(m, 100, 80, 0, 190) is None, "贴边应返回 None 走比例兜底"


@case("silhouette_width 取跨度分位数")
def _t_width():
    m = np.zeros((100, 200), bool)
    m[10:90, 50:110] = True
    assert pp.silhouette_width(m, 80, 60, 10, 90) == 60, "宽度量错"


@case("whiten_background 抠图遮罩挡住浅色衣物")
def _t_whiten_barrier():
    img = np.full((60, 60, 3), 250, np.uint8)     # 近白背景墙
    img[20:40, 25:35] = 245                        # 与墙只差 5 级的浅色衣服
    barrier = np.zeros((60, 60), bool)
    barrier[20:40, 25:35] = True
    kept, _ = pp.whiten_background(img.copy(), 14, barrier)
    assert (kept[20:40, 25:35] == 245).all(), "浅色衣物被泛洪刷白了"
    assert (kept[0, 0] == 255).all(), "背景没被纯白化"
    leaked, _ = pp.whiten_background(img.copy(), 14, None)
    assert (leaked[20:40, 25:35] == 255).all(), "无遮罩时应复现旧行为（会被吞）"


@case("head_geometry 剪影缺失时走比例兜底")
def _t_geom_fallback():
    f = pp.Face(bbox=(100, 100, 400, 500), score=1.0, right_eye=(200, 250),
                left_eye=(400, 250), nose=(300, 330), mouth_right=(240, 400),
                mouth_left=(360, 400))
    g = pp.head_geometry(f, None)
    assert abs(g.chin_off - f.iod * pp.CHIN_IOD_RATIO) < 1e-6
    assert abs(g.crown_off - g.chin_off * pp.CROWN_RATIO) < 1e-6
    assert g.crown_src == "ratio" and g.head_h > 0


@case("模型 sha256 校验（防 git-lfs 指针）")
def _t_model():
    if not os.path.isfile(MODEL):
        raise Skip(f"缺少 {MODEL}")
    assert os.path.getsize(MODEL) == pp.YUNET_BYTES, "模型大小不符（可能是 LFS 指针）"
    assert pp.sha256_of(MODEL) == pp.YUNET_SHA256, "sha256 不匹配"


# ---------------------------------------------------------------- 样本相关用例
@case("旋转符号约定（施加角 = -复测角）")
def _t_rot_sign():
    det, img = _need_face()
    f = det.detect(img)[0]
    base = pp.refine_eyes(img, f).roll_angle
    for applied in (-4.0, -2.0, 2.0, 4.0):
        est = pp.refine_eyes(rotate_about(img, applied, f.eye_center),
                             det.detect(rotate_about(img, applied, f.eye_center))[0]).roll_angle
        assert abs(est - (base - applied)) < 0.6, \
            f"施加 {applied:+.1f}° 后复测 {est:+.2f}°，期望 {base - applied:+.2f}°"


@case("瞳孔精修跨尺度稳定（误差恒定）")
def _t_pupil_stable():
    det, img = _need_face()
    f = det.detect(img)[0]
    errs = []
    for applied in (-4, -2, -1, 0, 1, 2, 4, 5):
        r = rotate_about(img, applied, f.eye_center)
        est = pp.refine_eyes(r, det.detect(r)[0]).roll_angle
        errs.append(est - (f.roll_angle - applied))
    spread = max(errs) - min(errs)
    assert spread < 0.4, f"精修后误差漂移 {spread:.2f}°（应 <0.4°）"
    assert max(abs(e) for e in errs) < 1.0, f"精修后最大误差 {max(abs(e) for e in errs):.2f}°"


@case("头廓实测值（头顶/下巴/头高/耳间宽）")
def _t_geom_values():
    det = yUNET_or_none()
    if det is None or not (os.path.isfile(COMPOSITED) and os.path.isfile(ALPHA)):
        raise Skip("需要 YuNet 模型 + 抠图合成图 + alpha（先跑一次主脚本）")
    img = cv2.imread(COMPOSITED)
    mask = np.load(ALPHA) > 128
    f = pp.refine_eyes(img, det.detect(img)[0])
    g = pp.head_geometry(f, mask)
    # 样本图人工标定：头顶≈772 下巴≈2336 头高≈1564 耳间宽≈1283
    assert abs(g.crown_y - 772) < 25, f"头顶 {g.crown_y:.0f} 偏标称 772"
    assert abs(g.chin_y - 2336) < 35, f"下巴 {g.chin_y:.0f} 偏标称 2336"
    assert abs(g.head_h - 1564) < 45, f"头高 {g.head_h:.0f} 偏标称 1564"
    assert abs(g.head_w - 1283) < 45, f"耳间宽 {g.head_w:.0f} 偏标称 1283"
    assert g.crown_src == "silhouette" and g.width_src == "silhouette"


@case("源图与成品用同一套量法（缩放一致性）")
def _t_consistency():
    det = yUNET_or_none()
    if det is None or not os.path.isfile(COMPOSITED):
        raise Skip("需要 YuNet 模型 + 抠图合成图")
    img = cv2.imread(COMPOSITED)
    mask = np.load(ALPHA) > 128 if os.path.isfile(ALPHA) else pp.subject_mask(img)
    f = pp.refine_eyes(img, det.detect(img)[0])
    g = pp.head_geometry(f, mask)
    args = pp.parse_args(["x", "y"])
    s = pp.solve_scale(g, args.top_margin, args.target_head_h, [])
    m = pp.place_matrix(g, f.eye_center, 0.0, s, args.top_margin)
    canvas = pp.warp(img, m, s)
    barrier = (np.load(ALPHA) > 128).astype(np.uint8) * 255 if os.path.isfile(ALPHA) else None
    cbar = pp.warp(barrier, m, s, nearest=True) > 127 if barrier is not None else None
    go, _ = pp.measure_output(det, canvas, args, f.bbox[3] / img.shape[0], cbar)
    drift = abs(go.head_h - g.head_h * s) / (g.head_h * s)
    assert drift < 0.05, f"成品复测头高与源图推算差 {drift * 100:.1f}%（量法不一致）"


# ---------------------------------------------------------------- 运行器
if __name__ == "__main__":
    bad = sum(1 for r in RESULTS if r[1] in ("FAIL", "ERROR"))
    skipped = sum(1 for r in RESULTS if r[1] == "SKIP")
    passed = sum(1 for r in RESULTS if r[1] == "PASS")
    for name, status, msg in RESULTS:
        mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "-", "ERROR": "!"}[status]
        print(f" {mark} {name}" + (f"  —— {msg}" if msg and status != "PASS" else ""))
    print(f"\n{passed} passed, {skipped} skipped, {bad} failed")
    sys.exit(1 if bad else 0)

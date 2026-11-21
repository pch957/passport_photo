#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""证件照规格复核 / 头廓可视化（独立脚本，process_passport.py 的调试图同款）。

给一张人像照（通常是处理好的 354×472 证件照，也可以是原图），量出人脸框、
5 个关键点、头顶 / 眼线 / 下巴三条线并画在图上，同时打印规格自检表。
加 --ref 再并排一张参考图，就是 process_passport.py --debug 出来的那种左右对照。

    venv/bin/python3 make_debug.py passport_output.jpg
    venv/bin/python3 make_debug.py out.jpg check.png --ref IMG_20261121_161043.jpg --rembg
    venv/bin/python3 make_debug.py 影楼给的.jpg --no-spec        # 只想看量出来的数

与 process_passport.py 共用同一套量法（瞳孔亚像素精修 + 剪影头顶 + 瞳距比例下巴 +
耳廓带跨度分位数），所以这里的数字和主脚本自检表可以直接对账。

两点差异要心里有数：这里量的是磁盘上的 JPEG，主脚本自检量的是压缩前的内存画布，
差 ±3px 属正常；背景纯白一项这里按 ≥250 判定（JPEG 有损压缩会让边缘像素偏离 255）。
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

import process_passport as pp

# 只暴露复核常用的开关，其余参数继承主脚本默认值，避免两处定义漂移
FORWARDED = ("--model --no-download --score-thr --nms-thr --detect-long --refine "
             "--lbf-model --download-lbf --chin-iod-ratio --crown-ratio --ear-ratio "
             "--rembg-model --no-matting").split()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    base = pp.parse_args(["_", "_"])          # 拿主脚本的全部默认值
    p = argparse.ArgumentParser(
        description="证件照头廓可视化与规格复核（与 process_passport.py 同一套量法）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("photo", help="要复核的照片（一般是 354x472 成品）")
    p.add_argument("out", nargs="?", help="输出 PNG，默认 <photo>.debug.png")
    p.add_argument("--ref", help="并排显示的参考图（一般是原图）")
    p.add_argument("--rembg", action="store_true", help="先抠图再量剪影（照片不是白底时用）")
    p.add_argument("--ref-rembg", action="store_true", help="参考图先抠图（原图通常不是白底）")
    p.add_argument("--no-spec", dest="spec", action="store_false", help="只画线，不打印规格表")
    for flag in FORWARDED:
        name = flag[2:].replace("-", "_")
        default = getattr(base, name)
        if isinstance(default, bool):
            p.add_argument(flag, dest=name, action="store_true", default=False)
        else:
            p.add_argument(flag, dest=name, type=type(default) if default is not None else str,
                           default=default)
    args = p.parse_args(argv)
    for flag in FORWARDED:                    # store_true 的开关没传时回落到主脚本默认
        name = flag[2:].replace("-", "_")
        if getattr(args, name) is False and isinstance(getattr(base, name), bool):
            setattr(args, name, getattr(base, name))
    args.out = args.out or os.path.splitext(args.photo)[0] + ".debug.png"
    return args


def measure(path: str, args: argparse.Namespace, rembg: bool):
    """读图 → 检测 → 精修 → 头廓。返回 (BGR, Face, HeadGeom)，失败返回 None。"""
    rgb = pp.load_rgb(path)
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    if rembg:
        bgr, alpha = pp.composite_white(
            pp.remove_background(rgb, args.rembg_model, not args.no_matting))
        mask = pp.subject_mask(bgr, alpha)
    else:
        mask = pp.subject_mask(bgr)
        if mask.mean() > 0.9:                 # 背景不接近白，剪影量不出头顶
            mask = None
            pp.log("    [!] 背景不是白色，头顶/头宽退回瞳距比例（加 --rembg 可强制抠图）")

    # 小图先放大再检测：YuNet 在长边 480~1280 上最稳，量完再除回去
    k = 1.0
    if max(bgr.shape[:2]) < 640:
        k = 1024.0 / max(bgr.shape[:2])
        big = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        big_mask = (cv2.resize(mask.astype(np.uint8) * 255, None, fx=k, fy=k,
                               interpolation=cv2.INTER_NEAREST) > 127) if mask is not None else None
    else:
        big, big_mask = bgr, mask

    detector = pp.build_detector(args)
    faces = detector.detect(big)
    if not faces:
        pp.log(f"[!] {path} 里没检测到人脸")
        return None
    face = pp.refine_eyes(big, faces[0])
    geom = pp.head_geometry(face, big_mask, args.chin_iod_ratio,
                            args.crown_ratio, args.ear_ratio)
    pp.log(f"[-] {os.path.basename(path)}  score={face.score:.3f} 瞳距={face.iod / k:.1f}px "
           f"倾斜={face.roll_angle:.2f}° 头顶={geom.crown_src} 头宽={geom.width_src}")
    return bgr, face.scaled(1.0 / k), geom.scaled(1.0 / k)


def panel(bgr: np.ndarray, face: pp.Face, geom: pp.HeadGeom, label: str,
          height: int = pp.CANVAS_H) -> np.ndarray:
    """缩放到统一高度并画线：人脸框、关键点、头顶/眼线/下巴、耳间宽两条竖线。"""
    h, w = bgr.shape[:2]
    k = height / h
    img = cv2.resize(bgr, (max(1, int(round(w * k))), height), interpolation=cv2.INTER_AREA)
    pp.draw_face_marks(img, face, k)
    pp.draw_head_lines(img, geom, k)
    cx = face.eye_center[0] * k
    half = geom.head_w * k / 2.0
    for dx in (-half, half):
        cv2.line(img, (int(cx + dx), int(geom.crown_y * k)),
                 (int(cx + dx), int(geom.chin_y * k)), pp.COLOR_BOX, 1)
    return pp.caption(img, label)


def spec_rows(bgr: np.ndarray, geom: pp.HeadGeom, face: pp.Face):
    h, w = bgr.shape[:2]
    top = geom.eye_y - geom.crown_off
    gap = h - top - geom.head_h
    w_ok = pp.HEAD_W_MIN <= geom.head_w <= pp.HEAD_W_MAX
    # 与主脚本同一取舍：宽度超限若是“缩放无解”，记 WARN 而不是 FAIL
    w_state: bool | str = "warn" if (not w_ok and pp.width_unavoidable(geom)) else w_ok
    pure = float((bgr == 255).all(axis=2).mean())
    near = float((bgr >= 250).all(axis=2).mean())
    rows = [(
        "画布尺寸", f"{w} x {h} px", f"{pp.CANVAS_W} x {pp.CANVAS_H} px",
        (w, h) == (pp.CANVAS_W, pp.CANVAS_H)), (
        "头部高度", f"{geom.head_h:.0f} px", f"{pp.HEAD_H_MIN}-{pp.HEAD_H_MAX} px",
        pp.HEAD_H_MIN <= geom.head_h <= pp.HEAD_H_MAX), (
        "头部宽度(含耳)", f"{geom.head_w:.0f} px", f"{pp.HEAD_W_MIN}-{pp.HEAD_W_MAX} px",
        w_state), (
        "头顶留白", f"{top:.0f} px", f"{pp.TOP_MARGIN_MIN}-{pp.TOP_MARGIN_MAX} px",
        pp.TOP_MARGIN_MIN <= top <= pp.TOP_MARGIN_MAX), (
        "下巴距下边缘", f"{gap:.0f} px", f"≥{pp.BOTTOM_GAP_MIN} px", gap >= pp.BOTTOM_GAP_MIN), (
        "人脸水平", f"{face.roll_angle:.2f}°", "≤1.5°", abs(face.roll_angle) <= 1.5), (
        "背景纯白", f"纯白 {pure * 100:.1f}% / 近白 {near * 100:.1f}%", "RGB(255,255,255)",
        near > 0.15)]
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    got = measure(args.photo, args, args.rembg)
    if got is None:
        return 1
    bgr, face, geom = got
    panels = [panel(bgr, face, geom,
                    f"head {geom.head_h:.0f}px  ear {geom.head_w:.0f}px  roll {face.roll_angle:.2f}deg")]
    if args.ref:
        ref = measure(args.ref, args, args.ref_rembg)
        if ref:
            rb, rf, rg = ref
            panels.insert(0, panel(rb, rf, rg,
                                   f"ref  head {rg.head_h:.0f}px  ear {rg.head_w:.0f}px"
                                   f"  roll {rf.roll_angle:.2f}deg"))
    sep = np.full((pp.CANVAS_H, 10, 3), 40, np.uint8)
    joined: list[np.ndarray] = []
    for i, p in enumerate(panels):
        if i:
            joined.append(sep)
        joined.append(p)
    cv2.imwrite(args.out, np.hstack(joined))
    if args.spec:
        ok = pp.print_report(spec_rows(bgr, geom, face))
        pp.log(f"\n[+] 已保存 {args.out}")
        return 0 if ok else 1
    pp.log(f"\n[+] 已保存 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

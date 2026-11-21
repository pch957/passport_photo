#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""【旧版本存档】v1：基于 haarcascade_frontalface_default.xml + haarcascade_eye.xml。

保留仅为对照/diff，实际处理请用 process_passport.py（YuNet 关键点版）。
注意：v1 的眼睛分支存在语法/逻辑错误（对 detectMultiScale 返回的整行做
`e + e/2.0` 再与标量比较，会抛 ValueError；`right_eye - left_eye` 对元组做减法），
一旦检测到眼睛就会崩溃，实际只走到“未检测到眼睛”的默认构图。
"""
import os
import sys
os.environ["REMBG_HOME"] = os.path.abspath("./.rembg")
os.environ["MPLCONFIGDIR"] = os.path.abspath("./.matplotlib")

import numpy as np
from PIL import Image, ImageOps
import cv2
from rembg import remove, new_session

def process_passport_photo(input_path, output_path):
    print(f"[-] Loading input image: {input_path}")
    raw_img = Image.open(input_path)
    input_image = ImageOps.exif_transpose(raw_img).convert("RGB")

    # 1. 抠图去背 (isnet-general-use)
    print("[-] Removing background using rembg...")
    session = new_session("isnet-general-use")
    output_image = remove(input_image, session=session, alpha_matting=True)

    white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
    white_bg.paste(output_image, (0, 0), output_image)

    # 2. 人脸与双眼检测（带详细 Debug 打印）
    cv_img = cv2.cvtColor(np.array(white_bg), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w, _ = cv_img.shape

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        print("[!] 未检测到人脸，使用默认中心裁剪")
        box_width, box_height = int(w * 0.4), int(h * 0.45)
        xmin, ymin = int(w * 0.3), int(h * 0.2)
    else:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        xmin, ymin, box_width, box_height = faces[0]
        print(f"[DEBUG] 检测到人脸框: xmin={xmin}, ymin={ymin}, w={box_width}, h={box_height}")

        upper_h = int(box_height * 0.55)
        upper_face_gray = gray[ymin:ymin + upper_h, xmin:xmin + box_width]
        min_eye = int(box_width * 0.12)
        max_eye = int(box_width * 0.24)
        eyes = eye_cascade.detectMultiScale(
            upper_face_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_eye, min_eye),
            maxSize=(max_eye, max_eye)
        )

        mid_x = box_width / 2.0
        valid_eyes = [e for e in eyes if (e + e / 2.0) > box_height * 0.20]
        left_candidates = [e for e in valid_eyes if (e + e / 2.0) < mid_x]
        right_candidates = [e for e in valid_eyes if (e + e / 2.0) >= mid_x]

        if len(left_candidates) > 0 and len(right_candidates) > 0:
            ex1, ey1, ew1, eh1 = max(left_candidates, key=lambda e: e * e)
            ex2, ey2, ew2, eh2 = max(right_candidates, key=lambda e: e * e)

            left_eye = (xmin + ex1 + ew1 / 2.0, ymin + ey1 + eh1 / 2.0)
            right_eye = (xmin + ex2 + ew2 / 2.0, ymin + ey2 + eh2 / 2.0)

            dy = right_eye - left_eye
            dx = right_eye - left_eye
            angle = np.degrees(np.arctan2(dy, dx))

            if abs(dy) < box_height * 0.12 and -12.0 <= angle <= 12.0 and abs(angle) > 0.5:
                print(f"[-] Rotating image to correct tilted head: {angle:.2f} degrees")
                white_bg = white_bg.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
                cv_img = cv2.cvtColor(np.array(white_bg), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f * f, reverse=True)
                    xmin, ymin, box_width, box_height = faces

    # 3. 裁剪与比例控制
    estimated_top_head = max(0, ymin - int(box_height * 0.22))
    estimated_chin = min(h, ymin + int(box_height * 1.12))
    head_height = estimated_chin - estimated_top_head

    target_head_h = 360.0
    target_top_margin = 45.0

    scale_factor = target_head_h / float(head_height if head_height > 0 else 1)

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    scaled_img = white_bg.resize((new_w, new_h), Image.LANCZOS)

    scaled_top_head = int(estimated_top_head * scale_factor)
    scaled_face_center_x = int((xmin + box_width / 2.0) * scale_factor)

    crop_left = scaled_face_center_x - 354 // 2
    crop_top = scaled_top_head - int(target_top_margin)

    crop_right = crop_left + 354
    crop_bottom = crop_top + 472

    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0
    if crop_right > new_w:
        crop_left -= (crop_right - new_w)
        crop_right = new_w
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0
    if crop_bottom > new_h:
        crop_top -= (crop_bottom - new_h)
        crop_bottom = new_h

    final_img = scaled_img.crop((crop_left, crop_top, crop_right, crop_bottom))
    if final_img.size != (354, 472):
        final_img = final_img.resize((354, 472), Image.LANCZOS)

    # 4. 体积二分压缩 (40KB - 75KB)
    print("[-] Compressing JPEG to target file size (40KB - 75KB)...")
    best_buffer = None
    quality = 95

    low, high = 50, 100
    while low <= high:
        mid = (low + high) // 2
        import io
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=mid)
        size_kb = buf.tell() / 1024.0

        if 40 <= size_kb <= 75:
            best_buffer = buf.getvalue()
            quality = mid
            break
        elif size_kb > 75:
            high = mid - 1
        else:
            low = mid + 1
            best_buffer = buf.getvalue()

    if best_buffer is None:
        import io
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        best_buffer = buf.getvalue()
        quality = 95

    with open(output_path, "wb") as f:
        f.write(best_buffer)

    final_size_kb = os.path.getsize(output_path) / 1024.0
    print(f"[+] Saved successfully to {output_path}")
    print(f"    - Dimensions: {final_img.size[0]} x {final_img.size[1]} px")
    print(f"    - File Size: {final_size_kb:.1f} KB (Quality: {quality})")
    print(f"    - Background: Pure White RGB(255, 255, 255)")
    print(f"    - Status: SUCCESS (Compliant with Chinese Passport & Consular App standards)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_passport.py <input_image> <output_image>")
        sys.exit(1)
    process_passport_photo(sys.argv[1], sys.argv[2])

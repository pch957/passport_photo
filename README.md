# 中国护照电子证件照处理脚本

把一张普通人像照片离线处理成符合**中国护照 / "中国领事" App** 要求的 354×472 电子证件照：自动抠图换纯白底、水平校正、按规范缩放构图、压缩到 40–75 KB，并输出一份逐项合规自检表。全程本地运行，不上传照片，**不使用任何生图模型重绘人脸**。

人脸定位用 OpenCV 自带的 **YuNet 神经网络检测器**（`cv2.FaceDetectorYN`，模型 230 KB）加瞳孔亚像素精修，可选 **FacemarkLBF** 68 点精修；不使用 `haarcascade_eye.xml` 这类老式 Haar 眼睛级联。

---

## 1. 环境准备

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
# 等价于：venv/bin/pip install rembg opencv-contrib-python-headless pillow numpy
```

- Python 3.12+（本项目在 3.14 上验证）
- `opencv-contrib-*` 是必需的：`cv2.face.createFacemarkLBF` 在 contrib 包里。只用 YuNet 的话装 `opencv-python-headless` 也够。
- 首次运行 rembg 会下载抠图模型（`isnet-general-use`，约 170 MB）到 `./.rembg/`。

模型文件放在 `models/`，缺失时脚本自动下载并校验 sha256：

| 模型 | 大小 | 说明 |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | 230 KB | 必需。YuNet 人脸检测 + 5 关键点 |
| `lbfmodel.yaml` | 54 MB | 可选。FacemarkLBF 68 点，能量出真实下颌最低点 |

> YuNet 模型在 opencv_zoo 仓库里走 git-LFS，**必须**从 `media.githubusercontent.com` 下载；用 `raw.githubusercontent.com` 拿回来的会是 131 字节的 LFS 指针文件。脚本已处理，手动下载时注意。

## 2. 快速开始

```bash
venv/bin/python3 process_passport.py IMG_20261121_161043.jpg passport_output.jpg --debug --strict
```

输出：

- `passport_output.jpg` — 354×472、600 DPI、40–75 KB 的成品
- `passport_output.debug.png` — 左右对照调试图（见 §5）
- 终端打印逐项自检表

## 3. 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--spec` | `passport` | 证件类型，见第 4 节；画布/头高/留白/背景/体积全部跟着它切换 |
| `--debug [路径]` | 关 | 输出调试叠加图，不给路径则用 `<输出>.debug.png` |
| `--strict` | 关 | 自检不通过时以非零码退出（`WARN` 不算不通过） |
| `--no-write` | 关 | 只分析不写文件，用来试参数 |
| `--refine {off,auto,lbf}` | `auto` | FacemarkLBF 精修；`auto` = 本地有模型才用 |
| `--download-lbf` | 关 | 允许联网下载 54 MB 的 `lbfmodel.yaml` |
| `--target-head-h` | 跟随规格 | 目标头部高度 px，显式传参可覆盖规格里的构图目标值 |
| `--top-margin` | 跟随规格 | 目标头顶留白 px |
| `--detect-long` | 1024 | 检测时的图像长边（YuNet 在 480–1280 最稳） |
| `--no-rotate` / `--max-roll` | — / 15 | 关闭水平校正 / 倾斜超过该角度就不校正 |
| `--no-bg-remove` | 关 | 跳过抠图（背景不会变纯白，仅做几何裁切） |
| `--no-whiten` / `--whiten-tol` | — / 14 | 关闭背景纯白化 / 泛洪容差 |
| `--rembg-model` / `--no-matting` | `isnet-general-use` / 关 | 抠图模型 / 关闭 alpha matting（更快，发丝边缘略硬） |
| `--min-kb` / `--max-kb` | 跟随规格 | 目标体积区间 |
| `--chin-iod-ratio` / `--crown-ratio` / `--ear-ratio` | 1.77 / 0.95 / 1.15 | 人体测量比例（剪影不可用时的兜底推算） |

完整列表看 `--help`。

## 4. 支持的证件类型 `--spec`

画布、头部区间、留白、背景色、体积区间、DPI 都写在脚本顶部的 `SPECS` 表里，
`--spec` 选哪一行就整体切换成哪套规格。数字都按官方来源核对，出处写在代码注释里。

| `--spec` | 类型 | 画布 px | 头部/脸部 px | 背景 | 体积 |
| --- | --- | --- | --- | --- | --- |
| `passport` | 中国护照 | 354×472 | 头高 330–390 | 纯白 | 40–75 KB |
| `idcard` | 居民身份证 | 358×441 | 头高 270–320 | 纯白 | 14–30 KB |
| `one_inch` | 一寸 | 295×413 | 头高 250–300 | 纯白 | 10–200 KB |
| `two_inch` | 二寸 | 413×579 | 头高 350–430 | 纯白 | 15–200 KB |
| `us_visa` | 美国签证 | 600×600 | 头高 300–414 | 纯白 | 60–240 KB |
| `uk_visa` | 英国签证 | 600×750 | 头高 483–567 | 浅灰 | 50 KB–6 MB |
| `schengen` | 申根签证 | 827×1063 | 脸高 756–850 | 浅灰 | 120 KB–1.2 MB |
| `au_passport` | 澳大利亚护照 | 827×1063 | 脸高 756–850 | 纯白 | 120 KB–1.2 MB |
| `au_visa` | 澳大利亚签证 | 1200×1600 | 脸高 1136–1280 | 纯白 | 70 KB–3.5 MB |
| `ca_visa` | 加拿大签证 | 1200×1800 | 头高 810–900 | 纯白 | 200 KB–5 MB |

```bash
venv/bin/python3 process_passport.py 照片.jpg 成品.jpg --spec au_passport --strict
```

加新类型只要往 `SPECS` 里添一行（画布、头高区间、留白区间、下巴留白、背景色、
体积区间、DPI、构图目标值），全流程都读这套常量，不用改处理逻辑。

## 5. 自检表怎么读

```
  项目            实测               要求              结果
  -------------------------------------------------------------
  画布尺寸        354 x 472 px       354 x 472 px      [PASS]
  头部高度        336 px             330-390 px        [PASS]
  头部宽度(含耳)  276 px             177-260 px        [WARN]
  头顶留白        44 px              35-59 px          [PASS]
  下巴距下边缘    92 px              ≥82 px            [PASS]
  人脸水平        0.06°              ≤1.5°             [PASS]
  背景纯白        纯白 39.2% 画面    RGB(255,255,255)  [PASS]
  文件格式        JPEG q=95 dpi=600  24bit JPEG        [PASS]
  文件体积        43.5 KB            40-75 KB          [PASS]
```

**关于 `头部宽度(含耳) WARN`**：规范同时要求头高 330–390 与两耳之间 177–260，两者之比上限 260/330 ≈ 0.79。耳朵外张、发量厚的头型（示例照片实测 0.82）**数学上无法同时满足**——把宽度压到 260 px，头高就只剩 316 px，跌破下限。脚本按「构图留白 > 头部高度 > 头部宽度」取舍，宽度超限报 `WARN` 并给出换算过程，**不会把头部缩到规范下限以下**（人脸识别看的是头部尺寸，缩头是更糟的选择）。若你更在意宽度那一项，用 `--target-head-h 316` 可反过来满足宽度、牺牲高度。

## 6. 调试图

`--debug` 生成的左右对照图，用来肉眼复核测量是否靠谱：

- **绿框** = 人脸框；**红点** = 5 关键点（左右瞳孔 / 鼻尖 / 左右嘴角）
- **品红线** = 头顶；**黄线** = 眼线；**红线** = 下巴
- **绿竖线**（`make_debug.py`）= 两耳之间的跨度
- 左栏是原图（含推算的头廓），右栏是成品（画的是命令的构图线，可与左栏对账）

## 7. 复核工具 `make_debug.py`

对**任意**一张人像照（影楼给的成品、别处做的照片）量一遍头廓并画线，同一套量法，数字可与主脚本对账：

```bash
venv/bin/python3 make_debug.py passport_output.jpg                 # 默认输出 <photo>.debug.png
venv/bin/python3 make_debug.py out.jpg check.png --ref 原图.jpg --ref-rembg
venv/bin/python3 make_debug.py 随便一张人像.jpg --no-spec          # 只画线不打印规格表
```

注意两点差异：这里量的是磁盘上的 JPEG，主脚本自检量的是压缩前的内存画布，差 ±3 px 属正常；背景纯白一项按 ≥250 判定（JPEG 有损压缩会让边缘像素偏离 255）。

## 8. 测试

```bash
venv/bin/python3 test_geometry.py
```

13 项标定/回归测试，覆盖：缩放约束求解（可行域 / 冲突 / 夹取）、剪影头顶与耳间宽、泛洪白化的遮罩屏障回归、旋转符号约定、瞳孔精修跨尺度稳定性、头廓实测标定值、源图↔成品量法一致性、模型 sha256。**改动任何测量或缩放逻辑后必须跑。**

## 9. 文件清单

```
process_passport.py            主脚本
make_debug.py                  规格复核 / 头廓可视化
test_geometry.py               标定与回归测试
process_passport.v1-haar.py    v1 存档（Haar 眼睛级联版，已废弃，仅作对照）
models/                        YuNet / FacemarkLBF 模型（git 忽略）
.dsh/skills/china-passport-photo/SKILL.md   配套 skill 规范
```

## 10. 常见问题

**Q: 报 `Size does not match`。**
A: `cv2.FaceDetectorYN.detect()` 前必须 `setInputSize()` 成当前图片尺寸，脚本已按图设置；自行改代码时注意。

**Q: FacemarkLBF 报 `index 1 is out of bounds for axis 1 with size 1`。**
A: `fit()` 返回的关键点形状是 `(68,1,2)`，要 `reshape(-1,2)`。

**Q: 浅色衣服上出现白色缺口。**
A: 背景泛洪白化以抠图遮罩为屏障（`--no-bg-remove` 时没有遮罩，可能漏刷或误刷）。可用 `--whiten-tol` 调容差，或 `--no-whiten` 关掉。

**Q: 检测不到人脸 / 关键点乱跳。**
A: 试 `--detect-long 640` 或 `1280`；仍不行看 `--debug` 图，必要时 `--refine lbf` 用 68 点重新定位下巴。

**Q: 体积不到 40 KB。**
A: 脚本会自动升 quality 到上限；仍不足说明原图细节太少，`--max-kb` 放宽或换更高分辨率的原图。

## 11. 许可

MIT（见 [LICENSE](LICENSE)）。

仓库里**不含任何人像照片**：示例输入图与成品被 `.gitignore` 排除（`*.jpg` / `*.png`），模型权重也不入库（脚本按 sha256 自动下载），clone 下来自己准备输入图即可。

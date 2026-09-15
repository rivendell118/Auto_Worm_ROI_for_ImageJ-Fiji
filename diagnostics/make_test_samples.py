"""Write synthetic TIFFs that provoke every verdict check_image() can return.

A tester with real acquisitions cannot produce most of these on demand: a folder
of good 16-bit images never exercises the colour, stack or unreadable paths, and
those are exactly the paths the 0.4.0 pre-check exists to guard. This writes one
folder per scenario, so each can be pointed at directly, and then runs the real
check_image() over everything it wrote and fails if any file lands on a verdict
other than the one it was generated for.

Run it with the build interpreter -- batch_worm_roi imports torch, scipy and cv2
at module level, so the verification step needs those installed:

    <python with numpy, Pillow, torch> diagnostics/make_test_samples.py

The output is disposable; nothing here is needed to build a release. Pass
--verify-only to re-run the checks against a directory generated earlier.
"""

from __future__ import print_function

import argparse
import io
import os
import sys
from collections import namedtuple

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
DEFAULT_OUT = os.path.join(HERE, "test_samples")

HEIGHT = 256
WIDTH = 192

# code=None means accepted. "value" is compared for equality when it is a plain
# value, passed to a predicate when it is callable, and skipped when it is None
# -- the "unreadable" detail is an exception message that varies by Pillow
# version, so asserting its exact text would break on an upgrade without
# anything actually being wrong.
Sample = namedtuple("Sample", "folder name make code value note warning")


def _worm_field(seed, count=7, height=HEIGHT, width=WIDTH, amplitude=190.0):
    """A dark field with bright, curved, worm-like objects.

    Not a substitute for real data -- the model will not segment these the way it
    segments an acquisition -- but the shapes are elongated and separated, so a
    run produces plausible ROIs and a plausible count rather than noise.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    field = 40.0 + 10.0 * np.sin(xx / 41.0) + 6.0 * np.cos(yy / 33.0)
    field += rng.normal(0.0, 2.5, (height, width))
    for _ in range(count):
        cy = rng.uniform(0.12, 0.88) * height
        cx = rng.uniform(0.12, 0.88) * width
        length = rng.uniform(0.30, 0.55) * height
        half_width = length * rng.uniform(0.05, 0.09)
        angle = rng.uniform(0.0, np.pi)
        dy, dx = yy - cy, xx - cx
        along = dx * np.cos(angle) + dy * np.sin(angle)
        across = -dx * np.sin(angle) + dy * np.cos(angle)
        # A gentle bend and tapered ends make the object a worm rather than an
        # ellipse; the Gaussian cross-section keeps it round in the middle and
        # the taper closes both ends.
        across = across - 0.30 * half_width * np.sin(2.0 * np.pi * along / length)
        taper = np.clip(1.0 - (2.0 * along / length) ** 6, 0.0, 1.0)
        field += amplitude * np.exp(-((across / half_width) ** 2)) * taper ** 0.25
    field -= field.min()
    return field / max(field.max(), 1e-9)


def _u8(field):
    return (field * 255.0 + 0.5).astype(np.uint8)


def _u16(field):
    return (field * 65535.0 + 0.5).astype(np.uint16)


# frombytes rather than fromarray(mode=...): Pillow 12 deprecated the mode
# argument, and frombytes is also the only way to state the byte order, which is
# the whole point of the big-endian sample below.
def _image(mode, values):
    # The shape comes from the array, not from HEIGHT/WIDTH, so the degenerate
    # one-pixel-wide samples below are built by the same code as the normal ones.
    return Image.frombytes(mode, (values.shape[1], values.shape[0]), values.tobytes())


def _blank(mode="L", fill=0):
    return Image.new(mode, (WIDTH, HEIGHT), fill)


def _one_pixel():
    """An otherwise empty plane holding a single 1. The pre-check asks whether a
    plane has *any* signal, not whether it looks like an image, so this counts."""
    values = np.zeros((HEIGHT, WIDTH), np.uint8)
    values[HEIGHT // 2, WIDTH // 2] = 1
    return values


def _valid_tiff_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="TIFF")
    return buffer.getvalue()


def _cut(image, keep):
    """The first `keep` fraction of a valid file: the header still parses, the
    pixel data does not. This is what an interrupted copy leaves behind."""
    data = _valid_tiff_bytes(image)
    return data[:max(16, int(len(data) * keep))]


def _png_bytes(image):
    buffer = io.BytesIO()
    image.convert("L").save(buffer, format="PNG")
    return buffer.getvalue()


def _bmp_bytes(image):
    buffer = io.BytesIO()
    image.convert("L").save(buffer, format="BMP")
    return buffer.getvalue()


def _garbage(seed, size=4096):
    return np.random.default_rng(seed).integers(0, 256, size, dtype=np.uint8).tobytes()


# --------------------------------------------------------------------------
# The sample table. Seeds are fixed so two runs produce identical files, which
# keeps a bug report reproducible.
# --------------------------------------------------------------------------
SAMPLES = [
    # 01 -- everything the pre-check accepts, one file per supported mode.
    Sample("01_ok", "ok_gray8.tif",
           lambda: _image("L", _u8(_worm_field(1))), None, None, None, ""),
    Sample("01_ok", "ok_gray16.tif",
           lambda: _image("I;16", _u16(_worm_field(2)).astype("<u2")), None, None, None, ""),
    # The one that must never be rejected: big-endian 16-bit is ~40% of the
    # TIFFs in this lab, and it is not colour.
    Sample("01_ok", "ok_gray16_bigendian.tif",
           lambda: _image("I;16B", _u16(_worm_field(3)).astype(">u2")), None, None, None, ""),
    Sample("01_ok", "ok_int32.tif",
           lambda: _image("I", (_worm_field(4) * 100000.0).astype("<i4")), None, None, None, ""),
    Sample("01_ok", "ok_float32.tif",
           lambda: _image("F", _worm_field(5).astype("<f4")), None, None, None, ""),

    # 02 -- real signal outside plane 0, which is the mismatch the guard exists
    # to prevent: the ROIs would describe plane 0 while ImageJ measures another.
    Sample("02_stack", "stack_two_planes.tif",
           lambda: [_image("I;16", _u16(_worm_field(11)).astype("<u2")),
                    _image("I;16", _u16(_worm_field(12)).astype("<u2"))], "stack", 2, None, ""),
    Sample("02_stack", "stack_three_planes.tif",
           lambda: [_image("L", _u8(_worm_field(13))),
                    _image("L", _u8(_worm_field(14))),
                    _image("L", _u8(_worm_field(15)))], "stack", 3, None, ""),
    # A single non-zero pixel is still signal. This pins the predicate to
    # "any non-zero", not to a brightness threshold someone might add later --
    # get this wrong and a genuinely mis-measured image is accepted.
    Sample("02_stack", "stack_one_pixel.tif",
           lambda: [_image("L", _u8(_worm_field(16))),
                    _image("L", _one_pixel())], "stack", 2, None, ""),

    # 03 -- every colour mode Pillow can put in a TIFF.
    Sample("03_colour", "colour_rgb.tif",
           lambda: _image("RGB", np.repeat(_u8(_worm_field(21))[:, :, None], 3, axis=2)),
           "colour", "RGB", None, ""),
    Sample("03_colour", "colour_rgba.tif",
           lambda: _image("RGBA", np.repeat(_u8(_worm_field(22))[:, :, None], 4, axis=2)),
           "colour", "RGBA", None, ""),
    Sample("03_colour", "colour_palette.tif",
           lambda: _image("L", _u8(_worm_field(23))).convert("P"), "colour", "P", None, ""),
    # Grey plus an alpha channel: looks grey, is not what the model was fed.
    Sample("03_colour", "colour_gray_alpha.tif",
           lambda: _image("LA", np.repeat(_u8(_worm_field(24))[:, :, None], 2, axis=2)),
           "colour", "LA", None, ""),
    Sample("03_colour", "colour_cmyk.tif",
           lambda: _image("CMYK", np.repeat(_u8(_worm_field(25))[:, :, None], 4, axis=2)),
           "colour", "CMYK", None, ""),

    # 04 -- readable name, unreadable content.
    Sample("04_unreadable", "unreadable_truncated_half.tif",
           lambda: _cut(_image("I;16", _u16(_worm_field(31)).astype("<u2")), 0.5),
           "unreadable", None, None, ""),
    Sample("04_unreadable", "unreadable_header_only.tif",
           lambda: _cut(_image("L", _u8(_worm_field(32))), 0.02),
           "unreadable", None, None, ""),
    Sample("04_unreadable", "unreadable_garbage.tif",
           lambda: _garbage(33), "unreadable", None, None, ""),
    Sample("04_unreadable", "unreadable_empty.tif",
           lambda: b"", "unreadable", None, None, ""),

    # 05 -- a real image in the wrong container; ImageJ cannot open it either.
    Sample("05_not_tiff", "not_tiff_actually_png.tif",
           lambda: _png_bytes(_image("L", _u8(_worm_field(41)))), "format", "PNG", None, ""),
    Sample("05_not_tiff", "not_tiff_actually_bmp.tif",
           lambda: _bmp_bytes(_image("L", _u8(_worm_field(42)))), "format", "BMP", None, ""),

    # 06 -- what some writers append: extra planes that hold no signal at all.
    # These are accepted, with a remark in the log rather than a refusal.
    Sample("06_empty_overlay", "overlay_one_empty_plane.tif",
           lambda: [_image("L", _u8(_worm_field(51))), _blank()],
           None, None, "1 empty extra plane(s)", ""),
    Sample("06_empty_overlay", "overlay_two_empty_planes.tif",
           lambda: [_image("L", _u8(_worm_field(52))), _blank(), _blank()],
           None, None, "2 empty extra plane(s)", ""),

    # 07 -- 1-bit is accepted, but the models were never trained on it.
    Sample("07_binary", "binary_1bit.tif",
           lambda: _image("L", _u8(_worm_field(61))).point(lambda v: 255 if v > 110 else 0).convert("1"),
           None, None, None, "binary"),

    # 08 -- two files, one set of output names.
    Sample("08_stem_collision", "A.tif",
           lambda: _image("L", _u8(_worm_field(71))), None, None, None, ""),
    Sample("08_stem_collision", "A.tiff",
           lambda: _image("L", _u8(_worm_field(72))), None, None, None, ""),

    # 09 -- one good image among every kind of bad one, to check that the batch
    # is refused as a whole and that nothing at all is written.
    Sample("09_mixed", "good_gray16.tif",
           lambda: _image("I;16", _u16(_worm_field(81)).astype("<u2")), None, None, None, ""),
    Sample("09_mixed", "bad_rgb.tif",
           lambda: _image("RGB", np.repeat(_u8(_worm_field(82))[:, :, None], 3, axis=2)),
           "colour", "RGB", None, ""),
    Sample("09_mixed", "bad_stack.tif",
           lambda: [_image("L", _u8(_worm_field(83))),
                    _image("L", _u8(_worm_field(84)))], "stack", 2, None, ""),
    Sample("09_mixed", "bad_truncated.tif",
           lambda: _cut(_image("L", _u8(_worm_field(85))), 0.4), "unreadable", None, None, ""),

    # 10 -- files the pre-check cannot reject, because nothing is wrong with
    # them as *images*, that then fail in the middle of a run. Less than two
    # pixels across in one direction is a valid single-plane greyscale TIFF and
    # reads fine, but the bounding-box geometry downstream raises on it. The
    # engine skips the image, names it Failed_<name>, records the reason in
    # batch_summary.csv and finishes the rest -- which is the behaviour to watch
    # here, and the one a tester has no real data to trigger.
    Sample("10_midrun_failure", "ok_control.tif",
           lambda: _image("I;16", _u16(_worm_field(91)).astype("<u2")), None, None, None, ""),
    Sample("10_midrun_failure", "too_narrow_1x512.tif",
           lambda: _image("I;16", np.zeros((512, 1), np.uint16)), None, None, None, ""),
    Sample("10_midrun_failure", "too_short_512x1.tif",
           lambda: _image("I;16", np.zeros((1, 512), np.uint16)), None, None, None, ""),
    Sample("10_midrun_failure", "too_small_1x1.tif",
           lambda: _image("I;16", np.zeros((1, 1), np.uint16)), None, None, None, ""),
]


READ_ME = u"""自动圈虫 0.4.0 CUDA for ImageJ —— 测试样例说明
====================================================================

这些 TIFF 是 diagnostics\\make_test_samples.py 生成的合成图，用来触发插件
的各种预检与报错分支。它们**不是真实拍摄的数据**，模型在这些图上的分割结果
没有科学意义，只用来验证程序的行为是否正确。

每个子目录对应一种情况。用法：在插件里把「输入目录」指向某个子目录，输出目录
选一个空目录（**不要**选成输入目录本身），然后开始处理。

  目录                     应该发生什么
  ------------------------------------------------------------------
  01_ok                    正常处理完，全部成功，没有报错。
                           含 8 位、16 位小端、16 位大端（I;16B）、
                           32 位整数、32 位浮点各一张。
                           16 位大端那张是本机最常见的数据类型，绝不能
                           被当成「彩色图」拒掉。
                           注意：这五张是合成图，文件名里也没有可解析的
                           条数，所以每张都会显示 REVIEW_COUNT_MISMATCH
                           （圈出的条数对不上默认的 expected=10）。这是
                           正常的，不是故障；重点看的是「有没有跑完、有没
                           有报错」。

  02_stack                 整批被拒绝，报错列出的每个文件都写着
                           "has N planes"。这是对的：本程序只读第 0 页，
                           而 ImageJ 测量的是当前选中的层，两者不是同
                           一个平面，圈出来的 ROI 会和测量结果对不上。
                           处理前就应该被拒，输出目录里不该出现任何文件。
                           其中 stack_one_pixel.tif 的第 2 页只有一个
                           非零像素，仍然要算「有信号」被拒。

  03_colour                整批被拒绝，报错写着 "is not single-channel
                           greyscale (PIL mode XXX)"。同上，RGB / RGBA /
                           调色板 / 带 alpha 的灰度 / CMYK 都不能处理。
                           注意 colour_gray_alpha.tif 看起来是灰度图，
                           但它带 alpha 通道，也必须被拒。

  04_unreadable            整批被拒绝，报错写着 "cannot be read: ..."。
                           四张分别是：拷到一半的 TIFF、只有文件头的
                           TIFF、4 KB 随机字节、0 字节空文件。
                           这类文件如果能过预检，会在批处理跑到一半时
                           把整批带崩，所以要在这里拦住。

  05_not_tiff              整批被拒绝，报错写着 "is not a TIFF (PNG)"。
                           改成 .tif 后缀的 PNG 和 BMP：Pillow 能打开，
                           ImageJ 打不开。

  06_empty_overlay         **正常处理完**，但日志里会多一行备注
                           "N empty extra plane(s); treated as overlay"。
                           这两张是多页 TIFF，多出来的页整页都是 0。
                           有些采集软件会把叠加层写成这种多余页，没有
                           信号就不会造成「测错平面」，所以放行、只记
                           一笔，不拒绝。

  07_binary                **正常处理完**，但会弹出（GUI）或记录（批处理）
                           一条 1 位二值图的警告。模型没有用二值图标定
                           过，分割结果可能不准。这张是真正会被处理的，
                           和 02/03/04/05 的「拒绝」不同。

  08_stem_collision        整批被拒绝，报错写着 A.tif / A.tiff 会写到
                           同一组结果文件名（结果按去掉扩展名的文件名
                           命名，A_RoiSet.zip 会被后一个覆盖前一个）。
                           单独打开这两个文件都是正常的，只有放在同一
                           个目录里才会撞。

  09_mixed                 整批被拒绝，报错里同时列出彩色、多页和损坏
                           三种原因，而 good_gray16.tif 本身没问题。
                           这是故意的：一张坏图就拒绝整批，避免跑出一
                           半之后才发现。输出目录里同样不该有任何文件。

  10_midrun_failure        **这是唯一一组会「跑到一半失败」的样例**，用来
                           看批处理的容错：ok_control.tif 正常处理完，另
                           外三张在跑到它们时报错跳过，整批不中断。
                           预期结果：
                             - 日志里出现三行 Failed_xxx.tif: ERROR ...
                             - 最后一行是 Failed images (3): ...
                             - 处理进度继续走到 4/4，不是停在 3/4
                             - batch_summary.csv 里这三行 qc_status 是
                               FAILED、error 列写着原因，ok_control.tif
                               那一行照常是正常结果
                             - 结果目录里**只有** ok_control_RoiSet.zip，
                               这三张失败的不留任何文件（这一点是 0.4.0
                               修过的：程序在画 QC 图之前就已经写好了
                               RoiSet.zip，中途失败会把那个 zip 留在原地，
                               而 ImageJ 只要看到有 RoiSet.zip 就会去测量，
                               于是「已失败」的图反而被测进了结果表）
                           为什么这三张会失败：它们的宽度或高度只有 1 个
                           像素（1x1、1x512、512x1）。这不是坏文件，预检
                           也拦不住——它是合法的单页灰度 TIFF，读得出来，
                           只是后面算包围盒时没有意义，所以只能到那一步
                           才失败。真实数据里不会出现这种图，这正是需要
                           合成样例的原因。
                           跑这一组时注意 GUI 的「保存日志」按钮：这类
                           中途失败最需要把日志交给开发的人。


也可以自己造：改 make_test_samples.py 里的 SAMPLES 表，或直接跑

    python diagnostics\\make_test_samples.py <输出目录>

它会先写文件，再用程序自己的 check_image() 逐个复核，判定和预期不一致就
报错退出。所以这些样例本身也是「预检逻辑有没有被改坏」的回归检查。
"""


def write_sample(path, payload):
    if isinstance(payload, (bytes, bytearray)):
        with open(path, "wb") as handle:
            handle.write(payload)
        return
    pages = list(payload) if isinstance(payload, (list, tuple)) else [payload]
    if len(pages) == 1:
        pages[0].save(path, format="TIFF")
    else:
        pages[0].save(path, format="TIFF", append_images=pages[1:])


def matches(expected, actual):
    if expected is None:
        return True
    if callable(expected):
        return bool(expected(actual))
    return expected == actual


def verify(root, verbose):
    """Re-run the program's own pre-check over the generated files."""
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    from batch_worm_roi import check_image, find_stem_collisions

    failures = []
    for sample in SAMPLES:
        path = os.path.join(root, sample.folder, sample.name)
        label = "%s/%s" % (sample.folder, sample.name)
        if not os.path.exists(path):
            failures.append("%s: MISSING" % label)
            continue
        found = check_image(path)
        why = []
        if found.code != sample.code:
            why.append("code %r, expected %r" % (found.code, sample.code))
        elif not matches(sample.value, found.value):
            why.append("value %r, expected %r" % (found.value, sample.value))
        if sample.note and sample.note not in found.note:
            why.append("note %r, expected to contain %r" % (found.note, sample.note))
        if found.warning != sample.warning:
            why.append("warning %r, expected %r" % (found.warning, sample.warning))
        if why:
            failures.append("%s: %s" % (label, "; ".join(why)))
        elif verbose:
            print("  ok   %-46s code=%-11s value=%r"
                  % (label, found.code, found.value))

    # Two verdicts are about a folder rather than a file, so they cannot be
    # checked one path at a time.
    collision_dir = os.path.join(root, "08_stem_collision")
    if os.path.isdir(collision_dir):
        names = sorted(os.listdir(collision_dir))
        collisions = find_stem_collisions([os.path.join(collision_dir, n) for n in names])
        if not collisions:
            failures.append("08_stem_collision: find_stem_collisions found nothing")
        elif verbose:
            print("  ok   %-46s collides as %r" % ("08_stem_collision", collisions))

    mixed_dir = os.path.join(root, "09_mixed")
    if os.path.isdir(mixed_dir):
        from batch_worm_roi import _reject_unsupported_images
        paths = [os.path.join(mixed_dir, n) for n in sorted(os.listdir(mixed_dir))]
        try:
            _reject_unsupported_images(paths)
            failures.append("09_mixed: _reject_unsupported_images did not raise")
        except ValueError as exc:
            text = str(exc)
            missing = [code for code in ("cannot be read", "has 2 planes",
                                         "PIL mode RGB") if code not in text]
            if missing:
                failures.append("09_mixed: rejection message lacks %r" % missing)
            elif verbose:
                print("  ok   %-46s refused with all three reasons"
                      % "09_mixed")
    return failures


def main():
    parser = argparse.ArgumentParser(
        description="Write synthetic TIFFs covering every check_image() verdict.")
    parser.add_argument("outdir", nargs="?", default=DEFAULT_OUT,
                        help="where to write the samples (default: %(default)s)")
    parser.add_argument("--verify-only", action="store_true",
                        help="do not write anything, just re-check what is there")
    parser.add_argument("--force", action="store_true",
                        help="write into a directory that already has content")
    parser.add_argument("--quiet", action="store_true",
                        help="only print failures")
    args = parser.parse_args()

    verbose = not args.quiet
    root = os.path.abspath(args.outdir)

    if not args.verify_only:
        if os.path.isdir(root) and os.listdir(root) and not args.force:
            print("refusing to write into %s: it is not empty. Pass --force to "
                  "overwrite, or pick another directory." % root)
            return 2
        written = 0
        for sample in SAMPLES:
            folder = os.path.join(root, sample.folder)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            write_sample(os.path.join(folder, sample.name), sample.make())
            written += 1
        with open(os.path.join(root, "READ_ME_测试样例.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(READ_ME)
        if verbose:
            print("wrote %d files under %s" % (written, root))

    if verbose:
        print("verifying with check_image():")
    failures = verify(root, verbose)
    if failures:
        print()
        print("%d sample(s) did not behave as expected:" % len(failures))
        for line in failures:
            print("  FAIL " + line)
        return 1
    if verbose:
        print("all %d samples match their expected verdict" % len(SAMPLES))
    return 0


if __name__ == "__main__":
    sys.exit(main())

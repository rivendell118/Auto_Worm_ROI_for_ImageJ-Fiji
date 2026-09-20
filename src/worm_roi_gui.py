#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动圈虫图形界面。

该程序只调用现有 batch_worm_roi.py 和已备份的模型，
不会修改模型文件或备份目录。
"""

from __future__ import annotations

import ctypes
import base64
import csv
import itertools
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageOps, ImageTk


APP_VERSION = "0.5.0-beta"
APP_DIR = Path(__file__).resolve().parent
# 打包后 __file__ 指向 PyInstaller 解包目录,不是 exe 所在目录。
# exe 可执行目录才是模型与配置的正确根目录。
if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
else:
    APP_ROOT = APP_DIR


def _state_directory() -> Path:
    """Keep mutable settings/logs outside the installation directory."""
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        directory = Path(os.environ["LOCALAPPDATA"]) / "AutoWorm"
    else:
        directory = Path.home() / ".autoworm"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Portable fallback for unusually restricted user profiles.
        directory = APP_ROOT
    return directory


STATE_DIR = _state_directory()
CONFIG_PATH = STATE_DIR / "settings.json"
MODEL_ROOT = APP_ROOT / "models"
IMAGEJ_BRIDGE_DIR = os.environ.get("AUTOWORM_IMAGEJ_BRIDGE_DIR", "").strip()
IMAGEJ_MEASUREMENT_MODE = os.environ.get("AUTOWORM_IMAGEJ_MODE", "") == "1"
# Which bridge the plug-in that started us speaks. The 0.5.0 jar sets "2"; every
# older jar sets nothing, so an empty string here means the jar in Fiji predates
# the plane field. That combination is only dangerous with 明场ROI on -- an old
# jar would measure slice 1 of every stack and say nothing -- so the check lives
# in _start_processing and only fires for that one case.
IMAGEJ_BRIDGE_PROTOCOL = os.environ.get("AUTOWORM_IMAGEJ_BRIDGE_PROTOCOL", "").strip()
IMAGEJ_BRIDGE_PROTOCOL_REQUIRED = "2"
CUDA_MIN_DRIVER_MAJOR = 580

# One name per notification inside the bridge directory. Zero padded because the
# Java side sorts by file name to read them in the order they were written.
_BRIDGE_SEQUENCE = itertools.count()
_BRIDGE_NAME = "%020d.properties"


def _imagej_bridge_pending() -> bool:
    """True while the plug-in has not finished with every notification written.

    It may still be measuring a previous batch, which is not an error, but the
    next one writes to the same output folder and its combined table would land
    on top of the previous one. The receiver deletes each notification once it
    has handled it -- not when it reads it, which takes milliseconds against a
    measurement that takes minutes -- so "no .properties file left" means
    exactly "all measured".
    """
    if not IMAGEJ_BRIDGE_DIR:
        return False
    return any(Path(IMAGEJ_BRIDGE_DIR).glob("*.properties"))


def _write_imagej_bridge_file(
        status: str, input_dir: str = "", output_dir: str = "", message: str = "",
        successful_images=None, qc_plane: int = 1, measured_plane: int = 1) -> str | None:
    """Atomically write one ASCII notification for the Java plug-in to read.

    Every call gets its own file name. A single fixed path would be overwritten
    by whichever batch finishes next while the plug-in is still busy with the
    previous one, and that earlier batch would then be measured by nobody.

    ``successful_images`` lists the images this batch finished, as the input
    folder names them, and the plug-in measures those and only those. None means
    "no list" and leaves the receiver on its older rule (measure every image
    that has a ROI ZIP), which is all the diagnostic --headless-run path needs.

    ``measured_plane`` is the 1-based page the plug-in is to measure, and every
    image of a brightfield batch shares it because the two plane numbers are
    asked once per batch. It is written as a bare decimal rather than base64,
    like ok_count: Properties.load handles plain digits without any of the
    escaping rules that make a hand-built .properties file fragile. A jar from
    before this field existed ignores it and measures slice 1, which is why the
    protocol check happens before the batch rather than here.

    ``qc_plane`` is the page the QC image was cut from, and the 0.5.0 plug-in
    does not read it -- the QC PNG is already the brightfield picture, so it has
    nothing to look up. It is written anyway to keep the notification and
    batch_summary.csv describing the run the same way; a future reader that wants
    to say where a QC came from does not have to guess.

    Returns None on success, or an error string describing the failure.
    """
    if not IMAGEJ_BRIDGE_DIR:
        return None

    def encoded(value: str) -> str:
        return base64.b64encode(value.encode("utf-8")).decode("ascii")

    lines = [
        "status=" + status,
        "input_b64=" + encoded(input_dir),
        "output_b64=" + encoded(output_dir),
        "message_b64=" + encoded(message),
        "qc_plane=" + str(int(qc_plane)),
        "measured_plane=" + str(int(measured_plane)),
    ]
    if successful_images is not None:
        names = sorted({str(name).strip() for name in successful_images
                        if str(name).strip()})
        lines.append("ok_count=" + str(len(names)))
        lines.append("ok_b64=" + encoded("\n".join(names)))

    path = Path(IMAGEJ_BRIDGE_DIR) / (_BRIDGE_NAME % next(_BRIDGE_SEQUENCE))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
        os.replace(temporary, path)
        return None
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"


def _bridge_planes(job) -> tuple[int, int]:
    """The (qc_plane, measured_plane) this job's images were written with.

    Both are 1 unless the batch ran with Brightfield ROI on, in which case the
    plug-in is told to measure the fluorescence plane -- the analysis images came
    off the brightfield plane, so measuring slice 1 would report numbers from
    whichever plane happens to be first.

    The two plane numbers are a property of the batch because the dialog asks
    once for the whole run, so one pair is right for every image it finished,
    mixed folders of single-plane files included: those were processed as
    single-plane files and their ROI was made from plane 1, so reporting any
    other number would be a lie about what the ROI covers.
    """
    if not job or not job.get("brightfield_roi"):
        return 1, 1
    try:
        fluorescence = int(job["fluorescence_plane"])
        brightfield = int(job["brightfield_plane"])
    except (KeyError, TypeError, ValueError):
        return 1, 1
    if fluorescence < 1 or brightfield < 1:
        return 1, 1
    return brightfield, fluorescence

MODEL_CONFIGS = {
    "high": {
        "label": "高清晰度图像",
        "version": "0.1.2",
        "checkpoint": MODEL_ROOT / "0.1.2" / "worm.pt",
        "tip_checkpoint": MODEL_ROOT / "0.1.2" / "tip.pt",
    },
    "low": {
        "label": "低清晰度图像",
        "version": "0.2.2",
        "checkpoint": MODEL_ROOT / "0.2.2" / "worm.pt",
        "tip_checkpoint": MODEL_ROOT / "0.2.2" / "tip.pt",
    },
}

# 明场模型单独一个目录，不占荧光模型的版本号：0.1.x/0.2.x 已经用掉，明场再挤进
# 同一套编号，早晚会和下一版荧光模型撞上。
#
# 它不是 MODEL_CONFIGS 的第三个模式。明场是一个勾选项，不是一种「处理模式」，
# 加进去会连累 _mode_label 的三元表达式、两个 Radiobutton，以及按模式查
# low_clarity_split 的那段推理。
#
# 本轮这两个文件还不存在（权重等用户给数据后再训），所以开启明场ROI 会在预检里
# 明确报「缺少模型」并拒绝启动——这正是要的：绝不能静默回退到高清或低清模型。
BRIGHTFIELD_MODEL_CONFIG = {
    "label": "明场图像",
    "version": "brightfield-0.1.0",
    "checkpoint": MODEL_ROOT / "brightfield-0.1.0" / "worm.pt",
    "tip_checkpoint": MODEL_ROOT / "brightfield-0.1.0" / "tip.pt",
}

THEMES = {
    "dark": {
        "bg": "#14181e", "panel": "#1b2129", "panel2": "#202832",
        "text": "#e8edf2", "muted": "#9aa7b4", "border": "#36414d",
        "accent": "#2f8cff", "accent_hover": "#58a2ff", "terminal": "#0d1117",
        "tree_sel": "#244e78", "danger": "#ef5b5b", "warning": "#ffd34d", "ok": "#4ec98d",
    },
    "light": {
        "bg": "#eef1f5", "panel": "#ffffff", "panel2": "#f7f9fb",
        "text": "#1c2630", "muted": "#637180", "border": "#c9d1da",
        "accent": "#1673d1", "accent_hover": "#0e62b5", "terminal": "#f5f7fa",
        "tree_sel": "#cce3fa", "danger": "#c53f3f", "warning": "#c58a00", "ok": "#21885c",
    },
}


def enable_windows_dpi_awareness() -> None:
    """让高 DPI 显示器上的文字和图片保持清晰。"""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


class HoverTooltip:
    """Show a small annotation while the pointer rests on a widget."""

    def __init__(self, widget: tk.Widget, text_getter, delay_ms: int = 350) -> None:
        self.widget = widget
        self.text_getter = text_getter
        self.delay_ms = delay_ms
        self.after_id: str | None = None
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: tk.Event | None = None) -> None:
        self._hide()
        self.after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self) -> None:
        self.after_id = None
        if self.window is not None or not self.widget.winfo_exists():
            return
        text = str(self.text_getter())
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        try:
            self.window.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        self.window.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.window, text=text, justify="left", padx=8, pady=5,
            background="#fffbd6", foreground="#202020",
            relief="solid", borderwidth=1,
            font=("Microsoft YaHei UI", 8),
        ).pack()

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


class CompactCheckbutton(tk.Canvas):
    """A checkbox whose clickable canvas is only slightly larger than its box."""

    BOX_SIZE = 16
    TARGET_SIZE = 20  # 20² / 16² = 1.5625 times the visible box area.

    def __init__(self, parent: tk.Widget, variable: tk.BooleanVar, command) -> None:
        super().__init__(
            parent, width=self.TARGET_SIZE, height=self.TARGET_SIZE,
            bd=0, highlightthickness=0, takefocus=True, cursor="hand2")
        self.variable = variable
        self.command = command
        self.colors = {
            "bg": "#ffffff", "fg": "#202020", "accent": "#1673d1",
            "muted": "#637180",
        }
        self.variable.trace_add("write", self._variable_changed)
        self.bind("<Button-1>", self._toggle)
        self.bind("<space>", self._toggle)
        self.bind("<Return>", self._toggle)
        self._draw()

    def _variable_changed(self, *_args) -> None:
        self._draw()

    def _toggle(self, _event: tk.Event | None = None) -> str:
        if str(self.cget("state")) != "disabled":
            self.variable.set(not self.variable.get())
            self.command()
        return "break"

    def set_state(self, state: str) -> None:
        self.configure(state=state, cursor="" if state == "disabled" else "hand2")
        self._draw()

    def set_colors(self, bg: str, fg: str, accent: str, muted: str) -> None:
        self.colors.update(bg=bg, fg=fg, accent=accent, muted=muted)
        self.configure(bg=bg)
        self._draw()

    def _draw(self) -> None:
        if not self.winfo_exists():
            return
        self.delete("all")
        offset = (self.TARGET_SIZE - self.BOX_SIZE) // 2
        end = offset + self.BOX_SIZE - 1
        disabled = str(self.cget("state")) == "disabled"
        border = self.colors["muted"]
        if self.variable.get():
            fill = self.colors["muted"] if disabled else self.colors["accent"]
            self.create_rectangle(offset, offset, end, end, outline=fill, fill=fill, width=1)
            tick = self.colors["bg"]
            self.create_line(offset + 3, offset + 8, offset + 7, offset + 12,
                             offset + 13, offset + 4, fill=tick, width=2,
                             capstyle="round", joinstyle="round")
        else:
            self.create_rectangle(offset, offset, end, end, outline=border,
                                  fill=self.colors["bg"], width=1)


class WormRoiGui:
    PAGE_SIZE = 4

    def __init__(self, root: tk.Tk, initial_path: str | None = None) -> None:
        self.root = root
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1380, max(1040, screen_width - 100))
        window_height = min(880, max(680, screen_height - 100))
        self.root.minsize(min(1080, screen_width - 40), min(700, screen_height - 60))
        window_x = max(0, (screen_width - window_width) // 2)
        window_y = max(0, (screen_height - window_height) // 2)
        self.root.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
        if os.name == "nt":
            # Windows 显示缩放下逻辑分辨率可能大于实际工作区，最大化可避免底部被任务栏裁切。
            self.root.state("zoomed")

        self.config_data = self._load_config()
        self.language_var = tk.StringVar(value=self.config_data.get("language", "zh"))
        if self.language_var.get() not in {"zh", "en"}:
            self.language_var.set("zh")
        self.root.title(self._tr(
            f"秀丽隐杆线虫荧光定量 · 自动圈虫 {APP_VERSION}",
            f"C. elegans Fluorescence Quantification · Auto Worm ROI {APP_VERSION}"))
        self.mode_var = tk.StringVar(value=self.config_data.get("mode", "high"))
        if self.mode_var.get() not in MODEL_CONFIGS:
            self.mode_var.set("high")
        try:
            configured_count = int(self.config_data.get("worm_count", 10))
        except (TypeError, ValueError):
            configured_count = 10
        self.last_valid_worm_count = max(1, configured_count)
        self.worm_count_var = tk.StringVar(value=str(self.last_valid_worm_count))
        self.shape_refinement_var = tk.BooleanVar(value=bool(
            self.config_data.get("shape_refinement", self.config_data.get("reserve2", False))))
        # 必须由实验员主动开启；旧配置中没有此键时始终保持关闭。
        self.manual_head_annotation_var = tk.BooleanVar(value=bool(
            self.config_data.get("manual_head_annotation", False)))
        self.segment_selection_var = tk.BooleanVar(value=bool(
            self.config_data.get("segment_selection", False)))
        try:
            segment_start = float(self.config_data.get("segment_start", 0.0))
            segment_end = float(self.config_data.get("segment_end", 1.0))
        except (TypeError, ValueError):
            segment_start, segment_end = 0.0, 1.0
        if not (0.0 <= segment_start < segment_end <= 1.0):
            segment_start, segment_end = 0.0, 1.0
        self.segment_start_var = tk.DoubleVar(value=segment_start)
        self.segment_end_var = tk.DoubleVar(value=segment_end)
        if self.segment_selection_var.get():
            self.manual_head_annotation_var.set(True)
        # 明场ROI。开关与两个层号都从配置读回来，但层号只作为对话框的默认值：
        # 每一批都重新问一遍，因为「这个文件夹的荧光在第几层」是数据的事，不是
        # 偏好设置的事，记在配置里只会在换一批图时静默用错。
        self.brightfield_roi_var = tk.BooleanVar(value=bool(
            self.config_data.get("brightfield_roi", False)))
        self.fluorescence_plane_var = tk.IntVar(value=max(
            1, self._configured_int("fluorescence_plane", 1)))
        self.brightfield_plane_var = tk.IntVar(value=max(
            1, self._configured_int("brightfield_plane", 2)))
        self.theme_var = tk.StringVar(value=self.config_data.get("theme", "dark"))
        if self.theme_var.get() not in THEMES:
            self.theme_var.set("dark")
        self.output_path = self.config_data.get("output_path", "")
        self.recent_paths = [p for p in self.config_data.get("recent_paths", []) if Path(p).exists()][:10]

        candidate = Path(initial_path).expanduser() if initial_path else None
        saved_folder = Path(self.config_data.get("current_folder", ""))
        if candidate and candidate.exists():
            self.current_folder = candidate if candidate.is_dir() else candidate.parent
        elif saved_folder.is_dir():
            self.current_folder = saved_folder
        else:
            self.current_folder = APP_ROOT

        self.process: threading.Thread | None = None
        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_requested = False
        self.close_when_stopped = False
        self.bridge_notification_written = False
        self.active_job: dict[str, object] | None = None
        # Text of the exception that ended the batch, kept until the ImageJ
        # bridge file is written: that file is the only channel the Fiji side
        # reads, and its message field is where the reason has to end up.
        self.pending_error_text = ""
        self.qc_files: list[Path] = []
        self.qc_status: dict[str, str] = {}
        self.qc_attention: dict[str, bool] = {}
        # 每张 QC 图的两个层号 (qc_plane, measured_plane)，按图名索引。
        #
        # 存的是这两个数的**差**而不是「明场开关」：混批里单层图的 QC 明明来自
        # 荧光层，可它那一行的开关列与整批相同，只看开关会给它标上一个并不存在
        # 的「明场层」。而 qc_plane != measured_plane 恰好只在「这张图的 QC 真的
        # 取自另一层」时成立——批量预检不允许两层相同，单层图则两层都是 1。
        #
        # 键缺失表示这张 QC 来自旧版汇总表，层号无从得知。
        self.qc_planes: dict[str, tuple[int, int]] = {}
        self.qc_page = 0
        self.selected_qc: Path | None = None
        self.tree_paths: dict[str, Path] = {}
        self.preview_images: list[ImageTk.PhotoImage] = []
        self.preview_resize_job: str | None = None
        self.annotation_path: Path | None = None
        self.annotation_image: Image.Image | None = None
        self.annotation_photo: ImageTk.PhotoImage | None = None
        self.annotation_arrows: list[object] = []
        self.annotation_boundaries: list[object] = []
        self.annotation_exclusions: list[object] = []
        self.annotation_mode_var = tk.StringVar(value="arrow")
        self.annotation_display_box: tuple[float, float, float] | None = None
        self.annotation_drag_start: tuple[float, float] | None = None
        self.annotation_boundary_draft: list[tuple[float, float]] = []
        self.segment_drag_handle: str | None = None

        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self._build_ui()
        self._apply_theme()
        self._load_folder(self.current_folder, add_recent=False)
        self._rebuild_file_menu()
        self._rebuild_settings_menu()
        self.root.after(100, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if candidate and candidate.is_file():
            self._add_recent(candidate)
            # quiet: this is the image ImageJ handed over at launch, not a choice
            # the user just made, so an unsupported one is reported in the log
            # rather than in a modal dialog before they have done anything.
            self.root.after(250, lambda: self._open_path(candidate, quiet=True))

    # ---------- 配置 ----------
    def _load_config(self) -> dict:
        # Read the old portable setting once for backward compatibility, but all
        # subsequent writes go to the per-user state directory.
        for path in (CONFIG_PATH, APP_ROOT / ".worm_roi_gui.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError):
                continue
        return {}

    def _configured_int(self, key: str, default: int) -> int:
        """One integer out of the saved settings, or the default.

        Anything unusable -- absent, a string a hand edit broke, a float -- reads
        as the default rather than raising. These are dialog defaults, so the
        cost of being wrong is a number the user is about to be asked anyway.
        """
        try:
            return int(self.config_data.get(key, default))
        except (TypeError, ValueError):
            return default

    def _save_config(self) -> None:
        data = {
            "language": self.language_var.get(),
            "mode": self.mode_var.get(),
            "worm_count": self.last_valid_worm_count,
            "shape_refinement": self.shape_refinement_var.get(),
            "manual_head_annotation": self.manual_head_annotation_var.get(),
            "brightfield_roi": self.brightfield_roi_var.get(),
            "fluorescence_plane": int(self.fluorescence_plane_var.get()),
            "brightfield_plane": int(self.brightfield_plane_var.get()),
            "segment_selection": self.segment_selection_var.get(),
            "segment_start": round(float(self.segment_start_var.get()), 2),
            "segment_end": round(float(self.segment_end_var.get()), 2),
            "theme": self.theme_var.get(),
            "output_path": self.output_path,
            "recent_paths": self.recent_paths[:10],
            "current_folder": str(self.current_folder),
        }
        try:
            temporary = CONFIG_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, CONFIG_PATH)
        except OSError as exc:
            self._terminal_write(self._tr(
                f"警告：无法保存界面设置：{exc}\n",
                f"WARNING: Cannot save interface settings: {exc}\n"), "warning")

    # ---------- 界面布局 ----------
    def _tr(self, chinese: str, english: str) -> str:
        return english if self.language_var.get() == "en" else chinese

    def _unsupported_reason(self, found) -> str:
        """Bilingual phrasing for a check_image() rejection.

        The values are the raw frame count / mode / error text, so nothing here
        has to parse the wording the library uses.
        """
        code, value = found.code, found.value
        if code == "stack":
            return self._tr("%d 层" % value, "%d planes" % value)
        if code == "colour":
            return self._tr("像素模式 %s，不是单通道灰度" % value,
                            "PIL pixel mode %s is not single-channel greyscale" % value)
        if code == "format":
            return self._tr("%s 文件，不是 TIFF；ImageJ 无法打开" % value,
                            "not a TIFF (%s); ImageJ cannot open it" % value)
        # Keep the underlying error: it is the only clue for a corrupt file.
        return self._tr("无法读取（%s）" % str(value)[:120],
                        "cannot be read (%s)" % str(value)[:120])

    def _mode_label(self, mode: str | None = None) -> str:
        selected = mode or self.mode_var.get()
        return self._tr(
            "高清晰度图像" if selected == "high" else "低清晰度图像",
            "High-clarity images" if selected == "high" else "Low-clarity images")

    def _build_ui(self) -> None:
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.topbar = tk.Frame(self.root, height=52)
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_propagate(False)
        self.topbar.columnconfigure(3, weight=1)

        self.file_button = tk.Menubutton(self.topbar, text=self._tr("文件", "File"), relief="flat", padx=18,
                                         font=("Microsoft YaHei UI", 10))
        self.file_button.grid(row=0, column=0, sticky="ns", padx=(8, 0))
        self.file_menu = tk.Menu(self.file_button, tearoff=False)
        self.file_button.configure(menu=self.file_menu)

        self.settings_button = tk.Menubutton(self.topbar, text=self._tr("设置", "Settings"), relief="flat", padx=18,
                                             font=("Microsoft YaHei UI", 10))
        self.settings_button.grid(row=0, column=1, sticky="ns")
        self.settings_menu = tk.Menu(self.settings_button, tearoff=False)
        self.settings_button.configure(menu=self.settings_menu)

        self.folder_label = tk.Label(self.topbar, anchor="w", padx=14,
                                     font=("Microsoft YaHei UI", 9))
        self.folder_label.grid(row=0, column=3, sticky="ew")

        self.open_output_button = tk.Button(
            self.topbar, text=self._tr("打开结果", "Open Results"), command=self._open_output_folder,
            relief="flat", padx=14, font=("Microsoft YaHei UI", 9)
        )
        self.open_output_button.grid(row=0, column=4, padx=4, pady=9)
        self.stop_button = tk.Button(
            self.topbar, text=self._tr("停止", "Stop"), command=self._stop_processing, state="disabled",
            relief="flat", padx=14, font=("Microsoft YaHei UI", 9)
        )
        self.stop_button.grid(row=0, column=5, padx=4, pady=9)
        self.run_button = tk.Button(
            self.topbar, text=self._tr("开始处理", "Start"), command=self._start_processing,
            relief="flat", padx=18, font=("Microsoft YaHei UI", 9, "bold")
        )
        self.run_button.grid(row=0, column=6, padx=(4, 10), pady=9)

        self.main_pane = tk.PanedWindow(self.root, orient="horizontal", sashwidth=5,
                                        bd=0, relief="flat", showhandle=False)
        self.main_pane.grid(row=1, column=0, sticky="nsew")

        self.left_pane = tk.PanedWindow(self.main_pane, orient="vertical", sashwidth=5,
                                        bd=0, relief="flat", showhandle=False, width=300)
        self.right_pane = tk.PanedWindow(self.main_pane, orient="vertical", sashwidth=5,
                                         bd=0, relief="flat", showhandle=False)
        self.main_pane.add(self.left_pane, minsize=235, width=300)
        self.main_pane.add(self.right_pane, minsize=650)

        self.resource_panel = self._panel(self.left_pane, self._tr("资源管理器", "File Browser"))
        self.settings_panel = self._panel(self.left_pane, self._tr("图像处理", "Image Processing"))
        self.left_pane.add(self.resource_panel, minsize=255)
        self.left_pane.add(self.settings_panel, minsize=335, height=365)

        self.monitor_panel = self._panel(self.right_pane, self._tr("监视器", "Monitor"))
        self.terminal_panel = self._panel(self.right_pane, self._tr("终端", "Log"))
        self.right_pane.add(self.monitor_panel, minsize=390)
        self.right_pane.add(self.terminal_panel, minsize=185, height=260)

        self._build_resource_panel()
        self._build_simple_settings()
        self._build_monitor()
        self._build_terminal()

    def _panel(self, parent: tk.Widget, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bd=0, highlightthickness=1)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        # Keep every title and optional title-bar action inside one full-width
        # header.  Adding an action directly to ``outer`` would create a second
        # outer grid column while the body still occupied only the first one,
        # leaving the panel body and its right edge visibly misaligned.
        header = tk.Frame(outer, bd=0, highlightthickness=0)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        label = tk.Label(header, text=title, anchor="w", padx=12, pady=7,
                         font=("Microsoft YaHei UI", 10, "bold"))
        label.grid(row=0, column=0, sticky="ew")
        body = tk.Frame(outer)
        body.grid(row=1, column=0, sticky="nsew")
        outer.header = header  # type: ignore[attr-defined]
        outer.body = body  # type: ignore[attr-defined]
        outer.title_label = label  # type: ignore[attr-defined]
        return outer

    def _build_resource_panel(self) -> None:
        body = self.resource_panel.body  # type: ignore[attr-defined]
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(body, show="tree", selectmode="browse")
        tree_scroll = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewOpen>>", self._on_tree_open)
        self.tree.bind("<Double-1>", self._on_tree_double_click)
        self.tree.bind("<Button-3>", self._show_tree_context)
        self.tree_menu = tk.Menu(self.tree, tearoff=False)
        self._rebuild_context_menus()

    def _build_simple_settings(self) -> None:
        body = self.settings_panel.body  # type: ignore[attr-defined]
        body.columnconfigure(0, weight=1)
        self.processing_mode_label = tk.Label(
            body, text=self._tr("处理模式", "Processing mode"), anchor="w",
            font=("Microsoft YaHei UI", 9, "bold"))
        self.processing_mode_label.grid(row=0, column=0, sticky="ew", padx=14, pady=(13, 5))
        self.high_radio = tk.Radiobutton(
            body, text=self._mode_label("high"), variable=self.mode_var,
            value="high", anchor="w", command=self._mode_changed,
            font=("Microsoft YaHei UI", 9), highlightthickness=0
        )
        self.high_radio.grid(row=1, column=0, sticky="ew", padx=13, pady=2)
        self.low_radio = tk.Radiobutton(
            body, text=self._mode_label("low"), variable=self.mode_var,
            value="low", anchor="w", command=self._mode_changed,
            font=("Microsoft YaHei UI", 9), highlightthickness=0
        )
        self.low_radio.grid(row=2, column=0, sticky="ew", padx=13, pady=2)
        count_row = tk.Frame(body)
        count_row.grid(row=3, column=0, sticky="ew", padx=14, pady=(10, 3))
        count_row.columnconfigure(1, weight=1)
        self.worm_count_label = tk.Label(
            count_row, text=self._tr("预设虫数 n", "Expected worms n"),
            anchor="w", font=("Microsoft YaHei UI", 9)
        )
        self.worm_count_label.grid(row=0, column=0, sticky="w")
        self.worm_count_tooltip = HoverTooltip(
            self.worm_count_label, self._worm_count_range_text)
        validate_count = (self.root.register(lambda value: value == "" or value.isdigit()), "%P")
        self.worm_count_spinbox = tk.Spinbox(
            count_row, from_=1, to=9999, increment=1, width=7, justify="center",
            textvariable=self.worm_count_var, command=self._worm_count_changed,
            validate="key", validatecommand=validate_count,
            font=("Microsoft YaHei UI", 9), relief="flat", bd=1
        )
        self.worm_count_spinbox.grid(row=0, column=1, sticky="e")
        self.worm_count_spinbox.bind("<Return>", self._worm_count_changed)
        self.worm_count_spinbox.bind("<FocusOut>", self._worm_count_changed)
        self.settings_option_rows = []
        self.settings_option_labels = []
        self.shape_refinement_check, self.shape_refinement_label = self._compact_check_option(
            body, row=4, text=self._tr("平滑修复", "Smoothing repair"),
            variable=self.shape_refinement_var,
            command=self._shape_refinement_changed)
        self.manual_head_annotation_check, self.manual_head_annotation_label = self._compact_check_option(
            body, row=5, text=self._tr(
                "手动标注（头向/分界/排除区）",
                "Manual annotation (head/boundary/exclusion)"),
            variable=self.manual_head_annotation_var,
            command=self._manual_head_annotation_changed)
        self.segment_selection_check, self.segment_selection_label = self._compact_check_option(
            body, row=6, text=self._tr(
                "部分圈画（需手动标注头向）",
                "Partial ROI (manual head direction required)"),
            variable=self.segment_selection_var,
            command=self._segment_selection_changed, pady=(3, 0))
        # 明场ROI 放在最后：它是唯一一个「会改变本次处理的图从哪一层来」的选项，
        # 与上面几个后处理开关不同类，隔开一行更清楚。必须走 _compact_check_option，
        # 否则 settings_option_rows/labels 里没有它，深浅色主题不会给它上色。
        self.brightfield_roi_check, self.brightfield_roi_label = self._compact_check_option(
            body, row=7, text=self._tr(
                "明场ROI（多层 TIFF 需开启）",
                "Brightfield ROI (required for multi-plane TIFFs)"),
            variable=self.brightfield_roi_var,
            command=self._brightfield_roi_changed, pady=(6, 0))
        self.segment_range_frame = tk.Frame(body)
        self.segment_range_frame.grid(row=8, column=0, sticky="ew", padx=13, pady=(0, 2))
        self.segment_range_frame.columnconfigure(0, weight=1)
        self.segment_range_canvas = tk.Canvas(
            self.segment_range_frame, height=48, bd=0, highlightthickness=0,
            cursor="hand2")
        self.segment_range_canvas.grid(row=0, column=0, sticky="ew")
        self.segment_range_canvas.bind("<Configure>", self._draw_segment_range)
        self.segment_range_canvas.bind("<ButtonPress-1>", self._segment_range_press)
        self.segment_range_canvas.bind("<B1-Motion>", self._segment_range_motion)
        self.segment_range_canvas.bind("<ButtonRelease-1>", self._segment_range_release)
        self.output_hint = tk.Label(body, anchor="w", justify="left", wraplength=250,
                                    font=("Microsoft YaHei UI", 8))
        self.output_hint.grid(row=9, column=0, sticky="ew", padx=15, pady=(3, 8))
        self._update_output_hint()

    def _compact_check_option(self, parent: tk.Widget, row: int, text: str,
                              variable: tk.BooleanVar, command,
                              pady=2) -> tuple[CompactCheckbutton, tk.Label]:
        option_row = tk.Frame(parent)
        option_row.grid(row=row, column=0, sticky="w", padx=13, pady=pady)
        checkbox = CompactCheckbutton(option_row, variable, command)
        checkbox.grid(row=0, column=0, sticky="w")
        label = tk.Label(
            option_row, text=text, anchor="w", justify="left",
            font=("Microsoft YaHei UI", 9))
        label.grid(row=0, column=1, sticky="w", padx=(4, 0))
        self.settings_option_rows.append(option_row)
        self.settings_option_labels.append(label)
        return checkbox, label

    def _build_monitor(self) -> None:
        body = self.monitor_panel.body  # type: ignore[attr-defined]
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self.preview_grid = tk.Frame(body)
        self.preview_grid.grid(row=0, column=0, sticky="nsew", padx=7, pady=7)
        for row in range(2):
            self.preview_grid.rowconfigure(row, weight=1, uniform="previewrow")
        for col in range(2):
            self.preview_grid.columnconfigure(col, weight=1, uniform="previewcol")
        self.preview_canvases: list[tk.Canvas] = []
        for idx in range(self.PAGE_SIZE):
            canvas = tk.Canvas(self.preview_grid, highlightthickness=1, bd=0, cursor="hand2")
            canvas.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=4, pady=4)
            canvas.bind("<Button-1>", lambda _event, i=idx: self._select_preview(i))
            canvas.bind("<Double-1>", lambda _event, i=idx: self._open_preview(i))
            canvas.bind("<Configure>", self._schedule_preview_redraw)
            self.preview_canvases.append(canvas)

        controls = tk.Frame(body)
        controls.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        controls.columnconfigure(1, weight=1)
        self.prev_button = tk.Button(controls, text=self._tr("上一页", "Previous"), command=self._previous_page,
                                     relief="flat", font=("Microsoft YaHei UI", 9))
        self.prev_button.grid(row=0, column=0)
        self.page_label = tk.Label(
            controls, text=self._tr("第 0 / 0 页", "Page 0 / 0"),
            font=("Microsoft YaHei UI", 9))
        self.page_label.grid(row=0, column=1)
        self.monitor_status = tk.Label(
            controls, text=self._tr("尚无 QC 图", "No QC images"), anchor="e",
                                       font=("Microsoft YaHei UI", 8))
        self.monitor_status.grid(row=0, column=2, padx=10)
        self.next_button = tk.Button(controls, text=self._tr("下一页", "Next"), command=self._next_page,
                                     relief="flat", font=("Microsoft YaHei UI", 9))
        self.next_button.grid(row=0, column=3)

        # 人工箭头、分界和排除区标注视图与 QC 网格共用监视器区域。
        self.annotation_frame = tk.Frame(body)
        self.annotation_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=7, pady=7)
        self.annotation_frame.rowconfigure(0, weight=1)
        self.annotation_frame.columnconfigure(0, weight=1)
        self.annotation_canvas = tk.Canvas(
            self.annotation_frame, highlightthickness=1, bd=0, cursor="crosshair")
        self.annotation_canvas.grid(row=0, column=0, sticky="nsew")
        self.annotation_canvas.bind("<ButtonPress-1>", self._annotation_press)
        self.annotation_canvas.bind("<B1-Motion>", self._annotation_motion)
        self.annotation_canvas.bind("<ButtonRelease-1>", self._annotation_release)
        # 分界线按 ImageJ 的多边形选择来画：左击落点、右击结束，鼠标移动时拉橡皮筋。
        self.annotation_canvas.bind("<Motion>", self._annotation_pointer_motion)
        self.annotation_canvas.bind("<ButtonPress-3>", self._annotation_finish_boundary)
        self.root.bind_all("<Escape>", self._annotation_cancel_draft)
        self.annotation_canvas.bind("<Configure>", self._schedule_annotation_redraw)

        annotation_controls = tk.Frame(self.annotation_frame)
        annotation_controls.grid(row=1, column=0, sticky="ew", pady=(7, 0))
        annotation_controls.columnconfigure(0, weight=1)
        self.annotation_help = tk.Label(
            annotation_controls,
            text=self._tr(
                "从对应虫体内部按住左键，拖向头部后松开（箭头尖端为头部）",
                "Drag from inside each worm toward its head; the arrow tip marks the head."),
            anchor="w", font=("Microsoft YaHei UI", 9))
        self.annotation_help.grid(row=0, column=0, columnspan=8, sticky="ew", padx=(3, 8))
        self.annotation_arrow_mode = tk.Radiobutton(
            annotation_controls, text=self._tr("头向箭头", "Head arrow"), variable=self.annotation_mode_var,
            value="arrow", command=self._annotation_mode_changed,
            font=("Microsoft YaHei UI", 9), highlightthickness=0)
        self.annotation_arrow_mode.grid(row=1, column=0, padx=(3, 2), sticky="w")
        self.annotation_boundary_mode = tk.Radiobutton(
            annotation_controls, text=self._tr("人工分界线", "Manual boundary"), variable=self.annotation_mode_var,
            value="boundary", command=self._annotation_mode_changed,
            font=("Microsoft YaHei UI", 9), highlightthickness=0)
        self.annotation_boundary_mode.grid(row=1, column=1, padx=2, sticky="w")
        self.annotation_exclusion_mode = tk.Radiobutton(
            annotation_controls, text=self._tr("排除区域", "Exclusion region"), variable=self.annotation_mode_var,
            value="exclusion", command=self._annotation_mode_changed,
            font=("Microsoft YaHei UI", 9), highlightthickness=0)
        self.annotation_exclusion_mode.grid(row=1, column=2, padx=2, sticky="w")
        self.annotation_count_label = tk.Label(
            annotation_controls, text=self._tr("箭头 0", "Arrows 0"),
            font=("Microsoft YaHei UI", 9, "bold"))
        self.annotation_count_label.grid(row=1, column=3, padx=6)
        self.annotation_undo_button = tk.Button(
            annotation_controls, text=self._tr("撤销当前", "Undo current"), command=self._undo_annotation,
            relief="flat", font=("Microsoft YaHei UI", 9))
        self.annotation_undo_button.grid(row=1, column=4, padx=3)
        self.annotation_clear_button = tk.Button(
            annotation_controls, text=self._tr("清空当前", "Clear current"), command=self._clear_annotations,
            relief="flat", font=("Microsoft YaHei UI", 9))
        self.annotation_clear_button.grid(row=1, column=5, padx=3)
        self.annotation_save_button = tk.Button(
            annotation_controls, text=self._tr("保存标注", "Save annotations"), command=self._save_head_annotations,
            relief="flat", font=("Microsoft YaHei UI", 9, "bold"))
        self.annotation_save_button.grid(row=1, column=6, padx=3)
        self.annotation_back_button = tk.Button(
            annotation_controls, text=self._tr("返回 QC", "Back to QC"), command=self._close_head_annotation,
            relief="flat", font=("Microsoft YaHei UI", 9))
        self.annotation_back_button.grid(row=1, column=7, padx=(3, 0))
        self.annotation_frame.grid_remove()

    def _build_terminal(self) -> None:
        # The log exists only in this widget, so give the user an obvious way to
        # get it out; a tester reporting a problem otherwise has nothing to send.
        header = self.terminal_panel.header  # type: ignore[attr-defined]
        self.save_log_button = tk.Button(
            header, text=self._tr("保存日志", "Save Log"), command=self._save_terminal_log,
            relief="flat", bd=0, highlightthickness=0, padx=9, pady=1,
            cursor="hand2", font=("Microsoft YaHei UI", 8)
        )
        self.save_log_button.grid(row=0, column=1, sticky="e", padx=(4, 8), pady=4)
        body = self.terminal_panel.body  # type: ignore[attr-defined]
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self.terminal = tk.Text(
            body, wrap="word", state="disabled", undo=False, padx=10, pady=8,
            font=("Cascadia Mono", 9), cursor="arrow", spacing1=1, spacing3=1
        )
        terminal_scroll = ttk.Scrollbar(body, orient="vertical", command=self.terminal.yview)
        self.terminal.configure(yscrollcommand=terminal_scroll.set)
        self.terminal.grid(row=0, column=0, sticky="nsew")
        terminal_scroll.grid(row=0, column=1, sticky="ns")
        self.terminal.bind("<Double-1>", self._terminal_double_click)
        self.terminal.bind("<Button-3>", self._show_terminal_context)
        self.terminal_menu = tk.Menu(self.terminal, tearoff=False)
        self._rebuild_context_menus()
        self._terminal_write(
            self._tr(
                "自动圈虫图形界面已就绪。\n"
                "选择包含 TIFF 图片的文件夹，然后点击“开始处理”。\n"
                "双击资源树、QC 预览或终端中的路径可打开对应文件。\n\n",
                "Auto Worm ROI is ready.\n"
                "Select a folder containing TIFF images, then click Start.\n"
                "Double-click a file, QC preview, or path in the log to open it.\n\n"),
            "info",
        )

    # ---------- 菜单 ----------
    def _rebuild_context_menus(self) -> None:
        if hasattr(self, "tree_menu"):
            self.tree_menu.delete(0, "end")
            self.tree_menu.add_command(
                label=self._tr("打开", "Open"), command=self._open_tree_selection)
            self.tree_menu.add_command(
                label=self._tr("在资源管理器中显示", "Show in File Explorer"),
                command=self._reveal_tree_selection)
            self.tree_menu.add_separator()
            self.tree_menu.add_command(
                label=self._tr("将此文件夹设为当前目录", "Use this folder as input"),
                command=self._use_tree_folder)
        if hasattr(self, "terminal_menu"):
            self.terminal_menu.delete(0, "end")
            self.terminal_menu.add_command(
                label=self._tr("复制", "Copy"),
                command=lambda: self.terminal.event_generate("<<Copy>>"))
            self.terminal_menu.add_command(
                label=self._tr("打开当前行中的文件/目录", "Open file/folder on current line"),
                command=self._open_terminal_line)
            self.terminal_menu.add_separator()
            self.terminal_menu.add_command(
                label=self._tr("保存日志…", "Save log…"), command=self._save_terminal_log)
            self.terminal_menu.add_command(
                label=self._tr("清空终端", "Clear log"), command=self._clear_terminal)

    def _rebuild_file_menu(self) -> None:
        menu = self.file_menu
        menu.delete(0, "end")
        menu.add_command(label=self._tr("打开文件…", "Open File…"),
                         command=self._choose_file, accelerator="Ctrl+O")
        menu.add_command(label=self._tr("打开文件夹…", "Open Folder…"),
                         command=self._choose_folder, accelerator="Ctrl+Shift+O")
        recent = tk.Menu(menu, tearoff=False)
        if self.recent_paths:
            for item in self.recent_paths:
                recent.add_command(label=item, command=lambda p=item: self._open_recent(p))
            recent.add_separator()
            recent.add_command(label=self._tr("清除最近记录", "Clear Recent"), command=self._clear_recent)
        else:
            recent.add_command(label=self._tr("（无最近文件）", "(No recent files)"), state="disabled")
        menu.add_cascade(label=self._tr("打开最近文件", "Open Recent"), menu=recent)
        menu.add_separator()
        menu.add_command(label=self._tr("另存为…", "Save As…"),
                         command=self._save_as, accelerator="Ctrl+Shift+S")
        menu.add_separator()
        menu.add_command(label=self._tr("退出", "Exit"), command=self._on_close, accelerator="Alt+F4")
        self.root.bind_all("<Control-o>", lambda _event: self._choose_file())
        self.root.bind_all("<Control-Shift-O>", lambda _event: self._choose_folder())
        self.root.bind_all("<Control-Shift-S>", lambda _event: self._save_as())

    def _rebuild_settings_menu(self) -> None:
        menu = self.settings_menu
        menu.delete(0, "end")
        mode_menu = tk.Menu(menu, tearoff=False)
        mode_menu.add_radiobutton(label=self._mode_label("high"), variable=self.mode_var,
                                  value="high", command=self._mode_changed)
        mode_menu.add_radiobutton(label=self._mode_label("low"), variable=self.mode_var,
                                  value="low", command=self._mode_changed)
        menu.add_cascade(label=self._tr("模式切换", "Processing Mode"), menu=mode_menu)
        menu.add_command(
            label=self._tr(
                f"预设虫数 n：{self.last_valid_worm_count}…",
                f"Expected worms n: {self.last_valid_worm_count}…"),
            command=self._prompt_worm_count)
        menu.add_checkbutton(label=self._tr("平滑修复", "Smoothing repair"),
                             variable=self.shape_refinement_var,
                             command=self._shape_refinement_changed)
        menu.add_checkbutton(label=self._tr(
                                 "手动标注（头向/分界/排除区）",
                                 "Manual annotation (head/boundary/exclusion)"),
                             variable=self.manual_head_annotation_var,
                             command=self._manual_head_annotation_changed)
        menu.add_checkbutton(label=self._tr(
                                 "部分圈画（需手动标注头向）",
                                 "Partial ROI (manual head direction required)"),
                             variable=self.segment_selection_var,
                             command=self._segment_selection_changed)
        menu.add_checkbutton(label=self._tr(
                                 "明场ROI（多层 TIFF 需开启）",
                                 "Brightfield ROI (required for multi-plane TIFFs)"),
                             variable=self.brightfield_roi_var,
                             command=self._brightfield_roi_changed)
        menu.add_separator()
        theme_menu = tk.Menu(menu, tearoff=False)
        theme_menu.add_radiobutton(label=self._tr("深色", "Dark"), variable=self.theme_var,
                                   value="dark", command=self._theme_changed)
        theme_menu.add_radiobutton(label=self._tr("浅色", "Light"), variable=self.theme_var,
                                   value="light", command=self._theme_changed)
        menu.add_cascade(label=self._tr("主题", "Theme"), menu=theme_menu)
        language_menu = tk.Menu(menu, tearoff=False)
        language_menu.add_radiobutton(label="中文", variable=self.language_var,
                                      value="zh", command=self._language_changed)
        language_menu.add_radiobutton(label="English", variable=self.language_var,
                                      value="en", command=self._language_changed)
        menu.add_cascade(label=self._tr("语言", "Language"), menu=language_menu)
        menu.add_separator()
        menu.add_command(label=self._tr("修改结果保存路径…", "Change output folder…"),
                         command=self._choose_output_path)
        menu.add_command(label=self._tr("恢复默认保存路径", "Restore default output folder"),
                         command=self._reset_output_path)
        menu.add_separator()
        menu.add_command(label=self._tr("CUDA 与系统信息", "CUDA and System Information"),
                         command=self._show_cuda_info)

    def _cuda_runtime_info(self) -> tuple[list[str], list[str]]:
        """Return display lines and blocking CUDA compatibility problems."""
        lines = [self._tr(
            "运行模式：ImageJ 插件桥接" if IMAGEJ_MEASUREMENT_MODE else "运行模式：独立界面",
            "Mode: ImageJ plug-in bridge" if IMAGEJ_MEASUREMENT_MODE else "Mode: standalone GUI")]
        if IMAGEJ_MEASUREMENT_MODE and not IMAGEJ_BRIDGE_DIR:
            # AUTOWORM_IMAGEJ_MODE is only ever set by the plug-in itself, so this
            # pair cannot happen by accident: the installed jar and this program
            # are different versions, and the plug-in is waiting for a
            # notification that will never come. It would measure nothing and say
            # nothing, which is the one outcome worth interrupting for.
            lines.append(self._tr(
                "警告：未收到桥接目录，ImageJ 收不到本次结果。"
                "请确认 Auto_Worm_ROI.jar 与 AutoWormGUI.exe 来自同一个发布包。",
                "WARNING: no bridge directory was given, so ImageJ will not receive "
                "these results. Check that Auto_Worm_ROI.jar and AutoWormGUI.exe "
                "come from the same release."))
        problems = []
        try:
            import torch
            lines.append("PyTorch: " + str(torch.__version__))
            lines.append("Bundled CUDA runtime: " + str(torch.version.cuda or "none"))
            lines.append("Bundled cuDNN: " + str(torch.backends.cudnn.version() or "none"))
            available = bool(torch.cuda.is_available())
            lines.append("CUDA available: " + str(available))
            if available:
                lines.append("GPU: " + torch.cuda.get_device_name(0))
            else:
                problems.append(self._tr(
                    "CUDA 不可用：需要 NVIDIA CUDA GPU 和兼容驱动。",
                    "CUDA is unavailable: an NVIDIA CUDA GPU and compatible driver are required."))
        except Exception as exc:
            lines.append("PyTorch/CUDA probe failed: " + str(exc))
            problems.append(self._tr(
                "无法初始化内置 CUDA 运行库。",
                "The bundled CUDA runtime could not be initialized."))

        driver = ""
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            driver = completed.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass
        lines.append("NVIDIA driver: " + (driver or "not detected"))
        if driver:
            try:
                if int(driver.split(".", 1)[0]) < CUDA_MIN_DRIVER_MAJOR:
                    problems.append(self._tr(
                        f"NVIDIA 驱动 {driver} 过旧；内置 CUDA 13.x 至少需要 580 系列驱动。",
                        f"NVIDIA driver {driver} is too old; bundled CUDA 13.x requires driver 580 or newer."))
            except ValueError:
                pass
        lines.append(self._tr(
            "要求：Windows 10/11、NVIDIA CUDA GPU、驱动 580 或更高；无需安装 CUDA Toolkit。",
            "Requires Windows 10/11, an NVIDIA CUDA GPU and driver 580 or newer; "
            "CUDA Toolkit is not required."))
        return lines, problems

    def _show_cuda_info(self) -> None:
        lines, problems = self._cuda_runtime_info()
        message = "\n".join(lines)
        title = self._tr("CUDA 与系统信息", "CUDA and System Information")
        if problems:
            messagebox.showwarning(title, message + "\n\n" + "\n".join(problems), parent=self.root)
        else:
            messagebox.showinfo(title, message, parent=self.root)

    def _language_changed(self) -> None:
        if self.language_var.get() not in {"zh", "en"}:
            self.language_var.set("zh")
        self._apply_language()
        self._apply_theme()
        self._save_config()
        self._terminal_write(self._tr(
            "语言已切换：中文。\n", "Language switched to English.\n"), "info")

    def _apply_language(self) -> None:
        self.root.title(self._tr(
            f"秀丽隐杆线虫荧光定量 · 自动圈虫 {APP_VERSION}",
            f"C. elegans Fluorescence Quantification · Auto Worm ROI {APP_VERSION}"))
        self.file_button.configure(text=self._tr("文件", "File"))
        self.settings_button.configure(text=self._tr("设置", "Settings"))
        self.open_output_button.configure(text=self._tr("打开结果", "Open Results"))
        self.stop_button.configure(text=self._tr("停止", "Stop"))
        self.run_button.configure(text=self._tr(
            "处理中…" if self._is_processing() else "开始处理",
            "Processing…" if self._is_processing() else "Start"))
        self.resource_panel.title_label.configure(text=self._tr("资源管理器", "File Browser"))
        self.settings_panel.title_label.configure(text=self._tr("图像处理", "Image Processing"))
        self.terminal_panel.title_label.configure(text=self._tr("终端", "Log"))
        self.save_log_button.configure(text=self._tr("保存日志", "Save Log"))
        monitor_title = (self._tr("监视器", "Monitor") if self.annotation_path is None else
                         self._tr("监视器 · 手动标注 · ", "Monitor · Manual annotation · ") +
                         self.annotation_path.name)
        self.monitor_panel.title_label.configure(text=monitor_title)
        self.processing_mode_label.configure(text=self._tr("处理模式", "Processing mode"))
        self.high_radio.configure(text=self._mode_label("high"))
        self.low_radio.configure(text=self._mode_label("low"))
        self.worm_count_label.configure(text=self._tr("预设虫数 n", "Expected worms n"))
        self.shape_refinement_label.configure(text=self._tr("平滑修复", "Smoothing repair"))
        self.manual_head_annotation_label.configure(text=self._tr(
            "手动标注（头向/分界/排除区）",
            "Manual annotation (head/boundary/exclusion)"))
        self.segment_selection_label.configure(text=self._tr(
            "部分圈画（需手动标注头向）",
            "Partial ROI (manual head direction required)"))
        self.brightfield_roi_label.configure(text=self._tr(
            "明场ROI（多层 TIFF 需开启）",
            "Brightfield ROI (required for multi-plane TIFFs)"))
        self.prev_button.configure(text=self._tr("上一页", "Previous"))
        self.next_button.configure(text=self._tr("下一页", "Next"))
        self.annotation_arrow_mode.configure(text=self._tr("头向箭头", "Head arrow"))
        self.annotation_boundary_mode.configure(text=self._tr("人工分界线", "Manual boundary"))
        self.annotation_exclusion_mode.configure(text=self._tr("排除区域", "Exclusion region"))
        self.annotation_undo_button.configure(text=self._tr("撤销当前", "Undo current"))
        self.annotation_clear_button.configure(text=self._tr("清空当前", "Clear current"))
        self.annotation_save_button.configure(text=self._tr("保存标注", "Save annotations"))
        self.annotation_back_button.configure(text=self._tr("返回 QC", "Back to QC"))
        self._rebuild_file_menu()
        self._rebuild_settings_menu()
        self._rebuild_context_menus()
        self._update_output_hint()
        self._annotation_mode_changed()
        self._draw_segment_range()
        self._draw_previews()

    # ---------- 资源管理器 ----------
    def _load_folder(self, folder: Path, add_recent: bool = True) -> None:
        if self._is_processing():
            self._terminal_write(self._tr(
                "处理期间不能切换输入文件夹。\n",
                "The input folder cannot be changed while processing.\n"), "warning")
            return
        if self.annotation_path is not None:
            self._close_head_annotation()
        try:
            folder = folder.resolve()
        except OSError:
            return
        if not folder.is_dir():
            return
        self.current_folder = folder
        if add_recent:
            self._add_recent(folder)
        self.folder_label.configure(text=str(folder))
        self.tree.delete(*self.tree.get_children())
        self.tree_paths.clear()
        root_item = self.tree.insert("", "end", text=f"📁 {folder.name or str(folder)}", open=True)
        self.tree_paths[root_item] = folder
        self._populate_tree_node(root_item, folder)
        self._update_output_hint()
        self._scan_qc_images(reset_page=True)
        self._save_config()

    def _populate_tree_node(self, item: str, folder: Path) -> None:
        for child in self.tree.get_children(item):
            self.tree.delete(child)
        try:
            entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except (OSError, PermissionError) as exc:
            error_item = self.tree.insert(item, "end", text=self._tr(
                f"⚠ 无法读取：{exc}", f"⚠ Cannot read: {exc}"))
            self.tree_paths[error_item] = folder
            return
        for path in entries:
            if path.name.startswith("."):
                continue
            icon = "📁" if path.is_dir() else self._file_icon(path)
            node = self.tree.insert(item, "end", text=f"{icon} {path.name}")
            self.tree_paths[node] = path
            if path.is_dir():
                self.tree.insert(node, "end", text="")

    @staticmethod
    def _file_icon(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            return "▧"
        if suffix in {".png", ".jpg", ".jpeg"}:
            return "▣"
        if suffix in {".zip", ".csv"}:
            return "▫"
        return "•"

    def _on_tree_open(self, _event: tk.Event) -> None:
        item = self.tree.focus()
        path = self.tree_paths.get(item)
        if path and path.is_dir():
            children = self.tree.get_children(item)
            if len(children) == 1 and not self.tree.item(children[0], "text"):
                self._populate_tree_node(item, path)

    def _on_tree_double_click(self, _event: tk.Event) -> None:
        item = self.tree.focus()
        path = self.tree_paths.get(item)
        if path and path.is_file():
            self._open_path(path)

    def _show_tree_context(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.tree.focus(item)
            self.tree_menu.tk_popup(event.x_root, event.y_root)

    def _tree_selected_path(self) -> Path | None:
        selected = self.tree.selection()
        return self.tree_paths.get(selected[0]) if selected else None

    def _open_tree_selection(self) -> None:
        path = self._tree_selected_path()
        if path:
            self._open_path(path)

    def _reveal_tree_selection(self) -> None:
        path = self._tree_selected_path()
        if path:
            self._reveal_in_explorer(path)

    def _use_tree_folder(self) -> None:
        path = self._tree_selected_path()
        if path:
            self._load_folder(path if path.is_dir() else path.parent)

    # ---------- 文件操作 ----------
    def _choose_file(self) -> None:
        if self._is_processing():
            self._terminal_write(self._tr(
                "处理期间不能切换输入文件。\n",
                "The input file cannot be changed while processing.\n"), "warning")
            return
        filename = filedialog.askopenfilename(
            parent=self.root, title=self._tr("打开文件", "Open File"),
            initialdir=str(self.current_folder),
            filetypes=[
                (self._tr("TIFF 图片", "TIFF images"), "*.tif *.tiff"),
                (self._tr("图像文件", "Image files"), "*.tif *.tiff *.png *.jpg *.jpeg"),
                (self._tr("所有文件", "All files"), "*.*")]
        )
        if not filename:
            return
        path = Path(filename)
        self._load_folder(path.parent, add_recent=False)
        self._add_recent(path)
        self._open_path(path)

    def _choose_folder(self) -> None:
        if self._is_processing():
            self._terminal_write(self._tr(
                "处理期间不能切换输入文件夹。\n",
                "The input folder cannot be changed while processing.\n"), "warning")
            return
        folder = filedialog.askdirectory(
            parent=self.root, title=self._tr("打开文件夹", "Open Folder"),
            initialdir=str(self.current_folder))
        if folder:
            self._load_folder(Path(folder))

    def _open_recent(self, value: str) -> None:
        if self._is_processing():
            self._terminal_write(self._tr(
                "处理期间不能切换输入路径。\n",
                "The input path cannot be changed while processing.\n"), "warning")
            return
        path = Path(value)
        if not path.exists():
            messagebox.showwarning(
                self._tr("路径不存在", "Path Not Found"),
                self._tr(f"找不到：\n{path}", f"Cannot find:\n{path}"), parent=self.root)
            self.recent_paths = [p for p in self.recent_paths if p != value]
            self._rebuild_file_menu()
            return
        if path.is_dir():
            self._load_folder(path)
        else:
            self._load_folder(path.parent, add_recent=False)
            self._add_recent(path)
            self._open_path(path)

    def _add_recent(self, path: Path) -> None:
        value = str(path.resolve())
        self.recent_paths = [p for p in self.recent_paths if os.path.normcase(p) != os.path.normcase(value)]
        self.recent_paths.insert(0, value)
        self.recent_paths = self.recent_paths[:10]
        self._rebuild_file_menu()
        self._save_config()

    def _clear_recent(self) -> None:
        self.recent_paths.clear()
        self._rebuild_file_menu()
        self._save_config()

    def _save_as(self) -> None:
        source = self.selected_qc
        if not source or not source.is_file():
            selected = self._tree_selected_path()
            source = selected if selected and selected.is_file() else None
        if not source:
            messagebox.showinfo(
                self._tr("另存为", "Save As"),
                self._tr(
                    "请先在监视器中选择一张 QC 图，\n或在资源管理器中选择一个文件。",
                    "Select a QC image in the monitor or a file in the file browser first."),
                parent=self.root)
            return
        target = filedialog.asksaveasfilename(
            parent=self.root, title=self._tr("另存为", "Save As"), initialdir=str(source.parent),
            initialfile=source.name, defaultextension=source.suffix,
            filetypes=[
                (self._tr(f"{source.suffix.upper()} 文件", f"{source.suffix.upper()} file"),
                 f"*{source.suffix}"),
                (self._tr("所有文件", "All files"), "*.*")]
        )
        if not target:
            return
        try:
            shutil.copy2(source, Path(target))
            self._terminal_write(self._tr(
                f"已另存为：{target}\n", f"Saved as: {target}\n"), "success")
        except OSError as exc:
            messagebox.showerror(self._tr("保存失败", "Save Failed"), str(exc), parent=self.root)

    # ---------- 设置 ----------
    def _mode_changed(self) -> None:
        label = self._mode_label()
        self._terminal_write(self._tr(
            f"模式已切换：{label}\n", f"Processing mode: {label}\n"), "info")
        self._save_config()

    def _worm_count_range_text(self) -> str:
        count = self.last_valid_worm_count
        return self._tr(
            f"识别数量必须等于 n={count}；不匹配时需人工复核",
            f"Detected count must equal n={count}; mismatches require manual review")

    def _validated_worm_count(self, show_error: bool = False) -> int | None:
        try:
            count = int(self.worm_count_var.get().strip())
            if count < 1:
                raise ValueError
        except (TypeError, ValueError):
            if show_error:
                messagebox.showerror(
                    self._tr("预设虫数无效", "Invalid Expected Count"),
                    self._tr("预设虫数 n 必须是正整数。",
                             "Expected worms n must be a positive integer."),
                    parent=self.root)
                self.worm_count_spinbox.focus_set()
            return None
        return count

    def _worm_count_changed(self, _event: tk.Event | None = None) -> None:
        count = self._validated_worm_count(show_error=False)
        if count is None:
            self.worm_count_var.set(str(self.last_valid_worm_count))
            return
        self.last_valid_worm_count = count
        self.worm_count_var.set(str(count))
        if hasattr(self, "annotation_count_label"):
            self._update_annotation_count()
        self._rebuild_settings_menu()
        self._save_config()

    def _prompt_worm_count(self) -> None:
        count = simpledialog.askinteger(
            self._tr("预设虫数 n", "Expected worms n"),
            self._tr("请输入每张图片的预设虫数 n（正整数）：",
                     "Enter the expected number of worms per image (positive integer):"),
            parent=self.root, initialvalue=self.last_valid_worm_count, minvalue=1
        )
        if count is not None:
            self.worm_count_var.set(str(count))
            self._worm_count_changed()

    def _prompt_planes(self) -> tuple[int, int] | None:
        """问一次荧光层与明场层，返回 (荧光层, 明场层)；取消返回 None。

        每批问一次而不是记住上一次的答案：同一台机器上不同批次的采集设置不同，
        「上次是 1/2」既可能是快捷方式也可能是陷阱，而选错层的后果是一整批看起来
        完全正常的错误数字。

        两个 askinteger 是仅有的两个对话框，取消任何一个都返回 None。在此之前
        没有做过任何有副作用的事，所以取消就是什么都没发生。
        """
        fluorescence = simpledialog.askinteger(
            self._tr("荧光层（用于测量）", "Fluorescence plane (measured)"),
            self._tr(
                "第几层是荧光图？这一层用来测量数值（CTCF 等）。\n\n"
                "本批所有多层 TIFF 共用这个层号。",
                "Which plane holds the fluorescence image? This is the plane the "
                "numbers are measured from.\n\nEvery multi-plane TIFF in this batch "
                "uses the same plane number."),
            parent=self.root, initialvalue=int(self.fluorescence_plane_var.get()),
            minvalue=1)
        if fluorescence is None:
            return None
        brightfield = simpledialog.askinteger(
            self._tr("明场层（用于圈 ROI）", "Brightfield plane (segmented)"),
            self._tr(
                f"第几层是明场图？ROI 在这一层上圈出，然后套用到第 {fluorescence} 层"
                "的荧光图上测量。\n\n"
                "本批所有多层 TIFF 共用这个层号。",
                f"Which plane holds the brightfield image? The ROIs are drawn on this "
                f"plane and then applied to the fluorescence image on plane "
                f"{fluorescence} for measurement.\n\nEvery multi-plane TIFF in this "
                "batch uses the same plane number."),
            parent=self.root, initialvalue=int(self.brightfield_plane_var.get()),
            minvalue=1)
        if brightfield is None:
            return None
        if brightfield == fluorescence:
            messagebox.showerror(
                self._tr("层号冲突", "Plane Numbers Clash"),
                self._tr(
                    f"荧光层和明场层都填了第 {fluorescence} 层，两者必须不同："
                    "明场层用来圈出 ROI，荧光层用来测量数值，同一层无法同时担当。\n\n"
                    "请重新开始处理并填写两个不同的层号。",
                    f"The fluorescence plane and the brightfield plane are both "
                    f"{fluorescence}; they must differ. The brightfield plane is where "
                    "the ROIs come from and the fluorescence plane is where the numbers "
                    "come from, and one plane cannot be both.\n\nStart processing again "
                    "and give two different numbers."),
                parent=self.root)
            return None
        return fluorescence, brightfield

    def _setting_changed(self) -> None:
        self._save_config()

    def _shape_refinement_changed(self) -> None:
        state = self._tr("已开启", "enabled") if self.shape_refinement_var.get() else self._tr("已关闭", "disabled")
        self._terminal_write(self._tr(
            f"平滑修复：{state}\n", f"Smoothing repair: {state}\n"), "info")
        self._save_config()

    def _manual_head_annotation_changed(self) -> None:
        enabled = self.manual_head_annotation_var.get()
        if not enabled and self.segment_selection_var.get():
            self.manual_head_annotation_var.set(True)
            self.root.bell()
            self._terminal_write(self._tr(
                "部分圈画已开启，不能关闭手动头向标注。\n",
                "Manual head annotation cannot be disabled while Partial ROI is enabled.\n"),
                "warning")
            self._rebuild_settings_menu()
            self._save_config()
            return
        state = self._tr("已开启", "enabled") if enabled else self._tr("已关闭", "disabled")
        self._terminal_write(self._tr(
            f"手动标注（头向/分界/排除区）：{state}\n",
            f"Manual annotation (head/boundary/exclusion): {state}\n"), "info")
        if not enabled and self.annotation_path is not None:
            self._close_head_annotation()
        self._save_config()

    def _segment_selection_changed(self) -> None:
        enabled = self.segment_selection_var.get()
        if enabled and not self.manual_head_annotation_var.get():
            self.manual_head_annotation_var.set(True)
            self._terminal_write(self._tr(
                "部分圈画已自动开启手动头向标注。\n",
                "Partial ROI automatically enabled manual head annotation.\n"), "info")
        start, end = self._segment_range_values()
        state = self._tr("已开启", "enabled") if enabled else self._tr("已关闭", "disabled")
        self._terminal_write(self._tr(
            "部分圈画：%s；头=0，尾=1，范围 %.2f–%.2f\n" % (state, start, end),
            "Partial ROI: %s; head=0, tail=1, range %.2f–%.2f\n" %
            (state, start, end)), "info")
        self._draw_segment_range()
        self._rebuild_settings_menu()
        self._save_config()

    def _brightfield_roi_changed(self) -> None:
        enabled = self.brightfield_roi_var.get()
        state = self._tr("已开启", "enabled") if enabled else self._tr("已关闭", "disabled")
        self._terminal_write(self._tr(
            f"明场ROI：{state}\n", f"Brightfield ROI: {state}\n"), "info")
        if enabled:
            self._terminal_write(self._tr(
                "处理多层 TIFF 时会在开始处理前询问荧光层与明场层（每批问一次）；"
                "单层 TIF 不受影响，照常处理。\n",
                "Multi-plane TIFFs will ask for the fluorescence and brightfield "
                "plane numbers before the batch starts (once per batch); "
                "single-plane TIFFs are unaffected.\n"), "info")
        self._rebuild_settings_menu()
        self._save_config()

    def _segment_range_values(self) -> tuple[float, float]:
        start = min(max(float(self.segment_start_var.get()), 0.0), 0.99)
        end = min(max(float(self.segment_end_var.get()), start + 0.01), 1.0)
        return round(start, 2), round(end, 2)

    def _segment_range_geometry(self) -> tuple[float, float, float]:
        width = max(80, self.segment_range_canvas.winfo_width())
        return 16.0, float(width - 16), 29.0

    def _draw_segment_range(self, _event: tk.Event | None = None) -> None:
        if not hasattr(self, "segment_range_canvas"):
            return
        canvas = self.segment_range_canvas
        colors = THEMES[self.theme_var.get()]
        canvas.delete("all")
        canvas.configure(bg=colors["panel"])
        left, right, y = self._segment_range_geometry()
        start, end = self._segment_range_values()
        start_x = left + start * (right - left)
        end_x = left + end * (right - left)
        muted = colors["border"] if self.segment_selection_var.get() else colors["muted"]
        active = colors["accent"] if self.segment_selection_var.get() else colors["border"]
        canvas.create_text(left, 7, text=self._tr("头  %.2f", "Head  %.2f") % start, anchor="nw",
                           fill=colors["text"], font=("Microsoft YaHei UI", 8))
        canvas.create_text(right, 7, text=self._tr("%.2f  尾", "%.2f  Tail") % end, anchor="ne",
                           fill=colors["text"], font=("Microsoft YaHei UI", 8))
        canvas.create_line(left, y, right, y, fill=muted, width=4)
        canvas.create_line(start_x, y, end_x, y, fill=active, width=6)
        for x, tag in ((start_x, "start"), (end_x, "end")):
            canvas.create_oval(x-7, y-7, x+7, y+7, fill=active,
                               outline="#ffffff", width=2, tags=("handle", tag))

    def _segment_range_press(self, event: tk.Event) -> None:
        if self._is_processing() or not self.segment_selection_var.get():
            self.root.bell()
            return
        left, right, _y = self._segment_range_geometry()
        start, end = self._segment_range_values()
        start_x = left + start * (right - left)
        end_x = left + end * (right - left)
        self.segment_drag_handle = (
            "start" if abs(event.x - start_x) <= abs(event.x - end_x) else "end")
        self._set_segment_range_from_x(event.x)

    def _segment_range_motion(self, event: tk.Event) -> None:
        if self.segment_drag_handle is not None:
            self._set_segment_range_from_x(event.x)

    def _segment_range_release(self, event: tk.Event) -> None:
        if self.segment_drag_handle is not None:
            self._set_segment_range_from_x(event.x)
            self.segment_drag_handle = None
            start, end = self._segment_range_values()
            self._terminal_write(self._tr(
                "区段范围已设置：头=0，尾=1，选择 %.2f–%.2f\n" % (start, end),
                "Partial ROI range set: head=0, tail=1, selected %.2f–%.2f\n" %
                (start, end)), "info")
            self._save_config()

    def _set_segment_range_from_x(self, x: float) -> None:
        left, right, _y = self._segment_range_geometry()
        value = round(min(max((float(x) - left) / max(right - left, 1.0), 0.0), 1.0), 2)
        start, end = self._segment_range_values()
        if self.segment_drag_handle == "start":
            self.segment_start_var.set(min(value, end - 0.01))
        elif self.segment_drag_handle == "end":
            self.segment_end_var.set(max(value, start + 0.01))
        self._draw_segment_range()

    def _theme_changed(self) -> None:
        self._apply_theme()
        self._save_config()

    def _choose_output_path(self) -> None:
        if self._is_processing():
            self._terminal_write(self._tr(
                "处理期间不能修改结果保存路径。\n",
                "The output folder cannot be changed while processing.\n"), "warning")
            return
        initial = self.output_path or str(self.current_folder / "_auto_roi")
        folder = filedialog.askdirectory(
            parent=self.root, title=self._tr("选择结果保存路径", "Select Output Folder"),
            initialdir=initial)
        if folder:
            self.output_path = str(Path(folder).resolve())
            self._update_output_hint()
            self._scan_qc_images(reset_page=True)
            self._save_config()
            self._terminal_write(self._tr(
                f"结果保存路径：{self.output_path}\n",
                f"Output folder: {self.output_path}\n"), "info")

    def _reset_output_path(self) -> None:
        if self._is_processing():
            self._terminal_write(self._tr(
                "处理期间不能修改结果保存路径。\n",
                "The output folder cannot be changed while processing.\n"), "warning")
            return
        self.output_path = ""
        self._update_output_hint()
        self._scan_qc_images(reset_page=True)
        self._save_config()
        self._terminal_write(self._tr(
            "已恢复默认保存路径：当前文件夹\\_auto_roi\n",
            "Default output restored: current folder\\_auto_roi\n"), "info")

    def _effective_output_dir(self) -> Path:
        return Path(self.output_path) if self.output_path else self.current_folder / "_auto_roi"

    def _monitor_output_dir(self) -> Path:
        if self.active_job is not None:
            return Path(str(self.active_job["output_dir"]))
        return self._effective_output_dir()

    def _is_processing(self) -> bool:
        return bool(self.process and self.process.is_alive())

    def _set_processing_controls(self, running: bool) -> None:
        normal = "disabled" if running else "normal"
        self.file_button.configure(state=normal)
        self.settings_button.configure(state=normal)
        self.high_radio.configure(state=normal)
        self.low_radio.configure(state=normal)
        self.worm_count_spinbox.configure(state=normal)
        self.shape_refinement_check.set_state(normal)
        self.manual_head_annotation_check.set_state(normal)
        self.segment_selection_check.set_state(normal)
        self.brightfield_roi_check.set_state(normal)
        self.segment_range_canvas.configure(state=normal)
        for button in (self.annotation_undo_button, self.annotation_clear_button,
                       self.annotation_save_button, self.annotation_back_button):
            button.configure(state=normal)
        self.run_button.configure(
            state="disabled" if running else "normal",
            text=self._tr("处理中…" if running else "开始处理",
                          "Processing…" if running else "Start"))
        self.stop_button.configure(state="normal" if running else "disabled")

    def _update_output_hint(self) -> None:
        if not hasattr(self, "output_hint"):
            return
        output = self.output_path if self.output_path else self._tr(
            "当前文件夹\\_auto_roi（默认）", "Current folder\\_auto_roi (default)")
        self.output_hint.configure(text=self._tr(
            f"结果保存至：\n{output}", f"Results saved to:\n{output}"))

    # ---------- 处理任务/终端 ----------
    def _validate_runtime(self) -> list[str]:
        problems = []
        # 打包后不再依赖外部 python 或独立脚本,只校验模型文件即可。
        config = MODEL_CONFIGS[self.mode_var.get()]
        for key, label in [
                ("checkpoint", self._tr("主分割模型", "segmentation model")),
                ("tip_checkpoint", self._tr("头尾精修模型", "head/tail refinement model"))]:
            path = config[key]
            if not path.is_file():
                problems.append(self._tr(
                    f"{label}不存在：{path}", f"Missing {label}: {path}"))
        if self.brightfield_roi_var.get():
            # 第一道防线。权重缺失时必须说清楚缺哪个文件、该在哪儿，而不是让批处理
            # 跑起来再报一个 FileNotFoundError，更不能悄悄改用主模型。
            # 0.5.0 起明场权重随包发布，所以这里只剩「装坏了 / 被挪走了」一种成因，
            # 提示也照这个说——原来那句「本版本尚未随包发布明场模型」现在与事实相反。
            missing = []
            for key, label in [
                    ("checkpoint", self._tr("明场分割模型", "brightfield segmentation model")),
                    ("tip_checkpoint", self._tr(
                        "明场头尾精修模型", "brightfield head/tail refinement model"))]:
                path = BRIGHTFIELD_MODEL_CONFIG[key]
                if not path.is_file():
                    missing.append(self._tr(
                        f"{label}不存在：{path}", f"Missing {label}: {path}"))
            if missing:
                problems.extend(missing)
                problems.append(self._tr(
                    "明场模型文件缺失，明场ROI 功能无法启动。可以试试重新解压一份完整的"
                    "发布包（上面列出的路径必须在包里存在）；或关闭左栏的「明场ROI」，"
                    "只处理单层 TIFF。",
                    "The brightfield model files are missing, so Brightfield ROI cannot "
                    "start. Re-extract a complete release package (the paths listed above "
                    "must exist inside it), or turn the option off and process "
                    "single-plane TIFFs only."))
        _, cuda_problems = self._cuda_runtime_info()
        problems.extend(cuda_problems)
        return problems

    def _start_processing(self) -> None:
        if self.process and self.process.is_alive():
            messagebox.showinfo(
                self._tr("正在处理", "Processing"),
                self._tr("已有一个处理任务在运行。", "A processing job is already running."),
                parent=self.root)
            return
        if (self.brightfield_roi_var.get() and IMAGEJ_MEASUREMENT_MODE
                and IMAGEJ_BRIDGE_PROTOCOL != IMAGEJ_BRIDGE_PROTOCOL_REQUIRED):
            # 最危险的一种搭配：新界面 + 旧 jar。旧 jar 不认识通知里的层号，
            # 只会测第 1 个切片——荧光层是第 2 层时每一张都静默测错，而且
            # 没有任何地方会报出来。这里直接拒绝启动。
            #
            # 明场ROI 关闭时不挡：那种情况下新旧混搭测的都是第 1 层，与旧版本
            # 行为一致，拦下来只会白耽误事。
            messagebox.showerror(
                self._tr("需要同版本的 ImageJ 插件", "ImageJ Plug-in Version Mismatch"),
                self._tr(
                    "明场ROI 需要与界面同版本的 ImageJ 插件（当前界面 "
                    f"{APP_VERSION}，插件协议 {IMAGEJ_BRIDGE_PROTOCOL or '未知'}）。\n\n"
                    "旧版插件不认识层号，会一律测量第 1 层，而荧光层不是第 1 层时"
                    "每一次测量都指向错误的平面，且不会有任何提示。\n\n"
                    "请把插件更新到与本界面相同的版本（把 Auto_Worm_ROI.jar 复制到 "
                    "Fiji 的 plugins 目录后重启 Fiji），或关闭「明场ROI」后再开始处理。",
                    "Brightfield ROI needs the ImageJ plug-in from the same release as "
                    f"this window (window {APP_VERSION}, plug-in protocol "
                    f"{IMAGEJ_BRIDGE_PROTOCOL or 'unknown'}).\n\n"
                    "An older plug-in does not know about plane numbers and would "
                    "measure slice 1 of every image, silently pointing at the wrong "
                    "plane whenever the fluorescence image is not the first one.\n\n"
                    "Update the plug-in to match this window (copy Auto_Worm_ROI.jar "
                    "into Fiji's plugins folder and restart Fiji), or turn Brightfield "
                    "ROI off and process again."),
                parent=self.root)
            return
        if _imagej_bridge_pending():
            messagebox.showinfo(
                self._tr("等待 ImageJ", "Waiting for ImageJ"),
                self._tr(
                    "ImageJ 正在测量上一批结果，测量完成后才能开始下一批。",
                    "ImageJ is still measuring the previous batch. Please wait for it to finish "
                    "before starting the next."),
                parent=self.root)
            return
        worm_count = self._validated_worm_count(show_error=True)
        if worm_count is None:
            return
        self.last_valid_worm_count = worm_count
        if self.annotation_path is not None:
            self._close_head_annotation()
        problems = self._validate_runtime()
        if problems:
            messagebox.showerror(self._tr("无法启动", "Cannot Start"),
                                 "\n".join(problems), parent=self.root)
            return
        try:
            tiffs = [p for p in self.current_folder.iterdir()
                     if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
        except OSError as error:
            # The folder was readable when it was chosen and is not any more: it
            # was removed or renamed, a drive was unplugged, a share's permission
            # was taken away. This was the only file-system call on the start
            # button's path without a guard, and the windowed build has nowhere
            # to print a traceback -- the button simply did nothing at all, after
            # the whole pre-flight pass had already run.
            messagebox.showerror(
                self._tr("无法读取文件夹", "Cannot Read the Folder"),
                self._tr(
                    f"无法读取当前文件夹，请重新选择：\n{self.current_folder}\n\n{error}",
                    f"Cannot read the current folder; please choose it again:\n"
                    f"{self.current_folder}\n\n{error}"),
                parent=self.root)
            return
        if not tiffs:
            messagebox.showwarning(
                self._tr("未找到 TIFF", "No TIFF Images Found"),
                self._tr(
                    f"当前文件夹中没有 .tif/.tiff 文件：\n{self.current_folder}",
                    f"The current folder contains no .tif/.tiff files:\n{self.current_folder}"),
                parent=self.root)
            return
        from batch_worm_roi import check_image, find_stem_collisions
        sorted_tiffs = sorted(tiffs)
        unsupported = []
        binary = []
        stacks = []
        for path in sorted_tiffs:
            found = check_image(path)
            if found.code is None:
                if found.warning == "binary":
                    binary.append(path.name)
                continue
            if found.code == "stack":
                # 多层 TIFF 不再一律拒收。它是不是能处理，取决于明场ROI 有没有
                # 打开、层号是多少，所以收进 stacks 等下面分别处理。check_image
                # 本身保持纯函数不变，判定仍是同一处。
                stacks.append((path.name, int(found.value)))
                continue
            unsupported.append(self._tr("%s：%s", "%s: %s") % (
                path.name, self._unsupported_reason(found)))
        if unsupported:
            messagebox.showerror(
                self._tr("图像格式不支持", "Unsupported Image Format"),
                self._tr(
                    "以下图像不是单通道、单层、单时间点的灰度 TIFF，本版本无法处理：\n\n",
                    "These images are not single-channel, single-plane, "
                    "single-timepoint greyscale TIFFs and cannot be processed by "
                    "this version:\n\n") +
                "\n".join(unsupported[:10]) +
                (self._tr("\n\n…… 共 %d 张不合格，以上列出前 10 张",
                          "\n\n… %d unsupported in total; the first 10 are listed above")
                 % len(unsupported) if len(unsupported) > 10 else "") +
                self._tr(
                    "\n\n圈虫只读取文件的第一页，而 ImageJ 测量的是它窗口中当前选中的"
                    "通道、Z 层和时间点；两者会指向不同的平面，结果无法对应。\n"
                    "请先用 ImageJ 的 Image > Stacks > Stack to Images 拆成单页 TIFF，"
                    "只保留要测量的通道，再重新处理。",
                    "\n\nThis program reads only the first plane of a file, while "
                    "ImageJ measures the channel, Z-slice and timepoint currently "
                    "selected in its window, so the two would describe different "
                    "planes.\nSplit the files into single-plane TIFFs first with "
                    "Image > Stacks > Stack to Images, keep the channel you want, "
                    "then run again."),
                parent=self.root)
            return
        if stacks and not self.brightfield_roi_var.get():
            # 需求 3：多层 TIFF 在未开启明场ROI 时的专用提示。它替代了 0.4.x 那句
            # 笼统的「图像格式不支持」——那句话没说清该怎么办，而这里的下一步是
            # 明确的：要么开明场ROI 指定层号，要么先拆成单页。
            listed = "\n".join(self._tr("  %s：%d 层", "  %s: %d planes") % entry
                               for entry in stacks[:10])
            if len(stacks) > 10:
                listed += self._tr("\n  …… 共 %d 张，以上列出前 10 张",
                                   "\n  … %d in total; the first 10 are listed above") % len(stacks)
            messagebox.showerror(
                self._tr("需要打开明场ROI", "Brightfield ROI Required"),
                self._tr(
                    "需要打开明场ROI才可以处理多层TIFF。\n\n"
                    "以下图像在第 1 层之外还有数据：\n",
                    "Brightfield ROI must be turned on to process multi-plane TIFFs."
                    "\n\nThese images carry data beyond plane 1:\n") +
                listed +
                self._tr(
                    "\n\n请勾选左栏的「明场ROI」后重新处理：程序会在开始前询问荧光层和"
                    "明场层的层号，明场层用来圈出 ROI，荧光层用来测量数值。\n"
                    "若这一批不需要明场图，请先用 ImageJ 的 "
                    "Image > Stacks > Stack to Images 把文件拆成单页 TIFF，"
                    "只保留要测量的那一层，再重新处理。",
                    "\n\nTurn on Brightfield ROI in the left panel and process again: "
                    "the program will ask which plane holds the fluorescence image and "
                    "which holds the brightfield one, draw the ROIs on the brightfield "
                    "plane and take the measurements from the fluorescence plane.\n"
                    "If this batch has no brightfield image, split the files into "
                    "single-plane TIFFs first with Image > Stacks > Stack to Images, "
                    "keep the plane you want to measure, and process again."),
                parent=self.root)
            return
        fluorescence_plane, brightfield_plane = 1, 1
        if stacks:
            # 需求 1：每批问一次，整批通用。这里仍在主线程、批量线程尚未创建，
            # 所以取消路径没有任何副作用——没建目录、没禁用控件、没写日志。
            planes = self._prompt_planes()
            if planes is None:
                return
            fluorescence_plane, brightfield_plane = planes
            deepest = max(planes)
            too_shallow = [entry for entry in stacks if entry[1] < deepest]
            if too_shallow:
                messagebox.showerror(
                    self._tr("层号超出文件的层数", "Plane Number Beyond the File"),
                    self._tr(
                        "这些多层 TIFF 没有第 %d 层（要处理荧光层 %d、明场层 %d）：\n\n",
                        "These multi-plane TIFFs have no plane %d (fluorescence plane "
                        "%d, brightfield plane %d):\n\n") % (
                            deepest, fluorescence_plane, brightfield_plane) +
                    "\n".join(self._tr("  %s：只有 %d 层", "  %s: only %d planes") % entry
                              for entry in too_shallow[:10]) +
                    self._tr("\n\n请重新核对层号后重试。",
                             "\n\nCheck the plane numbers and try again."),
                    parent=self.root)
                return
            self.fluorescence_plane_var.set(fluorescence_plane)
            self.brightfield_plane_var.set(brightfield_plane)
        else:
            # 整批都是单层图：没有层可选，两个层号都是 1。明场ROI 开着也一样 ——
            # 每张图都按单层图处理，ROI 与测量都来自第 1 层，通知里报第 2 层会是
            # 假话。用户配的那两个数只作下次对话框的默认值，不当作本批的事实。
            fluorescence_plane = brightfield_plane = 1
        collisions = find_stem_collisions([str(path) for path in sorted_tiffs])
        if collisions:
            messagebox.showerror(
                self._tr("图像重名，结果会互相覆盖", "Duplicate Image Names"),
                self._tr(
                    "以下图像去掉扩展名后重名，会写到同一组结果文件：\n\n",
                    "These images share a name once the extension is removed, so "
                    "they would write the same result files:\n\n") +
                "\n".join("  " + " / ".join(names)
                          for _, names in collisions[:10]) +
                (self._tr("\n\n…… 共 %d 组重名，以上列出前 10 组",
                          "\n\n… %d clashing groups in total; the first 10 are listed above")
                 % len(collisions) if len(collisions) > 10 else "") +
                self._tr(
                    "\n\n结果文件按「文件名去掉扩展名」命名，所以 A.tif 和 A.tiff 会写到"
                    "同一组文件，后处理的那张会无声覆盖前一张。请只保留其中一张。",
                    "\n\nResults are named after the file name without its extension, "
                    "so A.tif and A.tiff both write the same set of files and the "
                    "second one silently overwrites the first. Keep only one of each "
                    "pair."),
                parent=self.root)
            return
        if binary:
            # A warning, not a refusal: a 1-bit image is processed normally, but
            # the models this program ships have only seen 8- and 16-bit
            # acquisitions. Confirming costs one click and cannot be missed,
            # which a log line can.
            messagebox.showwarning(
                self._tr("二值图像提示", "Binary Image Warning"),
                self._tr(
                    "以下图像是二值（1 位）图像：本模型未针对二值图片进行训练，"
                    "分割结果可能不准确。\n处理会照常继续进行，点击确定即可。\n\n",
                    "These images are 1-bit (binary). The model was not trained on "
                    "binary images, so the segmentation may be inaccurate.\n"
                    "Processing will continue normally; just click OK.\n\n") +
                "\n".join(binary[:10]) +
                (self._tr("\n\n…… 共 %d 张，以上列出前 10 张",
                          "\n\n… %d in total; the first 10 are listed above")
                 % len(binary) if len(binary) > 10 else ""),
                parent=self.root)
        segment_start, segment_end = self._segment_range_values()
        if self.segment_selection_var.get():
            self.manual_head_annotation_var.set(True)
            from manual_head_annotation import load_image_annotations
            incomplete = []
            for path in sorted(tiffs):
                arrow_count = len(load_image_annotations(path))
                if arrow_count != worm_count:
                    incomplete.append(self._tr(
                        "%s：箭头 %d/%d" % (path.name, arrow_count, worm_count),
                        "%s: arrows %d/%d" % (path.name, arrow_count, worm_count)))
            if incomplete:
                messagebox.showerror(
                    self._tr("部分圈画缺少头向箭头", "Missing Head Arrows for Partial ROI"),
                    self._tr(
                        "部分圈画要求每条虫都有一个手动标注的头向箭头。\n\n",
                        "Partial ROI requires one manually annotated head arrow per worm.\n\n") +
                    "\n".join(incomplete[:10]) +
                    (self._tr("\n……", "\n…") if len(incomplete) > 10 else ""), parent=self.root)
                return
        output_dir = self._effective_output_dir()
        if os.path.normcase(os.path.realpath(str(output_dir))) == os.path.normcase(
                os.path.realpath(str(self.current_folder))):
            # Processing unlinks same-named leftovers (stale measurement tables
            # and QC reports) and rewrites batch_summary.csv. Aimed at the image
            # folder those are the experimenter's own files, so refuse here
            # rather than let the batch discover it after the run has started.
            messagebox.showerror(
                self._tr("结果目录不能是图片目录",
                         "Output Folder Cannot Be the Image Folder"),
                self._tr(
                    "结果目录与当前图片文件夹是同一个：\n%s\n\n"
                    "处理时会清掉上一轮留下的同名结果文件，放在图片目录里会连带影响"
                    "该目录中的原有文件。\n"
                    "请另选一个结果目录，或点击菜单“设置 → 恢复默认保存路径”"
                    "回到图片目录下的 _auto_roi 子目录。" % output_dir,
                    "The output folder is the same as the image folder:\n%s\n\n"
                    "Processing clears same-named result files left by earlier "
                    "runs, which would also affect files already in the image "
                    "folder.\nChoose a different output folder, or use "
                    "\"Settings > Restore default output folder\" to go back to "
                    "the _auto_roi subfolder." % output_dir),
                parent=self.root)
            return
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror(
                self._tr("无法创建结果目录", "Cannot Create Output Folder"),
                str(exc), parent=self.root)
            return

        config = MODEL_CONFIGS[self.mode_var.get()]
        from batch_worm_roi import run_gui_batch
        batch_args = (str(self.current_folder), str(output_dir),
                      str(config["checkpoint"]), str(config["tip_checkpoint"]))
        source_stems = {p.stem.lower() for p in tiffs}
        batch_kwargs = dict(
            # The batch is given the list this pass was checked against, rather
            # than left to scan the folder again. Left to itself it would glob the
            # directory a second time and pick up anything that arrived in between
            # -- an acquisition or sync still writing into the folder. That image
            # would be segmented, get a ROI ZIP and a QC image, and appear in
            # batch_summary.csv as a success, but it is not in the completed list
            # built below from this same snapshot, so ImageJ would never measure
            # it. ROI present, summary green, no fluorescence: the one outcome
            # nobody can notice from the output. Passing the list keeps the
            # pre-check, the batch, and the completion manifest on one snapshot.
            input_paths=[str(path) for path in sorted_tiffs],
            standard_count=worm_count,
            allowed_count_min=worm_count,
            allowed_count_max=worm_count,
            ignore_filename_count=True,
            shape_refinement=bool(self.shape_refinement_var.get()),
            low_clarity_split=self.mode_var.get() == "low",
            manual_head_annotation=bool(self.manual_head_annotation_var.get()),
            segment_selection=bool(self.segment_selection_var.get()),
            segment_start=segment_start,
            segment_end=segment_end,
            measurement_backend="imagej" if IMAGEJ_MEASUREMENT_MODE else "python",
            brightfield_roi=bool(self.brightfield_roi_var.get()),
            # 两个整数在上面的预检里已经问过并确认过，这里只是把它们传下去；批量
            # 线程里不会再弹任何对话框。
            fluorescence_plane=fluorescence_plane,
            brightfield_plane=brightfield_plane,
            brightfield_checkpoint_path=str(BRIGHTFIELD_MODEL_CONFIG["checkpoint"]),
            brightfield_tip_checkpoint_path=str(BRIGHTFIELD_MODEL_CONFIG["tip_checkpoint"]),
            should_cancel=lambda: self.stop_requested,
        )
        self.stop_requested = False
        # A persistent ImageJ bridge emits one completion manifest per batch.
        self.bridge_notification_written = False
        self.close_when_stopped = False
        self.pending_error_text = ""
        self.active_job = {
            "input_dir": str(self.current_folder),
            "output_dir": str(output_dir),
            "source_stems": source_stems,
            # File names as the folder lists them. Reported to ImageJ when the
            # batch ends, so it measures what finished rather than everything it
            # can find a ROI ZIP for.
            "source_names": {p.name for p in tiffs},
            "completed_stems": set(),
            # (image name, reason) for images this batch could not process. They
            # are reported to ImageJ at the end: a missing image in the results
            # table is otherwise unexplained on the Fiji side.
            "failures": [],
            "mode": self.mode_var.get(),
            "worm_count": worm_count,
            "segment_selection": bool(self.segment_selection_var.get()),
            "segment_start": segment_start,
            "segment_end": segment_end,
            # 桥接通知要按这批的模式报层号，所以与 batch_kwargs 同源，都取上面
            # 预检里确认过的值，而不是左栏配置里那两个（它们只是对话框的默认值，
            # 单层图整批时与事实不符）。
            "brightfield_roi": bool(self.brightfield_roi_var.get()),
            "fluorescence_plane": fluorescence_plane,
            "brightfield_plane": brightfield_plane,
        }
        self.qc_status.clear()
        self.qc_attention.clear()
        self.qc_planes.clear()
        self._set_processing_controls(True)
        self._terminal_write("\n" + "=" * 72 + "\n", "muted")
        self._terminal_write(self._tr(
            f"处理目录：{self.current_folder}\n", f"Processing: {self.current_folder}\n"), "info")
        self._terminal_write(self._tr(
            f"模式：{self._mode_label()}\n", f"Mode: {self._mode_label()}\n"), "info")
        if self.mode_var.get() == "low":
            self._terminal_write(self._tr(
                "低清粘连拆分：结合头部缝隙和虫体分界。\n",
                "Low-clarity split: combines head gaps and worm boundaries.\n"), "info")
        self._terminal_write(
            self._tr(f"预设虫数：n={worm_count}；要求数量完全相等\n",
                     f"Expected worms: n={worm_count}; exact count required\n"), "info")
        enabled = self._tr("已开启", "enabled")
        disabled = self._tr("已关闭", "disabled")
        if self.brightfield_roi_var.get():
            self._terminal_write(self._tr(
                "明场ROI：已开启；荧光层 %d，明场层 %d\n" % (
                    int(self.fluorescence_plane_var.get()),
                    int(self.brightfield_plane_var.get())),
                "Brightfield ROI: enabled; fluorescence plane %d, brightfield plane %d\n" % (
                    int(self.fluorescence_plane_var.get()),
                    int(self.brightfield_plane_var.get()))), "info")
        else:
            self._terminal_write(
                self._tr("明场ROI：已关闭\n", "Brightfield ROI: disabled\n"), "info")
        self._terminal_write(
            self._tr("平滑修复：%s\n", "Smoothing repair: %s\n") %
            (enabled if self.shape_refinement_var.get() else disabled), "info")
        self._terminal_write(
            self._tr("手动标注（头向/分界/排除区）：%s\n",
                     "Manual annotation (head/boundary/exclusion): %s\n") %
            (enabled if self.manual_head_annotation_var.get() else disabled), "info")
        self._terminal_write(
            self._tr("部分圈画：%s%s\n", "Partial ROI: %s%s\n") % (
                enabled if self.segment_selection_var.get() else disabled,
                (self._tr("；头=0，尾=1，范围 %.2f–%.2f",
                          "; head=0, tail=1, range %.2f–%.2f") %
                 (segment_start, segment_end)) if self.segment_selection_var.get() else ""), "info")
        self._terminal_write(self._tr(
            "推理精度：FP32；实际计算设备将在模型载入后显示。\n",
            "Inference precision: FP32; the device will appear after model loading.\n"), "muted")
        self._terminal_write(self._tr(
            f"输出目录：{output_dir}\n\n", f"Output: {output_dir}\n\n"), "info")
        self._save_config()

        thread = threading.Thread(
            target=self._run_gui_batch_thread, args=(batch_args, batch_kwargs, run_gui_batch),
            daemon=False)
        self.process = thread
        thread.start()

    def _run_gui_batch_thread(self, batch_args, batch_kwargs, run_func) -> None:
        """后台线程:进程内运行 run_gui_batch,把每行状态推入事件队列。"""
        def on_status(text: str) -> None:
            self.event_queue.put(("line", text))
            if ": worms=" in text or text.startswith("Output:"):
                self.event_queue.put(("refresh_qc", None))

        def on_image_failed(name: str, detail: str) -> None:
            self.event_queue.put(("image_failed", (name, detail)))

        try:
            run_func(*batch_args, on_status=on_status,
                     on_image_failed=on_image_failed, **batch_kwargs)
            self.event_queue.put(("done", 0))
        except BaseException as exc:
            self.event_queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _stop_processing(self) -> None:
        process = self.process
        if process and process.is_alive():
            self.stop_requested = True
            self.stop_button.configure(state="disabled")
            self._terminal_write(self._tr(
                "\n正在停止：当前图片将完整写出，之后不再处理新图片…\n",
                "\nStopping: the current image will finish writing; no new images will start…\n"),
                "warning")

    def _poll_events(self) -> None:
        keep_polling = True
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "line":
                    line = str(payload)
                    self._terminal_write(line, "normal")
                    if self._record_qc_status_from_line(line):
                        self._draw_previews()
                elif kind == "refresh_qc":
                    self._scan_qc_images(reset_page=False)
                elif kind == "image_failed":
                    if self.active_job is not None:
                        self.active_job["failures"].append(payload)
                elif kind == "done":
                    if self._processing_finished(int(payload)):
                        keep_polling = False
                        return
                elif kind == "error":
                    self._terminal_write(f"\nERROR: {payload}\n", "error")
                    # Keep the text: the bridge file written by
                    # _processing_finished is the only place the Fiji side can
                    # read it from.
                    self.pending_error_text = str(payload)
                    if self._processing_finished(-1):
                        keep_polling = False
                        return
        except queue.Empty:
            pass
        except Exception as error:
            # This pump is the only consumer of the worker's queue, so anything
            # thrown while draining it used to take the after() call at the
            # bottom with it. The window then stayed up and looked alive while
            # nothing from the queue was ever drawn again -- and because
            # _on_close waits for this pump to write the batch's last
            # notification, the window could no longer be closed either, leaving
            # the Task Manager as the only way out. One failure is not worth
            # that: say what happened and keep the pump running.
            self._report_pump_failure(error)
        finally:
            # Not re-armed when _processing_finished destroyed the window, which
            # is the only way out that returns True.
            if keep_polling:
                self.root.after(100, self._poll_events)

    def _report_pump_failure(self, error: BaseException) -> None:
        """Put a failure in the event pump where the user can see it.

        A windowed build has no stderr, so Tk's default handler for a callback
        that raised writes the traceback somewhere nobody will ever look. The
        terminal panel is the one place a user of this program reads.
        """
        try:
            self._terminal_write(self._tr(
                "\n界面刷新出错，已跳过这一条更新：\n",
                "\nThe window hit an error while refreshing. This update was skipped:\n")
                + "%s: %s\n" % (type(error).__name__, error), "error")
        except Exception:
            # Reporting is best effort; the traceback below is the record.
            pass
        traceback.print_exc()

    def _processing_finished(self, return_code: int) -> bool:
        job = self.active_job
        self._set_processing_controls(False)
        self._scan_qc_images_safely(reset_page=False)
        if self.stop_requested:
            self._terminal_write(self._tr(
                "\n处理已由用户停止。\n", "\nProcessing stopped by the user.\n"), "warning")
        elif return_code == 0:
            output = Path(str(job["output_dir"])) if job else self._effective_output_dir()
            self._terminal_write(self._tr(
                f"\n处理完成。结果位于：{output}\n",
                f"\nFinished. Results are in: {output}\n"), "success")
        else:
            self._terminal_write(self._tr(
                "\n错误：处理失败，请复制上方消息。\n",
                "\nERROR: Processing failed. Please copy the messages above.\n"), "error")
        self.process = None
        self.active_job = None
        if IMAGEJ_BRIDGE_DIR:
            failures = list(job.get("failures", ())) if job else []
            # 明场批次要按荧光层测量，所以层号随通知一起过去；关闭时是 (1, 1)。
            planes = _bridge_planes(job)
            if return_code == 0 and not self.stop_requested and job:
                self._notify_imagej(
                    "complete", str(job["input_dir"]), str(job["output_dir"]),
                    self._failure_message(failures),
                    successful_images=self._successful_images(job, failures),
                    planes=planes)
            elif self.stop_requested:
                # A stopped batch still finished some images, and run_gui_batch
                # keeps their ROI sets and summary rows on purpose -- the README
                # tells the experimenter the finished part survives a cancel.
                # Reporting "cancelled" threw that away: the plug-in measures
                # nothing at all for a cancelled batch, so those images ended up
                # with no fluorescence and no CTCF despite their ROI sets sitting
                # in the output folder. Only a batch where nothing finished has
                # nothing to measure.
                stopped = self._successful_images(job, failures, only_finished=True) or []
                if stopped:
                    message = self._tr(
                        "本批已由用户提前停止，只测量已完成的 %d 张图像。",
                        "This batch was stopped early by the user; only the %d image(s) "
                        "that finished are measured.") % len(stopped)
                    failed_text = self._failure_message(failures)
                    self._notify_imagej(
                        "complete", str(job["input_dir"]), str(job["output_dir"]),
                        message + ("\n\n" + failed_text if failed_text else ""),
                        successful_images=stopped, planes=planes)
                else:
                    self._notify_imagej("cancelled", "", "")
            else:
                # The batch never started, so there is nothing to measure and the
                # exception text is the whole story. Without it Fiji could only
                # say "check the Auto Worm log", which is in the window the user
                # has just been told to close.
                self._notify_imagej(
                    "error", "", "",
                    self.pending_error_text or self._failure_message(failures))
            self._save_config()
            if self.close_when_stopped:
                self.root.destroy()
                return True
            return False
        if self.close_when_stopped:
            self._save_config()
            self.root.destroy()
            return True
        return False

    def _failure_message(self, failures: list[tuple[str, str]]) -> str:
        """Text for the ImageJ bridge file when images had to be skipped.

        The Fiji side has no other way to learn why an image is missing from the
        measurement table, so the list goes into the message field even when the
        batch otherwise succeeded. Empty string when nothing failed.
        """
        if not failures:
            return ""
        lines = [self._tr(
            "原界面有 %d 张图像处理失败，已跳过。结果目录中没有它们的 ROI，"
            "ImageJ 不会测量它们：",
            "%d image(s) failed in the original UI and were skipped. They have no "
            "ROI in the output folder, so ImageJ will not measure them:") % len(failures)]
        lines.extend("  Failed_%s: %s" % (name, detail) for name, detail in failures[:10])
        if len(failures) > 10:
            lines.append(self._tr("  …… 共 %d 张失败", "  … %d failed in total")
                         % len(failures))
        return "\n".join(lines)

    def _successful_images(self, job, failures: list[tuple[str, str]],
                           only_finished: bool = False) -> list[str] | None:
        """The images this batch finished, by file name, for the plug-in to measure.

        The plug-in measures from this list rather than from "every image that
        has a ROI ZIP": a ZIP is written before an image has been fully
        processed, so an image that fails afterwards leaves one behind whenever
        the partial results cannot be deleted, and it would then be measured as
        if it had succeeded. The names are compared exactly -- both sides list
        the same folder through the same system, so nothing has to be normalised
        and no image can be matched to the wrong file.

        only_finished narrows the list to the images whose progress line the
        window had already seen, which is what a batch the user stopped may
        measure: the rest were never written out. Those stems are collected
        lowercased, so the comparison lowercases too.

        None means the list could not be built, which leaves the receiver on its
        older rule.
        """
        if not job:
            return None
        sources = job.get("source_names")
        if not isinstance(sources, set):
            return None
        failed = {name for name, _ in failures}
        if only_finished:
            finished = job.get("completed_stems")
            if not isinstance(finished, set):
                return None
            return sorted(name for name in sources
                          if name not in failed
                          and os.path.splitext(name)[0].lower() in finished)
        return sorted(name for name in sources if name not in failed)

    def _notify_imagej(
            self, status: str, input_dir: str, output_dir: str, message: str = "",
            successful_images=None, planes: tuple[int, int] = (1, 1)) -> None:
        if not IMAGEJ_BRIDGE_DIR or self.bridge_notification_written:
            return
        error = _write_imagej_bridge_file(
            status, input_dir, output_dir, message, successful_images=successful_images,
            qc_plane=planes[0], measured_plane=planes[1])
        if error is None:
            self.bridge_notification_written = True
        else:
            self._terminal_write(
                self._tr(
                    f"\n[Auto Worm] 无法写入 ImageJ 桥接结果：{error}\n",
                    f"\n[Auto Worm] Failed to write the ImageJ bridge result: {error}\n"),
                "error")
            messagebox.showerror(
                self._tr("无法通知 ImageJ", "Cannot Notify ImageJ"),
                self._tr(
                    "结果已生成，但无法写入 ImageJ 桥接文件，ImageJ 将收不到本次测量。\n" +
                    "结果目录：\n" + output_dir + "\n\n" + error,
                    "Results were produced, but the ImageJ bridge file could not be written; " +
                    "ImageJ will not receive this measurement.\n" +
                    "Output folder:\n" + output_dir + "\n\n" + error),
                parent=self.root)

    def _terminal_write(self, text: str, tag: str = "normal") -> None:
        if not hasattr(self, "terminal"):
            return
        self.terminal.configure(state="normal")
        self.terminal.insert("end", text, tag)
        self.terminal.see("end")
        self.terminal.configure(state="disabled")

    def _clear_terminal(self) -> None:
        self.terminal.configure(state="normal")
        self.terminal.delete("1.0", "end")
        self.terminal.configure(state="disabled")

    def _save_terminal_log(self) -> None:
        """Write what the log window shows to a file the user picks.

        The log lives in the text widget and nowhere else: closing the window
        loses it, so a report about a failed batch has nothing to attach unless
        it is saved first. batch_summary.csv does record the failure reason, but
        not the surrounding progress lines or the device and driver information
        printed at startup."""
        content = self.terminal.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo(
                self._tr("保存日志", "Save Log"),
                self._tr("日志窗口还是空的，没有可保存的内容。",
                         "The log is empty, there is nothing to save."),
                parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title=self._tr("保存日志", "Save Log"),
            defaultextension=".txt",
            initialfile="autoworm_log_%s.txt" % time.strftime("%Y%m%d_%H%M%S"),
            filetypes=[(self._tr("文本文件", "Text file"), "*.txt"), ("*.*", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
        except OSError as exc:
            messagebox.showerror(
                self._tr("保存日志", "Save Log"),
                self._tr("无法写入该文件：\n%s", "Could not write the file:\n%s") % exc,
                parent=self.root)
            return
        self._terminal_write(
            self._tr("日志已保存到 %s\n", "Log saved to %s\n") % path, "success")

    def _show_terminal_context(self, event: tk.Event) -> None:
        self.terminal.mark_set("insert", f"@{event.x},{event.y}")
        self.terminal_menu.tk_popup(event.x_root, event.y_root)

    def _terminal_double_click(self, event: tk.Event) -> str:
        self.terminal.mark_set("insert", f"@{event.x},{event.y}")
        self._open_terminal_line()
        return "break"

    def _terminal_current_line(self) -> str:
        index = self.terminal.index("insert")
        return self.terminal.get(f"{index} linestart", f"{index} lineend").strip()

    def _open_terminal_line(self) -> None:
        line = self._terminal_current_line()
        path = self._extract_path_from_terminal_line(line)
        if path and path.exists():
            self._open_path(path)
        else:
            self.root.bell()

    def _extract_path_from_terminal_line(self, line: str) -> Path | None:
        for prefix in ("Processing:", "Output:", "Finished. Results are in:",
                       "处理目录：", "输出目录：", "处理完成。结果位于："):
            if line.startswith(prefix):
                candidate = Path(line[len(prefix):].strip())
                if candidate.exists():
                    return candidate
        match = re.match(r"^(.+?\.tiff?)(?::|$)", line, flags=re.IGNORECASE)
        if match:
            value = Path(match.group(1).strip())
            if not value.is_absolute():
                value = self.current_folder / value
            return value
        candidate = Path(line.strip().strip('"'))
        return candidate if candidate.exists() else None

    # ---------- 人工头部方向标注 ----------
    def _open_path(self, path: Path, quiet: bool = False) -> None:
        """开启人工标注时，TIFF 在内置监视器中打开；其他文件保持原行为。

        quiet=True is for the launch-time restore of the image ImageJ passed on
        the command line: that happens before the user has asked for anything, so
        an unsupported file must not raise a modal dialog at startup.
        """
        if (self.manual_head_annotation_var.get()
                and path.suffix.lower() in {".tif", ".tiff"}):
            self._show_head_annotation(path, quiet=quiet)
        else:
            self._open_external(path)

    def _show_head_annotation(self, path: Path, quiet: bool = False) -> None:
        if self._is_processing():
            self._terminal_write(self._tr(
                "处理期间不能打开或修改头部标注。\n",
                "Annotations cannot be opened or changed while processing.\n"), "warning")
            return
        from batch_worm_roi import check_image
        found = check_image(path)
        # 决策 5：明场ROI 开启时标注画在明场层上。圈出的 ROI 就是画在这一层上的，
        # 画在荧光层上会与虫子对不上，箭头保存后也落不到明场圈出的虫身上。层号
        # 沿用左栏那个值 —— 标注是一张一张打开的，没有「整批问一次」的时机。
        brightfield_annotation = found.code == "stack" and bool(self.brightfield_roi_var.get())
        if found.code is not None and not brightfield_annotation:
            # _start_processing refuses these too. Annotating one first would
            # mean drawing arrows on plane 0 of an image the batch rejects, and
            # only finding out when the run is started.
            detail = self._unsupported_reason(found)
            if quiet:
                self._terminal_write(self._tr(
                    "该图像无法标注：%s %s\n" % (path.name, detail),
                    "Cannot annotate this image: %s %s\n" % (path.name, detail)),
                    "warning")
                return
            messagebox.showerror(
                self._tr("图像格式不支持", "Unsupported Image Format"),
                self._tr(
                    "无法标注该图像：%s\n%s\n\n"
                    "圈虫只读取文件的第一页，标注结果无法对应到多通道/堆栈图像中"
                    "你想测量的平面。",
                    "Cannot annotate this image: %s\n%s\n\n"
                    "This program reads only the first plane, so annotations "
                    "cannot be tied to the plane you mean to measure.") % (
                        path.name, detail),
                parent=self.root)
            return
        annotation_plane = 1
        if brightfield_annotation:
            annotation_plane = int(self.brightfield_plane_var.get())
            frame_count = int(found.value)
            if annotation_plane > frame_count:
                detail = self._tr(
                    "层号超出文件的层数：%s 只有 %d 层，没有第 %d 层。\n\n"
                    "请核对左栏「明场ROI」的明场层层号后重试。" % (
                        path.name, frame_count, annotation_plane),
                    "Plane out of range: %s has %d planes, so there is no plane %d.\n\n"
                    "Check the brightfield plane number under Brightfield ROI." % (
                        path.name, frame_count, annotation_plane))
                if quiet:
                    self._terminal_write(self._tr(
                        "该图像无法标注：%s\n" % detail, "Cannot annotate: %s\n" % detail),
                        "warning")
                else:
                    messagebox.showerror(
                        self._tr("层号超出文件的层数", "Plane Out Of Range"), detail,
                        parent=self.root)
                return
        try:
            from manual_head_annotation import (
                enhanced_tiff_rgb, load_image_annotations, load_image_boundary_guides,
                load_image_exclusion_regions)
            image = enhanced_tiff_rgb(path, plane=annotation_plane)
            arrows = load_image_annotations(path, current_size=image.size)
            boundaries = load_image_boundary_guides(path, current_size=image.size)
            exclusions = load_image_exclusion_regions(path, current_size=image.size)
        except (OSError, ValueError) as exc:
            messagebox.showerror(self._tr("无法打开 TIFF", "Cannot Open TIFF"),
                                 f"{path}\n\n{exc}", parent=self.root)
            return
        self.annotation_path = path.resolve()
        self.annotation_image = image
        self.annotation_arrows = list(arrows)
        self.annotation_boundaries = list(boundaries)
        self.annotation_exclusions = list(exclusions)
        self.annotation_mode_var.set("arrow")
        self.annotation_drag_start = None
        self.annotation_boundary_draft = []
        self.preview_grid.grid_remove()
        self.page_label.master.grid_remove()
        self.annotation_frame.grid()
        title = self._tr("监视器 · 手动标注 · ", "Monitor · Manual annotation · ") + path.name
        if annotation_plane > 1:
            # 说明白底图取自哪一层：这张图画在明场层上，而测量走的是荧光层。
            title += self._tr(f" · 第 {annotation_plane} 层（明场）",
                              f" · plane {annotation_plane} (brightfield)")
        self.monitor_panel.title_label.configure(text=title)  # type: ignore[attr-defined]
        self._add_recent(path)
        self._annotation_mode_changed()
        self._update_annotation_count()
        self.root.after_idle(self._draw_annotation)

    def _close_head_annotation(self) -> None:
        if self.annotation_path is None:
            return
        self._save_head_annotations(silent=True)
        self.annotation_frame.grid_remove()
        self.preview_grid.grid()
        self.page_label.master.grid()
        self.monitor_panel.title_label.configure(
            text=self._tr("监视器", "Monitor"))  # type: ignore[attr-defined]
        self.annotation_path = None
        self.annotation_image = None
        self.annotation_photo = None
        self.annotation_arrows = []
        self.annotation_boundaries = []
        self.annotation_exclusions = []
        self.annotation_display_box = None
        self.annotation_drag_start = None
        self.annotation_boundary_draft = []
        self._draw_previews()

    def _schedule_annotation_redraw(self, _event: tk.Event | None = None) -> None:
        if self.annotation_path is not None:
            self.root.after_idle(self._draw_annotation)

    def _draw_annotation(self) -> None:
        image = self.annotation_image
        if image is None or self.annotation_path is None:
            return
        canvas = self.annotation_canvas
        colors = THEMES[self.theme_var.get()]
        canvas.delete("all")
        canvas.configure(bg=colors["terminal"], highlightbackground=colors["border"])
        width, height = max(40, canvas.winfo_width()), max(40, canvas.winfo_height())
        scale = min((width - 4) / image.width, (height - 4) / image.height)
        scale = max(scale, 0.01)
        display_size = (max(1, int(round(image.width * scale))),
                        max(1, int(round(image.height * scale))))
        shown = image.resize(display_size, Image.Resampling.LANCZOS)
        self.annotation_photo = ImageTk.PhotoImage(shown)
        x0 = (width - display_size[0]) / 2.0
        y0 = (height - display_size[1]) / 2.0
        self.annotation_display_box = (x0, y0, scale)
        canvas.create_image(x0, y0, image=self.annotation_photo, anchor="nw", tags="base_image")
        for index, region in enumerate(self.annotation_exclusions, 1):
            coordinates = []
            for px, py in region.points:
                coordinates.extend(self._image_to_canvas(float(px), float(py)))
            if len(coordinates) >= 6:
                canvas.create_polygon(
                    *coordinates, fill="#ff28be", outline="#ff66d0", width=3,
                    stipple="gray25", tags="saved_exclusion")
                canvas.create_text(
                    coordinates[0] + 7, coordinates[1] + 7, text=f"X{index}",
                    anchor="nw", fill="#ffffff",
                    font=("Microsoft YaHei UI", 9, "bold"), tags="saved_exclusion")
        for index, boundary in enumerate(self.annotation_boundaries, 1):
            coordinates = []
            for px, py in boundary.points:
                coordinates.extend(self._image_to_canvas(float(px), float(py)))
            if len(coordinates) >= 4:
                # 直线段：算法里的墙就是按折线画的，显示成曲线会让两者对不上。
                canvas.create_line(
                    *coordinates, fill="#00e8ff", width=3, tags="saved_boundary")
                canvas.create_text(
                    coordinates[0] + 7, coordinates[1] + 7, text=f"B{index}",
                    anchor="nw", fill="#ffffff",
                    font=("Microsoft YaHei UI", 9, "bold"), tags="saved_boundary")
        for index, arrow in enumerate(self.annotation_arrows, 1):
            sx, sy = self._image_to_canvas(float(arrow.tail_x), float(arrow.tail_y))
            hx, hy = self._image_to_canvas(float(arrow.head_x), float(arrow.head_y))
            canvas.create_line(
                sx, sy, hx, hy, fill="#ff2020", width=3, arrow="last",
                arrowshape=(14, 17, 6), tags="saved_head_arrow")
            canvas.create_oval(sx - 3, sy - 3, sx + 3, sy + 3,
                               fill="#ff2020", outline="", tags="saved_head_arrow")
            canvas.create_text(
                hx + 8, hy + 8, text=str(index), anchor="nw", fill="#ffffff",
                font=("Microsoft YaHei UI", 9, "bold"), tags="saved_head_arrow")

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        if not self.annotation_display_box:
            return x, y
        x0, y0, scale = self.annotation_display_box
        return x0 + x * scale, y0 + y * scale

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        image = self.annotation_image
        if image is None or not self.annotation_display_box:
            return None
        x0, y0, scale = self.annotation_display_box
        ix, iy = (x - x0) / scale, (y - y0) / scale
        if not (0 <= ix < image.width and 0 <= iy < image.height):
            return None
        return float(ix), float(iy)

    def _annotation_minimum_length(self) -> float:
        """分界线/排除区允许的最短长度（图像对角线的一定比例）。"""
        image = self.annotation_image
        if image is None:
            return 10.0
        return max(10.0, max(5.0, 0.005 * (image.width ** 2 + image.height ** 2) ** 0.5))

    def _annotation_press(self, event: tk.Event) -> None:
        point = self._canvas_to_image(event.x, event.y)
        if self.annotation_mode_var.get() == "boundary":
            # 多边形取点：左击落一个节点，右击结束（见 _annotation_finish_boundary）。
            if point is not None:
                self.annotation_boundary_draft.append(point)
                self._draw_annotation_draft(point)
            return
        self.annotation_drag_start = point
        self.annotation_boundary_draft = [point] if (
            point is not None and self.annotation_mode_var.get() == "exclusion") else []

    def _annotation_pointer_motion(self, event: tk.Event) -> None:
        """鼠标移动时把橡皮筋从最后一个节点拉到光标。"""
        if self.annotation_mode_var.get() == "boundary" and self.annotation_boundary_draft:
            self._draw_annotation_draft(self._canvas_to_image(event.x, event.y))

    def _draw_annotation_draft(self, cursor: tuple[float, float] | None) -> None:
        """画出分界线草稿：已落的节点、它们之间的直线，以及到光标的橡皮筋。"""
        points = self.annotation_boundary_draft
        self.annotation_canvas.delete("draft_annotation")
        if not points:
            return
        coordinates = []
        for point in points:
            coordinates.extend(self._image_to_canvas(*point))
        if len(coordinates) >= 4:
            self.annotation_canvas.create_line(
                *coordinates, fill="#00e8ff", width=3, dash=(5, 2),
                tags="draft_annotation")
        for index in range(0, len(coordinates), 2):
            x, y = coordinates[index], coordinates[index + 1]
            self.annotation_canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3, fill="#00e8ff", outline="",
                tags="draft_annotation")
        if cursor is not None:
            last = self._image_to_canvas(*points[-1])
            here = self._image_to_canvas(*cursor)
            self.annotation_canvas.create_line(
                last[0], last[1], here[0], here[1], fill="#00e8ff", width=2,
                dash=(4, 3), tags="draft_annotation")

    def _annotation_finish_boundary(self, _event: tk.Event | None = None) -> bool:
        """右击结束分界线：节点够长就存下来，否则响一声丢掉。"""
        if self.annotation_mode_var.get() != "boundary":
            return False
        points = self.annotation_boundary_draft
        self.annotation_boundary_draft = []
        self.annotation_canvas.delete("draft_annotation")
        if len(points) < 2:
            return False
        length = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
                     for a, b in zip(points, points[1:]))
        if length < self._annotation_minimum_length():
            self.root.bell()
            return False
        # 双击会在同一处落两个节点，去掉重合并保留其余节点原样：
        # 多边形的拐点是实验员点出来的，不该像手绘曲线那样被抽稀。
        simplified = [points[0]]
        for point in points[1:]:
            if ((point[0] - simplified[-1][0]) ** 2 +
                    (point[1] - simplified[-1][1]) ** 2) ** 0.5 >= 1.0:
                simplified.append(point)
        from manual_head_annotation import BoundaryGuide
        self.annotation_boundaries.append(BoundaryGuide(tuple(simplified)))
        self._save_head_annotations(silent=True)
        self._update_annotation_count()
        self._draw_annotation()
        return True

    def _annotation_cancel_draft(self, _event: tk.Event | None = None) -> None:
        """Esc 放弃正在画的分界线。"""
        if not self.annotation_boundary_draft:
            return
        self.annotation_boundary_draft = []
        self.annotation_canvas.delete("draft_annotation")

    def _annotation_motion(self, event: tk.Event) -> None:
        if self.annotation_mode_var.get() == "boundary":
            self._draw_annotation_draft(self._canvas_to_image(event.x, event.y))
            return
        start = self.annotation_drag_start
        end = self._canvas_to_image(event.x, event.y)
        self.annotation_canvas.delete("draft_annotation")
        if start is None or end is None:
            return
        if self.annotation_mode_var.get() == "exclusion":
            points = self.annotation_boundary_draft
            if not points or ((end[0] - points[-1][0]) ** 2 +
                              (end[1] - points[-1][1]) ** 2) ** 0.5 >= 2.0:
                points.append(end)
            coordinates = []
            for point in points:
                coordinates.extend(self._image_to_canvas(*point))
            if len(coordinates) >= 4:
                self.annotation_canvas.create_line(
                    *coordinates, fill="#ff28be", width=3, smooth=True,
                    splinesteps=12, dash=(5, 2), tags="draft_annotation")
                if len(coordinates) >= 6:
                    self.annotation_canvas.create_line(
                        coordinates[-2], coordinates[-1], coordinates[0], coordinates[1],
                        fill="#ff28be", width=2, dash=(5, 2), tags="draft_annotation")
        else:
            sx, sy = self._image_to_canvas(*start)
            hx, hy = self._image_to_canvas(*end)
            self.annotation_canvas.create_line(
                sx, sy, hx, hy, fill="#ff2020", width=3, arrow="last",
                arrowshape=(14, 17, 6), dash=(5, 2), tags="draft_annotation")

    def _annotation_release(self, event: tk.Event) -> None:
        if self.annotation_mode_var.get() == "boundary":
            # 分界线由右击结束，松开左键不提交任何东西。
            return
        start = self.annotation_drag_start
        end = self._canvas_to_image(event.x, event.y)
        self.annotation_drag_start = None
        self.annotation_canvas.delete("draft_annotation")
        if start is None or end is None or self.annotation_image is None:
            self.annotation_boundary_draft = []
            return
        minimum = max(5.0, 0.005 * (self.annotation_image.width ** 2 +
                                    self.annotation_image.height ** 2) ** 0.5)
        if self.annotation_mode_var.get() == "exclusion":
            points = self.annotation_boundary_draft
            if not points or points[-1] != end:
                points.append(end)
            length = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
                         for a, b in zip(points, points[1:]))
            self.annotation_boundary_draft = []
            if length < max(10.0, minimum):
                self.root.bell()
                return
            # 限制单条曲线点数，保留约 3 像素以上的手绘变化。
            simplified = [points[0]]
            for point in points[1:-1]:
                if ((point[0] - simplified[-1][0]) ** 2 +
                        (point[1] - simplified[-1][1]) ** 2) ** 0.5 >= 3.0:
                    simplified.append(point)
            simplified.append(points[-1])
            area_twice = abs(sum(
                first[0] * second[1] - second[0] * first[1]
                for first, second in zip(simplified, simplified[1:] + simplified[:1])))
            if len(simplified) < 3 or area_twice < 2.0 * max(25.0, minimum ** 2):
                self.root.bell()
                return
            from manual_head_annotation import ExclusionRegion
            self.annotation_exclusions.append(ExclusionRegion(tuple(simplified)))
        else:
            length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
            if length < minimum:
                self.root.bell()
                return
            from manual_head_annotation import HeadDirection
            self.annotation_arrows.append(HeadDirection(start[0], start[1], end[0], end[1]))
        self._save_head_annotations(silent=True)
        self._update_annotation_count()
        self._draw_annotation()

    def _annotation_mode_changed(self) -> None:
        mode = self.annotation_mode_var.get()
        help_text = {
            "arrow": self._tr(
                "从对应虫体内部按住左键，拖向头部后松开（箭头尖端为头部）",
                "Drag from inside each worm toward its head; the arrow tip marks the head."),
            "boundary": self._tr(
                "沿两条粘连虫之间的缝隙，从头端到尾端左击依次点出青色分界线的节点，"
                "右击结束；Esc 放弃这一条",
                "Along the gap between touching worms, left-click to place each cyan "
                "boundary point from head end to tail end; right-click to finish, "
                "Esc to discard."),
            "exclusion": self._tr(
                "沿脏背景或其他无需识别区域的外围按住左键画一圈，松开后自动闭合",
                "Draw around dirty background or other ignored areas; release to close the region."),
        }
        self.annotation_help.configure(text=help_text.get(mode, help_text["arrow"]))
        self.annotation_canvas.configure(
            cursor="crosshair" if mode in {"arrow", "boundary"} else "pencil")
        self.annotation_drag_start = None
        self.annotation_boundary_draft = []
        self.annotation_canvas.delete("draft_annotation")
        self._update_annotation_count()

    def _update_annotation_count(self) -> None:
        count = len(self.annotation_arrows)
        boundary_count = len(self.annotation_boundaries)
        exclusion_count = len(self.annotation_exclusions)
        expected = self.last_valid_worm_count
        colors = THEMES[self.theme_var.get()]
        mode = self.annotation_mode_var.get()
        self.annotation_count_label.configure(
            text=self._tr(
                f"箭头 {count}/{expected} · 分界 {boundary_count} · 排除 {exclusion_count}",
                f"Arrows {count}/{expected} · Boundaries {boundary_count} · Exclusions {exclusion_count}"),
            fg=(colors["ok"] if count == expected else colors["danger"])
            if mode == "arrow" else colors["text"])
        current_count = len(self._current_annotation_collection())
        self.annotation_undo_button.configure(state="normal" if current_count else "disabled")
        self.annotation_clear_button.configure(state="normal" if current_count else "disabled")

    def _save_head_annotations(self, silent: bool = False) -> None:
        if self.annotation_path is None or self.annotation_image is None:
            return
        try:
            from manual_head_annotation import save_image_annotations
            destination = save_image_annotations(
                self.annotation_path, self.annotation_image.size, self.annotation_arrows,
                boundaries=self.annotation_boundaries, exclusions=self.annotation_exclusions)
            if not silent:
                self._terminal_write(self._tr(
                    f"已保存 {len(self.annotation_arrows)} 个头部箭头、"
                    f"{len(self.annotation_boundaries)} 条人工分界线、"
                    f"{len(self.annotation_exclusions)} 个排除区域：{destination}\n",
                    f"Saved {len(self.annotation_arrows)} head arrows, "
                    f"{len(self.annotation_boundaries)} manual boundaries, and "
                    f"{len(self.annotation_exclusions)} exclusion regions: {destination}\n"),
                    "success")
        except (OSError, ValueError) as exc:
            messagebox.showerror(self._tr("标注保存失败", "Annotation Save Failed"),
                                 str(exc), parent=self.root)

    def _undo_annotation(self) -> None:
        collection = self._current_annotation_collection()
        if collection:
            collection.pop()
            self._save_head_annotations(silent=True)
            self._update_annotation_count()
            self._draw_annotation()

    def _clear_annotations(self) -> None:
        mode = self.annotation_mode_var.get()
        collection = self._current_annotation_collection()
        if not collection:
            return
        if not messagebox.askyesno(
                self._tr("清空人工标注", "Clear Manual Annotations"),
                self._tr(
                    "确定清空当前图片的所有%s吗？" % {
                        "boundary": "人工分界线", "exclusion": "排除区域"
                    }.get(mode, "头部箭头"),
                    "Clear all %s for the current image?" % {
                        "boundary": "manual boundaries", "exclusion": "exclusion regions"
                    }.get(mode, "head arrows")),
                parent=self.root):
            return
        collection.clear()
        self._save_head_annotations(silent=True)
        self._update_annotation_count()
        self._draw_annotation()

    def _current_annotation_collection(self) -> list[object]:
        return {
            "boundary": self.annotation_boundaries,
            "exclusion": self.annotation_exclusions,
        }.get(self.annotation_mode_var.get(), self.annotation_arrows)

    # ---------- QC 监视器 ----------
    def _scan_qc_images_safely(self, reset_page: bool = False) -> None:
        """Scan for QC output without letting a bad file end the batch handling.

        This runs at the top of _processing_finished, before the job is cleared,
        so an exception here leaves the batch half-finished: the controls stay
        disabled, self.process keeps a thread that has already ended, and
        _on_close -- which waits for the pump to finish that batch -- can then
        never close the window.

        Losing the QC thumbnails for one refresh is a small thing to trade for
        that. The file most likely to cause it is batch_summary.csv in the
        output folder, which is a file the QC workflow invites the experimenter
        to open.
        """
        try:
            self._scan_qc_images(reset_page=reset_page)
        except Exception as error:
            self._report_pump_failure(error)

    def _scan_qc_images(self, reset_page: bool = False) -> None:
        output = self._monitor_output_dir()
        if self.active_job is not None:
            allowed_stems = set(self.active_job.get("completed_stems", set()))
            self._load_qc_status_summary(output)
        else:
            self.qc_status.clear()
            self.qc_attention.clear()
            allowed_stems = self._load_qc_status_summary(output)
        try:
            files = sorted(output.rglob("*_QC.png"), key=lambda p: (p.stat().st_mtime, p.name.lower())) if output.is_dir() else []
        except OSError:
            files = []
        if allowed_stems is not None:
            files = [path for path in files
                     if self._qc_source_stem(path) in allowed_stems]
        old_selected = self.selected_qc
        self.qc_files = files
        if reset_page:
            self.qc_page = 0
            self.selected_qc = None
        elif old_selected not in files:
            self.selected_qc = None
        max_page = max(0, (len(files) - 1) // self.PAGE_SIZE)
        self.qc_page = min(self.qc_page, max_page)
        self._draw_previews()

    def _schedule_preview_redraw(self, _event: tk.Event | None = None) -> None:
        if self.preview_resize_job:
            self.root.after_cancel(self.preview_resize_job)
        self.preview_resize_job = self.root.after(100, self._draw_previews)

    def _draw_previews(self) -> None:
        self.preview_resize_job = None
        colors = THEMES[self.theme_var.get()]
        self.preview_images.clear()
        start = self.qc_page * self.PAGE_SIZE
        current = self.qc_files[start:start + self.PAGE_SIZE]
        for idx, canvas in enumerate(self.preview_canvases):
            canvas.delete("all")
            canvas.configure(bg=colors["terminal"], highlightbackground=colors["border"])
            width = max(160, canvas.winfo_width())
            height = max(130, canvas.winfo_height())
            if idx >= len(current):
                canvas.create_text(width // 2, height // 2,
                                   text=self._tr("等待 QC 图像", "Waiting for QC images"),
                                   fill=colors["muted"],
                                   font=("Microsoft YaHei UI", 10))
                continue
            path = current[idx]
            try:
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                image.thumbnail((max(20, width - 12), max(20, height - 12)), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.preview_images.append(photo)
                canvas.create_image(width // 2, height // 2, image=photo, anchor="center")
                source = self._source_label_for_qc(path)
                status = self._status_for_qc(path)
                needs_review = status.startswith("REVIEW")
                attention = self._attention_for_qc(path)
                overlay_text = self._tr(
                    f"需人工复核 ｜ {source}", f"Manual review ｜ {source}") if needs_review else source
                text_id = canvas.create_text(10, 9, text=overlay_text, anchor="nw", fill="#ffffff",
                                             font=("Microsoft YaHei UI", 9, "bold"), width=max(120, width - 28))
                bbox = canvas.bbox(text_id)
                if bbox:
                    rect = canvas.create_rectangle(bbox[0] - 5, bbox[1] - 3, bbox[2] + 5, bbox[3] + 3,
                                                   fill="#8c2635" if needs_review else "#111820", outline="")
                    canvas.tag_lower(rect, text_id)
                if needs_review:
                    canvas.create_rectangle(2, 2, width - 3, height - 3, outline=colors["danger"], width=3)
                elif attention:
                    canvas.create_rectangle(2, 2, width - 3, height - 3, outline=colors["warning"], width=3)
                elif path == self.selected_qc:
                    canvas.create_rectangle(2, 2, width - 3, height - 3, outline=colors["accent"], width=3)
            except (OSError, ValueError) as exc:
                canvas.create_text(width // 2, height // 2,
                                   text=self._tr(f"无法读取\n{path.name}\n{exc}",
                                                 f"Cannot read\n{path.name}\n{exc}"),
                                   justify="center", fill=colors["danger"], font=("Microsoft YaHei UI", 9))
        pages = (len(self.qc_files) + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        current_page = self.qc_page + 1 if pages else 0
        self.page_label.configure(text=self._tr(
            f"第 {current_page} / {pages} 页", f"Page {current_page} / {pages}"))
        # 需求 2：明场ROI 启动时监视器上只有明场 QC，所以状态行直接写明，免得
        # 用户以为看到的是荧光底图。判据与每张图的标注同源（两个层号不同）。
        brightfield_qc = any(qc != measured
                             for qc, measured in self.qc_planes.values())
        self.monitor_status.configure(text=(self._tr(
            f"共 {len(self.qc_files)} 张 QC 图（明场）" if brightfield_qc
            else f"共 {len(self.qc_files)} 张 QC 图",
            f"{len(self.qc_files)} QC images (brightfield)" if brightfield_qc
            else f"{len(self.qc_files)} QC images")
            if self.qc_files else self._tr("尚无 QC 图", "No QC images")))
        self.prev_button.configure(state="normal" if self.qc_page > 0 else "disabled")
        self.next_button.configure(state="normal" if self.qc_page + 1 < pages else "disabled")

    @staticmethod
    def _qc_source_stem(path: Path) -> str:
        stem = path.stem
        return (stem[:-3] if stem.lower().endswith("_qc") else stem).lower()

    def _load_qc_status_summary(self, output: Path) -> set[str] | None:
        # batch_summary.csv is working material, so it lives under other/ rather
        # than in the output folder itself. The name comes from batch_worm_roi so
        # there is one place to change it; the import is local because that module
        # is only otherwise imported when a batch starts.
        from batch_worm_roi import OTHER_DIRNAME
        summary = output / OTHER_DIRNAME / "batch_summary.csv"
        if not summary.is_file():
            return None
        source_stems: set[str] = set()
        try:
            with summary.open("r", newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    image_name = (row.get("image") or "").strip()
                    status = (row.get("qc_status") or "").strip()
                    try:
                        attention = any(int(row.get(field) or 0) > 0 for field in (
                            "shape_refined_count", "shape_review_count",
                            "low_clarity_split_count"))
                    except ValueError:
                        attention = False
                    if image_name:
                        source_stems.add(Path(image_name).stem.lower())
                    if image_name and status:
                        self.qc_status[image_name.lower()] = status
                        self.qc_attention[image_name.lower()] = attention
                    try:
                        qc_plane = int(row.get("qc_plane") or 0)
                        measured_plane = int(row.get("measured_plane") or 0)
                    except ValueError:
                        qc_plane = measured_plane = 0
                    if image_name and qc_plane >= 1 and measured_plane >= 1:
                        # 旧版本的汇总表没有这两列，读出来是 0，那就什么都不记：
                        # 监视器会说不出层号，而不是编一个出来。
                        self.qc_planes[image_name.lower()] = (qc_plane, measured_plane)
        except (OSError, UnicodeDecodeError, csv.Error):
            # UnicodeDecodeError is the one that matters: this file is the QC
            # workflow's own output and the README sends the experimenter to it,
            # so an Excel round-trip on a zh-CN machine rewrites it as GBK. The
            # sibling reader in batch_worm_roi.py has always caught it; this one
            # used to let it escape, and the folder is persisted, so the window
            # failed to open at every later launch until the file was renamed.
            return None
        return source_stems

    def _record_qc_status_from_line(self, line: str) -> bool:
        match = re.match(
            r"^(.+?\.tiff?):\s+worms=\d+,\s+expected=\d+,\s+(\S+)",
            line.strip(), flags=re.IGNORECASE)
        if not match:
            return False
        image_path = Path(match.group(1))
        self.qc_status[image_path.name.lower()] = match.group(2)
        if self.active_job is not None:
            completed = self.active_job.get("completed_stems")
            if isinstance(completed, set):
                completed.add(image_path.stem.lower())
        return True

    def _status_for_qc(self, qc_path: Path) -> str:
        stem = qc_path.stem
        source_stem = stem[:-3] if stem.endswith("_QC") else stem
        for image_name, status in self.qc_status.items():
            if Path(image_name).stem.lower() == source_stem.lower():
                return status
        return ""

    def _attention_for_qc(self, qc_path: Path) -> bool:
        stem = qc_path.stem
        source_stem = stem[:-3] if stem.endswith("_QC") else stem
        for image_name, attention in self.qc_attention.items():
            if Path(image_name).stem.lower() == source_stem.lower():
                return bool(attention)
        return False

    def _planes_for_qc(self, qc_path: Path) -> tuple[int, int] | None:
        """(qc_plane, measured_plane) for this QC, or None when unknown.

        None rather than a pair of ones for "unknown": a summary written before
        the plane columns existed genuinely does not say, and printing a plane
        number for it would be inventing a fact about somebody's data.
        """
        stem = qc_path.stem
        source_stem = stem[:-3] if stem.endswith("_QC") else stem
        for image_name, planes in self.qc_planes.items():
            if Path(image_name).stem.lower() == source_stem.lower():
                return planes
        return None

    def _source_label_for_qc(self, qc_path: Path) -> str:
        source = self._source_path_for_qc(qc_path)
        stem = qc_path.stem
        source_stem = stem[:-3] if stem.lower().endswith("_qc") else stem
        if source:
            try:
                display = str(source.relative_to(self.current_folder))
            except ValueError:
                display = str(source)
        else:
            display = f"{source_stem}.tif"
        label = self._tr(f"来源：{display}", f"Source: {display}")
        planes = self._planes_for_qc(qc_path)
        # 需求 2：说清 QC 来自哪张 tif 的第几层。两个层号不同，才说明这张 QC 真的
        # 取自另一层（单层图与关闭功能时两层都是 1，标「明场」会是假话）。
        if planes is not None and planes[0] != planes[1]:
            label += self._tr(f" · 第 {planes[0]} 层（明场）",
                              f" · plane {planes[0]} (brightfield)")
        return label

    def _source_path_for_qc(self, qc_path: Path) -> Path | None:
        stem = qc_path.stem
        source_stem = stem[:-3] if stem.lower().endswith("_qc") else stem
        for suffix in (".tif", ".tiff", ".TIF", ".TIFF"):
            candidate = self.current_folder / f"{source_stem}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _preview_path(self, slot: int) -> Path | None:
        index = self.qc_page * self.PAGE_SIZE + slot
        return self.qc_files[index] if 0 <= index < len(self.qc_files) else None

    def _select_preview(self, slot: int) -> None:
        path = self._preview_path(slot)
        if path:
            self.selected_qc = path
            self._draw_previews()

    def _open_preview(self, slot: int) -> None:
        path = self._preview_path(slot)
        if path:
            self.selected_qc = path
            source = self._source_path_for_qc(path)
            if self.manual_head_annotation_var.get() and source is not None:
                self._show_head_annotation(source)
            else:
                self._open_external(path)

    def _previous_page(self) -> None:
        if self.qc_page > 0:
            self.qc_page -= 1
            self.selected_qc = None
            self._draw_previews()

    def _next_page(self) -> None:
        pages = (len(self.qc_files) + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        if self.qc_page + 1 < pages:
            self.qc_page += 1
            self.selected_qc = None
            self._draw_previews()

    # ---------- 主题 ----------
    def _apply_theme(self) -> None:
        colors = THEMES[self.theme_var.get()]
        self.root.configure(bg=colors["bg"])
        self.style.configure("Treeview", background=colors["panel"], fieldbackground=colors["panel"],
                             foreground=colors["text"], bordercolor=colors["border"], rowheight=27,
                             font=("Microsoft YaHei UI", 9))
        self.style.map("Treeview", background=[("selected", colors["tree_sel"])],
                       foreground=[("selected", colors["text"])])
        self.style.configure("Vertical.TScrollbar", background=colors["panel2"], troughcolor=colors["panel"])

        for pane in (self.main_pane, self.left_pane, self.right_pane):
            pane.configure(bg=colors["border"])
        self.topbar.configure(bg=colors["panel"])
        self.folder_label.configure(bg=colors["panel"], fg=colors["muted"])
        for button in (self.file_button, self.settings_button):
            button.configure(bg=colors["panel"], fg=colors["text"], activebackground=colors["panel2"],
                             activeforeground=colors["text"])
        self.run_button.configure(bg=colors["accent"], fg="#ffffff", activebackground=colors["accent_hover"],
                                  activeforeground="#ffffff", disabledforeground="#d7e5f5")
        self.stop_button.configure(bg=colors["panel2"], fg=colors["danger"], activebackground=colors["border"],
                                   activeforeground=colors["danger"], disabledforeground=colors["muted"])
        self.open_output_button.configure(bg=colors["panel2"], fg=colors["text"], activebackground=colors["border"],
                                          activeforeground=colors["text"])
        self.save_log_button.configure(bg=colors["panel2"], fg=colors["text"], activebackground=colors["border"],
                                       activeforeground=colors["text"])

        for panel in (self.resource_panel, self.settings_panel, self.monitor_panel, self.terminal_panel):
            panel.configure(bg=colors["panel"], highlightbackground=colors["border"])
            panel.header.configure(bg=colors["panel2"])  # type: ignore[attr-defined]
            panel.title_label.configure(bg=colors["panel2"], fg=colors["text"])  # type: ignore[attr-defined]
            panel.body.configure(bg=colors["panel"])  # type: ignore[attr-defined]
        self.preview_grid.configure(bg=colors["panel"])
        self.annotation_frame.configure(bg=colors["panel"])
        self.annotation_canvas.configure(bg=colors["terminal"], highlightbackground=colors["border"])
        annotation_controls = self.annotation_help.master
        annotation_controls.configure(bg=colors["panel"])
        self.annotation_help.configure(bg=colors["panel"], fg=colors["text"])
        self.annotation_count_label.configure(bg=colors["panel"])
        for radio in (self.annotation_arrow_mode, self.annotation_boundary_mode,
                      self.annotation_exclusion_mode):
            radio.configure(bg=colors["panel"], fg=colors["text"],
                            activebackground=colors["panel"], activeforeground=colors["text"],
                            selectcolor=colors["panel2"])
        for button in (self.annotation_undo_button, self.annotation_clear_button,
                       self.annotation_save_button, self.annotation_back_button):
            button.configure(bg=colors["panel2"], fg=colors["text"],
                             activebackground=colors["border"], activeforeground=colors["text"],
                             disabledforeground=colors["muted"])

        controls = self.page_label.master
        controls.configure(bg=colors["panel"])
        self.page_label.configure(bg=colors["panel"], fg=colors["text"])
        self.monitor_status.configure(bg=colors["panel"], fg=colors["muted"])
        for button in (self.prev_button, self.next_button):
            button.configure(bg=colors["panel2"], fg=colors["text"], activebackground=colors["border"],
                             activeforeground=colors["text"], disabledforeground=colors["muted"])

        settings_body = self.settings_panel.body  # type: ignore[attr-defined]
        settings_body.configure(bg=colors["panel"])
        for widget in settings_body.winfo_children():
            if isinstance(widget, tk.Label):
                widget.configure(bg=colors["panel"], fg=colors["muted"] if widget is self.output_hint else colors["text"])
            elif isinstance(widget, (tk.Radiobutton, tk.Checkbutton)):
                widget.configure(bg=colors["panel"], fg=colors["text"], activebackground=colors["panel"],
                                 activeforeground=colors["text"], selectcolor=colors["panel2"])
        for row in self.settings_option_rows:
            row.configure(bg=colors["panel"])
        for label in self.settings_option_labels:
            label.configure(bg=colors["panel"], fg=colors["text"])
        for checkbox in (self.shape_refinement_check, self.manual_head_annotation_check,
                         self.segment_selection_check):
            checkbox.set_colors(colors["panel"], colors["text"],
                                colors["accent"], colors["muted"])
        self.segment_range_frame.configure(bg=colors["panel"])
        self._draw_segment_range()
        count_row = self.worm_count_spinbox.master
        count_row.configure(bg=colors["panel"])
        self.worm_count_label.configure(bg=colors["panel"], fg=colors["text"])
        self.worm_count_spinbox.configure(
            bg=colors["panel2"], fg=colors["text"], buttonbackground=colors["panel2"],
            insertbackground=colors["text"], highlightbackground=colors["border"],
            highlightcolor=colors["accent"], selectbackground=colors["tree_sel"],
            selectforeground=colors["text"]
        )

        self.terminal.configure(bg=colors["terminal"], fg=colors["text"], insertbackground=colors["text"],
                                selectbackground=colors["tree_sel"], selectforeground=colors["text"])
        self.terminal.tag_configure("normal", foreground=colors["text"])
        self.terminal.tag_configure("info", foreground=colors["accent_hover"])
        self.terminal.tag_configure("muted", foreground=colors["muted"])
        self.terminal.tag_configure("warning", foreground="#e5ad42")
        self.terminal.tag_configure("error", foreground=colors["danger"])
        self.terminal.tag_configure("success", foreground=colors["ok"])

        for menu in (self.file_menu, self.settings_menu, self.tree_menu, self.terminal_menu):
            menu.configure(bg=colors["panel"], fg=colors["text"], activebackground=colors["accent"],
                           activeforeground="#ffffff", selectcolor=colors["accent"])
        self._rebuild_file_menu()
        self._rebuild_settings_menu()
        self._draw_previews()
        self._update_annotation_count()
        if self.annotation_path is not None:
            self._draw_annotation()

    # ---------- 外部打开/退出 ----------
    def _open_output_folder(self) -> None:
        folder = self._monitor_output_dir()
        if folder.exists():
            self._open_external(folder)
        else:
            messagebox.showinfo(
                self._tr("结果目录", "Results Folder"),
                self._tr(f"目录尚未生成：\n{folder}",
                         f"The folder has not been created yet:\n{folder}"),
                parent=self.root)

    def _open_external(self, path: Path) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            self._add_recent(path)
        except OSError as exc:
            messagebox.showerror(self._tr("无法打开", "Cannot Open"),
                                 f"{path}\n\n{exc}", parent=self.root)

    def _reveal_in_explorer(self, path: Path) -> None:
        try:
            if os.name == "nt" and path.is_file():
                subprocess.Popen(["explorer", "/select,", str(path)])
            else:
                self._open_external(path if path.is_dir() else path.parent)
        except OSError as exc:
            messagebox.showerror(self._tr("无法显示", "Cannot Show Item"),
                                 str(exc), parent=self.root)

    def _on_close(self) -> None:
        process = self.process
        if process is not None:
            if process.is_alive():
                if not messagebox.askyesno(
                        self._tr("确认退出", "Confirm Exit"),
                        self._tr(
                            "处理任务仍在运行。是否在当前图片完整写出后停止并退出？",
                            "Processing is still running. Stop after the current image is fully written and exit?"),
                        parent=self.root):
                    return
                self.stop_requested = True
                self.close_when_stopped = True
                self.stop_button.configure(state="disabled")
                self._terminal_write(
                    self._tr(
                        "\n已请求退出：正在等待当前图片和汇总表安全写出…\n",
                        "\nExit requested: waiting for the current image and summary to finish writing…\n"),
                    "warning")
                return
            # The worker thread has already ended but its done/error event is still in the
            # queue. Do NOT set stop_requested (that would misreport a successful batch as
            # cancelled); just let _poll_events consume the event and _processing_finished
            # write complete/error before closing the window.
            self.close_when_stopped = True
            return
        if self.annotation_path is not None:
            self._save_head_annotations(silent=True)
        # Nothing is overwritten here: this is a new name in the queue, and the
        # plug-in reads it after any batch still waiting, so a just-finished
        # batch is measured before this tells it the window is gone.
        if IMAGEJ_BRIDGE_DIR:
            _write_imagej_bridge_file("closed")
        self._save_config()
        self.root.destroy()


def main() -> int:
    enable_windows_dpi_awareness()
    # 打包后自检模式:无界面直接跑批处理,验证模型定位与推理。
    # windowed(console=False)打包下 stdout 被抑制,状态与错误写入日志以便诊断。
    if "--headless-run" in sys.argv:
        log_path = STATE_DIR / "gui_headless.log"
        idx = sys.argv.index("--headless-run")
        folder = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else ""

        def headless_log(msg):
            try:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(str(msg) + "\n")
            except OSError:
                pass

        if not folder:
            headless_log("ERROR: --headless-run 需要一个文件夹路径参数")
            return 2
        try:
            mode = "low" if "--low-clarity" in sys.argv else "high"
            config = MODEL_CONFIGS[mode]
            headless_log("MODEL_ROOT: %s" % MODEL_ROOT)
            headless_log("checkpoint: %s" % config["checkpoint"])
            segment_enabled = "--enable-segment-selection" in sys.argv
            def headless_float(flag, default):
                try:
                    position = sys.argv.index(flag)
                    return float(sys.argv[position + 1])
                except (ValueError, IndexError, TypeError):
                    return default
            def headless_int(flag, default):
                try:
                    position = sys.argv.index(flag)
                    return int(sys.argv[position + 1])
                except (ValueError, IndexError, TypeError):
                    return default
            def headless_text(flag, default):
                try:
                    position = sys.argv.index(flag)
                    value = sys.argv[position + 1].strip()
                    return value or default
                except (ValueError, IndexError, TypeError):
                    return default
            # 无界面自检没有对话框可弹，层号只能从命令行来；不写就用 1 / 2，
            # 与界面里那三个配置项的默认值一致。
            brightfield = "--brightfield-roi" in sys.argv
            fluorescence_plane = headless_int("--fluorescence-plane", 1)
            brightfield_plane = headless_int("--brightfield-plane", 2)
            brightfield_checkpoint = headless_text(
                "--brightfield-checkpoint", str(BRIGHTFIELD_MODEL_CONFIG["checkpoint"]))
            brightfield_tip_checkpoint = headless_text(
                "--brightfield-tip-checkpoint", str(BRIGHTFIELD_MODEL_CONFIG["tip_checkpoint"]))
            if brightfield:
                headless_log("明场ROI：已开启；荧光层 %d，明场层 %d" %
                             (fluorescence_plane, brightfield_plane))
            from batch_worm_roi import run_gui_batch
            run_gui_batch(
                folder, "", str(config["checkpoint"]), str(config["tip_checkpoint"]),
                on_status=headless_log,
                low_clarity_split=mode == "low",
                manual_head_annotation=("--manual-head-annotation" in sys.argv or segment_enabled),
                segment_selection=segment_enabled,
                segment_start=headless_float("--segment-start", 0.0),
                segment_end=headless_float("--segment-end", 1.0),
                measurement_backend="imagej" if IMAGEJ_MEASUREMENT_MODE else "python",
                brightfield_roi=brightfield,
                fluorescence_plane=fluorescence_plane,
                brightfield_plane=brightfield_plane,
                brightfield_checkpoint_path=brightfield_checkpoint,
                brightfield_tip_checkpoint_path=brightfield_tip_checkpoint)
            _write_imagej_bridge_file(
                "complete", str(Path(folder).resolve()),
                str(Path(folder).resolve() / "_auto_roi"),
                qc_plane=brightfield_plane if brightfield else 1,
                measured_plane=fluorescence_plane if brightfield else 1)
            headless_log("HEADLESS_OK")
            return 0
        except Exception as exc:
            headless_log("HEADLESS_ERROR: " + traceback.format_exc())
            _write_imagej_bridge_file("error", message=str(exc))
            return 1
    root = tk.Tk()
    # 兼容某些 Windows 启动方式将未加引号的空格路径拆成多个参数。
    initial_path = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    try:
        WormRoiGui(root, initial_path)
        root.mainloop()
        return 0
    except Exception:
        details = traceback.format_exc()
        try:
            (STATE_DIR / "gui_crash.log").write_text(details, encoding="utf-8")
        except OSError:
            pass
        try:
            messagebox.showerror("界面启动失败", f"详细信息已写入 gui_crash.log\n\n{details[-1200:]}")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

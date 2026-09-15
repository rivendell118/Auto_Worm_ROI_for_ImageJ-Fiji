"""启动隐藏的 Tk 窗口，检查人工箭头、分界和排除区标注是否可用。"""

import sys
import json
import types
import tkinter as tk
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import worm_roi_gui as gui  # noqa: E402


def main():
    state = ROOT / "validation_outputs" / "manual_head_gui_state"
    state.mkdir(parents=True, exist_ok=True)
    gui.CONFIG_PATH = state / "settings.json"
    gui.APP_ROOT = ROOT
    try:
        gui.CONFIG_PATH.unlink()
    except FileNotFoundError:
        pass

    root = tk.Tk()
    root.withdraw()
    app = gui.WormRoiGui(root)
    root.update_idletasks()
    # Panel actions belong to the shared full-width title bar, not to a second
    # outer grid column.  A second outer column previously made the log body and
    # its scrollbar stop short of the right panel edge.
    assert app.save_log_button.master is app.terminal_panel.header
    assert app.terminal_panel.title_label.master is app.terminal_panel.header
    assert app.terminal_panel.grid_size()[0] == 1
    assert set(str(app.terminal_panel.header.grid_info()["sticky"])) == set("ew")
    assert set(str(app.terminal_panel.body.grid_info()["sticky"])) == set("nsew")
    assert app.manual_head_annotation_var.get() is False
    assert app.settings_panel.title_label.cget("text") == "图像处理"
    setting_label = "手动标注（头向/分界/排除区）"
    assert app.manual_head_annotation_label.cget("text") == setting_label
    labels = [app.settings_menu.entrycget(index, "label")
              for index in range(app.settings_menu.index("end") + 1)
              if app.settings_menu.type(index) != "separator"]
    assert setting_label in labels
    assert "部分圈画（需手动标注头向）" in labels
    assert "语言" in labels
    assert app.high_radio.cget("text") == "高清晰度图像"
    assert app.low_radio.cget("text") == "低清晰度图像"
    assert app.shape_refinement_label.cget("text") == "平滑修复"
    assert app.segment_selection_label.cget("text") == "部分圈画（需手动标注头向）"
    target_ratio = ((app.shape_refinement_check.TARGET_SIZE ** 2) /
                    (app.shape_refinement_check.BOX_SIZE ** 2))
    assert 1.45 <= target_ratio <= 1.65
    assert app._worm_count_range_text() == "识别数量必须等于 n=10；不匹配时需人工复核"
    assert not hasattr(app, "worm_count_range_label")
    assert app.worm_count_tooltip.text_getter() == app._worm_count_range_text()

    app.language_var.set("en")
    app._language_changed()
    assert app.file_button.cget("text") == "File"
    assert app.settings_button.cget("text") == "Settings"
    assert app.settings_panel.title_label.cget("text") == "Image Processing"
    assert app.high_radio.cget("text") == "High-clarity images"
    assert app.manual_head_annotation_label.cget("text") == \
        "Manual annotation (head/boundary/exclusion)"
    assert app.worm_count_tooltip.text_getter() == \
        "Detected count must equal n=10; mismatches require manual review"
    english_labels = [app.settings_menu.entrycget(index, "label")
                      for index in range(app.settings_menu.index("end") + 1)
                      if app.settings_menu.type(index) != "separator"]
    assert "Language" in english_labels
    assert json.loads(gui.CONFIG_PATH.read_text(encoding="utf-8"))["language"] == "en"
    app.language_var.set("zh")
    app._language_changed()

    image_path = WORKSPACE / "test" / "test06" / "N2-2.tif"
    app.manual_head_annotation_var.set(True)
    app._show_head_annotation(image_path)
    root.update_idletasks()
    assert app.annotation_path == image_path.resolve()
    assert app.annotation_image.size == (2048, 2048)
    displayed = np.asarray(app.annotation_image)
    assert np.array_equal(displayed[..., 0], displayed[..., 1])
    assert np.array_equal(displayed[..., 1], displayed[..., 2])
    # This folder has to have been produced by this version: 0.4.2 moved the QC
    # overlays into other/ and the summary beside them, so an _auto_roi left by
    # an earlier build has neither where this looks for it.
    yellow_output = WORKSPACE / "test" / "test06" / "_auto_roi"
    qc_image = yellow_output / "other" / "N2-2_QC.png"
    app._load_qc_status_summary(yellow_output)
    assert app._attention_for_qc(qc_image) is True
    assert app._status_for_qc(qc_image) == "PASS"
    assert len(app.annotation_arrows) == 10
    assert "箭头 10/10" in app.annotation_count_label.cget("text")
    assert app.annotation_boundary_mode.cget("text") == "人工分界线"
    assert app.annotation_exclusion_mode.cget("text") == "排除区域"
    app.annotation_mode_var.set("boundary")
    app._annotation_mode_changed()
    assert "头端到尾端" in app.annotation_help.cget("text")
    assert "右击结束" in app.annotation_help.cget("text")
    assert app.annotation_canvas.cget("cursor") == "crosshair"
    # 多边形取点：左击落节点、鼠标移动只拉橡皮筋、右击结束、Esc 放弃。
    # 存盘换成空实现，免得把标注写进测试图片旁边。
    saved = app._save_head_annotations
    app._save_head_annotations = lambda silent=False: None
    app.annotation_display_box = (0.0, 0.0, 1.0)
    clicks = [(100.0, 100.0), (100.0, 300.0), (400.0, 300.0)]

    def press(x, y):
        app._annotation_press(types.SimpleNamespace(
            x=app._image_to_canvas(x, y)[0], y=app._image_to_canvas(x, y)[1]))

    boundaries_before = len(app.annotation_boundaries)
    press(*clicks[0])
    assert app.annotation_boundary_draft == [clicks[0]]
    assert app.annotation_boundaries == []
    app._annotation_pointer_motion(types.SimpleNamespace(x=300, y=100))
    assert len(app.annotation_canvas.find_withtag("draft_annotation")) > 0
    press(*clicks[1])
    press(*clicks[2])
    assert len(app.annotation_boundary_draft) == 3
    # 松开左键在分界线模式下什么都不做。
    app._annotation_release(types.SimpleNamespace(x=400, y=300))
    assert len(app.annotation_boundary_draft) == 3
    assert len(app.annotation_boundaries) == boundaries_before
    assert app._annotation_finish_boundary() is True
    assert len(app.annotation_boundaries) == boundaries_before + 1
    assert app.annotation_boundary_draft == []
    assert app.annotation_boundaries[-1].points == tuple(clicks)
    assert app.annotation_canvas.find_withtag("draft_annotation") == ()
    # 只有两个节点、长度又不够时应当拒绝并存不下来。
    press(100.0, 100.0)
    press(101.0, 100.0)
    assert app._annotation_finish_boundary() is False
    assert len(app.annotation_boundaries) == boundaries_before + 1
    # Esc 丢掉草稿。
    press(100.0, 100.0)
    press(600.0, 600.0)
    app._annotation_cancel_draft()
    assert app.annotation_boundary_draft == []
    assert len(app.annotation_boundaries) == boundaries_before + 1
    app._undo_annotation()
    assert len(app.annotation_boundaries) == boundaries_before
    app._save_head_annotations = saved
    app.annotation_mode_var.set("exclusion")
    app._annotation_mode_changed()
    assert "无需识别区域" in app.annotation_help.cget("text")
    app.segment_selection_var.set(True)
    app._segment_selection_changed()
    assert app.manual_head_annotation_var.get() is True
    app.segment_drag_handle = "start"
    left, right, _ = app._segment_range_geometry()
    app._set_segment_range_from_x(left + 0.25 * (right - left))
    app.segment_drag_handle = "end"
    app._set_segment_range_from_x(left + 0.75 * (right - left))
    app.segment_drag_handle = None
    assert app._segment_range_values() == (0.25, 0.75)
    app.manual_head_annotation_var.set(False)
    app._manual_head_annotation_changed()
    assert app.manual_head_annotation_var.get() is True
    app._close_head_annotation()
    app._save_config()
    root.destroy()
    print("MANUAL_ANNOTATION_GUI_OK: dual range slider forces manual head direction")


if __name__ == "__main__":
    main()

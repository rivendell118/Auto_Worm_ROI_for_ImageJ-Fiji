# -*- coding: utf-8 -*-
"""PyInstaller runtime hook: 预加载 torch 原生 DLL,修复 c10.dll WinError 1114。

原理:torch/__init__.py 的 _load_dll_libraries() 用 LoadLibraryExW(0x1100)的
SAFE 搜索标志加载 torch\\lib\\*.dll——不搜 PATH、不搜 _internal 根目录,只搜
DLL 自身目录 + System32 + AddDllDirectory 目录。打包产物里 PyInstaller 只把
MSVC 运行时(v14.29)放 _internal 根,而不在 torch 的 SAFE 搜索范围内,导致
依赖 MSVCP140.dll/VCRUNTIME140.dll 等较新导出的 c10.dll/shm.dll/torch_cpu.dll/
torch_python.dll 初始化失败。

此 hook 在 frozen 启动时(任何 torch import 之前)运行:
  1) 用 os.add_dll_directory 注册 _internal 根与 torch\\lib(保留句柄防止 GC 回收);
  2) 用 ctypes.CDLL(unsafe 搜索,含 PATH)把 torch\\lib 下所有 DLL 预先加载进进程。
  二者结合后,torch 的 SAFE 加载会命中已加载的模块,不再重跑 DllMain。
"""
import ctypes
import glob
import os
import sys

_handles = []
_log_path = None


def _log(msg):
    try:
        global _log_path
        if _log_path is None:
            state_root = os.environ.get("LOCALAPPDATA")
            if state_root:
                state_root = os.path.join(state_root, "AutoWorm")
            else:
                state_root = os.path.join(os.path.expanduser("~"), ".autoworm")
            os.makedirs(state_root, exist_ok=True)
            _log_path = os.path.join(state_root, "torch_hook.log")
        with open(_log_path, "a", encoding="utf-8") as fh:
            fh.write(str(msg) + "\n")
    except OSError:
        pass


def _run():
    try:
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(sys.executable))
        torch_lib = os.path.join(base, "torch", "lib")
        paths = [base] + ([torch_lib] if os.path.isdir(torch_lib) else [])
        # 1) 把 _internal 根与 torch\lib 加进 PATH(供 unsafe 的 ctypes.CDLL 使用)
        os.environ["PATH"] = os.pathsep.join([p for p in paths if p]) + os.pathsep + os.environ.get("PATH", "")
        # 2) add_dll_directory 注册进 DLL SAFE 搜索目录,句柄保留避免被 GC
        for p in paths:
            if os.path.isdir(p):
                try:
                    _handles.append(os.add_dll_directory(p))
                except Exception as exc:
                    _log("add_dll_directory(%s) failed: %s" % (p, exc))
        # 3) 预加载 torch\lib 下全部 DLL(unsafe 搜索,含 PATH)
        if os.path.isdir(torch_lib):
            dlls = sorted(glob.glob(os.path.join(torch_lib, "*.dll")))
            for dll in dlls:
                try:
                    ctypes.CDLL(dll)
                except Exception as exc:
                    _log("preload %s FAILED: %s" % (os.path.basename(dll), exc))
            _log("hook: preloaded %d torch DLLs" % len(dlls))
        _log("hook: done")
    except Exception as exc:
        # hook 失败不应拖垮整个应用
        _log("hook exception: %s" % exc)


_run()

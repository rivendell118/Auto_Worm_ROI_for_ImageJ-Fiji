@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~1"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

echo [1/4] Compile ImageJ plug-in...
if not exist "build\plugin_classes" mkdir "build\plugin_classes"
rem dist\ 被 .gitignore 排除，新克隆的仓库里没有。jar 是先写临时文件再改名过去的，
rem 目标目录不存在时报的是 NoSuchFileException（一条看不出所以然的 Java 堆栈），
rem 而不是「目录不存在」。这里先建好，免得第一次构建就撞上。
if not exist "dist" mkdir "dist"
javac --release 8 -encoding UTF-8 -cp "lib\ij.jar" -d "build\plugin_classes" ^
  "plugin_src\Auto_Worm_ROI.java" "plugin_src\Auto_Worm_Annotations.java"
if errorlevel 1 exit /b 1
copy /y "plugin_src\plugins.config" "build\plugin_classes\plugins.config" >nul
jar --create --file "dist\Auto_Worm_ROI.jar" -C "build\plugin_classes" .
if errorlevel 1 exit /b 1

echo [2/4] Build original Auto Worm GUI with CUDA/ImageJ bridge...
"%PYTHON_EXE%" -m PyInstaller --clean --noconfirm "gui_imagej.spec"
if errorlevel 1 exit /b 1

echo [3/4] Synchronize runtime DLLs and copy model files...
"%PYTHON_EXE%" "postbuild.py"
if errorlevel 1 exit /b 1

echo [4/4] Verify outputs...
if not exist "dist\Auto_Worm_ROI.jar" exit /b 1
if not exist "dist\AutoWormImageJ\AutoWormGUI.exe" exit /b 1
echo Build complete: %cd%\dist
endlocal

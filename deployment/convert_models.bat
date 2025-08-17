@echo off
REM Batch script to convert PyTorch models to ONNX format
REM Run this from the deployment directory

echo Converting PyTorch models to ONNX format...
echo.

REM Activate virtual environment
call ..\.venv\Scripts\activate.bat

REM Convert best_full_model.pth
echo Converting best_full_model.pth...
python convert_to_onnx.py --input ../models/best_full_model.pth --output best_full_model.onnx
echo.

REM Convert best_base_model.pth  
echo Converting best_base_model.pth...
python convert_to_onnx.py --input ../models/best_base_model.pth --output best_base_model.onnx
echo.

REM Convert best_incremental_model.pth if it exists
if exist "../models/best_incremental_model.pth" (
    echo Converting best_incremental_model.pth...
    python convert_to_onnx.py --input ../models/best_incremental_model.pth --output best_incremental_model.onnx
    echo.
)

echo Conversion completed!
echo ONNX models are ready for deployment.
pause


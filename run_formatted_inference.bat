@echo off
REM Quick script to run formatted inference and generate JSON output
REM This avoids ONNX Runtime issues by using PyTorch only

echo ================================================================
echo AUDIO-TO-BLENDSHAPES FORMATTED INFERENCE
echo ================================================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\Activate.ps1
echo.

REM Navigate to inference directory and run the formatted inference
echo Running formatted inference...
cd inference
python format_specific_inference.py --model ../models/best_full_model.pth --output formatted_result.json --fps 15
echo.

echo ================================================================
echo INFERENCE COMPLETED!
echo ================================================================
echo.
echo JSON results saved to: inference/formatted_result.json
echo.
echo The output format matches your specified structure with:
echo - Session info with unique session ID and timestamp
echo - Frame count and audio chunk count  
echo - Individual frames with blendshapes and head pose
echo - All 52 MediaPipe blendshape names
echo - Head position (x, y, z) and rotation (quaternion)
echo.
pause


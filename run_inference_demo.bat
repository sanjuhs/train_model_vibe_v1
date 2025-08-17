@echo off
REM Complete inference demonstration script
REM Shows PyTorch and ONNX inference with the same audio

echo ================================================================
echo AUDIO-TO-BLENDSHAPES INFERENCE DEMONSTRATION
echo ================================================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

REM Step 1: Convert models to ONNX (if not already done)
echo Step 1: Converting PyTorch models to ONNX...
cd deployment
if not exist "best_full_model.onnx" (
    python convert_to_onnx.py --input ../models/best_full_model.pth --output best_full_model.onnx
) else (
    echo ONNX model already exists, skipping conversion...
)
echo.

REM Step 2: Run PyTorch inference demo
echo Step 2: Running PyTorch inference demo...
cd ../inference
python pytorch_inference_demo.py --model ../models/best_full_model.pth --output pytorch_demo_results
echo.

REM Step 3: Run ONNX Runtime inference demo  
echo Step 3: Running ONNX Runtime inference demo...
cd ../deployment
python onnx_inference_demo.py --model best_full_model.onnx --output onnx_demo_results
echo.

echo ================================================================
echo DEMONSTRATION COMPLETED SUCCESSFULLY!
echo ================================================================
echo.
echo Results saved to:
echo - PyTorch results: inference/pytorch_demo_results/
echo - ONNX results: deployment/onnx_demo_results/
echo.
echo The models are now ready for deployment:
echo - PyTorch models: Use for Python applications
echo - ONNX models: Use for JavaScript/web deployment
echo.
pause


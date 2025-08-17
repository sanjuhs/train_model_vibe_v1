import ctypes, os

pyd = r"C:\Users\USER\Desktop\coding\python\train_LSTM\.venv\Lib\site-packages\onnxruntime\capi\onnxruntime_pybind11_state.pyd" 
try:
    ctypes.WinDLL(pyd)
    print("✅ Loaded successfully")
except OSError as e:
    err = ctypes.GetLastError()
    print(f"❌ Load failed: {e!r} (Windows error {err})")

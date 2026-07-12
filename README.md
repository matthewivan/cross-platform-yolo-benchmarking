How to actually run it (per board)

First, on each board, convert the model:

- Pi Zero 2W: --target onnx
- Radxa Zero 3W: --target rknn --rknn-name rk3566
- Khadas Edge 2: --target rknn --rknn-name rk3588
- Jetson AGX Orin: --target tensorrt ← must build on the Jetson itself, engines are hardware-specific.

Rename output to 640_yolov8n.onnx / .rknn / .engine so detector.py auto-finds it.

For custom rknn detection script:
```bash
python3 benchmark.py \
  --external-cmd "python3 your_rknn_script.py --model 640_yolov8n.rknn --imgs ./imgs" \
  --latency-unit s \
  --frameworks rknn \
  --output radxa_10_benchmark_results_640model.csv
```

Then run the benchmark:
```bash
python3 benchmark.py --frameworks onnx     --imgsz 640 --output rpi_..._640model.csv     # Pi
python3 benchmark.py --frameworks rknn     --imgsz 640 --output radxa_..._640model.csv   # Radxa/Khadas
python3 benchmark.py --frameworks tensorrt --imgsz 640 --output jetson_..._640model.csv  # Jetson
```
Then: `python3 create_plot.py → boxplots + temp scatters for all boards.`

Two gotchas for the Jetson specifically

- Run cat /sys/class/thermal/thermal_zone*/type on it once. The auto-detect looks for CPU/SoC/tj, but if it grabs the wrong one, force it with --thermal-zone N.
freq_start_MHz may log as empty — psutil doesn't read Orin clocks reliably. If you need clocks there, tegrastats is the real source (not wired in; say the word and I'll add it).

- The biggest remaining unknown is whether Ultralytics' .rknn / .engine loading works cleanly on your exact JetPack + rknn-toolkit versions — those move around. Want me to add a raw-RKNN fallback backend in detector.py as a safety net?

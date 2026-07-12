# Cross-Platform YOLO Benchmarking

Thermal/load-stress benchmark for YOLOv8n object detection across single-board
computers: **Raspberry Pi Zero 2W**, **Radxa Zero 3W**, **Khadas Edge 2**, and
**Jetson AGX Orin**. It pins the CPU at a fixed load with `stress-ng`, runs
inference, and records latency alongside temperature and clock frequency to show
how each board throttles under heat.

## Filename convention

All result CSVs use one uniform pattern:

```
{slug}_benchmark_{res}.csv
```

| Board            | slug     | Example (640)            |
|------------------|----------|--------------------------|
| Pi Zero 2W       | `rpi`    | `rpi_benchmark_640.csv`    |
| Radxa Zero 3W    | `radxa`  | `radxa_benchmark_640.csv`  |
| Khadas Edge 2    | `khadas` | `khadas_benchmark_640.csv` |
| Jetson AGX Orin  | `jetson` | `jetson_benchmark_640.csv` |

`create_plot.py` prints the full expected list at startup, so you never have to
open the file to check names.

## How to actually run it (per board)

### 1. Convert the model (on each board)

| Board           | Convert command                                            |
|-----------------|------------------------------------------------------------|
| Pi Zero 2W      | `--target onnx`                                            |
| Radxa Zero 3W   | `--target rknn --rknn-name rk3566`                        |
| Khadas Edge 2   | `--target rknn --rknn-name rk3588`                        |
| Jetson AGX Orin | `--target tensorrt` — **must build on the Jetson itself**, engines are hardware-specific |

```bash
python3 convert_model.py --pt 640_yolov8n_v9.pt --imgsz 640 --target onnx
```

Rename the output to `640_yolov8n.onnx` / `.rknn` / `.engine` so `detector.py`
auto-finds it (or pass `--model` explicitly).

### 2. Run the benchmark

```bash
python3 benchmark.py --frameworks onnx     --imgsz 640 --output rpi_benchmark_640.csv     # Pi
python3 benchmark.py --frameworks rknn     --imgsz 640 --output radxa_benchmark_640.csv   # Radxa
python3 benchmark.py --frameworks rknn     --imgsz 640 --output khadas_benchmark_640.csv  # Khadas
python3 benchmark.py --frameworks tensorrt --imgsz 640 --output jetson_benchmark_640.csv  # Jetson
```

### Using your own RKNN detection script

If you'd rather run your own RKNN script instead of the built-in `detector.py`,
point `benchmark.py` at it with `--external-cmd`. Your script must print
`inference time: <value>` once per frame. Use `--latency-unit s` if it prints
**seconds** (values are converted to ms automatically):

```bash
python3 benchmark.py \
  --external-cmd "python3 your_rknn_script.py --model 640_yolov8n.rknn --imgs ./imgs" \
  --latency-unit s \
  --frameworks rknn \
  --output radxa_benchmark_640.csv
```

With `--external-cmd`, `--frameworks` is just the label written to the CSV — pick
whatever tag you want for the plots.

### 3. Plot

```bash
python3 create_plot.py    # boxplots + latency-vs-temp scatters for all boards
```

Missing CSVs just print `[skip] missing …` and are left out — no crash. You only
need the files for boards/resolutions you've actually run.

## Jetson AGX Orin gotchas

- **Thermal zone:** run `cat /sys/class/thermal/thermal_zone*/type` once. The
  auto-detect looks for CPU/SoC/tj, but if it grabs the wrong zone, force it with
  `--thermal-zone N`.
- **Clock frequency:** `freq_start_MHz` may log empty — `psutil` doesn't read Orin
  clocks reliably. `tegrastats` is the authoritative source there (not currently
  wired in).

## Known unknowns

- Whether Ultralytics' `.rknn` / `.engine` loading works cleanly depends on your
  exact JetPack + rknn-toolkit versions, which move around. If Ultralytics gives
  trouble, use `--external-cmd` with a raw-runtime script (RKNN-Lite / TensorRT)
  as a fallback.

## Files

| File               | Purpose                                                        |
|--------------------|----------------------------------------------------------------|
| `benchmark.py`     | Orchestrator: stress load, run inference, log latency/temp/clock to CSV |
| `detector.py`      | Uniform per-frame inference timer (ONNX / RKNN / TensorRT via Ultralytics) |
| `convert_model.py` | Export `.pt` → ONNX / RKNN / TensorRT engine                   |
| `create_plot.py`   | Config-driven boxplots + latency-vs-temperature scatters       |

## Adding a new board

Add one line to the `BOARDS` dict in `create_plot.py`:

```python
"New Board": {"slug": "xyz", "color": "purple"},
```

Filenames generate themselves from the slug — nothing else to edit.

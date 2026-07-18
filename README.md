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

For Jetson, build separately named FP16 and INT8 engines. INT8 calibration must
use a representative dataset YAML from the same problem domain:

```bash
python3 convert_model.py --pt models/yolov8n_second_buoy.pt --imgsz 640 \
  --target tensorrt --precision fp16 \
  --output models/yolov8n_second_buoy_640_fp16.engine

python3 convert_model.py --pt models/yolov8n_second_buoy.pt --imgsz 640 \
  --target tensorrt --precision int8 --data path/to/data.yaml --batch 8 \
  --output models/yolov8n_second_buoy_640_int8.engine
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

The built-in TensorRT runner accepts either one real image or a directory. The
directory is preloaded and cycled until `--frames` is reached, keeping disk I/O
outside the measured loop:

```bash
python3 benchmark.py --frameworks tensorrt \
  --model models/yolov8n_second_buoy_640_fp16.engine \
  --images ./images --imgsz 640 --frames 200 --duration 60 \
  --output jetson_benchmark_640_fp16.csv
```

`--duration` is the CPU-load preheat period. The stressor remains active for the
whole inference loop after preheating.

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

Both supplied external runners already use real image directories. They and the
built-in runner print `measured loop elapsed: <seconds>` after initialization.
`benchmark.py` uses that value for `fps`; `process_fps` includes child-process
startup, while `inference_fps` is derived from inference-only mean latency.

### 3. Plot

```bash
python3 create_plot.py    # boxplots + latency-vs-temp scatters for all boards
```

Missing CSVs just print `[skip] missing …` and are left out — no crash. You only
need the files for boards/resolutions you've actually run.

## Complete Jetson AGX Orin guide

The Jetson workflow differs from the other boards because the portable PyTorch
model is compiled into a TensorRT engine before benchmarking:

```text
PyTorch .pt
    -> TensorRT export and optimization on the Jetson
TensorRT .engine
    -> detector.py loads the engine and runs GPU inference
benchmark.py
    -> applies CPU load, collects latency and tegrastats, and writes CSV
```

A TensorRT `.engine` is tied to the Jetson hardware and its CUDA, TensorRT, and
JetPack versions. Build every engine on the Jetson that will run it. Rebuild the
engine after changing JetPack, TensorRT, model weights, resolution, precision,
or target hardware.

Official references:

- [NVIDIA JetPack setup](https://docs.nvidia.com/jetson/agx-orin-devkit/user-guide/setup_jetpack.html)
- [NVIDIA tegrastats](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)
- [NVIDIA Orin power and clock controls](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- [Ultralytics Jetson setup](https://docs.ultralytics.com/guides/nvidia-jetson)
- [Ultralytics TensorRT export](https://docs.ultralytics.com/integrations/tensorrt)

### 1. Prepare the hardware

Use the correct high-power supply, active cooling, and unobstructed airflow.
Avoid desktop applications, suspend, and other background workloads during a
run. Keep the power supply, physical location, fan policy, and cooling setup
identical for FP16 and INT8.

An NVMe SSD is preferable, although the measured TensorRT loop preloads images
and therefore excludes repeated image reads from its timing.

### 2. Identify the Jetson software and hardware

```bash
cat /etc/nv_tegra_release
uname -a
python3 --version
cat /proc/device-tree/model
dpkg -l | grep -E 'nvidia-jetpack|tensorrt|cuda|cudnn'
```

Record this information with the results. PyTorch wheels must match the JetPack,
Python, CUDA, and ARM64 environment; an ordinary desktop CUDA wheel may install
but fail to use the Jetson GPU.

### 3. Install and verify JetPack

If the Jetson already boots but is missing the SDK components:

```bash
sudo apt update
sudo apt install nvidia-jetpack
sudo reboot
```

After reboot, verify CUDA and TensorRT:

```bash
nvcc --version
dpkg-query -W tensorrt
which trtexec
trtexec --version
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

On some JetPack releases, `trtexec` is under `/usr/src/tensorrt/bin/` rather
than on `PATH`:

```bash
/usr/src/tensorrt/bin/trtexec --version
```

If the OS itself needs to be flashed, use NVIDIA SDK Manager or the installation
method documented for the exact AGX Orin carrier/module. Do not mix packages
from unrelated JetPack releases.

### 4. Put the project on the Jetson

Clone it if it is in Git:

```bash
git clone <repository-url>
cd cross-platform-yolo-benchmarking
```

Alternatively, copy it from another Linux machine:

```bash
rsync -av cross-platform-yolo-benchmarking/ \
  <jetson-user>@<jetson-ip>:~/cross-platform-yolo-benchmarking/
```

On the Jetson, verify the required inputs:

```bash
cd ~/cross-platform-yolo-benchmarking
ls -lh models/yolov8n_second_buoy.pt
find images -maxdepth 1 -type f | head
```

### 5. Create the Python environment

Install the OS tools used by the harness:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-opencv stress-ng
```

Use `--system-site-packages` so the environment can see the TensorRT bindings
installed by JetPack:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

Install the Ultralytics, PyTorch, and Torchvision versions documented for the
detected JetPack release. JetPack 5 and JetPack 6 often require Jetson-specific
ARM64 PyTorch wheels. Follow the matching section in the
[Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)
instead of assuming the normal PyPI Torch wheel is compatible.

Verify the complete Python stack:

```bash
python3 -c "import torch, ultralytics, tensorrt as trt; \
print('Torch:', torch.__version__); \
print('CUDA available:', torch.cuda.is_available()); \
print('Torch CUDA:', torch.version.cuda); \
print('Ultralytics:', ultralytics.__version__); \
print('TensorRT:', trt.__version__)"
```

Verify an actual CUDA operation:

```bash
python3 -c "import torch; \
print(torch.cuda.get_device_name(0)); \
x=torch.ones((1024,1024),device='cuda'); \
print(x.sum())"
```

Do not continue until `torch.cuda.is_available()` is `True`.

### 6. Verify the benchmark utilities

```bash
which stress-ng
stress-ng --version
which tegrastats
tegrastats --interval 1000
```

Stop the last command with Ctrl+C after several samples. It should report fields
such as `GR3D_FREQ`, `cpu@`, `gpu@`, `soc0@`, and `VDD_IN`.

Inspect power modes:

```bash
sudo nvpmodel -q
sudo nvpmodel -q --verbose
cat /etc/nvpmodel.conf
```

### 7. Choose a power and clock policy

There are two valid but different experiments.

For controlled maximum performance, select maximum power first and then lock
the clocks:

```bash
sudo nvpmodel -m 0
sudo reboot
sudo nvpmodel -q
sudo jetson_clocks
sudo jetson_clocks --show
```

Only reboot if `nvpmodel` requests it or as part of establishing a clean run.
Mode `0` is normally MAXN, but verify it using `nvpmodel -q` and the local
`/etc/nvpmodel.conf`. Always run `nvpmodel` before `jetson_clocks`; changing the
power mode after locking clocks may require another reboot.

For realistic dynamic performance, select the intended deployment power mode
but do not run `jetson_clocks`. Leave normal dynamic clocks and fan control
active. Do not mix fixed-clock and dynamic-clock trials in one CSV.

Maximum fan can be requested with:

```bash
sudo jetson_clocks --fan
```

Use it only if maximum cooling is explicitly part of the experiment. It reduces
thermal throttling and therefore changes what the test measures.

### 8. Prepare INT8 calibration data

FP16 needs no calibration dataset. INT8 requires an Ultralytics dataset YAML
whose validation images represent actual deployment conditions. Prefer the YAML
used to train the buoy model.

Example `data.yaml`:

```yaml
path: /home/<jetson-user>/datasets/buoys
train: images/train
val: images/val

names:
  0: Black Buoy
  1: Blue Buoy
  2: Green Buoy
  3: Maroon Buoy
  4: Or
  5: Orange Buoy
  6: Red Buoy
  7: Wader
  8: White Buoy
  9: Yellow Buoy
  10: Zebra Buoy
```

Verify that the paths resolve:

```bash
find /home/<jetson-user>/datasets/buoys/images/val -type f | head
```

Calibration images should cover representative cameras, exposure, water and
background conditions, object sizes, colors, and empty scenes. Evaluate INT8
accuracy on a separate labeled validation/test set after benchmarking.

### 9. Build TensorRT engines on the Jetson

Activate the environment whenever opening a new terminal:

```bash
cd ~/cross-platform-yolo-benchmarking
source .venv/bin/activate
```

Build FP16:

```bash
python3 convert_model.py \
  --pt models/yolov8n_second_buoy.pt \
  --imgsz 640 \
  --target tensorrt \
  --precision fp16 \
  --output models/yolov8n_second_buoy_640_fp16.engine
```

Build calibrated INT8:

```bash
python3 convert_model.py \
  --pt models/yolov8n_second_buoy.pt \
  --imgsz 640 \
  --target tensorrt \
  --precision int8 \
  --data /home/<jetson-user>/datasets/buoys/data.yaml \
  --batch 8 \
  --workspace 4 \
  --output models/yolov8n_second_buoy_640_int8.engine
```

If export runs out of memory, remove `--workspace 4` to use automatic workspace
selection, reduce it to `--workspace 2`, or reduce `--batch` from 8 to 4 or 1.
For a preliminary calibration, `--fraction 0.5` uses half of the calibration
dataset; use the full representative set for final results when practical.

An optional FP32 baseline is:

```bash
python3 convert_model.py \
  --pt models/yolov8n_second_buoy.pt \
  --imgsz 640 \
  --target tensorrt \
  --precision fp32 \
  --output models/yolov8n_second_buoy_640_fp32.engine
```

Verify all outputs:

```bash
ls -lh models/*.engine
```

Engine construction is preparation, not part of a benchmark trial. Build once
and reuse the engine for every trial.

### 10. Smoke-test the engines

FP16:

```bash
python3 detector.py \
  --framework tensorrt \
  --model models/yolov8n_second_buoy_640_fp16.engine \
  --images ./images \
  --imgsz 640 \
  --frames 20 \
  --warmup 10
```

INT8:

```bash
python3 detector.py \
  --framework tensorrt \
  --model models/yolov8n_second_buoy_640_int8.engine \
  --images ./images \
  --imgsz 640 \
  --frames 20 \
  --warmup 10
```

Each run should print 20 `inference time:` lines followed by:

```text
measured loop elapsed: <seconds> s
```

`detector.py` loads all images before timing and cycles them until `--frames` is
reached. Model loading and warmup are excluded from measured-loop time.

### 11. Smoke-test the benchmark harness

```bash
python3 benchmark.py \
  --frameworks tensorrt \
  --model models/yolov8n_second_buoy_640_fp16.engine \
  --images ./images \
  --imgsz 640 \
  --frames 50 \
  --trials 2 \
  --cpu-loads 0 \
  --duration 0 \
  --tegrastats-interval 500 \
  --output jetson_benchmark_640_fp16_smoke.csv
```

Inspect the result before committing to the full run:

```bash
head -n 3 jetson_benchmark_640_fp16_smoke.csv
```

Check that latency, FPS, `nvpmodel_mode`, GPU utilization/frequency,
temperatures, and VDD_IN power are populated. A very short smoke test may finish
before many tegrastats samples are collected.

### 12. Run the full FP16 and INT8 benchmarks

FP16:

```bash
python3 benchmark.py \
  --frameworks tensorrt \
  --model models/yolov8n_second_buoy_640_fp16.engine \
  --images ./images \
  --imgsz 640 \
  --frames 200 \
  --trials 10 \
  --cpu-loads 0 25 50 75 100 \
  --duration 60 \
  --tegrastats-interval 500 \
  --output jetson_benchmark_640_fp16.csv
```

INT8, using exactly the same conditions:

```bash
python3 benchmark.py \
  --frameworks tensorrt \
  --model models/yolov8n_second_buoy_640_int8.engine \
  --images ./images \
  --imgsz 640 \
  --frames 200 \
  --trials 10 \
  --cpu-loads 0 25 50 75 100 \
  --duration 60 \
  --tegrastats-interval 500 \
  --output jetson_benchmark_640_int8.csv
```

Each full command produces 50 rows: five CPU loads multiplied by ten trials.
For every nonzero load, `stress-ng` preheats for `--duration` seconds and remains
active throughout model loading, warmup, and measured inference. Load 0 starts
no stressor and has no preheat wait.

Allow the device to return to a consistent starting temperature before changing
precision, or alternate/run randomized configurations if controlling drift more
formally. Never change power, clocks, fan, images, resolution, frame count,
preheat, or background load between the FP16 and INT8 comparisons.

### 13. Understand the CSV metrics

| Column | Meaning |
|--------|---------|
| `mean_latency_ms` | Mean inference-only time reported by Ultralytics/TensorRT |
| `p50_latency_ms` | Median inference-only latency |
| `p95_latency_ms` | Slow-tail latency |
| `p99_latency_ms` | Extreme-tail latency |
| `fps` | Frames divided by measured-loop time; primary steady-loop throughput |
| `process_fps` | Includes Python startup, engine loading, and warmup |
| `inference_fps` | `1000 / mean_latency_ms`; inference-only theoretical rate |
| `gpu_util_percent_avg/max` | Average/maximum sampled GPU utilization |
| `gpu_freq_MHz_avg/max` | Average/maximum sampled GPU frequency |
| `cpu_temp_C_avg/max` | Sampled CPU temperature |
| `gpu_temp_C_avg/max` | Sampled GPU temperature |
| `soc_temp_C_avg/max` | Sampled SoC temperature |
| `vdd_in_mW_avg/max` | Sampled board input power |
| `nvpmodel_mode` | Power mode reported during the trial |

Normally `inference_fps >= fps >= process_fps`. Compare `p95` and `p99`, not
only the mean; throttling and contention often appear first in tail latency.
`freq_start_MHz` and `freq_end_MHz` may be empty because they use the generic
`psutil` CPU-frequency interface. Use tegrastats fields for Jetson GPU clocks.

Summarize both precisions:

```bash
python3 -c "import pandas as pd; \
[(print('\n'+p.upper()), print(pd.read_csv('jetson_benchmark_640_'+p+'.csv').groupby('cpu_load_percent')[['mean_latency_ms','p95_latency_ms','fps','inference_fps','gpu_temp_C_avg','vdd_in_mW_avg']].mean())) for p in ('fp16','int8')]"
```

### 14. Validate INT8 accuracy

Speed does not establish that INT8 is acceptable. Validate both engines against
the same labeled data:

```bash
yolo detect val \
  model=models/yolov8n_second_buoy_640_fp16.engine \
  data=/home/<jetson-user>/datasets/buoys/data.yaml \
  imgsz=640

yolo detect val \
  model=models/yolov8n_second_buoy_640_int8.engine \
  data=/home/<jetson-user>/datasets/buoys/data.yaml \
  imgsz=640
```

Compare mAP50, mAP50-95, precision, recall, and per-class results. Prefer a
separate test split rather than reporting accuracy on a small calibration set.

### 15. Plot FP16 and INT8 results

`create_plot.py` currently expects one standard Jetson filename per resolution:
`jetson_benchmark_640.csv`. To plot each precision with the current script, copy
one precision at a time and preserve the resulting images:

```bash
cp jetson_benchmark_640_fp16.csv jetson_benchmark_640.csv
python3 create_plot.py
mv latency_boxplot_640.png latency_boxplot_640_fp16.png
mv latency_vs_temp_640.png latency_vs_temp_640_fp16.png

cp jetson_benchmark_640_int8.csv jetson_benchmark_640.csv
python3 create_plot.py
mv latency_boxplot_640.png latency_boxplot_640_int8.png
mv latency_vs_temp_640.png latency_vs_temp_640_int8.png
```

Keep the original precision-specific CSV files. The plotting script would need a
small configuration extension to show FP16 and INT8 simultaneously as separate
series.

### 16. Repeat at other resolutions

Build a separate engine for every resolution and precision. For example:

```text
yolov8n_second_buoy_256_fp16.engine
yolov8n_second_buoy_256_int8.engine
yolov8n_second_buoy_320_fp16.engine
yolov8n_second_buoy_320_int8.engine
yolov8n_second_buoy_640_fp16.engine
yolov8n_second_buoy_640_int8.engine
```

Pass the matching `--imgsz` and engine to both export and benchmark commands.

### 17. Common Jetson failures

- **`ModuleNotFoundError: tensorrt`:** recreate the virtual environment with
  `--system-site-packages` and verify the system Python can import TensorRT.
- **`torch.cuda.is_available()` is false:** replace Torch/Torchvision with the
  Jetson ARM64 versions matching the installed JetPack release.
- **Engine will not deserialize:** rebuild it on this Jetson after checking its
  JetPack/TensorRT version. Engines are not portable like ONNX files.
- **INT8 cannot find calibration data:** use an absolute YAML path and verify its
  `path`, `train`, and `val` entries.
- **TensorRT export runs out of memory:** lower `--batch` or `--workspace`, or
  omit `--workspace` for automatic selection.
- **Telemetry columns are empty:** run `tegrastats --interval 500` directly and
  make sure the trial lasts long enough to collect samples.
- **The benchmark pauses for 60 seconds:** this is the requested preheat for
  every nonzero CPU load; stress continues through inference.
- **INT8 is not faster:** inspect inference-only latency before loop/process FPS,
  check clocks and throttling, and remember unsupported layers may use FP16.
- **Wrong thermal zone:** inspect `cat /sys/class/thermal/thermal_zone*/type` and
  pass `--thermal-zone N` if the cross-platform sensor selection is unsuitable.

## Jetson command runbook

This is the compact copy/paste version of the preceding guide. Replace values in
angle brackets before running a command.

### Inspect the system

```bash
cat /etc/nv_tegra_release
```

Shows the L4T/JetPack base release used to select compatible packages.

```bash
cat /proc/device-tree/model
```

Confirms the exact Jetson hardware.

```bash
python3 --version
```

Shows the Python ABI needed by Jetson-specific wheels.

```bash
dpkg -l | grep -E 'nvidia-jetpack|tensorrt|cuda|cudnn'
```

Lists installed NVIDIA runtime packages.

### Install the platform and harness dependencies

```bash
sudo apt update
sudo apt install -y nvidia-jetpack python3-pip python3-venv python3-opencv stress-ng
sudo reboot
```

Installs JetPack runtimes and the host tools used by the benchmark, then reboots.

```bash
nvcc --version
trtexec --version
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

Verifies CUDA, the TensorRT CLI, and TensorRT Python bindings.

```bash
cd ~/cross-platform-yolo-benchmarking
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

Creates an environment that retains access to JetPack packages and installs the
cross-platform benchmark dependencies. Install the JetPack-specific Torch,
Torchvision, and Ultralytics versions next, following the official Jetson guide.

```bash
python3 -c "import torch, ultralytics, tensorrt as trt; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__, trt.__version__)"
```

Confirms that the Python stack imports and Torch can see CUDA.

### Configure and verify the Jetson

```bash
which stress-ng
which tegrastats
tegrastats --interval 1000
```

Verifies stress and telemetry tools; stop tegrastats with Ctrl+C.

```bash
sudo nvpmodel -q --verbose
```

Displays the active power mode and its limits.

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show
```

Selects MAXN and fixed maximum clocks for a controlled peak-performance run.
Reboot first if `nvpmodel` says the mode change requires it.

### Build engines

```bash
source .venv/bin/activate
python3 convert_model.py --pt models/yolov8n_second_buoy.pt --imgsz 640 \
  --target tensorrt --precision fp16 \
  --output models/yolov8n_second_buoy_640_fp16.engine
```

Builds the reusable 640-pixel FP16 engine on the Jetson.

```bash
python3 convert_model.py --pt models/yolov8n_second_buoy.pt --imgsz 640 \
  --target tensorrt --precision int8 \
  --data /home/<jetson-user>/datasets/buoys/data.yaml \
  --batch 8 --workspace 4 \
  --output models/yolov8n_second_buoy_640_int8.engine
```

Builds the INT8 engine using representative calibration images.

```bash
ls -lh models/*.engine
```

Confirms that both engine files were produced.

### Smoke-test inference and the harness

```bash
python3 detector.py --framework tensorrt \
  --model models/yolov8n_second_buoy_640_fp16.engine \
  --images ./images --imgsz 640 --frames 20 --warmup 10
```

Checks FP16 engine loading, per-frame latency, and measured-loop timing.

```bash
python3 detector.py --framework tensorrt \
  --model models/yolov8n_second_buoy_640_int8.engine \
  --images ./images --imgsz 640 --frames 20 --warmup 10
```

Performs the same check for INT8.

```bash
python3 benchmark.py --frameworks tensorrt \
  --model models/yolov8n_second_buoy_640_fp16.engine \
  --images ./images --imgsz 640 --frames 50 --trials 2 \
  --cpu-loads 0 --duration 0 --tegrastats-interval 500 \
  --output jetson_benchmark_640_fp16_smoke.csv
```

Runs a quick no-stress harness test before the long experiment.

```bash
head -n 3 jetson_benchmark_640_fp16_smoke.csv
```

Checks that latency, throughput, power mode, and telemetry were written.

### Run the full benchmarks

```bash
python3 benchmark.py --frameworks tensorrt \
  --model models/yolov8n_second_buoy_640_fp16.engine \
  --images ./images --imgsz 640 --frames 200 --trials 10 \
  --cpu-loads 0 25 50 75 100 --duration 60 \
  --tegrastats-interval 500 \
  --output jetson_benchmark_640_fp16.csv
```

Runs 50 FP16 trials with controlled CPU load and thermal preheating.

```bash
python3 benchmark.py --frameworks tensorrt \
  --model models/yolov8n_second_buoy_640_int8.engine \
  --images ./images --imgsz 640 --frames 200 --trials 10 \
  --cpu-loads 0 25 50 75 100 --duration 60 \
  --tegrastats-interval 500 \
  --output jetson_benchmark_640_int8.csv
```

Runs the matching 50 INT8 trials under the same conditions.

### Validate, summarize, and plot

```bash
yolo detect val model=models/yolov8n_second_buoy_640_fp16.engine \
  data=/home/<jetson-user>/datasets/buoys/data.yaml imgsz=640
yolo detect val model=models/yolov8n_second_buoy_640_int8.engine \
  data=/home/<jetson-user>/datasets/buoys/data.yaml imgsz=640
```

Compares FP16 and INT8 accuracy on the same labeled validation set.

```bash
python3 -c "import pandas as pd; [(print('\n'+p.upper()), print(pd.read_csv('jetson_benchmark_640_'+p+'.csv').groupby('cpu_load_percent')[['mean_latency_ms','p95_latency_ms','fps','inference_fps','gpu_temp_C_avg','vdd_in_mW_avg']].mean())) for p in ('fp16','int8')]"
```

Prints per-load latency, throughput, temperature, and power summaries.

```bash
cp jetson_benchmark_640_fp16.csv jetson_benchmark_640.csv
python3 create_plot.py
mv latency_boxplot_640.png latency_boxplot_640_fp16.png
mv latency_vs_temp_640.png latency_vs_temp_640_fp16.png
```

Generates and preserves the current standard plots for FP16. Repeat with the
INT8 CSV and `_int8` output names.

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

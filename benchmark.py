#!/usr/bin/env python3
"""
benchmark.py — thermal/load stress harness for YOLOv8 across single-board computers.

For each (framework, cpu_load, trial) it:
  1. starts a fixed CPU load with stress-ng and preheats for --duration,
  2. runs detector.py and collects per-frame latencies (already in ms),
  3. records measured-loop/process throughput plus platform telemetry,
  4. writes one CSV row.

CHANGES vs the original:
  * FIXED the `run_inference(...) * 1000` bug. detector.py prints milliseconds,
    so we no longer multiply (which was repeating the list 1000x and inflating
    num_frames / fps by ~1000). We just wrap in np.array().
  * Thermal zone is AUTO-DETECTED by name (CPU/SoC/tj), because thermal_zone0
    is NOT the CPU sensor on Jetson and some Rockchip boards. Override with
    --thermal-zone N.
  * Extra detector args (--imgsz / --frames / --model) are passed through.
  * Jetson tegrastats data is sampled when the utility is available.
"""

import psutil
import subprocess
import time
import argparse
import numpy as np
import csv
import os
import glob
import shlex
import shutil
import tempfile
import re
from datetime import datetime


# -----------------------------------------------------------------------------
# Thermal zone discovery
# thermal_zone0 is often GPU/PMIC, not CPU — especially on Jetson. Pick by type.
# -----------------------------------------------------------------------------
def find_thermal_zone(prefer=("cpu", "soc", "tj", "tsens", "cpu-therm")):
    zones = {}
    for path in glob.glob("/sys/class/thermal/thermal_zone*/type"):
        try:
            with open(path) as f:
                zones[path.rsplit("/", 1)[0]] = f.read().strip().lower()
        except OSError:
            continue
    for key in prefer:
        for zdir, ztype in zones.items():
            if key in ztype:
                return os.path.join(zdir, "temp"), ztype
    # Fallback: zone0 (original behavior)
    return "/sys/class/thermal/thermal_zone0/temp", "zone0(fallback)"


def get_cpu_temp(temp_path):
    try:
        with open(temp_path) as f:
            return float(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def get_cpu_freq():
    # Note: on Jetson, psutil.cpu_freq() may return None; `tegrastats` is the
    # authoritative source there. We keep this for the ARM SBCs.
    try:
        freq = psutil.cpu_freq()
        return freq.current if freq else None
    except Exception:
        return None


def get_cpu_usage():
    return psutil.cpu_percent(interval=None)


# -----------------------------------------------------------------------------
# External inference runner (detector.py)
# -----------------------------------------------------------------------------
def run_inference(framework, imgsz, frames, model, image, images,
                  external_cmd, unit_scale):
    """
    Runs either the built-in detector.py OR your own script (--external-cmd),
    then parses every 'inference time: <value>' line from stdout.

    unit_scale converts the parsed value to milliseconds:
      1.0    if the script already prints ms   (detector.py)
      1000.0 if the script prints seconds       (your RKNN script)
    """
    if external_cmd:
        # Your own command, run verbatim. --framework/--imgsz/--frames are
        # ignored here (framework is just used as the CSV label).
        cmd = shlex.split(external_cmd)
    else:
        script_path = os.path.join(os.path.dirname(__file__), "detector.py")
        cmd = ["python3", script_path, "--framework", framework,
               "--imgsz", str(imgsz), "--frames", str(frames)]
        if model:
            cmd += ["--model", model]
        if image:
            cmd += ["--image", image]
        elif images:
            cmd += ["--images", images]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"inference command failed (exit {proc.returncode}): {err}")

    durations = []
    loop_elapsed = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("measured loop elapsed:"):
            try:
                loop_elapsed = float(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
            continue
        if not line.startswith("inference time:"):
            continue
        try:
            durations.append(float(line.split(":", 1)[1].strip().split()[0]) * unit_scale)
        except (ValueError, IndexError):
            continue
    if not durations:
        raise RuntimeError(f"No 'inference time:' lines parsed. stderr:\n{err}")
    return durations, loop_elapsed


# -----------------------------------------------------------------------------
# Jetson telemetry (optional; silently unavailable on non-Jetson systems)
# -----------------------------------------------------------------------------
def start_tegrastats(interval_ms):
    executable = shutil.which("tegrastats")
    if executable is None:
        return None
    log = tempfile.NamedTemporaryFile(prefix="yolo_tegrastats_", suffix=".log",
                                      delete=False)
    log.close()
    proc = subprocess.Popen([executable, "--interval", str(interval_ms),
                             "--logfile", log.name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, log.name


def stop_tegrastats(state):
    if state is None:
        return {}
    proc, path = state
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        lines = []
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return parse_tegrastats(lines)


def parse_tegrastats(lines):
    """Return trial averages/maxima from common Jetson tegrastats fields."""
    values = {key: [] for key in ("gpu_util_percent", "gpu_freq_MHz",
                                  "cpu_temp_C", "gpu_temp_C", "soc_temp_C",
                                  "vdd_in_mW")}
    for line in lines:
        gpu = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        gpu_freq = re.search(r"GR3D_FREQ\s+\d+%@(?:\[(?:\d+,)*|)(\d+)", line)
        if gpu is not None:
            values["gpu_util_percent"].append(float(gpu.group(1)))
        if gpu_freq is not None:
            values["gpu_freq_MHz"].append(float(gpu_freq.group(1)))
        for label, key in (("cpu", "cpu_temp_C"), ("gpu", "gpu_temp_C"),
                           ("soc0", "soc_temp_C"), ("soc", "soc_temp_C")):
            match = re.search(rf"(?:^|\s){label}@([0-9.]+)C", line, re.IGNORECASE)
            if match is not None:
                values[key].append(float(match.group(1)))
                if key == "soc_temp_C":
                    break
        power = re.search(r"VDD_IN\s+(\d+)mW", line)
        if power is not None:
            values["vdd_in_mW"].append(float(power.group(1)))
    result = {}
    for key, samples in values.items():
        result[f"{key}_avg"] = float(np.mean(samples)) if samples else None
        result[f"{key}_max"] = max(samples) if samples else None
    result["tegrastats_samples"] = max((len(v) for v in values.values()), default=0)
    return result


def get_nvpmodel_mode():
    executable = shutil.which("nvpmodel")
    if executable is None:
        return None
    try:
        proc = subprocess.run([executable, "-q"], capture_output=True, text=True,
                              timeout=5, check=False)
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return " | ".join(lines) or None
    except (OSError, subprocess.TimeoutExpired):
        return None


# -----------------------------------------------------------------------------
# CPU stress
# -----------------------------------------------------------------------------
def start_cpu_stress(cpu_load_percent):
    if cpu_load_percent <= 0:
        return None  # 0% load = no stressor at all

    cpu_count = psutil.cpu_count(logical=True)

    return subprocess.Popen(
        [
            "/usr/bin/stress-ng",
            "--cpu", str(cpu_count),
            "--cpu-load", str(cpu_load_percent),
            "--metrics-brief",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_stress(proc):
    if proc is None:
        return
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait()


# -----------------------------------------------------------------------------
# One trial
# -----------------------------------------------------------------------------
def run_test(cpu_load, duration, framework, imgsz, frames, model, image, images,
             temp_path, external_cmd, unit_scale, tegrastats_interval):
    stress_proc = start_cpu_stress(cpu_load)
    if stress_proc is not None and duration > 0:
        time.sleep(duration)  # controlled preheat; stress remains active during inference

    temp_start = get_cpu_temp(temp_path)
    freq_start = get_cpu_freq()
    usage_start = get_cpu_usage()

    telemetry = start_tegrastats(tegrastats_interval)
    t0 = time.perf_counter()
    try:
        durations, loop_elapsed = run_inference(
            framework, imgsz, frames, model, image, images, external_cmd, unit_scale)
        arr = np.array(durations)  # normalized to ms
    finally:
        telemetry_stats = stop_tegrastats(telemetry)
        stop_stress(stress_proc)
    t1 = time.perf_counter()

    temp_end = get_cpu_temp(temp_path)
    freq_end = get_cpu_freq()
    usage_end = get_cpu_usage()

    process_elapsed = t1 - t0
    measured_elapsed = loop_elapsed if loop_elapsed and loop_elapsed > 0 else None
    stats = {
        "framework": framework,
        "cpu_load_percent": cpu_load,
        "num_frames": len(arr),
        "total_time_s": process_elapsed,
        "measured_loop_time_s": measured_elapsed,
        "mean_latency_ms": float(arr.mean()),
        "p50_latency_ms": float(np.percentile(arr, 50)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "p99_latency_ms": float(np.percentile(arr, 99)),
        "fps": len(arr) / measured_elapsed if measured_elapsed else None,
        "process_fps": len(arr) / process_elapsed if process_elapsed > 0 else None,
        "inference_fps": 1000.0 / float(arr.mean()) if arr.mean() > 0 else None,
        "temp_start_C": temp_start,
        "temp_end_C": temp_end,
        "freq_start_MHz": freq_start,
        "freq_end_MHz": freq_end,
        "cpu_usage_start_percent": usage_start,
        "cpu_usage_end_percent": usage_end,
        "nvpmodel_mode": get_nvpmodel_mode(),
    }
    stats.update(telemetry_stats)
    return stats


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SBC thermal/load benchmark runner")
    parser.add_argument("--frameworks", nargs="+", default=["onnx"],
                        help="e.g. onnx (Pi/CPU), rknn (Radxa/Khadas), tensorrt (Jetson)")
    parser.add_argument("--cpu-loads", nargs="+", type=int,
                        default=[0, 25, 50, 75, 100])
    parser.add_argument("--duration", type=int, default=60,
                        help="CPU-stress preheat seconds; stress continues through inference")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frames", type=int, default=200,
                        help="Frames timed per trial")
    parser.add_argument("--model", default=None,
                        help="Explicit model path passed to detector.py")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", default=None,
                        help="Real image passed to built-in detector.py")
    inputs.add_argument("--images", default=None,
                        help="Image directory passed to built-in detector.py")
    parser.add_argument("--external-cmd", default=None,
                        help="Run YOUR OWN inference script instead of detector.py, "
                             "e.g. --external-cmd \"python3 my_rknn_bench.py --model m.rknn --imgs ./imgs\". "
                             "It must print 'inference time: <value>' per frame to stdout.")
    parser.add_argument("--latency-unit", choices=["ms", "s"], default="ms",
                        help="Unit your script prints in. Use 's' for the RKNN "
                             "script (prints seconds); values are converted to ms.")
    parser.add_argument("--thermal-zone", type=int, default=None,
                        help="Force /sys/class/thermal/thermal_zoneN (skip auto-detect)")
    parser.add_argument("--tegrastats-interval", type=int, default=500,
                        help="Jetson telemetry sample interval in milliseconds")
    parser.add_argument("--output", default="benchmark_results.csv")
    args = parser.parse_args()

    # Check for stress-ng before starting any benchmark trials.
    if any(load > 0 for load in args.cpu_loads):
        stress_ng = shutil.which("stress-ng")

        if stress_ng is None:
            parser.error(
                "stress-ng was not found.\n"
                "Install it with:\n"
                "  sudo apt update\n"
                "  sudo apt install stress-ng"
            )

        print(f"[stress] using {stress_ng}")

    if args.thermal_zone is not None:
        temp_path = f"/sys/class/thermal/thermal_zone{args.thermal_zone}/temp"
        ztype = f"zone{args.thermal_zone}(forced)"
    else:
        temp_path, ztype = find_thermal_zone()
    print(f"[thermal] using {temp_path}  (type={ztype})")

    unit_scale = 1000.0 if args.latency_unit == "s" else 1.0
    if args.external_cmd:
        print(f"[runner] external: {args.external_cmd}  (unit={args.latency_unit})")

    fieldnames = [
        "trial", "framework", "cpu_load_percent", "num_frames", "total_time_s",
        "measured_loop_time_s", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms",
        "p99_latency_ms", "fps", "process_fps", "inference_fps",
        "temp_start_C", "temp_end_C", "freq_start_MHz", "freq_end_MHz",
        "cpu_usage_start_percent", "cpu_usage_end_percent",
        "nvpmodel_mode", "tegrastats_samples",
        "gpu_util_percent_avg", "gpu_util_percent_max",
        "gpu_freq_MHz_avg", "gpu_freq_MHz_max",
        "cpu_temp_C_avg", "cpu_temp_C_max", "gpu_temp_C_avg", "gpu_temp_C_max",
        "soc_temp_C_avg", "soc_temp_C_max", "vdd_in_mW_avg", "vdd_in_mW_max",
    ]
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for framework in args.frameworks:
            for cpu_load in args.cpu_loads:
                for trial in range(1, args.trials + 1):
                    print(f"[{datetime.now():%H:%M:%S}] trial {trial} | "
                          f"{framework} | load={cpu_load}%")
                    stats = run_test(cpu_load, args.duration, framework,
                                     args.imgsz, args.frames, args.model,
                                     args.image, args.images, temp_path,
                                     args.external_cmd, unit_scale,
                                     args.tegrastats_interval)
                    stats["trial"] = trial
                    writer.writerow(stats)
                    csvfile.flush()


if __name__ == "__main__":
    main()

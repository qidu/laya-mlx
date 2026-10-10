echo "=== "aac6fef/laya-multilingual-mlx"  ====================================="
PYTHONPATH=. .venv/bin/python - <<'PY'
import platform, time
import mlx.core as mx
import serve_judge as sj
from laya_mlx.agent import Agent

MODEL = "aac6fef/laya-multilingual-mlx"   # same checkpoint on every device, so the
                                          # comparison measures the device, not the model
print("mlx", mx.__version__, "|", platform.platform(), "| default", mx.default_device())
print("metal_available", mx.metal.is_available())
print("device_info", mx.metal.device_info())

for dev in ("gpu", "metal", "cpu"):
    a = Agent(MODEL, device=dev, dtype="float16", batch_size=16)
    print(f"\n=== requested={dev}  resolved={a.device} ===")
    for label, qs, n in (("choice", sj.WARMUP_CHOICE, 1), ("noul", sj.WARMUP_NOUL, 2)):
        for _ in range(3):
            a.system_one(sj.WARMUP_STATE, qs)
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            r = a.system_one(sj.WARMUP_STATE, qs)
            mx.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / 20
        print(f"  {label}: {ms:9.2f} ms/call  {n*1000/ms:8.2f} q/s  answered={sorted(r['answers'])}")
    print(f"  peak MLX alloc: {mx.get_peak_memory()/2**20:.1f} MiB")
PY

echo "=== "convaiinnovations/laya"  ====================================="

PYTHONPATH=. .venv/bin/python - <<'PY'
import platform, time
import mlx.core as mx
import serve_judge as sj
from laya_mlx.agent import Agent

print("mlx", mx.__version__, "|", platform.platform(), "| default", mx.default_device())
print("metal_available", mx.metal.is_available())
print("device_info", mx.metal.device_info())

for dev in ("gpu", "metal", "cpu"):
    a = Agent("convaiinnovations/laya", device=dev, dtype="float16", batch_size=16)
    print(f"\n=== requested={dev}  resolved={a.device} ===")
    for label, qs, n in (("choice", sj.WARMUP_CHOICE, 1), ("noul", sj.WARMUP_NOUL, 2)):
        for _ in range(3):
            a.system_one(sj.WARMUP_STATE, qs)
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            r = a.system_one(sj.WARMUP_STATE, qs)
            mx.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / 20
        print(f"  {label}: {ms:9.2f} ms/call  {n*1000/ms:8.2f} q/s  answered={sorted(r['answers'])}")
PY

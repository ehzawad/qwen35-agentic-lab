#!/usr/bin/env python
"""Record the host the study ran on, and check a later host against it.

`env/requirements.lock.txt` pins the Python graph. It cannot pin the parts of
the apparatus that live below Python -- the kernel, glibc, the NVIDIA driver,
the CUDA runtime the torch wheel was built against, and which physical card is
registered. Those are what make a wheel list mean something, and they are what
a fresh clone on someone else's box will differ in first.

    scripts/record_host_apparatus.py record     # write env/host_apparatus.json
    scripts/record_host_apparatus.py check      # compare this host to it

`check` distinguishes the two kinds of difference, because they are not the
same kind of problem:

  MUST MATCH     the tested-apparatus fields whose drift invalidates a replay of
                 the original numbers -- interpreter, torch build, CUDA runtime,
                 registered GPU model. Reported as FAIL.
  MAY DIFFER     the host facts an independent replication is allowed to change
                 -- kernel, glibc, driver patch level, CPU, RAM, disk, hostname,
                 GPU UUID. Reported as NOTE, never as failure: a second A5000 is
                 a NEW RUN with a new run_id, not a defect.

Neither mode touches the GPU. The driver version is read from
/proc/driver/nvidia/version, `nvidia-smi` is invoked read-only for inventory,
and no CUDA context is ever created -- this script is safe to run while a stage
is on the card.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_RECORD = ROOT / "env" / "host_apparatus.json"
SCHEMA = "host_apparatus/1"

# The documented landmine. It is carried in the record itself, not only in
# prose, so that anything reading the record programmatically also learns the
# refusal rule.
FORBIDDEN_PACKAGES = {
    "flash-linear-attention": "flash_linear_attention",
    "causal-conv1d": "causal_conv1d",
}
LANDMINE = {
    "packages_that_must_stay_absent": sorted(FORBIDDEN_PACKAGES),
    "import_names": sorted(FORBIDDEN_PACKAGES.values()),
    "symptom": (
        "SIGSEGV in the forward pass: exit 139, no Python traceback, no partial "
        "output. Not a slowdown and not a numerical regression -- a crash."
    ),
    "versions_observed_failing": ["flash-linear-attention 0.5.2",
                                  "causal-conv1d 1.6.2.post1"],
    "on_stack": "torch 2.11.0+cu130, transformers 5.14.1, one NVIDIA RTX A5000",
    "why_they_look_required": (
        "transformers advertises them as the Gated DeltaNet fast path for this "
        "architecture and will happily use them if importable."
    ),
    "rule": (
        "Neither package may be installed. scripts/setup.sh omits them on "
        "purpose, env/requirements.lock.txt does not contain them, and "
        "`check` FAILS if either is importable. The torch fallback is slower "
        "and correct."
    ),
}


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(*cmd: str) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30, check=True).stdout.strip()
    except Exception:
        return None


def _relative_to_root(path: str) -> str:
    # Deliberately NOT resolved: `.venv/bin/python` is a symlink into the
    # uv-managed interpreter store outside the repository, and resolving it
    # would put an absolute home path back into the record.
    try:
        return pathlib.Path(path).relative_to(ROOT).as_posix()
    except ValueError:
        return path


def os_release() -> dict:
    fields = {}
    path = pathlib.Path("/etc/os-release")
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                fields[k] = v.strip('"')
    return {
        "pretty_name": fields.get("PRETTY_NAME"),
        "id": fields.get("ID"),
        "version_id": fields.get("VERSION_ID"),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "glibc": ".".join(str(p) for p in platform.libc_ver()[1:]) or None,
    }


def driver() -> dict:
    """Driver facts without opening the device."""
    nvrm, kmd_build = None, None
    proc = pathlib.Path("/proc/driver/nvidia/version")
    if proc.exists():
        text = proc.read_text()
        m = re.search(r"NVRM version:\s*(.*)", text)
        if m:
            kmd_build = m.group(1).strip()
            v = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", kmd_build)
            nvrm = v.group(1) if v else None
    smi = _run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    return {
        "nvidia_driver_version": nvrm or (smi.splitlines()[0].strip() if smi else None),
        "kernel_module_build_string": kmd_build,
        "read_from": "/proc/driver/nvidia/version" if nvrm else "nvidia-smi",
    }


def gpus() -> list[dict]:
    out = _run("nvidia-smi",
               "--query-gpu=index,name,uuid,pci.bus_id,memory.total",
               "--format=csv,noheader")
    found = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        found.append({"index": int(parts[0]), "name": parts[1], "uuid": parts[2],
                      "pci_bus_id": parts[3], "memory_total": parts[4]})
    return found


def python_stack() -> dict:
    stack: dict = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        # Repo-relative when the interpreter is the lab venv, so the tracked
        # record does not carry this machine's home directory.
        "executable": _relative_to_root(sys.executable),
        # `uv --version` prints "uv 0.11.21 (x86_64-unknown-linux-gnu)"; keep the
        # version, drop the target triple, which `os.machine` already records.
        "uv_version": next(iter((_run("uv", "--version") or "").split()[1:2]), None),
    }
    # torch is IMPORTED, because only `torch.__version__` carries the `+cu130`
    # build tag that distribution metadata drops, and `torch.version.cuda` is
    # the CUDA runtime the wheel was built against. Importing torch does not
    # create a CUDA context, so this stays safe while a stage is on the card --
    # nothing here calls torch.cuda.
    try:
        import torch  # noqa: PLC0415
        stack["torch_version"] = torch.__version__
        stack["torch_cuda_runtime"] = torch.version.cuda
        stack["torch_cudnn_build"] = torch.backends.cudnn.version()
    except Exception as exc:
        stack["torch_version"] = None
        stack["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    # Everything else comes from installed metadata rather than an import: no
    # reason to execute vLLM's module tree to learn its version number.
    import importlib.metadata as md  # noqa: PLC0415
    for dist in ("transformers", "trl", "peft", "accelerate", "datasets", "vllm",
                 "huggingface-hub", "safetensors", "bitsandbytes", "triton"):
        try:
            stack[dist] = md.version(dist)
        except Exception:
            stack[dist] = None
    return stack


def forbidden_present() -> list[str]:
    import importlib.util
    present = []
    for dist, mod in sorted(FORBIDDEN_PACKAGES.items()):
        try:
            if importlib.util.find_spec(mod) is not None:
                present.append(f"{dist} (import {mod})")
        except Exception:
            present.append(f"{dist} (import {mod}: import machinery raised)")
    return present


def capacity() -> dict:
    mem_total_kb = None
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            mem_total_kb = int(line.split()[1])
            break
    cpu_model = None
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    usage = shutil.disk_usage(ROOT)
    return {
        "cpu_model": cpu_model,
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_gib": round((mem_total_kb or 0) / 1024 / 1024, 1),
        "repo_filesystem_total_gib": round(usage.total / 1024**3, 1),
        "repo_filesystem_free_gib": round(usage.free / 1024**3, 1),
    }


def build_record() -> dict:
    present = forbidden_present()
    return {
        "schema": SCHEMA,
        "kind": "post_finalization_apparatus_identification",
        "recorded_at_utc": _utcnow(),
        "disclosure": (
            "ADDITIVE, DATED record of the host, written AFTER the "
            "preregistration was finalized. It registers nothing and amends "
            "nothing; it states what the apparatus was so a later host can be "
            "compared to it."
        ),
        "tested_apparatus": {
            "os": os_release(),
            "driver": driver(),
            "gpus_present": gpus(),
            "registered_gpu": {
                "name": "NVIDIA RTX A5000",
                "count": 1,
                "note": (
                    "The study is a single-card A5000 study. A second card may "
                    "be physically present and belong to another tenant; it is "
                    "not used, not probed and not reserved."
                ),
            },
            "python_stack": python_stack(),
            "capacity": capacity(),
        },
        "driver_policy": {
            "exact_tested_driver": (driver() or {}).get("nvidia_driver_version"),
            "for_original_replay": (
                "Use driver 610.43.02 with CUDA runtime 13.0. Numbers produced "
                "on a different driver are an independent replication, not a "
                "replay of the original run."
            ),
            "for_independent_replication": (
                "Any driver that supports the torch 2.11.0+cu130 wheel on "
                "Ampere is acceptable. It is a NEW run: new run_id, new locks, "
                "new ledger, new verdict, never an append to the original."
            ),
        },
        "landmine": LANDMINE,
        "forbidden_packages_present": present,
        "forbidden_packages_clean": not present,
        "host_observation": {
            "note": "Informational; not part of any check.",
            "hostname": os.uname().nodename,
            "repo_path": str(ROOT),
        },
    }


MUST_MATCH = (
    ("python_stack", "python_version"),
    ("python_stack", "torch_version"),
    ("python_stack", "torch_cuda_runtime"),
    ("python_stack", "transformers"),
    ("python_stack", "vllm"),
)


def cmd_record(args) -> int:
    record = build_record()
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    ap = record["tested_apparatus"]
    print(f"  {ap['os']['pretty_name']}  kernel {ap['os']['kernel_release']}  "
          f"glibc {ap['os']['glibc']}")
    print(f"  driver {ap['driver']['nvidia_driver_version']}  "
          f"cuda runtime {ap['python_stack']['torch_cuda_runtime']}  "
          f"torch {ap['python_stack']['torch_version']}")
    print(f"  python {ap['python_stack']['python_version']}  "
          f"uv {ap['python_stack']['uv_version']}")
    if record["forbidden_packages_present"]:
        print("  WARNING: recorded a host with the segfaulting packages present: "
              f"{record['forbidden_packages_present']}")
    return 0


def cmd_check(args) -> int:
    rec = json.loads(pathlib.Path(args.record).read_text())
    if rec.get("schema") != SCHEMA:
        raise SystemExit(f"REFUSED: {args.record} is schema {rec.get('schema')!r}, "
                         f"not {SCHEMA!r}")
    now = build_record()
    want, got = rec["tested_apparatus"], now["tested_apparatus"]

    fails, notes = [], []
    for section, key in MUST_MATCH:
        a, b = want[section].get(key), got[section].get(key)
        if a != b:
            fails.append(f"{section}.{key}: this host has {b!r}, the tested "
                         f"apparatus is {a!r}")

    want_gpu = (want.get("registered_gpu") or {}).get("name")
    names = {g["name"] for g in got.get("gpus_present", [])}
    if want_gpu and want_gpu not in names:
        fails.append(f"registered_gpu: no {want_gpu} on this host (found "
                     f"{sorted(names) or 'no GPU'}); the registered card is "
                     f"part of the apparatus, not a preference")

    if now["forbidden_packages_present"]:
        fails.append("landmine: " + ", ".join(now["forbidden_packages_present"])
                     + " is importable. It segfaults the forward pass on this "
                       "stack (exit 139, no traceback). Uninstall it.")

    for section, key in (("os", "kernel_release"), ("os", "glibc"),
                         ("os", "pretty_name"), ("driver", "nvidia_driver_version"),
                         ("capacity", "cpu_model"), ("capacity", "ram_total_gib")):
        a, b = want[section].get(key), got[section].get(key)
        if a != b:
            notes.append(f"{section}.{key}: {b!r} here vs {a!r} tested")
    want_uuids = {g["uuid"] for g in want.get("gpus_present", [])
                  if g["name"] == want_gpu}
    got_uuids = {g["uuid"] for g in got.get("gpus_present", []) if g["name"] == want_gpu}
    if want_uuids and got_uuids and not (want_uuids & got_uuids):
        notes.append(f"{want_gpu} UUID differs ({sorted(got_uuids)} vs "
                     f"{sorted(want_uuids)}): this is a DIFFERENT physical card, "
                     f"so it is a new run_id and a new ledger, never an append")

    for n in notes:
        print(f"NOTE (may differ): {n}")
    if fails:
        print(f"FAIL: {len(fails)} apparatus mismatch(es) that invalidate an "
              f"original replay:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("OK: this host matches the tested apparatus on every must-match field"
          + (f"; {len(notes)} field(s) differ and are allowed to" if notes else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", default=str(DEFAULT_RECORD))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--output", default=str(DEFAULT_RECORD))
    r.set_defaults(fn=cmd_record)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

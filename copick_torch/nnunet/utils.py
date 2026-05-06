import subprocess, sys, yaml


MEDNEXT_INSTALL = "pip install git+https://github.com/MIC-DKFZ/MedNeXt.git"


def check_mednext_installed():
    try:
        import nnunet_mednext  # noqa: F401
    except ModuleNotFoundError:
        print("[ERROR] MedNeXt is not installed. Run:")
        print(f"  {MEDNEXT_INSTALL}")
        sys.exit(1)
    _register_mednext_trainer()


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _run(cmd: list[str], env: dict):
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def _register_mednext_trainer():
    """Copy copick-torch's MedNeXt trainer into nnunetv2's trainer discovery directory."""
    from pathlib import Path
    import shutil, nnunetv2

    src = Path(__file__).parent / "mednext_trainer.py"
    dst = (
        Path(nnunetv2.__path__[0])
        / "training"
        / "nnUNetTrainer"
        / "variants"
        / "nnUNetTrainerMedNeXt.py"
    )

    if not dst.exists():
        shutil.copy2(src, dst)
        print("  [INFO] Registered MedNeXt trainer into nnUNet v2.")


def _scale_batch_size_for_ddp(cfg: dict, model: str, num_gpus: int):
    """
    Scale the plans JSON batch_size to num_gpus × planned_per_gpu_batch_size.

    nnUNet plans for single-GPU memory: batch_size=2 means 2 samples per GPU.
    In DDP the global batch is split across GPUs, so we multiply by num_gpus
    so each GPU still processes the same number of samples (same memory footprint).
    The updated plans file is written back in-place; re-run plan_and_preprocess to reset.
    """
    from pathlib import Path
    import json

    plans_name    = "nnUNetResEncUNetLPlans" if model == "resnecl" else "nnUNetPlans"
    configuration = cfg.get("configuration", "3d_fullres")
    dataset_dir   = f"Dataset{cfg['dataset_id']:03d}_{cfg['dataset_name']}"
    plans_file    = Path(cfg["nnunet_preprocessed"]) / dataset_dir / f"{plans_name}.json"

    if not plans_file.exists():
        return

    with open(plans_file) as f:
        plans = json.load(f)

    try:
        current = plans["configurations"][configuration]["batch_size"]
    except KeyError:
        return

    per_gpu_key = "_copick_torch_per_gpu_batch_size"
    per_gpu_bs  = plans.get(per_gpu_key, current)
    scaled      = per_gpu_bs * num_gpus

    if current == scaled:
        return

    plans[per_gpu_key] = per_gpu_bs
    plans["configurations"][configuration]["batch_size"] = scaled
    with open(plans_file, "w") as f:
        json.dump(plans, f, indent=4)

    print(
        f"  [plans] batch_size scaled {per_gpu_bs} → {scaled} "
        f"({per_gpu_bs}/GPU × {num_gpus} GPUs, same per-GPU memory)."
    )


def remove_prefix(text: str) -> str:
    if text is None:
        return None
    if text.startswith("local://"):
        text = text[8:]
    return text


def save_parameters_yaml(params: dict, output_path: str):
    class _SmartDumper(yaml.SafeDumper):
        pass

    def _represent_list(dumper, data):
        flow = all(x is None or isinstance(x, (str, int, float, bool)) for x in data)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)

    _SmartDumper.add_representer(list, _represent_list)
    with open(output_path, "w") as f:
        yaml.dump(params, f, Dumper=_SmartDumper, default_flow_style=False,
                  sort_keys=False, width=100, indent=2)


def get_config(config_path: str, name: str, process: str, user_id=None, session_id=None) -> dict:
    """Read a process config YAML from the CoPick overlay/static logs directory."""
    import os, glob, copick

    root = copick.from_file(config_path)
    overlay_root = remove_prefix(root.config.overlay_root)
    try:
        static_root = remove_prefix(root.config.static_root)
    except Exception:
        static_root = None

    if session_id is None:
        pattern = glob.glob(os.path.join(overlay_root, "logs", f"{process}_*{name}.yaml"))
        if len(pattern) == 0 and static_root is not None:
            pattern = glob.glob(os.path.join(static_root, "logs", f"{process}_*{name}.yaml"))
        fname = pattern[-1]
    else:
        fname = f"{process}-{user_id}_{session_id}_{name}.yaml"

    if os.path.exists(os.path.join(overlay_root, "logs", fname)):
        path = os.path.join(overlay_root, "logs", fname)
    elif static_root is not None and os.path.exists(os.path.join(static_root, "logs", fname)):
        path = os.path.join(static_root, "logs", fname)
    else:
        raise FileNotFoundError(f"Target config file not found: {fname}")

    with open(path) as f:
        return yaml.safe_load(f)

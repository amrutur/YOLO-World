# YOLO-World — Claude Code project instructions

## Environment

All Python, pip, and project shell commands must run inside the `yolow` conda
environment.

Because each `Bash` tool call spawns a fresh shell, prefix commands with the
activation:

```bash
source /home/amrutur/miniconda3/etc/profile.d/conda.sh && conda activate yolow && <command>
```

Alternatively, invoke the env's interpreter directly:

```
/home/amrutur/miniconda3/envs/yolow/bin/python
```

Do **not** run project Python/pip commands in the base conda env.

## Isolation — important

The user's shell sources ROS 2 Humble in `.bashrc`, which injects
`/opt/ros/humble/...` into `PYTHONPATH`. The user also has packages installed
in `~/.local/lib/python3.10/site-packages`. Without isolation, both of these
leak into the `yolow` env on `sys.path`, and pip will happily uninstall or
upgrade packages in `~/.local` thinking they are "installed" in the env.

This was the cause of a real incident where `pip install torch` inside `yolow`
silently removed torch from `~/.local`.

**Mitigation:** the `yolow` env has activation hooks that:

- unset `PYTHONPATH` (drops ROS leakage)
- export `PYTHONNOUSERSITE=1` (makes Python ignore `~/.local`)

These hooks live at:

```
~/miniconda3/envs/yolow/etc/conda/activate.d/isolate.sh
~/miniconda3/envs/yolow/etc/conda/deactivate.d/isolate.sh
```

They run automatically on `conda activate yolow`, so the standard prefix shown
above already gets you clean isolation. **Do not run pip inside `yolow`
without activating first** — going through the env's interpreter directly
(`~/miniconda3/envs/yolow/bin/python -m pip ...`) bypasses the hooks and
re-exposes the leak.

Verify isolation any time with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate yolow && \
  python -c "import sys; [print(p) for p in sys.path]"
```

Only `yolow` site-packages should appear.

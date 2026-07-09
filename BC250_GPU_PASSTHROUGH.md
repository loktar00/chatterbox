# BC-250 GPU passthrough for this LXC

This container can see the BC-250 and now has the LXC device nodes needed for
DRM/Vulkan visibility. ROCm/HIP compute is still unsafe on this host.

Current evidence from inside the container:

- PCI GPU: `01:00.0 [1002:13fe] Cyan Skillfish [BC-250]`
- ROCm agent from sysfs: `gfx1013`
- KFD sysfs major/minor: `235:0`
- DRM card sysfs major/minor: `226:1`
- DRM render sysfs major/minor: `226:128`
- Present nodes: `/dev/kfd`, `/dev/dri/renderD128`, `/dev/dri/card1`
- Vulkan check: `verify_bc250_vulkan_only.sh` passes and reports
  `AMD BC-250 (RADV GFX1013)`
- ROCm visibility check: `rocminfo` can see a `gfx1013` GPU agent, but compute
  tests are blocked by default because the host crashed during a native HIP
  smoke test
- Installed experimental ROCm Python env:
  `/root/chatterbox/.venv-rocm`
- ROCm PyTorch build in that env:
  `torch==2.6.0+rocm6.2.4`, `torchaudio==2.6.0+rocm6.2.4`

ROCm and PyTorch need `/dev/kfd` plus the DRM render node. Vulkan/RADV needs the
DRM render node. AMD's container docs describe `/dev/kfd` as the main ROCm
compute interface and `/dev/dri` as the DRI GPU interface.

BC-250 is not on AMD's official ROCm supported GPU table. Treat ROCm compute as
unsafe on this machine.

Crash evidence:

- PyTorch ROCm import and device enumeration succeeded after passthrough.
- The first PyTorch ROCm tensor kernel failed cleanly with:
  `HIP error: invalid device function`.
- The PyTorch ROCm wheel reports arch support for:
  `gfx900 gfx906 gfx908 gfx90a gfx942 gfx1030 gfx1100 gfx1101`.
- It does **not** include `gfx1013`, which is the BC-250's actual GPU arch.
- A minimal native HIP kernel compiled for `gfx1013` was then attempted and was
  followed by a full host crash/reboot.

Do not set `HSA_OVERRIDE_GFX_VERSION=gfx1030` for this Chatterbox test; BC-250
is `gfx1013`, and spoofing across ISA families is a stability/correctness risk.
The guarded scripts require `ALLOW_UNSAFE_BC250_ROCM=1` before any ROCm compute
can be run again.

## Host-side LXC config

Run this on the Proxmox host, replacing `$CTID` with this container's ID.

There is also a helper script prepared in this repo:

```bash
# Run on the Proxmox host, not inside this container.
/root/chatterbox/host_apply_bc250_passthrough.sh $CTID
```

If `/root/chatterbox` is not visible on the host, use the manual commands
below.

First confirm the host node names and major numbers:

```bash
ls -la /dev/kfd /dev/dri
stat -c '%n %t:%T %a %U:%G' /dev/kfd /dev/dri/card1 /dev/dri/renderD128
```

Then stop the container and add passthrough:

```bash
pct stop $CTID

cat >> /etc/pve/lxc/$CTID.conf <<'EOF'

# BC-250 GPU access for Vulkan/ROCm
lxc.cgroup2.devices.allow: c 226:* rwm
lxc.cgroup2.devices.allow: c 235:* rwm
lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir
lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=file
EOF

pct start $CTID
```

If the host reports a different KFD major than `235`, use that host value in
the `lxc.cgroup2.devices.allow: c <major>:* rwm` line instead.

## Permission note

For a quick validation pass, the least complicated host-side permission test is:

```bash
chmod 666 /dev/kfd /dev/dri/renderD128 /dev/dri/card1
```

That is intentionally permissive. After we prove the stack works, replace it
with a cleaner group/udev mapping for the container.

## Expected container validation

After restarting the container, run:

```bash
/root/chatterbox/verify_bc250_gpu.sh
```

Success criteria for device visibility:

- `/dev/kfd` exists.
- `/dev/dri/renderD128` exists.
- `vulkaninfo --summary` shows `RADV GFX1013`, not only `llvmpipe`.
- `rocminfo` can open KFD and lists a GPU agent.

Passing these checks does **not** make ROCm compute safe on this card.

## After passthrough works

Run this read-only visibility check:

```bash
/root/chatterbox/verify_bc250_gpu.sh
```

Do **not** run the ROCm compute checks unless you intentionally accept host crash
risk:

```bash
ALLOW_UNSAFE_BC250_ROCM=1 /root/chatterbox/verify_native_hip_gfx1013.sh
ALLOW_UNSAFE_BC250_ROCM=1 /root/chatterbox/verify_rocm_torch.sh
ALLOW_UNSAFE_BC250_ROCM=1 systemctl start chatterbox-api-rocm.service
```

The following scripts now refuse to run by default:

```bash
/root/chatterbox/verify_rocm_torch.sh
/root/chatterbox/verify_native_hip_gfx1013.sh
/root/chatterbox/run_api_rocm.sh
```

The existing CPU Turbo API service is separate and uses `/root/chatterbox/.venv`
on port 8000. It is the only enabled service:

```bash
systemctl is-enabled chatterbox-api.service        # enabled
systemctl is-active chatterbox-api.service         # active
systemctl is-enabled chatterbox-api-rocm.service   # disabled
systemctl is-active chatterbox-api-rocm.service    # inactive
```

The CPU service is forced to `CHATTERBOX_DEVICE=cpu` and clears CUDA/HIP/ROCR
visibility variables. A short synthesis test produced:

```bash
/root/chatterbox/exports/cpu_turbo_test.wav
```

The experimental ROCm service uses `/root/chatterbox/.venv-rocm` on port 8001
and is configured with `Restart=no` so a failing GPU test does not loop.

## Current practical path

ROCm/HIP is not a safe acceleration path for Chatterbox on this BC-250. Vulkan
RADV works at the device level, but upstream Chatterbox is a PyTorch model and
PyTorch does not provide a general Vulkan backend for this workload. Running
Chatterbox on the BC-250 through Vulkan would require a substantial port of the
model/runtime to a Vulkan-capable inference stack rather than a normal package
install or small local rebuild.

One CPU-only export probe has succeeded:

```bash
/root/chatterbox/export_voice_encoder_onnx.py
/root/chatterbox/exports/voice_encoder.forward.random_weights.onnx
```

That proves a small Chatterbox submodule can be lowered to ONNX for further
Vulkan-runtime experiments. It does not make the full PyTorch pipeline runnable
on Vulkan.

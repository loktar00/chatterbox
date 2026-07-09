# BC-250 Fork Handoff

This branch contains the BC-250 Vulkan acceleration work on top of upstream
`master`. ROCm/HIP is not the supported runtime path for this host.

## Branch State

- Branch: `bc250-vulkan-accel`
- Fork remote: `https://github.com/loktar00/chatterbox.git`
- Commit identity: `loktar00 <loktar69@hotmail.com>`
- Safe API: keep the CPU API on port `8000`
- Fast Vulkan API: use the guarded launcher only
- Current-state audit: `BC250_CURRENT_STATE.md`

Verify the current branch before pushing:

```bash
git status --short --branch
git log --format='%h %an <%ae> | %cn <%ce> | %s' master..HEAD
./verify_bc250_safe_stack.py --require-clean-git --require-t3-validation-artifact
```

## Push From This Container

This fresh container does not currently have GitHub credentials, `gh`, or an
SSH key. After adding credentials, push the branch:

```bash
git push -u fork bc250-vulkan-accel
```

For SSH instead of HTTPS:

```bash
git remote set-url fork git@github.com:loktar00/chatterbox.git
git push -u fork bc250-vulkan-accel
```

Do not embed a GitHub token in committed files.

## Push From Another Authenticated Machine

The local backup bundle preserves the branch without needing this container to
authenticate to GitHub.

On an authenticated machine:

```bash
git clone https://github.com/loktar00/chatterbox.git chatterbox-bc250
cd chatterbox-bc250
git fetch /path/to/chatterbox-bc250-vulkan-accel-<commit>.bundle bc250-vulkan-accel:bc250-vulkan-accel
git push -u origin bc250-vulkan-accel
```

Use the newest `/root/chatterbox-bc250-vulkan-accel-*.bundle`.

## Ignored Runtime Artifacts

The Git branch intentionally does not include generated runtime artifacts under
`exports/` or the built helper `.so` files. Dry-run the artifact bundle plan:

```bash
./bundle_bc250_artifacts.py --mode runtime-evidence
```

Create the artifact archive only when you want to spend the disk:

```bash
./bundle_bc250_artifacts.py --mode runtime-evidence --create /root/chatterbox-bc250-artifacts-runtime-evidence.tar.zst
```

Restore it into a fresh checkout:

```bash
tar -C /root/chatterbox --zstd -xf /root/chatterbox-bc250-artifacts-runtime-evidence.tar.zst
./verify_bc250_safe_stack.py --require-t3-validation-artifact
```

## Safe Startup

Keep the CPU API running as the safe fallback. For the fast-fused Vulkan path,
use the guarded launcher:

```bash
PORT=8003 ./run_api_vulkan_fast_fused_guarded.sh
```

For multiple BC-250 workers:

```bash
PORT=8003 CHATTERBOX_VK_DEVICE_SELECT=0000:01:00.0 ./run_api_vulkan_fast_fused_guarded.sh
PORT=8004 CHATTERBOX_VK_DEVICE_SELECT=0000:02:00.0 ./run_api_vulkan_fast_fused_guarded.sh
CHATTERBOX_ROUTER_BACKENDS=http://127.0.0.1:8003,http://127.0.0.1:8004 ./run_router.sh
```

Run one request at a time per worker until concurrency is explicitly tested.

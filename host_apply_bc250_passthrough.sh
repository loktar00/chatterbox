#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <CTID>" >&2
  echo "Run this on the Proxmox host, not inside the container." >&2
  exit 2
fi

ctid="$1"
conf="/etc/pve/lxc/${ctid}.conf"

if [[ ! -f "$conf" ]]; then
  echo "Missing LXC config: $conf" >&2
  echo "This script must be run on the Proxmox host." >&2
  exit 1
fi

echo "Checking host GPU device nodes..."
ls -la /dev/kfd /dev/dri
stat -c '%n major:minor(hex)=%t:%T mode=%a owner=%U:%G' \
  /dev/kfd /dev/dri/renderD128 /dev/dri/card1

backup="${conf}.bak.$(date +%Y%m%d-%H%M%S)"
cp -a "$conf" "$backup"
echo "Backed up $conf to $backup"

add_line() {
  local line="$1"
  if ! grep -Fxq "$line" "$conf"; then
    printf '%s\n' "$line" >> "$conf"
  fi
}

add_line ""
add_line "# BC-250 GPU access for Vulkan/ROCm"
add_line "lxc.cgroup2.devices.allow: c 226:* rwm"
add_line "lxc.cgroup2.devices.allow: c 235:* rwm"
add_line "lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir"
add_line "lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=file"

echo
echo "Updated $conf:"
tail -30 "$conf"
echo
echo "For first validation only, relax host node permissions:"
echo "  chmod 666 /dev/kfd /dev/dri/renderD128 /dev/dri/card1"
echo
echo "Then restart the container:"
echo "  pct restart $ctid"

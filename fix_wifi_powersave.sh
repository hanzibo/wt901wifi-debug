#!/usr/bin/env bash
# 关闭无线网卡省电模式。
#
# 这是 WT901WIFI 在 AP 模式下「丢包 + 间歇性卡顿」的根因:
# 省电模式下网卡会周期休眠, AP 把帧缓冲后突发送达, 实测到达间隔
# p99 高达 228ms、最大 548ms, 收包率也从 ~100Hz 掉到 ~70Hz。
# 关闭后 p99 降到 25ms、收包率恢复到 ~99Hz。
#
# 用法:
#   ./fix_wifi_powersave.sh            # 自动找已连接的无线网卡
#   ./fix_wifi_powersave.sh wlp8s0     # 指定网卡
#
# 该设置通过 NetworkManager 写入连接配置, 会持久保存(针对该 WiFi)。
# 换到别的 WiFi 需要重新执行一次。

set -euo pipefail

IFACE="${1:-}"
if [ -z "$IFACE" ]; then
    IFACE=$(nmcli -t -f DEVICE,TYPE,STATE device 2>/dev/null \
            | awk -F: '$2=="wifi" && $3=="connected"{print $1; exit}')
fi
if [ -z "$IFACE" ]; then
    echo "找不到已连接的无线网卡。用法: $0 [接口名]" >&2
    exit 1
fi
echo "无线接口: $IFACE"

mapfile -t CONNS < <(nmcli -t -f NAME,DEVICE connection show --active \
                     | awk -F: -v d="$IFACE" '$2==d{print $1}')
if [ ${#CONNS[@]} -eq 0 ]; then
    echo "找不到 $IFACE 上的活动连接" >&2
    exit 1
fi

for c in "${CONNS[@]}"; do
    nmcli connection modify "$c" wifi.powersave 2
    echo "已设置 '$c': wifi.powersave=2 (关闭省电)"
done

# 立即生效
nmcli device reapply "$IFACE" >/dev/null 2>&1 \
    || nmcli connection up "${CONNS[0]}" >/dev/null 2>&1 \
    || true

sleep 1
echo -n "当前状态: "
iw dev "$IFACE" get power_save 2>/dev/null || echo "无法读取"

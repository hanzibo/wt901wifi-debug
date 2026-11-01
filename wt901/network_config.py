"""Vendor combined network command; never log the returned payload."""
import ipaddress


def build_sta_command(ssid: str, password: str, server_ip: str, port: int) -> bytes:
    if not 1 <= len(ssid.encode('utf-8')) <= 32:
        raise ValueError('热点名称必须为 1～32 字节')
    if not 8 <= len(password) <= 63:
        raise ValueError('热点密码必须为 8～63 个字符')
    for name, value in (('热点名称', ssid), ('热点密码', password)):
        try:
            value.encode('ascii')
        except UnicodeEncodeError as exc:
            raise ValueError(f'设备配置指令只支持 ASCII {name}') from exc
        if any(char in value for char in ('"', ',', ';', '\r', '\n')):
            raise ValueError(f'{name}不能包含双引号、逗号、分号或换行')
    address = ipaddress.IPv4Address(server_ip)
    if (address.is_unspecified or address.is_loopback or address.is_multicast
            or address.is_link_local or address == ipaddress.IPv4Address('255.255.255.255')):
        raise ValueError('接收地址须为目标网络中的有效单播 IPv4')
    if not 1 <= port <= 65535:
        raise ValueError('接收端口无效')
    return f'IPWIFI:"{ssid}","{password}";UDP{address},{port}\r\n'.encode('ascii')

"""PC-side protocol reconstructed from OptUSBProj; not firmware verification."""
import math
import struct

JOINT_NAMES = tuple(f'{side}{i}' for side in ('R', 'L') for i in range(1, 7)) + ('H1', 'H2', 'Waist')
FINGER_NAMES = tuple(f'{side}{i}' for side in ('RH', 'LH') for i in range(1, 6))
# Historical PC limits only; hardware must be checked separately.
ARM_LIMITS = ((0,235),(90,240),(0,180),(0,118),(0,180),(50,121),
              (0,226),(90,234),(0,180),(0,119),(0,180),(51,122))


def encode(degrees, fingers):
    if len(degrees) != 15 or len(fingers) != 10:
        raise ValueError('需要十五个角度和十个手指控制量')
    angles = []
    for value in degrees:
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 327.67:
            raise ValueError('协议角度须在 0…327.67°，不能溢出 int16')
        # Decimal UI targets, explicitly rounded to 0.01 degrees.
        angles.append(round(value * 100))
    for value in fingers:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4095:
            raise ValueError('手指量须为 0…4095 的整数')
    frame = bytearray(b'\xaa\x55' + struct.pack('<15h', *angles))
    for group in (fingers[:5], fingers[5:]):
        bits = 0
        for value in group:
            bits = (bits << 12) | value
        frame.extend((bits << 4).to_bytes(8, 'big'))
    frame.extend(bytes(4))
    frame.extend(struct.pack('<H', sum(frame)))
    return bytes(frame)


def decode(frame):
    """Offline command inspection, NOT a robot feedback decoder."""
    if len(frame) != 54 or frame[:2] != b'\xaa\x55':
        raise ValueError('控制帧头或长度错误')
    if sum(frame[:-2]) != int.from_bytes(frame[-2:], 'little'):
        raise ValueError('校验和错误')
    if frame[48:52] != bytes(4):
        raise ValueError('保留字段非零')
    angles = [x / 100 for x in struct.unpack('<15h', frame[2:32])]
    fingers = []
    for group in (frame[32:40], frame[40:48]):
        bits = int.from_bytes(group, 'big')
        if bits & 15:
            raise ValueError('手指填充位非零')
        fingers.extend((bits >> (4 + 12 * i)) & 4095 for i in range(4, -1, -1))
    # Revalidate ranges; incoming bytes are never blindly transmitted.
    encode(angles, fingers)
    return angles, fingers

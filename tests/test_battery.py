import math
import unittest

from wt901.protocol import AttitudePacket, battery_percent_from_voltage


class BatteryPercentageTests(unittest.TestCase):
    def test_manual_bins_and_boundaries(self):
        cases = {
            4.20: 100,
            3.97: 100,
            3.96: 90,
            3.93: 90,
            3.92: 75,
            3.87: 75,
            3.82: 60,
            3.79: 50,
            3.77: 40,
            3.73: 30,
            3.70: 20,
            3.68: 15,
            3.50: 10,
            3.40: 5,
            3.39: 0,
            3.30: 0,
        }
        for voltage, expected in cases.items():
            with self.subTest(voltage=voltage):
                self.assertEqual(battery_percent_from_voltage(voltage), expected)

    def test_missing_or_invalid_voltage(self):
        for voltage in (0, -1, math.nan, math.inf, -math.inf):
            with self.subTest(voltage=voltage):
                self.assertIsNone(battery_percent_from_voltage(voltage))

    def test_packet_uses_decoded_centivolts(self):
        raw = bytearray(b"WT5500011381" + bytes(40) + b"\r\n")
        raw[46:48] = (393).to_bytes(2, "little")
        packet = AttitudePacket.parse(bytes(raw))
        self.assertEqual(packet.battery_v, 3.93)
        self.assertEqual(packet.battery_percent, 90)


if __name__ == "__main__":
    unittest.main()

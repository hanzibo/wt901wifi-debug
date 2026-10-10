import socket
import unittest
from wt901.gui_udp import GuiUdpSource


class UdpDiagnosticsTests(unittest.TestCase):
    def test_logging_does_not_change_parser_and_records_gaps(self):
        source=GuiUdpSource(port=0,host='127.0.0.1')
        sender=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        raw=b'WT5500011381'+bytes(40)+b'\r\n'
        try:
            sender.sendto(raw,source.sock.getsockname())
            self.assertEqual(len(source.poll_timed()),1)
            self.assertEqual(source.take_diagnostics(),[])
            source.enable_diagnostics(True)
            for _ in range(2):
                sender.sendto(b'bad'+raw,source.sock.getsockname())
                self.assertEqual(len(source.poll_timed()),1)
            rows=source.take_diagnostics()
            parsed=[r for r in rows if r['event']=='imu_parsed']
            self.assertEqual(len(parsed),2)
            self.assertIsNone(parsed[0]['gap_ms'])
            self.assertGreaterEqual(parsed[1]['gap_ms'],0)
            self.assertTrue(all(r['discarded_bytes']==3 for r in rows if r['event']=='udp_parse'))
            self.assertEqual(len([r for r in rows if r['event']=='udp_datagram']),2)
            self.assertEqual(source.take_diagnostics(),[])
        finally:
            sender.close();source.close()

    def test_bounded_buffer_reports_loss(self):
        source=GuiUdpSource(port=0,host='127.0.0.1')
        try:
            source.enable_diagnostics(True)
            for i in range(20005):source.diagnostic_event('test',value=i)
            rows=source.take_diagnostics()
            self.assertEqual(len(rows),20001)
            self.assertEqual(rows[-1]['dropped'],5)
            source.enable_diagnostics(False)
            self.assertEqual(source.take_diagnostics(),[])
        finally:source.close()


if __name__=='__main__':unittest.main()

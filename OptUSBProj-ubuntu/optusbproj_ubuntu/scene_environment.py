"""Explicit simulator lifecycle; no joint commands or forced process termination."""
import os
import subprocess
import socket
from pathlib import Path
from PySide6 import QtCore


DEFAULT_INSTALL=Path('/home/hzb/Software/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04')
DEFAULT_SCENE=Path(__file__).resolve().parents[2]/'robot-urdf-model-new/NowRobotWithHand1_teleop_neutral.ttt'


def port_in_use(port):
    try:
        with socket.create_connection(('127.0.0.1',int(port)),timeout=.2):return True
    except ConnectionRefusedError:return False
    except OSError:return True  # Ambiguous state must not launch a duplicate instance.


def launch_environment(install,scene,port,gpu=True):
    install=Path(install);scene=Path(scene)
    if not (install/'coppeliaSim.sh').is_file():raise ValueError('未找到 coppeliaSim.sh，请选择安装目录')
    if not scene.is_file() or scene.suffix.lower()!='.ttt':raise ValueError('请选择存在的 .ttt 场景文件')
    env=os.environ.copy()
    # Python/OpenCV/Qt plugin paths must not leak into CoppeliaSim's bundled Qt.
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH','QT_PLUGIN_PATH','QT_QPA_PLATFORM','LD_PRELOAD'):
        env.pop(key,None)
    if gpu:
        for key in ('LIBGL_ALWAYS_SOFTWARE','MESA_LOADER_DRIVER_OVERRIDE','GALLIUM_DRIVER',
                    '__NV_PRIME_RENDER_OFFLOAD_PROVIDER','DRI_PRIME'):
            env.pop(key,None)
        env.update(__NV_PRIME_RENDER_OFFLOAD='1',__GLX_VENDOR_LIBRARY_NAME='nvidia',
                   QT_QPA_PLATFORM='xcb')
    return subprocess.Popen(['bash',str(install/'coppeliaSim.sh'),str(scene.resolve()),
                             '-GzmqRemoteApi.rpcPort='+str(int(port))],cwd=str(install),env=env,
                            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                            start_new_session=True)


class EnvironmentAction(QtCore.QThread):
    result=QtCore.Signal(bool,str)
    def __init__(self,port,closing=False,parent=None):
        super().__init__(parent);self.port=port;self.closing=closing
    def run(self):
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        import zmq
        client=RemoteAPIClient('127.0.0.1',self.port)
        for option in (zmq.RCVTIMEO,zmq.SNDTIMEO):client.socket.setsockopt(option,2000)
        client.socket.setsockopt(zmq.LINGER,0)
        try:
            client.call('sim.getSimulationState',[])
            if self.closing:
                # Trigger the normal simulator exit path, never kill its process.
                try:client.call('sim.quitSimulator',[])
                except zmq.Again:pass  # Exit/dialog may prevent the final reply.
                self.result.emit(True,'已请求正常退出；如有保存/退出确认弹窗，请在 CoppeliaSim 中处理。')
            else:self.result.emit(True,'所选端口的 CoppeliaSim 已运行；不会重复打开或替换当前场景。')
        except Exception as exc:
            self.result.emit(False,'环境未响应：'+str(exc))
        finally:
            client.socket.close(0);client.context.term()

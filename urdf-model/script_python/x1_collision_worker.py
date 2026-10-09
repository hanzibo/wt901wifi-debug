"""Private JSON-lines worker. Run in OptUSBProj-yolo, NOT the ROS env."""
import json
import sys
import time

def emit(data):
    print(json.dumps(data, allow_nan=False), flush=True)


def main():
    try:
        from x1_collision_model import CollisionModel
        model = CollisionModel()
        emit(dict(type='ready', model_version=model.version))
    except Exception as exc:
        emit(dict(type='fatal', reason=str(exc)))
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            began = time.monotonic()
            original = None
            for alpha in (1., .5, .25, .125, .0625):
                goal = {k: v+alpha*(request['goal'][k]-v) for k,v in request['start'].items()}
                result = model.check_motion(request['start'], goal,
                    budget=min(.025, max(0., .09-(time.monotonic()-began))))
                if original is None:
                    original = result.copy()
                if result['safe']:
                    result['alpha'] = alpha
                    if alpha < 1:
                        result['reason'] = '安全步长截短，目标未达到：'+original['reason']
                    break
                if time.monotonic()-began >= .09:
                    break
            result['requested_result'] = original
            result['total_elapsed_ms'] = (time.monotonic()-began)*1000
            emit(dict(type='result', id=request['id'], **result))
        except Exception as exc:
            emit(dict(type='fatal', reason=str(exc)))
            return


if __name__ == '__main__':
    main()

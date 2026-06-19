import os
import json
import base64
import numpy as np
import cv2
import pytest

import app as app_module


# --- Helpers ---

@pytest.fixture(scope='module', autouse=True)
def load_models():
    """Attempt to load real models once for functional tests. If loading fails, skip functional tests."""
    try:
        # load_model will check MODEL_PATH and instantiate YOLO models
        app_module.load_model()
    except Exception as e:
        pytest.skip(f"Could not load models: {e}")
    yield


@pytest.fixture()
def images_list():
    # Load all images from tests/test_images (jpg/png/etc). Return list of tuples (filename, frame)
    img_dir = os.path.join(os.path.dirname(__file__), 'test_images')
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    imgs = []
    if not os.path.exists(img_dir):
        imgs.append(('__synthetic__', np.zeros((480,640,3), dtype=np.uint8)))
        return imgs
    for fn in sorted(os.listdir(img_dir)):
        if os.path.splitext(fn)[1].lower() in exts:
            path = os.path.join(img_dir, fn)
            frm = cv2.imread(path)
            if frm is not None:
                imgs.append((fn, frm))
    if not imgs:
        imgs.append(('__synthetic__', np.zeros((480,640,3), dtype=np.uint8)))
    return imgs


# --- Tests ---

def assert_results_structure(results, keys_required=None):
    assert isinstance(results, list)
    if keys_required:
        for r in results:
            for k in keys_required:
                assert k in r


def test_infer_and_detect_persons(images_list):
    """Functional test: run infer and detect_persons on each image and validate structure."""
    for name, frame in images_list:
        print(f"TEST: infer+detect_persons | image: {name}")
        # run model inference for PPE (may return empty list)
        try:
            ann_ppe, ppe_results = app_module.dispatch(frame.copy(), 'ppe', selected_ppe=['all'], reset=True)
            assert_results_structure(ppe_results, keys_required=['class','confidence','violation'])
        except Exception as e:
            pytest.fail(f"PPE detection failed on {name}: {e}")

        # run person detection
        try:
            ann_person, person_results = app_module.dispatch(frame.copy(), 'person', reset=True)
            assert isinstance(person_results, list)
            # if any results, they should have id and confidence
            for r in person_results:
                assert 'id' in r and 'confidence' in r
        except Exception as e:
            pytest.fail(f"Person detection failed on {name}: {e}")


def test_tripwire_functional(images_list):
    # Set a broad tripwire zone covering center
    app_module.set_tripwire_zone([(0.1,0.1),(0.9,0.1),(0.9,0.9),(0.1,0.9)])
    for name, frame in images_list:
        print(f"TEST: detect_tripwire | image: {name}")
        try:
            ann, res = app_module.dispatch(frame.copy(), 'tripwire', reset=True)
            assert isinstance(res, dict)
            assert set(res.keys()) >= {'breach','breach_count','persons'}
        except Exception as e:
            pytest.fail(f"Tripwire detection failed on {name}: {e}")


def test_tamper_functional(images_list):
    for name, frame in images_list:
        print(f"TEST: detect_tampering | image: {name}")
        try:
            # run warmup calls
            ann, res = app_module.dispatch(frame.copy(), 'tamper', reset=True)
            assert isinstance(res, dict)
            assert 'brightness' in res and 'motion' in res
        except Exception as e:
            pytest.fail(f"Tamper detection failed on {name}: {e}")


def test_bytes_and_b64(images_list):
    for name, frame in images_list:
        print(f"TEST: bytes_to_frame and to_b64 | image: {name}")
        _, buf = cv2.imencode('.jpg', frame)
        data = buf.tobytes()
        frame2 = app_module.bytes_to_frame(data)
        assert frame2 is not None
        b64 = app_module.to_b64(frame2)
        assert isinstance(b64, str)


def test_analytics_endpoints(temp_db):
    print("TEST: analytics endpoints (DB-backed) | expected: 200 responses")
    client = app_module.app.test_client()
    sid1 = app_module.log_session('ppe', 'test')
    app_module.log_person(sid1, 1, 0.75)
    app_module.log_ppe(sid1, [{'class': 'NO-Safety Vest', 'confidence': 1.0, 'violation': True}])
    app_module.log_tripwire(sid1, True, 1, 3)
    app_module.log_tamper(sid1, True, ['Camera Covered'], 10.0, 5.0)

    endpoints = [
        '/api/analytics/summary',
        '/api/analytics/ppe_class_counts',
        '/api/analytics/detections_over_time',
        '/api/analytics/violations_vs_ok',
        '/api/analytics/sessions_by_task',
        '/api/analytics/tamper_reasons',
        '/api/analytics/recent_sessions',
        '/api/analytics/compliance_trend',
    ]
    for ep in endpoints:
        rv = client.get(ep)
        print(f"Endpoint {ep} -> status {rv.status_code}")
        assert rv.status_code == 200


def test_dispatch_routes(images_list):
    print("TEST: dispatch routes to tasks")
    for name, frame in images_list:
        print(f" image: {name}")
        for task in ['person','tripwire','tamper','ppe']:
            ann, res = app_module.dispatch(frame.copy(), task, selected_ppe=['all'], reset=True)
            assert isinstance(ann, np.ndarray)
            assert isinstance(res, (list, dict))


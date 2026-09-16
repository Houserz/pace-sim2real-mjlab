"""Native MuJoCo playback of recorded encoder and fitted simulation trajectories.

No hardware access or new dynamics rollout: this displays the saved evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / 'logs/analysis/all_best172_heldout_chirp30_20260910'
XML = ROOT / 'src/pace_sim2real/assets/dobot/dobot.xml'
STAMPS = ['180339', '180630', '174929']


def load_record(stamp):
    report = json.loads((EVAL / f'20260910_{stamp}_report.json').read_text())
    parameter_path = ROOT / report['parameters']
    assert hashlib.sha256(parameter_path.read_bytes()).hexdigest() == report['parameter_sha256']
    assert hashlib.sha256(XML.read_bytes()).hexdigest() == report['model_sha256']
    saved = torch.load(parameter_path, map_location='cpu', weights_only=True)
    with np.load(EVAL / f'20260910_{stamp}_trajectories.npz') as archive:
        record = {key: archive[key].copy() for key in archive.files}
    assert record['joint_order'].tolist() == saved['joint_order'] == report['joint_order']
    assert record['measured'].shape == record['simulated'].shape == (12000, 12)
    assert np.isfinite(record['measured']).all() and np.isfinite(record['simulated']).all()
    # Real: recorded angles unchanged. Simulation export is q_sim - bias;
    # add bias ONLY there to recover the physical q_sim from the 49-parameter rollout.
    bias = saved['params'][36:48].numpy()
    record['real_q'] = record['measured'].astype(float)
    record['sim_q'] = record['simulated'].astype(float) + bias
    assert np.array_equal(record['real_q'], record['measured'].astype(float))
    assert np.allclose(record['sim_q'] - bias, record['simulated'])
    source = ROOT / 'data/dobot/all' / report['capture']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report['source_sha256']
    with np.load(source) as raw:
        assert np.array_equal(raw['dof_pos'][raw['phase'] == 'chirp'].astype(np.float32),
                              record['measured'])
    return record


def make_model(order):
    root = ET.parse(XML).getroot()
    root.find('compiler').set('meshdir', str(XML.parent / 'assets'))
    root.remove(root.find('sensor'))  # Playback needs no sensor references.
    world = root.find('worldbody')
    original = world.find('body')
    world.remove(original)
    for prefix, color in [('real', '0.1 0.65 1 0.70'), ('sim', '1 0.42 0.08 0.70')]:
        body = copy.deepcopy(original)
        for joint in list(body.findall('joint')):
            if joint.get('type') == 'free':
                body.remove(joint)
        for node in body.iter():
            if 'name' in node.attrib:
                node.set('name', prefix + '_' + node.get('name'))
            if node.tag == 'geom':
                node.set('contype', '0'); node.set('conaffinity', '0')
                node.set('rgba', color)
                # Show the visual meshes and feet, not the collision proxies.
                if node.get('type') != 'mesh' and 'foot' not in node.get('name', ''):
                    node.set('group', '3')
        world.append(body)
    ET.SubElement(world, 'light', pos='0 -2 3', dir='0 0 -1', diffuse='0.8 0.8 0.8')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    addresses = {prefix: np.array([model.joint(prefix+'_'+j).qposadr[0] for j in order])
                 for prefix in ['real', 'sim']}
    return model, addresses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', choices=STAMPS, default='180339')
    parser.add_argument('--speed', type=float, default=0.25)
    parser.add_argument('--start', type=float, default=0.0, help='Seconds into the 30 s chirp.')
    parser.add_argument('--side-by-side', action='store_true')
    parser.add_argument('--snapshot', type=Path, help='Render one frame and exit (offline check).')
    args = parser.parse_args()
    if not 0 < args.speed <= 4 or not 0 <= args.start < 30:
        parser.error('speed must be in (0,4], start in [0,30)')
    records = {stamp: load_record(stamp) for stamp in STAMPS}
    model, addresses = make_model(records[args.capture]['joint_order'])
    model.vis.global_.offwidth = 1100
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)
    opt = mujoco.MjvOption(); opt.geomgroup[3] = 0
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, .3]
    cam.distance = 1.55; cam.azimuth = 135; cam.elevation = -20
    state = dict(stamp=args.capture, t=args.start, speed=args.speed, paused=False,
                 side=args.side_by_side, mode=0)
    mutex = threading.Lock()
    real_body = model.body('real_link_trunk').id
    sim_body = model.body('sim_link_trunk').id
    original_y = model.body_pos[[real_body, sim_body], 1].copy()
    real_geoms = np.array([model.body(int(b)).name.startswith('real_') for b in model.geom_bodyid])
    base_alpha = model.geom_rgba[:, 3].copy()

    def frame():
        rec = records[state['stamp']]
        index = min(int(state['t'] / .0025), 11999)
        data.qpos[addresses['real']] = rec['real_q'][index]
        data.qpos[addresses['sim']] = rec['sim_q'][index]
        model.body_pos[[real_body, sim_body], 1] = original_y + (np.array([-.4, .4]) if state['side'] else 0)
        model.geom_rgba[:, 3] = base_alpha
        if state['mode'] == 1: model.geom_rgba[~real_geoms, 3] = 0
        if state['mode'] == 2: model.geom_rgba[real_geoms, 3] = 0
        data.time = index * .0025
        mujoco.mj_forward(model, data)
        error = np.rad2deg(rec['simulated'][index].astype(float)-rec['measured'][index])
        return float(np.sqrt(np.mean(error**2))), float(np.max(abs(error)))

    if args.snapshot:
        frame()
        with mujoco.Renderer(model, height=720, width=1100) as renderer:
            renderer.update_scene(data, camera=cam, scene_option=opt)
            from PIL import Image
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(args.snapshot)
        assert np.allclose(data.qpos[addresses['real']], records[state['stamp']]['real_q'][int(args.start/.0025)])
        print('PASS: model-frame playback, record order/hash checks and rendered snapshot.', flush=True)
        return

    def key(code):
        with mutex:
            if code == 32: state['paused'] = not state['paused']
            elif code == 258: state['side'] = not state['side']  # Tab
            elif code == 262: state['t'] = min(29.9975, state['t'] + .1)
            elif code == 263: state['t'] = max(0., state['t'] - .1)
            elif code in [49, 50, 51]: state.update(stamp=STAMPS[code-49], t=0.)
            elif code == 82: state['t'] = 0.
            elif code == 86: state['mode'] = (state['mode']+1) % 3  # V
            elif code in [61, 334]: state['speed'] = min(4., state['speed']*2)
            elif code in [45, 333]: state['speed'] = max(.03125, state['speed']/2)

    from mujoco import viewer as native_viewer
    print('Blue=recorded angles (no fitted bias); Orange=49-parameter simulation q. Space pause, Tab layout, '
          '1/2/3 capture, arrows seek 0.1s, +/- speed, V visibility, R restart.', flush=True)
    with native_viewer.launch_passive(model, data, key_callback=key,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = cam.lookat; viewer.cam.distance = cam.distance
        viewer.cam.azimuth = cam.azimuth; viewer.cam.elevation = cam.elevation
        viewer.opt.geomgroup[3] = 0
        last = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            with mutex, viewer.lock():
                if not state['paused']: state['t'] = (state['t'] + (now-last)*state['speed']) % 30
                rmse, peak = frame()
                text = (f"{state['stamp']} | t={state['t']:.3f}/30s | {state['speed']:g}x "
                        f"| {'PAUSED' if state['paused'] else 'PLAY'}\n"
                        f"Blue: recorded angles, NO fitted bias | Orange: simulated q, all 49 parameters\n"
                        f"Encoder-space fit RMSE={rmse:.2f} deg | max={peak:.2f} deg\n"
                        "Space pause | Tab layout | 1/2/3 data | arrows seek | +/- speed | V visibility | R restart\n"
                        "Real is unchanged; orange restores physical q_sim. Fit metric uses q_sim-bias. Offline playback.")
            last = now
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ''))
            viewer.sync()
            time.sleep(1/60)


if __name__ == '__main__':
    main()

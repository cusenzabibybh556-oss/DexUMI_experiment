import * as THREE from 'three';
import { GLTFLoader } from '/assets/vendor/GLTFLoader.js';
import { OrbitControls } from '/assets/vendor/OrbitControls.js';

const MODEL_URLS = {
  left: '/assets/inspire/left/inspire_hand_left.urdf',
  right: '/assets/inspire/right/inspire_hand_right.urdf',
};
const COMMAND_SUFFIXES = [
  'pinky_proximal_joint',
  'ring_proximal_joint',
  'middle_proximal_joint',
  'index_proximal_joint',
  'thumb_proximal_pitch_joint',
  'thumb_proximal_yaw_joint',
];
const COMMAND_MAX = 1000;
const DATA_STALE_MS = 350;
const STATE_DISPLAY_DEADBAND = [6, 6, 6, 6, 8, 12];

function numbers(value, fallback = [0, 0, 0]) {
  if (!value) return fallback;
  const parsed = value.trim().split(/\s+/).map(Number);
  return parsed.length === fallback.length && parsed.every(Number.isFinite) ? parsed : fallback;
}

function applyOrigin(object, origin) {
  const [x, y, z] = numbers(origin?.getAttribute('xyz'));
  const [roll, pitch, yaw] = numbers(origin?.getAttribute('rpy'));
  object.position.set(x, y, z);
  object.quaternion.setFromEuler(new THREE.Euler(roll, pitch, yaw, 'ZYX'));
}

function disposeObject(object) {
  object?.traverse((child) => {
    child.geometry?.dispose();
    if (Array.isArray(child.material)) child.material.forEach((material) => material.dispose());
    else child.material?.dispose();
  });
  object?.removeFromParent();
}

class InspireHandViewer {
  constructor(container, status, sideLabel) {
    this.container = container;
    this.status = status;
    this.sideLabel = sideLabel;
    this.active = false;
    this.ready = false;
    this.side = '';
    this.loadingSide = '';
    this.generation = 0;
    this.joints = new Map();
    this.mimics = [];
    this.targetAngles = new Map();
    this.lastStateAt = 0;
    this.lastCommandAt = 0;
    this.commandSource = '';
    this.displayState = null;
    this.initScene();
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#7b8282');
    this.camera = new THREE.OrthographicCamera(-.1, .1, .1, -.1, .005, 10);
    this.camera.up.set(0, -1, 0);
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = .08;
    this.controls.minZoom = .7;
    this.controls.maxZoom = 5;
    this.scene.add(new THREE.HemisphereLight('#f3f6f5', '#19231f', 2.2));
    const key = new THREE.DirectionalLight('#ffffff', 2.6);
    key.position.set(.3, -.35, .45);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight('#a8c7ff', 1.1);
    fill.position.set(-.3, .2, .1);
    this.scene.add(fill);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
    this.animate();
  }

  configure({ visible, model, side }) {
    const requestedSide = side === 'left' ? 'left' : 'right';
    this.active = Boolean(visible && model === 'inspire');
    if (!this.active) return;
    this.sideLabel.textContent = `${requestedSide.toUpperCase()} · PALM VIEW`;
    if (requestedSide !== this.side && requestedSide !== this.loadingSide) {
      this.loadModel(requestedSide);
    }
  }

  async loadModel(side) {
    const generation = ++this.generation;
    this.loadingSide = side;
    this.ready = false;
    this.setStatus('LOADING');
    try {
      const loaded = await this.loadUrdf(MODEL_URLS[side]);
      if (generation !== this.generation) {
        disposeObject(loaded.model);
        return;
      }
      disposeObject(this.model);
      this.model = loaded.model;
      this.joints = loaded.joints;
      this.mimics = loaded.mimics;
      this.targetAngles = loaded.targetAngles;
      this.side = side;
      this.loadingSide = '';
      this.scene.add(this.model);
      this.setCommand(Array(6).fill(COMMAND_MAX), true);
      this.model.updateMatrixWorld(true);
      this.resetCamera();
      this.ready = true;
      this.lastStateAt = 0;
      this.lastCommandAt = 0;
      this.commandSource = '';
      this.displayState = null;
      this.setStatus('WAITING', 'stale');
    } catch (error) {
      if (generation !== this.generation) return;
      this.loadingSide = '';
      console.error('Inspire model load failed', error);
      this.setStatus('MODEL ERROR', 'error');
    }
  }

  async loadUrdf(urdfUrl) {
    const response = await fetch(urdfUrl);
    if (!response.ok) throw new Error(`URDF ${response.status}`);
    const documentXml = new DOMParser().parseFromString(await response.text(), 'application/xml');
    if (documentXml.querySelector('parsererror')) throw new Error('invalid Inspire URDF');
    const robot = documentXml.querySelector('robot');
    const links = new Map();
    const joints = new Map();
    const mimics = [];
    const targetAngles = new Map();
    const childNames = new Set();
    const meshJobs = [];
    const loader = new GLTFLoader();

    robot.querySelectorAll(':scope > link').forEach((linkXml) => {
      const name = linkXml.getAttribute('name');
      const link = new THREE.Group();
      link.name = name;
      links.set(name, link);
      linkXml.querySelectorAll(':scope > visual').forEach((visual) => {
        const meshXml = visual.querySelector(':scope > geometry > mesh');
        if (!meshXml) return;
        const visualGroup = new THREE.Group();
        applyOrigin(visualGroup, visual.querySelector(':scope > origin'));
        link.add(visualGroup);
        const filename = meshXml.getAttribute('filename');
        const meshUrl = new URL(filename, response.url).pathname;
        const scale = numbers(meshXml.getAttribute('scale'), [1, 1, 1]);
        meshJobs.push(loader.loadAsync(meshUrl).then((gltf) => {
          const visualObject = gltf.scene;
          visualObject.name = `${name}_visual`;
          visualObject.scale.set(...scale);
          visualObject.traverse((object) => {
            if (!object.isMesh) return;
            object.castShadow = false;
            object.receiveShadow = false;
          });
          visualGroup.add(visualObject);
        }));
      });
    });

    robot.querySelectorAll(':scope > joint').forEach((jointXml) => {
      const name = jointXml.getAttribute('name');
      const type = jointXml.getAttribute('type');
      const parentName = jointXml.querySelector(':scope > parent')?.getAttribute('link');
      const childName = jointXml.querySelector(':scope > child')?.getAttribute('link');
      const parent = links.get(parentName);
      const child = links.get(childName);
      if (!parent || !child) return;
      childNames.add(childName);
      const origin = new THREE.Group();
      const motion = new THREE.Group();
      origin.name = `${name}_origin`;
      motion.name = name;
      applyOrigin(origin, jointXml.querySelector(':scope > origin'));
      parent.add(origin);
      origin.add(motion);
      motion.add(child);
      if (type !== 'revolute' && type !== 'continuous') return;
      const axis = numbers(jointXml.querySelector(':scope > axis')?.getAttribute('xyz'), [1, 0, 0]);
      const limit = jointXml.querySelector(':scope > limit');
      const lower = Number(limit?.getAttribute('lower') || 0);
      const upper = Number(limit?.getAttribute('upper') || 0);
      motion.userData.axis = new THREE.Vector3(...axis).normalize();
      motion.userData.lower = lower;
      motion.userData.upper = upper;
      motion.userData.currentAngle = lower;
      joints.set(name, motion);
      targetAngles.set(name, lower);
      const mimic = jointXml.querySelector(':scope > mimic');
      if (mimic) {
        mimics.push({
          source: mimic.getAttribute('joint'),
          target: name,
          multiplier: Number(mimic.getAttribute('multiplier') || 1),
          offset: Number(mimic.getAttribute('offset') || 0),
        });
      }
    });

    await Promise.all(meshJobs);
    const rootName = [...links.keys()].find((name) => !childNames.has(name));
    if (!rootName) throw new Error('Inspire URDF has no root link');
    const model = new THREE.Group();
    model.add(links.get(rootName));
    model.updateMatrixWorld(true);

    // Remove robot-mount rotations so both real left/right URDFs share a stable palm view.
    const palm = links.get('hand_base_link');
    if (palm) {
      const palmRotation = new THREE.Quaternion();
      palm.getWorldQuaternion(palmRotation);
      model.quaternion.copy(palmRotation.invert());
      model.updateMatrixWorld(true);
    }
    return { model, joints, mimics, targetAngles };
  }

  update(data) {
    if (!this.active || !this.ready) return;
    const now = performance.now();
    const state = Array.isArray(data?.hand_state) && data.hand_state.length === 6
      ? data.hand_state.map(Number) : null;
    const action = Array.isArray(data?.action) && data.action.length === 6
      ? data.action.map(Number) : null;
    if (data?.valid?.hand_state && state?.every(Number.isFinite)) {
      this.setCommand(this.filterState(state));
      this.lastStateAt = now;
      this.lastCommandAt = now;
      this.commandSource = 'STATE';
    } else if (data?.valid?.action && action?.every(Number.isFinite)) {
      this.setCommand(action);
      this.lastCommandAt = now;
      this.commandSource = 'ACTION';
    }
  }

  filterState(values) {
    if (!this.displayState) {
      this.displayState = [...values];
      return this.displayState;
    }
    this.displayState = values.map((value, index) => (
      Math.abs(value - this.displayState[index]) > STATE_DISPLAY_DEADBAND[index]
        ? value : this.displayState[index]
    ));
    return this.displayState;
  }

  setCommand(values, immediate = false) {
    COMMAND_SUFFIXES.forEach((suffix, index) => {
      const name = suffix;
      const joint = this.joints.get(name);
      if (!joint) return;
      const flexion = (COMMAND_MAX - THREE.MathUtils.clamp(values[index], 0, COMMAND_MAX)) / COMMAND_MAX;
      const angle = THREE.MathUtils.lerp(joint.userData.lower, joint.userData.upper, flexion);
      this.targetAngles.set(name, angle);
      if (immediate) joint.userData.currentAngle = angle;
    });
    this.mimics.forEach(({ source, target, multiplier, offset }) => {
      const joint = this.joints.get(target);
      const sourceAngle = this.targetAngles.get(source);
      if (!joint || !Number.isFinite(sourceAngle)) return;
      const angle = THREE.MathUtils.clamp(
        sourceAngle * multiplier + offset,
        joint.userData.lower,
        joint.userData.upper,
      );
      this.targetAngles.set(target, angle);
      if (immediate) joint.userData.currentAngle = angle;
    });
  }

  updateStatus(now) {
    if (!this.lastCommandAt) return this.setStatus('WAITING', 'stale');
    if (now - this.lastCommandAt > DATA_STALE_MS) return this.setStatus('STATE · STALE', 'stale');
    if (this.commandSource === 'STATE' && now - this.lastStateAt <= DATA_STALE_MS) {
      return this.setStatus('STATE · LIVE', 'ok');
    }
    return this.setStatus('ACTION · PREVIEW', 'stale');
  }

  setStatus(text, className = '') {
    this.status.textContent = text;
    this.status.className = className;
  }

  fitOrthographicCamera() {
    if (!this.model) return;
    const bounds = new THREE.Box3().setFromObject(this.model);
    const size = bounds.getSize(new THREE.Vector3());
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    const aspect = width / height;
    const viewHeight = Math.max(size.y * 1.16, (size.z * 1.16) / aspect, .14);
    this.camera.left = -viewHeight * aspect / 2;
    this.camera.right = viewHeight * aspect / 2;
    this.camera.top = viewHeight / 2;
    this.camera.bottom = -viewHeight / 2;
    this.camera.updateProjectionMatrix();
  }

  resetCamera() {
    if (!this.model) return;
    const bounds = new THREE.Box3().setFromObject(this.model);
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const distance = Math.max(size.x, size.y, size.z) * 3;
    this.camera.position.set(center.x - distance, center.y, center.z);
    this.camera.up.set(0, -1, 0);
    this.camera.zoom = 1;
    this.controls.target.copy(center);
    this.fitOrthographicCamera();
    this.camera.lookAt(center);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    if (this.model) this.fitOrthographicCamera();
  }

  animate = () => {
    requestAnimationFrame(this.animate);
    if (!this.active || !this.ready) return;
    this.joints.forEach((joint, name) => {
      const target = this.targetAngles.get(name) ?? joint.userData.lower;
      const current = joint.userData.currentAngle ?? joint.userData.lower;
      const angle = THREE.MathUtils.lerp(current, target, .18);
      joint.userData.currentAngle = angle;
      joint.quaternion.setFromAxisAngle(joint.userData.axis, angle);
    });
    this.updateStatus(performance.now());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}

const viewer = new InspireHandViewer(
  document.getElementById('inspireHandModel'),
  document.getElementById('inspireModelState'),
  document.getElementById('inspireModelSide'),
);
window.inspireHandViewer = {
  configure: (profile) => viewer.configure(profile),
  update: (data) => viewer.update(data),
  resetCamera: () => viewer.resetCamera(),
};
window.dispatchEvent(new CustomEvent('inspire-viewer-ready'));

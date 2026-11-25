import * as THREE from 'three';
import { STLLoader } from '/assets/vendor/STLLoader.js';
import { OrbitControls } from '/assets/vendor/OrbitControls.js';

const MODEL_ROOT = '/assets/o6';
const COMMAND_JOINTS = [
  'thumb_cmc_pitch', 'thumb_cmc_yaw', 'index_mcp_pitch',
  'middle_mcp_pitch', 'ring_mcp_pitch', 'pinky_mcp_pitch',
];
const MIMIC_JOINTS = [
  ['thumb_cmc_pitch', 'thumb_ip', 1.86],
  ['index_mcp_pitch', 'index_dip', 0.89],
  ['middle_mcp_pitch', 'middle_dip', 0.89],
  ['ring_mcp_pitch', 'ring_dip', 0.89],
  ['pinky_mcp_pitch', 'pinky_dip', 0.89],
];

const TACTILE_ROWS = 10;
const TACTILE_COLS = 4;
const PALM_ROWS = 14;
const PALM_COLS = 15;
const PALM_MESH_ROWS = 42;
const PALM_MESH_COLS = 45;
const PALM_LAYOUT = { link: 'hand_base_link', flipRows: true, flipCols: false };
const TACTILE_STALE_MS = 120;
const STATE_STALE_MS = 250;
const PATCH_OFFSET_M = 0.00035;
const PALM_PATCH_OFFSET_M = 0.0012;
const NEUTRAL = new THREE.Color('#aab3af');
const PATCH_NEUTRAL = new THREE.Color('#b4bdb9');
// Match the unpressed palm layer to the shell color.
const PALM_NEUTRAL = NEUTRAL.clone();
const PALM_COLOR_STOPS = [
  [0, PALM_NEUTRAL],
  [.08, PALM_NEUTRAL],
  [.40, new THREE.Color('#22c55e')],
  [.72, new THREE.Color('#facc15')],
  [1, new THREE.Color('#ef4444')],
];
const PALM_PREVIEW = new URLSearchParams(window.location.search).get('palmPreview');

// Approved front-view contour in normalized hand_base_link STL (y, z).
// Ordered clockwise from the finger side; the left shelf clears the thumb.
const PALM_FOOTPRINT = [
  [.22, .80], [.36, .82], [.56, .82], [.75, .79],
  [.84, .75], [.89, .69], [.90, .57], [.89, .43],
  [.86, .28], [.80, .19], [.70, .16], [.61, .16],
  [.53, .19], [.48, .25], [.45, .35], [.43, .46],
  [.39, .52], [.32, .54], [.18, .55], [.09, .57],
  [.07, .62], [.07, .68], [.09, .74], [.14, .78],
];
const PALM_Z_MIN = .16;
const PALM_Z_MAX = .82;

// The SDK does not document taxel orientation. These flags are deliberately
// per finger so a physical point-touch check can correct orientation locally.
const FINGER_PATCHES = [
  { link: 'thumb_distal', flipRows: true, flipCols: false },
  { link: 'index_distal', flipRows: true, flipCols: false },
  { link: 'middle_distal', flipRows: true, flipCols: false },
  { link: 'ring_distal', flipRows: true, flipCols: false },
  { link: 'pinky_distal', flipRows: true, flipCols: false },
];

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

function pressureColor(value, neutral = PATCH_NEUTRAL) {
  const ratio = Math.sqrt(THREE.MathUtils.clamp(Number(value) / 255, 0, 1));
  const stops = [
    [0, neutral],
    [.34, new THREE.Color('#22c55e')],
    [.68, new THREE.Color('#f59e0b')],
    [1, new THREE.Color('#ef4444')],
  ];
  for (let index = 1; index < stops.length; index += 1) {
    if (ratio <= stops[index][0]) {
      const [leftAt, left] = stops[index - 1];
      const [rightAt, right] = stops[index];
      return left.clone().lerp(right, (ratio - leftAt) / (rightAt - leftAt));
    }
  }
  return stops.at(-1)[1].clone();
}

function palmPressureColor(value) {
  const ratio = Math.sqrt(THREE.MathUtils.clamp(Number(value) / 255, 0, 1));
  for (let index = 1; index < PALM_COLOR_STOPS.length; index += 1) {
    const [rightAt, right] = PALM_COLOR_STOPS[index];
    if (ratio <= rightAt) {
      const [leftAt, left] = PALM_COLOR_STOPS[index - 1];
      return left.clone().lerp(right, (ratio - leftAt) / (rightAt - leftAt));
    }
  }
  return PALM_COLOR_STOPS.at(-1)[1].clone();
}

function palmFootprintEdges(height) {
  const crossings = [];
  for (let index = 0; index < PALM_FOOTPRINT.length; index += 1) {
    const [aY, aZ] = PALM_FOOTPRINT[index];
    const [bY, bZ] = PALM_FOOTPRINT[(index + 1) % PALM_FOOTPRINT.length];
    if (Math.abs(aZ - bZ) < 1e-9) {
      if (Math.abs(height - aZ) < 1e-9) crossings.push(aY, bY);
    } else if (height >= Math.min(aZ, bZ) - 1e-9
        && height <= Math.max(aZ, bZ) + 1e-9) {
      const t = (height - aZ) / (bZ - aZ);
      crossings.push(THREE.MathUtils.lerp(aY, bY, t));
    }
  }
  if (crossings.length < 2) throw new Error(`invalid palm outline at z=${height}`);
  return [Math.min(...crossings), Math.max(...crossings)];
}

function writeTextureColor(pixels, pixel, color) {
  const encoded = color.clone().convertLinearToSRGB();
  pixels[pixel] = Math.round(encoded.r * 255);
  pixels[pixel + 1] = Math.round(encoded.g * 255);
  pixels[pixel + 2] = Math.round(encoded.b * 255);
  pixels[pixel + 3] = 255;
}

function surfacePoint(geometry, y, z, bounds, offset = PATCH_OFFSET_M,
                      requireHit = false, faceCandidates = null) {
  const position = geometry.attributes.position;
  const index = geometry.index;
  const origin = new THREE.Vector3(bounds.max.x + .02, y, z);
  const ray = new THREE.Ray(origin, new THREE.Vector3(-1, 0, 0));
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const hit = new THREE.Vector3();
  const best = new THREE.Vector3();
  const bestNormal = new THREE.Vector3(1, 0, 0);
  let bestDistance = Infinity;
  const faceCount = faceCandidates ? faceCandidates.length
    : (index ? index.count / 3 : position.count / 3);

  for (let face = 0; face < faceCount; face += 1) {
    const triangle = faceCandidates ? faceCandidates[face] : face;
    const ia = index ? index.getX(triangle * 3) : triangle * 3;
    const ib = index ? index.getX(triangle * 3 + 1) : triangle * 3 + 1;
    const ic = index ? index.getX(triangle * 3 + 2) : triangle * 3 + 2;
    a.fromBufferAttribute(position, ia);
    b.fromBufferAttribute(position, ib);
    c.fromBufferAttribute(position, ic);
    if (y < Math.min(a.y, b.y, c.y) || y > Math.max(a.y, b.y, c.y)
        || z < Math.min(a.z, b.z, c.z) || z > Math.max(a.z, b.z, c.z)) continue;
    if (!ray.intersectTriangle(a, b, c, false, hit)) continue;
    const distance = origin.distanceToSquared(hit);
    if (distance >= bestDistance) continue;
    bestDistance = distance;
    best.copy(hit);
    THREE.Triangle.getNormal(a, b, c, bestNormal);
  }

  if (bestDistance < Infinity) {
    if (bestNormal.x < 0) bestNormal.negate();
    return { point: best.addScaledVector(bestNormal, offset), normal: bestNormal.clone() };
  }

  if (requireHit) throw new Error(`palm ray missed shell at y=${y}, z=${z}`);

  // The selected patch is inside the silhouette for the shipped O6 meshes.
  // This fallback keeps a future mesh revision renderable if an edge ray misses.
  return {
    point: new THREE.Vector3(bounds.max.x + offset, y, z),
    normal: new THREE.Vector3(1, 0, 0),
  };
}

function createPalmFaceBins(geometry, bounds) {
  const cellsY = 48;
  const cellsZ = 64;
  const bins = Array.from({ length: cellsY * cellsZ }, () => []);
  const size = bounds.getSize(new THREE.Vector3());
  const position = geometry.attributes.position;
  const index = geometry.index;
  const faceCount = index ? index.count / 3 : position.count / 3;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const cell = (coordinate, minimum, extent, count) => THREE.MathUtils.clamp(
    Math.floor((coordinate - minimum) / extent * count), 0, count - 1,
  );

  for (let face = 0; face < faceCount; face += 1) {
    a.fromBufferAttribute(position, index ? index.getX(face * 3) : face * 3);
    b.fromBufferAttribute(position, index ? index.getX(face * 3 + 1) : face * 3 + 1);
    c.fromBufferAttribute(position, index ? index.getX(face * 3 + 2) : face * 3 + 2);
    const yMin = Math.min(a.y, b.y, c.y);
    const yMax = Math.max(a.y, b.y, c.y);
    const zMin = Math.min(a.z, b.z, c.z);
    const zMax = Math.max(a.z, b.z, c.z);
    // One-cell padding also covers rays that land exactly on a bin boundary.
    const firstY = cell(yMin, bounds.min.y, size.y, cellsY) - 1;
    const lastY = cell(yMax, bounds.min.y, size.y, cellsY) + 1;
    const firstZ = cell(zMin, bounds.min.z, size.z, cellsZ) - 1;
    const lastZ = cell(zMax, bounds.min.z, size.z, cellsZ) + 1;
    for (let z = Math.max(0, firstZ); z <= Math.min(cellsZ - 1, lastZ); z += 1) {
      for (let y = Math.max(0, firstY); y <= Math.min(cellsY - 1, lastY); y += 1) {
        bins[z * cellsY + y].push(face);
      }
    }
  }

  return (y, z) => bins[
    cell(z, bounds.min.z, size.z, cellsZ) * cellsY
    + cell(y, bounds.min.y, size.y, cellsY)
  ];
}

function createPatchGeometry(sourceGeometry, rows = TACTILE_ROWS, cols = TACTILE_COLS) {
  sourceGeometry.computeBoundingBox();
  const bounds = sourceGeometry.boundingBox;
  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  const yHalf = size.y * .29;
  const zMin = bounds.min.z + size.z * .27;
  const zMax = bounds.min.z + size.z * .97;
  const positions = [];
  const normals = [];
  const uvs = [];
  const indices = [];

  for (let row = 0; row <= rows; row += 1) {
    const v = row / rows;
    const z = THREE.MathUtils.lerp(zMin, zMax, v);
    const tipProgress = THREE.MathUtils.smoothstep(v, .72, 1);
    const baseProgress = 1 - THREE.MathUtils.smoothstep(v, 0, .12);
    const rowHalfWidth = yHalf * (1 - tipProgress * .55 - baseProgress * .08);
    for (let col = 0; col <= cols; col += 1) {
      const u = col / cols;
      const y = THREE.MathUtils.lerp(center.y - rowHalfWidth, center.y + rowHalfWidth, u);
      const sample = surfacePoint(sourceGeometry, y, z, bounds);
      positions.push(sample.point.x, sample.point.y, sample.point.z);
      normals.push(sample.normal.x, sample.normal.y, sample.normal.z);
      uvs.push(u, v);
    }
  }

  const stride = cols + 1;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const lowerLeft = row * stride + col;
      const lowerRight = lowerLeft + 1;
      const upperLeft = lowerLeft + stride;
      const upperRight = upperLeft + 1;
      indices.push(lowerLeft, lowerRight, upperLeft, lowerRight, upperRight, upperLeft);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeBoundingSphere();
  return geometry;
}

function createPalmPatchGeometry(sourceGeometry) {
  sourceGeometry.computeBoundingBox();
  const bounds = sourceGeometry.boundingBox;
  const size = bounds.getSize(new THREE.Vector3());
  const candidateFaces = createPalmFaceBins(sourceGeometry, bounds);
  const positions = [], normals = [], uvs = [], indices = [];

  for (let row = 0; row <= PALM_MESH_ROWS; row += 1) {
    const v = row / PALM_MESH_ROWS;
    const height = THREE.MathUtils.lerp(PALM_Z_MIN, PALM_Z_MAX, v);
    const [leftEdge, rightEdge] = palmFootprintEdges(height);
    const z = bounds.min.z + size.z * height;
    for (let col = 0; col <= PALM_MESH_COLS; col += 1) {
      const u = col / PALM_MESH_COLS;
      const y = bounds.min.y + size.y * THREE.MathUtils.lerp(leftEdge, rightEdge, u);
      const sample = surfacePoint(
        sourceGeometry, y, z, bounds, PALM_PATCH_OFFSET_M, true,
        candidateFaces(y, z),
      );
      positions.push(sample.point.x, sample.point.y, sample.point.z);
      normals.push(sample.normal.x, sample.normal.y, sample.normal.z);
      uvs.push(u, v);
    }
  }
  const stride = PALM_MESH_COLS + 1;
  for (let row = 0; row < PALM_MESH_ROWS; row += 1) {
    for (let col = 0; col < PALM_MESH_COLS; col += 1) {
      const lowerLeft = row * stride + col;
      const lowerRight = lowerLeft + 1;
      const upperLeft = lowerLeft + stride;
      const upperRight = upperLeft + 1;
      indices.push(lowerLeft, lowerRight, upperLeft, lowerRight, upperRight, upperLeft);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeBoundingSphere();
  return geometry;
}

function createTactileTexture(rows = TACTILE_ROWS, cols = TACTILE_COLS,
                              neutral = PATCH_NEUTRAL) {
  const pixels = new Uint8Array(rows * cols * 4);
  for (let index = 0; index < rows * cols; index += 1) {
    writeTextureColor(pixels, index * 4, neutral);
  }
  const texture = new THREE.DataTexture(
    pixels, cols, rows, THREE.RGBAFormat, THREE.UnsignedByteType,
  );
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

class O6HandViewer {
  constructor(container, status) {
    this.container = container;
    this.status = status;
    this.active = false;
    this.ready = false;
    this.loading = null;
    this.side = 'left';
    this.joints = new Map();
    this.targetAngles = new Map();
    this.distalMeshes = new Map();
    this.tactilePatches = [];
    this.lastStateAt = 0;
    this.lastCommandAt = 0;
    this.lastTactileAt = 0;
    this.commandSource = '';
    this.tactileMatrices = null;
    this.patchFresh = false;
    this.lastPalmAt = 0;
    this.palmFresh = false;
    this.palmPreview = ['zero', 'gradient', 'reference'].includes(PALM_PREVIEW)
      ? PALM_PREVIEW : '';
  }

  setVisible(visible, side = 'left') {
    this.side = side === 'right' ? 'right' : 'left';
    this.active = visible;
    if (visible && !this.loading) this.loading = this.init();
  }

  async init() {
    try {
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color('#090c0d');
      this.camera = new THREE.OrthographicCamera(-.1, .1, .1, -.1, .005, 10);
      this.camera.up.set(0, 0, 1);
      this.renderer = new THREE.WebGLRenderer({
        antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: true,
      });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.renderer.shadowMap.enabled = false;
      this.container.appendChild(this.renderer.domElement);
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = .08;
      this.controls.minZoom = .7;
      this.controls.maxZoom = 5;
      this.scene.add(new THREE.HemisphereLight('#f1f5f3', '#1a2824', 2.1));
      const key = new THREE.DirectionalLight('#ffffff', 2.8);
      key.position.set(.25, -.3, .4);
      this.scene.add(key);
      const fill = new THREE.DirectionalLight('#9ec5ff', 1.2);
      fill.position.set(-.3, .25, .12);
      this.scene.add(fill);
      this.model = await this.loadUrdf();
      this.scene.add(this.model);
      this.model.updateMatrixWorld(true);
      this.createTactilePatches();
      if (this.palmPreview) this.applyPalmPreview(this.palmPreview);
      this.resetCamera();
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.container);
      this.resize();
      this.ready = true;
      this.setStatus('WAITING', 'stale');
      this.animate();
    } catch (error) {
      console.error('O6 model load failed', error);
      this.setStatus('MODEL ERROR', 'error');
    }
  }

  async loadUrdf() {
    const response = await fetch(`${MODEL_ROOT}/${this.side}/linkerhand_o6_${this.side}.urdf`);
    if (!response.ok) throw new Error(`URDF ${response.status}`);
    const documentXml = new DOMParser().parseFromString(await response.text(), 'application/xml');
    if (documentXml.querySelector('parsererror')) throw new Error('invalid O6 URDF');
    const robot = documentXml.querySelector('robot');
    const links = new Map();
    const meshJobs = [];
    const loader = new STLLoader();

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
        const filename = meshXml.getAttribute('filename').replace(/^\.\//, '');
        const url = `/assets/o6/${this.side}/${filename}`;
        meshJobs.push(loader.loadAsync(url).then((geometry) => {
          geometry.computeVertexNormals();
          const material = new THREE.MeshStandardMaterial({
            color: NEUTRAL, roughness: .64, metalness: .12,
          });
          const mesh = new THREE.Mesh(geometry, material);
          mesh.name = `${name}_visual`;
          visualGroup.add(mesh);
          if (FINGER_PATCHES.some((patch) => patch.link === name) || name === PALM_LAYOUT.link) {
            this.distalMeshes.set(name, { mesh, visualGroup });
          }
        }));
      });
    });

    const childNames = new Set();
    robot.querySelectorAll(':scope > joint').forEach((jointXml) => {
      const name = jointXml.getAttribute('name');
      const parentName = jointXml.querySelector(':scope > parent')?.getAttribute('link');
      const childName = jointXml.querySelector(':scope > child')?.getAttribute('link');
      const parent = links.get(parentName);
      const child = links.get(childName);
      if (!parent || !child) return;
      childNames.add(childName);
      const origin = new THREE.Group();
      origin.name = `${name}_origin`;
      applyOrigin(origin, jointXml.querySelector(':scope > origin'));
      const motion = new THREE.Group();
      motion.name = name;
      const axis = numbers(jointXml.querySelector(':scope > axis')?.getAttribute('xyz'), [1, 0, 0]);
      const limit = jointXml.querySelector(':scope > limit');
      const lower = Number(limit?.getAttribute('lower') || 0);
      const upper = Number(limit?.getAttribute('upper') || 0);
      motion.userData.axis = new THREE.Vector3(...axis).normalize();
      motion.userData.lower = lower;
      motion.userData.upper = upper;
      this.joints.set(name, motion);
      this.targetAngles.set(name, lower);
      parent.add(origin);
      origin.add(motion);
      motion.add(child);
    });
    await Promise.all(meshJobs);
    const rootName = [...links.keys()].find((name) => !childNames.has(name));
    if (!rootName) throw new Error('O6 URDF has no root link');
    const model = new THREE.Group();
    model.add(links.get(rootName));
    return model;
  }

  createTactilePatches() {
    this.tactilePatches = FINGER_PATCHES.map((layout) => {
      const target = this.distalMeshes.get(layout.link);
      if (!target) throw new Error(`missing tactile link ${layout.link}`);
      const texture = createTactileTexture();
      const material = new THREE.MeshBasicMaterial({
        map: texture,
        side: THREE.DoubleSide,
        polygonOffset: true,
        polygonOffsetFactor: -2,
        polygonOffsetUnits: -2,
      });
      const mesh = new THREE.Mesh(createPatchGeometry(target.mesh.geometry), material);
      mesh.name = `${layout.link}_tactile_patch`;
      mesh.renderOrder = 3;
      target.visualGroup.add(mesh);
      return { ...layout, mesh, texture, pixels: texture.image.data };
    });
    const palmTarget = this.distalMeshes.get(PALM_LAYOUT.link);
    if (!palmTarget) throw new Error('missing palm link');
    const palmTexture = createTactileTexture(PALM_ROWS, PALM_COLS, PALM_NEUTRAL);
    const palmMaterial = new THREE.MeshStandardMaterial({
      map: palmTexture, side: THREE.FrontSide, depthTest: true, depthWrite: true,
      roughness: .64, metalness: .12,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    });
    const palmMesh = new THREE.Mesh(
      createPalmPatchGeometry(palmTarget.mesh.geometry), palmMaterial,
    );
    palmMesh.name = 'palm_tactile_patch';
    palmMesh.renderOrder = 3;
    palmTarget.visualGroup.add(palmMesh);
    this.palmPatch = {
      ...PALM_LAYOUT, mesh: palmMesh, texture: palmTexture, pixels: palmTexture.image.data,
    };
  }

  update(data) {
    if (!this.ready) return;
    const now = performance.now();
    const state = Array.isArray(data?.hand_state) && data.hand_state.length === 6
      ? data.hand_state.map(Number) : null;
    const action = Array.isArray(data?.action) && data.action.length === 6
      ? data.action.map(Number) : null;
    if (data?.valid?.hand_state && state?.every(Number.isFinite)) {
      this.setCommand(state);
      this.lastStateAt = now;
      this.lastCommandAt = now;
      this.commandSource = 'STATE';
    } else if (data?.valid?.action && action?.every(Number.isFinite)) {
      this.setCommand(action);
      this.lastCommandAt = now;
      this.commandSource = 'ACTION';
    }
    const matrices = data?.tactile?.matrix;
    const tactileGap = Number(data?.sync?.tactile_dt_ms);
    const matrixValid = Array.isArray(matrices) && matrices.length === FINGER_PATCHES.length
      && matrices.every((matrix) => Array.isArray(matrix) && matrix.length >= TACTILE_ROWS);
    if (data?.valid?.tactile && matrixValid
        && (!Number.isFinite(tactileGap) || tactileGap <= TACTILE_STALE_MS)) {
      this.tactileMatrices = matrices;
      this.lastTactileAt = now;
      this.writeTactileTextures(matrices);
    }
    const palm = data?.palm?.matrix;
    const palmAge = Number(data?.sync?.palm_dt_ms);
    if (!this.palmPreview && data?.valid?.palm && Array.isArray(palm) && palm.length === PALM_ROWS
        && palm.every((row) => Array.isArray(row) && row.length === PALM_COLS)
        && (!Number.isFinite(palmAge) || palmAge <= TACTILE_STALE_MS)) {
      this.lastPalmAt = now;
      this.writePalmTexture(palm);
    }
  }

  setCommand(values) {
    COMMAND_JOINTS.forEach((name, index) => {
      const joint = this.joints.get(name);
      if (!joint) return;
      const flexion = (255 - THREE.MathUtils.clamp(values[index], 0, 255)) / 255;
      this.targetAngles.set(name, THREE.MathUtils.lerp(
        joint.userData.lower, joint.userData.upper, flexion,
      ));
    });
    MIMIC_JOINTS.forEach(([source, target, multiplier]) => {
      const targetJoint = this.joints.get(target);
      const angle = (this.targetAngles.get(source) || 0) * multiplier;
      this.targetAngles.set(target, THREE.MathUtils.clamp(
        angle, targetJoint?.userData.lower || 0, targetJoint?.userData.upper || angle,
      ));
    });
  }

  writeTactileTextures(matrices) {
    this.tactilePatches.forEach((patch, finger) => {
      const matrix = matrices[finger];
      for (let row = 0; row < TACTILE_ROWS; row += 1) {
        for (let col = 0; col < TACTILE_COLS; col += 1) {
          const sourceRow = patch.flipRows ? TACTILE_ROWS - 1 - row : row;
          const sourceCol = patch.flipCols ? TACTILE_COLS - 1 - col : col;
          const value = Number(matrix?.[sourceRow]?.[sourceCol]);
          const color = Number.isFinite(value) && value >= 0 ? pressureColor(value) : PATCH_NEUTRAL;
          const pixel = (row * TACTILE_COLS + col) * 4;
          writeTextureColor(patch.pixels, pixel, color);
        }
      }
      patch.texture.needsUpdate = true;
    });
    this.patchFresh = true;
  }

  clearTactileTextures() {
    this.tactilePatches.forEach((patch) => {
      for (let pixel = 0; pixel < patch.pixels.length; pixel += 4) {
        writeTextureColor(patch.pixels, pixel, PATCH_NEUTRAL);
      }
      patch.texture.needsUpdate = true;
    });
    this.patchFresh = false;
  }

  writePalmTexture(matrix) {
    const patch = this.palmPatch;
    for (let row = 0; row < PALM_ROWS; row += 1) {
      for (let col = 0; col < PALM_COLS; col += 1) {
        const sourceRow = patch.flipRows ? PALM_ROWS - 1 - row : row;
        const sourceCol = patch.flipCols ? PALM_COLS - 1 - col : col;
        const value = Number(matrix[sourceRow][sourceCol]);
        writeTextureColor(patch.pixels, (row * PALM_COLS + col) * 4,
          Number.isFinite(value) && value >= 0 ? palmPressureColor(value) : PALM_NEUTRAL);
      }
    }
    patch.texture.needsUpdate = true;
    this.palmFresh = true;
  }

  applyPalmPreview(mode) {
    if (!['zero', 'gradient', 'reference'].includes(mode)) return;
    const matrix = Array.from({ length: PALM_ROWS }, (_, row) =>
      Array.from({ length: PALM_COLS }, (_, col) => {
        if (mode === 'zero') return 0;
        if (mode === 'gradient') {
          return Math.round(255 * (row * PALM_COLS + col) / (PALM_ROWS * PALM_COLS - 1));
        }
        // The reference screenshot has a red spot toward the finger-side
        // outer palm. Source rows are reversed when written to the texture.
        const u = (col + .5) / PALM_COLS;
        const v = (PALM_ROWS - row - .5) / PALM_ROWS;
        const distance = ((u - .65) / .18) ** 2 + ((v - .72) / .20) ** 2;
        const ratio = THREE.MathUtils.clamp(
          (Math.exp(-.5 * distance) - .1) / .9, 0, 1,
        );
        return Math.round(255 * ratio * ratio);
      }));
    this.writePalmTexture(matrix);
    this.lastPalmAt = Infinity;
  }

  clearPalmTexture() {
    if (!this.palmFresh) return;
    for (let pixel = 0; pixel < this.palmPatch.pixels.length; pixel += 4) {
      writeTextureColor(this.palmPatch.pixels, pixel, PALM_NEUTRAL);
    }
    this.palmPatch.texture.needsUpdate = true;
    this.palmFresh = false;
  }

  updateTactileFreshness(now) {
    const fresh = this.tactileMatrices && now - this.lastTactileAt <= TACTILE_STALE_MS;
    if (!fresh && this.patchFresh) this.clearTactileTextures();
    return fresh;
  }

  updateStatus(now, tactileFresh) {
    let commandStatus = 'WAITING';
    let className = 'stale';
    if (this.lastCommandAt && now - this.lastCommandAt <= STATE_STALE_MS) {
      if (this.commandSource === 'STATE' && now - this.lastStateAt <= STATE_STALE_MS) {
        commandStatus = 'STATE · LIVE';
        className = 'ok';
      } else {
        commandStatus = 'ACTION · PREVIEW';
      }
    } else if (this.lastCommandAt) {
      commandStatus = 'STATE · STALE';
    }
    const tactileStatus = tactileFresh ? 'F · LIVE' : 'F · STALE';
    const palmStatus = this.palmPreview
      ? `P · ${this.palmPreview.toUpperCase()} PREVIEW`
      : (this.palmFresh ? 'P · LIVE' : 'P · STALE');
    if (!tactileFresh) className = 'stale';
    this.setStatus(`${commandStatus} / ${tactileStatus} / ${palmStatus}`, className);
  }

  setStatus(text, className = '') {
    this.status.textContent = text;
    this.status.className = className;
  }

  fitOrthographicCamera() {
    if (!this.model || !this.camera || !this.container) return;
    const bounds = new THREE.Box3().setFromObject(this.model);
    const size = bounds.getSize(new THREE.Vector3());
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    const aspect = width / height;
    const viewHeight = Math.max(size.z * 1.12, (size.y * 1.12) / aspect, .12);
    this.camera.left = -viewHeight * aspect / 2;
    this.camera.right = viewHeight * aspect / 2;
    this.camera.top = viewHeight / 2;
    this.camera.bottom = -viewHeight / 2;
    this.camera.updateProjectionMatrix();
  }

  resetCamera() {
    if (!this.model || !this.camera || !this.controls) return;
    const bounds = new THREE.Box3().setFromObject(this.model);
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const distance = Math.max(size.x, size.y, size.z) * 3;
    this.camera.position.set(center.x + distance, center.y, center.z);
    this.camera.up.set(0, 0, 1);
    this.camera.zoom = 1;
    this.controls.target.copy(center);
    this.fitOrthographicCamera();
    this.camera.lookAt(center);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  resize() {
    if (!this.renderer || !this.camera) return;
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.fitOrthographicCamera();
  }

  animate = () => {
    requestAnimationFrame(this.animate);
    if (!this.active || !this.ready) return;
    const now = performance.now();
    this.joints.forEach((joint, name) => {
      const target = this.targetAngles.get(name) || 0;
      const current = joint.userData.currentAngle || 0;
      const angle = THREE.MathUtils.lerp(current, target, .16);
      joint.userData.currentAngle = angle;
      joint.quaternion.setFromAxisAngle(joint.userData.axis, angle);
    });
    const tactileFresh = this.updateTactileFreshness(now);
    if (!this.palmPreview && this.palmFresh
        && now - this.lastPalmAt > TACTILE_STALE_MS) this.clearPalmTexture();
    this.updateStatus(now, tactileFresh);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}

const container = document.getElementById('o6HandModel');
const status = document.getElementById('o6ModelState');
const viewer = new O6HandViewer(container, status);
window.o6HandViewer = {
  setVisible: (visible, side) => viewer.setVisible(Boolean(visible), side),
  update: (data) => viewer.update(data),
  resetCamera: () => viewer.resetCamera(),
};
window.dispatchEvent(new CustomEvent('o6-viewer-ready'));

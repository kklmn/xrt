const viewport = document.querySelector("#viewport");
const runButton = document.querySelector("#runButton");
const rayBudget = document.querySelector("#rayBudget");
const raySize = document.querySelector("#raySize");
const pointOpacityInput = document.querySelector("#pointOpacity");
const lineOpacityInput = document.querySelector("#lineOpacity");
const scaleXInput = document.querySelector("#scaleX");
const scaleYInput = document.querySelector("#scaleY");
const scaleZInput = document.querySelector("#scaleZ");
const showMeshesInput = document.querySelector("#showMeshes");
const beamSelect = document.querySelector("#beamSelect");
const statusEl = document.querySelector("#status");
const elementList = document.querySelector("#elementList");
const beamList = document.querySelector("#beamList");

const canvas = document.createElement("canvas");
const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
viewport.appendChild(canvas);

const labelsLayer = document.createElement("div");
labelsLayer.className = "labels-layer";
viewport.appendChild(labelsLayer);

const coordReadout = document.createElement("div");
coordReadout.className = "coord-readout";
viewport.appendChild(coordReadout);

const colors = {
  source: [0.45, 0.76, 0.43],
  oe: [0.18, 0.67, 0.86],
  aperture: [0.88, 0.65, 0.26],
  screen: [0.78, 0.49, 1.0],
  element: [0.72, 0.75, 0.79],
  link: [0.36, 0.40, 0.45],
  grid: [0.18, 0.21, 0.25],
  coordBox: [0.31, 0.36, 0.42],
  axisX: [0.9, 0.24, 0.22],
  axisY: [0.3, 0.8, 0.35],
  axisZ: [0.25, 0.52, 1.0],
};

let scenePayload = null;
let beams = new Map();
let selectedBeam = "";
let sceneCenter = [0, 0, 0];
let sceneCenterModel = [0, 0, 0];
let sceneRadius = 1000;
let distance = 3000;
let yaw = -0.75;
let pitch = 0.42;
let target = [0, 0, 0];
let targetModel = [0, 0, 0];
let dragging = null;
let lastPointer = null;
let labelNodes = [];
let coordTicks = [];
let coordBounds = null;
let beamSequence = [];
let beamGroups = new Map();
let beamLinks = [];
let sceneScale = [1, 1, 1];
let showMeshes = true;

if (!gl) {
  setStatus("WebGL unavailable");
  throw new Error("WebGL is not available in this browser.");
}

const program = createProgram(gl, `
attribute vec3 a_position;
attribute vec3 a_color;
uniform mat4 u_matrix;
uniform float u_pointSize;
varying vec3 v_color;
void main() {
  gl_Position = u_matrix * vec4(a_position, 1.0);
  gl_PointSize = u_pointSize;
  v_color = a_color;
}
`, `
precision mediump float;
varying vec3 v_color;
void main() {
  gl_FragColor = vec4(v_color, 1.0);
}
`);

const beamProgram = createProgram(gl, `
attribute vec3 a_position;
attribute float a_colorAxis;
attribute float a_intensity;
uniform mat4 u_matrix;
uniform float u_pointSize;
uniform vec2 u_colorMinMax;
uniform float u_iMax;
uniform float u_opacity;
varying vec4 v_color;

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
  gl_Position = u_matrix * vec4(a_position, 1.0);
  gl_PointSize = u_pointSize;
  float span = max(u_colorMinMax.y - u_colorMinMax.x, 1.0e-12);
  float hue = clamp((a_colorAxis - u_colorMinMax.x) / span, 0.0, 1.0) * 0.85;
  float alpha = clamp(u_opacity * a_intensity / max(u_iMax, 1.0e-12), 0.0, 1.0);
  v_color = vec4(hsv2rgb(vec3(hue, 1.0, 1.0)), alpha);
}
`, `
precision mediump float;
uniform bool u_roundPoints;
varying vec4 v_color;
void main() {
  if (u_roundPoints) {
    vec2 c = gl_PointCoord - vec2(0.5, 0.5);
    if (dot(c, c) > 0.25) discard;
  }
  gl_FragColor = v_color;
}
`);

const attribs = {
  position: gl.getAttribLocation(program, "a_position"),
  color: gl.getAttribLocation(program, "a_color"),
};
const uniforms = {
  matrix: gl.getUniformLocation(program, "u_matrix"),
  pointSize: gl.getUniformLocation(program, "u_pointSize"),
};

const beamAttribs = {
  position: gl.getAttribLocation(beamProgram, "a_position"),
  colorAxis: gl.getAttribLocation(beamProgram, "a_colorAxis"),
  intensity: gl.getAttribLocation(beamProgram, "a_intensity"),
};
const beamUniforms = {
  matrix: gl.getUniformLocation(beamProgram, "u_matrix"),
  pointSize: gl.getUniformLocation(beamProgram, "u_pointSize"),
  colorMinMax: gl.getUniformLocation(beamProgram, "u_colorMinMax"),
  iMax: gl.getUniformLocation(beamProgram, "u_iMax"),
  opacity: gl.getUniformLocation(beamProgram, "u_opacity"),
  roundPoints: gl.getUniformLocation(beamProgram, "u_roundPoints"),
};

const buffers = {
  grid: makeBuffer(),
  axes: makeBuffer(),
  links: makeBuffer(),
  surfaces: makeBuffer(),
  edges: makeBuffer(),
  beams: makeBeamBuffer(),
  beamLines: makeBeamBuffer(),
};

function setStatus(text) {
  statusEl.textContent = text;
}

function makeBuffer() {
  return {
    position: gl.createBuffer(),
    color: gl.createBuffer(),
    count: 0,
    mode: gl.LINES,
    pointSize: 2,
  };
}

function makeBeamBuffer() {
  return {
    position: gl.createBuffer(),
    colorAxis: gl.createBuffer(),
    intensity: gl.createBuffer(),
    count: 0,
    mode: gl.POINTS,
    pointSize: 2,
    opacity: 0.75,
    roundPoints: true,
    colorMinMax: [0, 1],
    iMax: 1,
  };
}

function createProgram(glCtx, vertexSource, fragmentSource) {
  const vertex = compileShader(glCtx, glCtx.VERTEX_SHADER, vertexSource);
  const fragment = compileShader(glCtx, glCtx.FRAGMENT_SHADER, fragmentSource);
  const result = glCtx.createProgram();
  glCtx.attachShader(result, vertex);
  glCtx.attachShader(result, fragment);
  glCtx.linkProgram(result);
  if (!glCtx.getProgramParameter(result, glCtx.LINK_STATUS)) {
    throw new Error(glCtx.getProgramInfoLog(result));
  }
  return result;
}

function compileShader(glCtx, type, source) {
  const shader = glCtx.createShader(type);
  glCtx.shaderSource(shader, source);
  glCtx.compileShader(shader);
  if (!glCtx.getShaderParameter(shader, glCtx.COMPILE_STATUS)) {
    throw new Error(glCtx.getShaderInfoLog(shader));
  }
  return shader;
}

function upload(buffer, positions, colorValues, mode, pointSize = 2) {
  buffer.mode = mode;
  buffer.count = positions.length / 3;
  buffer.pointSize = pointSize;
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.position);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.color);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colorValues), gl.STATIC_DRAW);
}

function drawBuffer(buffer, matrix) {
  if (!buffer.count) return;
  gl.uniformMatrix4fv(uniforms.matrix, false, matrix);
  gl.uniform1f(uniforms.pointSize, buffer.pointSize);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.position);
  gl.vertexAttribPointer(attribs.position, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(attribs.position);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.color);
  gl.vertexAttribPointer(attribs.color, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(attribs.color);

  gl.drawArrays(buffer.mode, 0, buffer.count);
}

function uploadBeam(buffer, positions, colorAxis, intensity, mode, pointSize, opacity, roundPoints, colorMinMax, iMax) {
  buffer.mode = mode;
  buffer.count = positions.length / 3;
  buffer.pointSize = pointSize;
  buffer.opacity = opacity;
  buffer.roundPoints = roundPoints;
  buffer.colorMinMax = colorMinMax || finiteMinMax(colorAxis);
  buffer.iMax = iMax || finiteMax(intensity, 1e-12);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.position);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.colorAxis);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colorAxis), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.intensity);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(intensity), gl.STATIC_DRAW);
}

function drawBeamBuffer(buffer, matrix) {
  if (!buffer.count) return;
  gl.useProgram(beamProgram);
  gl.uniformMatrix4fv(beamUniforms.matrix, false, matrix);
  gl.uniform1f(beamUniforms.pointSize, buffer.pointSize);
  gl.uniform2fv(beamUniforms.colorMinMax, buffer.colorMinMax);
  gl.uniform1f(beamUniforms.iMax, buffer.iMax);
  gl.uniform1f(beamUniforms.opacity, buffer.opacity);
  gl.uniform1i(beamUniforms.roundPoints, buffer.roundPoints ? 1 : 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.position);
  gl.vertexAttribPointer(beamAttribs.position, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(beamAttribs.position);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.colorAxis);
  gl.vertexAttribPointer(beamAttribs.colorAxis, 1, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(beamAttribs.colorAxis);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.intensity);
  gl.vertexAttribPointer(beamAttribs.intensity, 1, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(beamAttribs.intensity);

  gl.drawArrays(buffer.mode, 0, buffer.count);
}

function finiteMinMax(values) {
  let min = Infinity;
  let max = -Infinity;
  values.forEach((value) => {
    if (Number.isFinite(value)) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  });
  if (!Number.isFinite(min)) {
    return [0, 1];
  }
  if (max <= min) {
    if (max === 0) return [-0.1, 0.1];
    return [min * 0.99, max * 1.01];
  }
  return [min, max];
}

function finiteMax(values, fallback = 1e-12) {
  let max = -Infinity;
  values.forEach((value) => {
    if (Number.isFinite(value)) max = Math.max(max, value);
  });
  return Number.isFinite(max) && max > 0 ? max : fallback;
}

function vec3From(point) {
  return [
    Number(point?.[0] || 0),
    Number(point?.[1] || 0),
    Number(point?.[2] || 0),
  ];
}

function scaledPoint(point) {
  return [
    point[0] * sceneScale[0],
    point[1] * sceneScale[1],
    point[2] * sceneScale[2],
  ];
}

function unscaledPoint(point) {
  return [
    point[0] / sceneScale[0],
    point[1] / sceneScale[1],
    point[2] / sceneScale[2],
  ];
}

function addVertex(positions, colorValues, point, color) {
  const scaled = scaledPoint(point);
  positions.push(scaled[0], scaled[1], scaled[2]);
  colorValues.push(color[0], color[1], color[2]);
}

function addSegment(positions, colorValues, a, b, color) {
  addVertex(positions, colorValues, a, color);
  addVertex(positions, colorValues, b, color);
}

function addBoxFrame(positions, colorValues, min, max, color) {
  const p = [
    [min[0], min[1], min[2]],
    [max[0], min[1], min[2]],
    [max[0], max[1], min[2]],
    [min[0], max[1], min[2]],
    [min[0], min[1], max[2]],
    [max[0], min[1], max[2]],
    [max[0], max[1], max[2]],
    [min[0], max[1], max[2]],
  ];
  [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    .forEach(([a, b]) => addSegment(positions, colorValues, p[a], p[b], color));
}

function collectModelPoints({ includeBeams = true } = {}) {
  const pts = [];
  if (scenePayload) {
    scenePayload.elements.forEach((el) => pts.push(vec3From(el.position)));
    scenePayload.links.forEach((link) => link.points.forEach((p) => pts.push(vec3From(p))));
  }
  if (includeBeams) {
    beamSequence.forEach((beam) => {
      const step = Math.max(1, Math.ceil(beam.points.length / 2000));
      for (let i = 0; i < beam.points.length; i += step) {
        pts.push(beam.points[i]);
      }
    });
  }
  return pts;
}

function computeBounds(points, { scaled = false } = {}) {
  const pts = scaled ? points.map(scaledPoint) : points;
  if (!pts.length) {
    return {
      min: [-1, -1, -1],
      max: [1, 1, 1],
      center: [0, 0, 0],
      radius: 1,
    };
  }

  const min = [...pts[0]];
  const max = [...pts[0]];
  pts.forEach((p) => {
    for (let i = 0; i < 3; i += 1) {
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }
  });
  const center = [
    (min[0] + max[0]) * 0.5,
    (min[1] + max[1]) * 0.5,
    (min[2] + max[2]) * 0.5,
  ];
  let radius = 1;
  pts.forEach((p) => {
    radius = Math.max(radius, length(sub(p, center)));
  });
  return { min, max, center, radius: Math.max(radius, 1) };
}

function expandedBounds(bounds) {
  const min = [...bounds.min];
  const max = [...bounds.max];
  for (let i = 0; i < 3; i += 1) {
    const span = Math.max(max[i] - min[i], 1);
    const pad = niceGridStep(span / 20) * 2;
    min[i] -= pad;
    max[i] += pad;
    if (min[i] < 0 && max[i] > 0) {
      min[i] = Math.min(min[i], -pad);
      max[i] = Math.max(max[i], pad);
    }
  }
  return { min, max };
}

function ticksForRange(min, max, targetCount) {
  const span = Math.max(max - min, 1e-9);
  const step = niceGridStep(span / Math.max(targetCount, 2));
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let value = start; value <= max + step * 0.25; value += step) {
    ticks.push(Number(value.toPrecision(12)));
    if (ticks.length > 64) break;
  }
  return ticks;
}

function fitScene() {
  updateSceneMetrics({ preserveTarget: false });
  distance = sceneRadius * 2.8;
}

function updateSceneMetrics({ preserveTarget = true } = {}) {
  const points = collectModelPoints();
  const modelBounds = computeBounds(points);
  const scaledBounds = computeBounds(points, { scaled: true });
  sceneCenterModel = modelBounds.center;
  sceneCenter = scaledBounds.center;
  sceneRadius = scaledBounds.radius;
  coordBounds = expandedBounds(modelBounds);
  if (preserveTarget) {
    target = scaledPoint(targetModel);
  } else {
    targetModel = [...sceneCenterModel];
    target = scaledPoint(targetModel);
  }
  distance = Math.max(distance, sceneRadius * 0.02);
}

function updateSceneScale({ refit = true } = {}) {
  sceneScale = [
    Math.pow(10, Number(scaleXInput.value)),
    Math.pow(10, Number(scaleYInput.value)),
    Math.pow(10, Number(scaleZInput.value)),
  ];
  if (scenePayload) {
    if (refit) {
      fitScene();
    } else {
      updateSceneMetrics({ preserveTarget: true });
    }
    buildStaticGeometry();
    buildBeamGeometry();
    draw();
  }
}

function renderScenePayload(payload, options = {}) {
  const { clearBeams = true, refit = true } = options;
  scenePayload = payload;
  if (clearBeams) {
    beams = new Map();
    selectedBeam = "";
    beamSequence = [];
    beamGroups = new Map();
    beamLinks = [];
  }
  if (refit) {
    fitScene();
  } else {
    updateSceneMetrics({ preserveTarget: true });
  }
  buildStaticGeometry();
  renderElementList();
  renderBeamList();
  draw();
}

function buildStaticGeometry() {
  buildGrid();
  buildAxes();
  buildLinks();
  buildElementGeometry();
}

function buildGrid() {
  const positions = [];
  const colorValues = [];
  coordTicks = [];
  const bounds = coordBounds || expandedBounds(computeBounds(collectModelPoints()));
  const min = bounds.min;
  const max = bounds.max;
  const xTicks = ticksForRange(min[0], max[0], 9);
  const yTicks = ticksForRange(min[1], max[1], 9);
  const zTicks = ticksForRange(min[2], max[2], 7);
  const x0 = clamp(0, min[0], max[0]);
  const y0 = clamp(0, min[1], max[1]);
  const z0 = clamp(0, min[2], max[2]);
  const backY = max[1];
  const sideX = min[0];

  xTicks.forEach((x) => {
    addSegment(positions, colorValues, [x, min[1], z0], [x, max[1], z0], colors.grid);
    addSegment(positions, colorValues, [x, backY, min[2]], [x, backY, max[2]], colors.grid);
    coordTicks.push({ position: [x, min[1], z0], text: formatCoord(x), axis: "x" });
  });
  yTicks.forEach((y) => {
    addSegment(positions, colorValues, [min[0], y, z0], [max[0], y, z0], colors.grid);
    addSegment(positions, colorValues, [sideX, y, min[2]], [sideX, y, max[2]], colors.grid);
    coordTicks.push({ position: [sideX, y, z0], text: formatCoord(y), axis: "y" });
  });
  zTicks.forEach((z) => {
    addSegment(positions, colorValues, [sideX, min[1], z], [sideX, max[1], z], colors.grid);
    addSegment(positions, colorValues, [min[0], backY, z], [max[0], backY, z], colors.grid);
    coordTicks.push({ position: [sideX, backY, z], text: formatCoord(z), axis: "z" });
  });

  addBoxFrame(positions, colorValues, min, max, colors.coordBox);
  addSegment(positions, colorValues, [min[0], y0, z0], [max[0], y0, z0], colors.axisX);
  addSegment(positions, colorValues, [x0, min[1], z0], [x0, max[1], z0], colors.axisY);
  addSegment(positions, colorValues, [x0, y0, min[2]], [x0, y0, max[2]], colors.axisZ);
  coordTicks.push({ position: [max[0], y0, z0], text: "X", axis: "x", major: true });
  coordTicks.push({ position: [x0, max[1], z0], text: "Y", axis: "y", major: true });
  coordTicks.push({ position: [x0, y0, max[2]], text: "Z", axis: "z", major: true });
  upload(buffers.grid, positions, colorValues, gl.LINES, 1);
}

function buildAxes() {
  const positions = [];
  const colorValues = [];
  const bounds = coordBounds || expandedBounds(computeBounds(collectModelPoints()));
  const min = bounds.min;
  const max = bounds.max;
  const origin = [
    clamp(0, min[0], max[0]),
    clamp(0, min[1], max[1]),
    clamp(0, min[2], max[2]),
  ];
  const size = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1) * 0.08;
  addSegment(positions, colorValues, origin, add(origin, [size, 0, 0]), colors.axisX);
  addSegment(positions, colorValues, origin, add(origin, [0, size, 0]), colors.axisY);
  addSegment(positions, colorValues, origin, add(origin, [0, 0, size]), colors.axisZ);
  upload(buffers.axes, positions, colorValues, gl.LINES, 2);
}

function buildLinks() {
  const positions = [];
  const colorValues = [];
  scenePayload.links.forEach((link) => {
    for (let i = 1; i < link.points.length; i += 1) {
      addVertex(positions, colorValues, vec3From(link.points[i - 1]), colors.link);
      addVertex(positions, colorValues, vec3From(link.points[i]), colors.link);
    }
  });
  upload(buffers.links, positions, colorValues, gl.LINES, 2);
}

function buildElementGeometry() {
  const surfacePositions = [];
  const surfaceColors = [];
  const edgePositions = [];
  const edgeColors = [];
  const minSize = Math.max(sceneRadius * 0.018, 1);
  const minDepth = Math.max(sceneRadius * 0.006, 0.2);

  scenePayload.elements.forEach((el) => {
    const c = colors[el.kind] || colors.element;
    const geom = el.geometry || {};
    const width = Math.max(Number(geom.width || 1), minSize);
    const height = Math.max(Number(geom.height || 1), minSize);
    const depth = Math.max(Number(geom.depth || 0.1), minDepth);
    const instances = el.renderInstances?.length ?
      el.renderInstances : [{ position: el.position }];

    instances.forEach((instance) => {
      const p = vec3From(instance.position);
      if (geom.shape === "aperture") {
        addAperture(edgePositions, edgeColors, p, width, height, depth, c);
      } else if (geom.shape === "lens") {
        addLens(surfacePositions, surfaceColors, edgePositions, edgeColors, p, width, height, depth, c);
      } else {
        addBox(surfacePositions, surfaceColors, edgePositions, edgeColors, p, width, height, depth, c);
      }
    });
  });
  upload(buffers.surfaces, surfacePositions, surfaceColors, gl.TRIANGLES, 1);
  upload(buffers.edges, edgePositions, edgeColors, gl.LINES, 1);
}

function addAperture(edgePositions, edgeColors, center, width, height, depth, color) {
  const outer = rectPoints(center, width * 1.45, height * 1.45, center[1]);
  const inner = rectPoints(center, width * 0.72, height * 0.72, center[1]);
  addPolyline(edgePositions, edgeColors, outer, color, true);
  addPolyline(edgePositions, edgeColors, inner, color, true);
  for (let i = 0; i < 4; i += 1) {
    addVertex(edgePositions, edgeColors, outer[i], color);
    addVertex(edgePositions, edgeColors, inner[i], color);
  }
}

function addLens(surfacePositions, surfaceColors, edgePositions, edgeColors, center, width, height, depth, color) {
  const front = rectPoints(center, width, height, center[1] - depth * 0.5);
  const waist = rectPoints(center, width * 0.45, height * 0.45, center[1]);
  const back = rectPoints(center, width, height, center[1] + depth * 0.5);
  addQuad(surfacePositions, surfaceColors, front[0], front[1], waist[1], waist[0], color);
  addQuad(surfacePositions, surfaceColors, front[1], front[2], waist[2], waist[1], color);
  addQuad(surfacePositions, surfaceColors, front[2], front[3], waist[3], waist[2], color);
  addQuad(surfacePositions, surfaceColors, front[3], front[0], waist[0], waist[3], color);
  addQuad(surfacePositions, surfaceColors, waist[0], waist[1], back[1], back[0], color);
  addQuad(surfacePositions, surfaceColors, waist[1], waist[2], back[2], back[1], color);
  addQuad(surfacePositions, surfaceColors, waist[2], waist[3], back[3], back[2], color);
  addQuad(surfacePositions, surfaceColors, waist[3], waist[0], back[0], back[3], color);
  addPolyline(edgePositions, edgeColors, front, color, true);
  addPolyline(edgePositions, edgeColors, waist, color, true);
  addPolyline(edgePositions, edgeColors, back, color, true);
  for (let i = 0; i < 4; i += 1) {
    addVertex(edgePositions, edgeColors, front[i], color);
    addVertex(edgePositions, edgeColors, waist[i], color);
    addVertex(edgePositions, edgeColors, waist[i], color);
    addVertex(edgePositions, edgeColors, back[i], color);
  }
}

function addBox(surfacePositions, surfaceColors, edgePositions, edgeColors, center, width, height, depth, color) {
  const x = width * 0.5;
  const y = depth * 0.5;
  const z = height * 0.5;
  const p = [
    [center[0] - x, center[1] - y, center[2] - z],
    [center[0] + x, center[1] - y, center[2] - z],
    [center[0] + x, center[1] + y, center[2] - z],
    [center[0] - x, center[1] + y, center[2] - z],
    [center[0] - x, center[1] - y, center[2] + z],
    [center[0] + x, center[1] - y, center[2] + z],
    [center[0] + x, center[1] + y, center[2] + z],
    [center[0] - x, center[1] + y, center[2] + z],
  ];
  addQuad(surfacePositions, surfaceColors, p[0], p[1], p[2], p[3], color);
  addQuad(surfacePositions, surfaceColors, p[4], p[7], p[6], p[5], color);
  addQuad(surfacePositions, surfaceColors, p[0], p[4], p[5], p[1], color);
  addQuad(surfacePositions, surfaceColors, p[1], p[5], p[6], p[2], color);
  addQuad(surfacePositions, surfaceColors, p[2], p[6], p[7], p[3], color);
  addQuad(surfacePositions, surfaceColors, p[3], p[7], p[4], p[0], color);
  [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(([a, b]) => {
    addVertex(edgePositions, edgeColors, p[a], color);
    addVertex(edgePositions, edgeColors, p[b], color);
  });
}

function rectPoints(center, width, height, y) {
  const x = width * 0.5;
  const z = height * 0.5;
  return [
    [center[0] - x, y, center[2] - z],
    [center[0] + x, y, center[2] - z],
    [center[0] + x, y, center[2] + z],
    [center[0] - x, y, center[2] + z],
  ];
}

function addQuad(positions, colorValues, a, b, c, d, color) {
  [a, b, c, a, c, d].forEach((point) => addVertex(positions, colorValues, point, color));
}

function addPolyline(positions, colorValues, points, color, closed = false) {
  for (let i = 1; i < points.length; i += 1) {
    addVertex(positions, colorValues, points[i - 1], color);
    addVertex(positions, colorValues, points[i], color);
  }
  if (closed && points.length > 2) {
    addVertex(positions, colorValues, points[points.length - 1], color);
    addVertex(positions, colorValues, points[0], color);
  }
}

function buildBeamGeometry() {
  const positions = [];
  const colorAxis = [];
  const intensities = [];
  const linePositions = [];
  const lineColorAxis = [];
  const lineIntensities = [];
  beamSequence.forEach((beam) => {
    const group = beamGroups.get(beam.elementId);
    if (isHiddenFootprintBeam(beam, group)) return;
    if (selectedBeam && beam.key !== selectedBeam) return;
    beam.points.forEach((point, index) => {
      addBeamVertex(positions, colorAxis, intensities, point, beam.colorAxis[index], beam.intensity[index]);
    });
  });
  beamLinks.forEach(({ start, end }) => {
    if (selectedBeam && selectedBeam !== start.key && selectedBeam !== end.key) return;
    const previousByIndex = new Map();
    start.indices.forEach((rayIndex, j) => previousByIndex.set(rayIndex, j));
    end.indices.forEach((rayIndex, j) => {
      const k = previousByIndex.get(rayIndex);
      if (k === undefined) return;
      addBeamVertex(linePositions, lineColorAxis, lineIntensities, start.points[k], start.colorAxis[k], start.intensity[k]);
      addBeamVertex(linePositions, lineColorAxis, lineIntensities, end.points[j], start.colorAxis[k], start.intensity[k]);
    });
    if (!start.indices.length || !end.indices.length) {
      const count = Math.min(start.points.length, end.points.length);
      for (let j = 0; j < count; j += 1) {
        addBeamVertex(linePositions, lineColorAxis, lineIntensities, start.points[j], start.colorAxis[j], start.intensity[j]);
        addBeamVertex(linePositions, lineColorAxis, lineIntensities, end.points[j], start.colorAxis[j], start.intensity[j]);
      }
    }
  });
  const allColorAxis = colorAxis.concat(lineColorAxis);
  const allIntensities = intensities.concat(lineIntensities);
  const globalColorMinMax = finiteMinMax(allColorAxis);
  const globalIMax = finiteMax(allIntensities, 1e-12);
  uploadBeam(
    buffers.beams, positions, colorAxis, intensities, gl.POINTS,
    Number(raySize.value), Number(pointOpacityInput.value), true,
    globalColorMinMax, globalIMax);
  uploadBeam(
    buffers.beamLines, linePositions, lineColorAxis, lineIntensities, gl.LINES,
    1, Number(lineOpacityInput.value), false, globalColorMinMax, globalIMax);
}

function addBeamVertex(positions, colorAxis, intensities, point, colorValue, intensityValue) {
  const scaled = scaledPoint(point);
  positions.push(scaled[0], scaled[1], scaled[2]);
  colorAxis.push(Number.isFinite(colorValue) ? colorValue : 0);
  intensities.push(Number.isFinite(intensityValue) ? Math.max(intensityValue, 0) : 1);
}

function isHiddenFootprintBeam(beam, group) {
  if (!group || beam.beamName !== "beamGlobal") return false;
  return group.beams.has("beamLocal") || group.beams.has("beamLocal1");
}

function renderElementList() {
  elementList.textContent = "";
  if (!scenePayload?.elements.length) {
    elementList.textContent = "No elements";
    return;
  }
  scenePayload.elements.forEach((el) => {
    const row = document.createElement("button");
    row.className = "row";
    row.type = "button";
    const color = rgbCss(colors[el.kind] || colors.element);
    row.innerHTML = `
      <span class="swatch" style="background:${color}"></span>
      <span class="row-main">
        <span class="row-name">${escapeHtml(el.name)}</span>
        <span class="row-meta">${escapeHtml(el.kind)}</span>
      </span>`;
    row.addEventListener("click", () => {
      targetModel = vec3From(el.position);
      target = scaledPoint(targetModel);
      draw();
    });
    elementList.appendChild(row);
  });
}

function renderBeamList() {
  beamSelect.textContent = "";
  beamList.textContent = "";
  const allOption = document.createElement("option");
  allOption.value = "";
  allOption.textContent = "All beams";
  beamSelect.appendChild(allOption);
  for (const [key, beam] of beams.entries()) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = key;
    beamSelect.appendChild(option);

    const row = document.createElement("button");
    row.className = "row";
    row.type = "button";
    row.innerHTML = `
      <span class="swatch" style="background:${rgbCss(beam.color)}"></span>
      <span class="row-main">
        <span class="row-name">${escapeHtml(key)}</span>
        <span class="row-meta">${beam.count} rays</span>
      </span>`;
    row.addEventListener("click", () => {
      selectedBeam = key;
      beamSelect.value = key;
      buildBeamGeometry();
      draw();
    });
    beamList.appendChild(row);
  }
  selectedBeam = "";
  beamSelect.value = "";
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

function rgbCss(color) {
  return `rgb(${Math.round(color[0] * 255)},${Math.round(color[1] * 255)},${Math.round(color[2] * 255)})`;
}

function beamColor(index) {
  const hue = ((index * 137.5) % 360) / 360;
  return hslToRgb(hue, 0.72, 0.58);
}

function hslToRgb(h, s, l) {
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [hue2rgb(p, q, h + 1 / 3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1 / 3)];
}

function renderBeams(messages) {
  beams = new Map();
  beamSequence = [];
  beamGroups = new Map();
  beamLinks = [];
  let index = 0;
  messages.forEach((message) => {
    if (!message.beam) return;
    const elementId = String(message.sender_id || message.sender_name || "");
    const elementName = String(message.sender_name || message.sender_id || "");
    let group = beamGroups.get(elementId);
    if (!group) {
      group = { id: elementId, name: elementName, beams: new Map() };
      beamGroups.set(elementId, group);
    }
    Object.entries(message.beam).forEach(([beamName, beam]) => {
      if (!beam.positions?.length) return;
      const key = `${elementName}:${beamName}`;
      const payload = {
        key,
        elementId,
        elementName,
        beamName,
        color: beamColor(index),
        count: beam.count,
        indices: beam.indices || [],
        energy: beam.energy || [],
        intensity: beam.intensity || [],
        sourceColorAxis: beam.colorAxis || [],
        points: beam.positions.map(vec3From),
      };
      payload.colorAxis = makeBeamColorAxis(payload);
      beams.set(key, payload);
      beamSequence.push(payload);
      group.beams.set(beamName, payload);
      index += 1;
    });
  });
  beamLinks = buildBeamLinks();
  updateSceneMetrics({ preserveTarget: true });
  buildStaticGeometry();
  renderBeamList();
  buildBeamGeometry();
  draw();
}

function buildBeamLinks() {
  const links = [];
  const internalLinks = new Set();
  Object.entries(scenePayload?.flow || {}).forEach(([elementId, operations]) => {
    Object.values(operations || {}).forEach((params) => {
      const sourceId = params?.beam;
      if (!sourceId) return;
      const startGroup = beamGroups.get(String(sourceId));
      const endGroup = beamGroups.get(String(elementId));
      const start = pickBeam(startGroup, ["beamLocal", "beamLocal2", "beamGlobal"]);
      const end = pickBeam(endGroup, ["beamLocal", "beamLocal1", "beamGlobal"]);
      if (start && end) {
        links.push({ start, end });
      }
      addInternalBeamLink(links, internalLinks, startGroup);
    });
  });
  beamGroups.forEach((group) => addInternalBeamLink(links, internalLinks, group));
  return links;
}

function pickBeam(group, names) {
  if (!group) return null;
  for (const name of names) {
    const beam = group.beams.get(name);
    if (beam) return beam;
  }
  return null;
}

function addInternalBeamLink(links, seen, group) {
  if (!group) return;
  const start = group.beams.get("beamLocal1");
  const end = group.beams.get("beamLocal2");
  if (!start || !end || seen.has(group.id)) return;
  links.push({ start, end });
  seen.add(group.id);
}

function makeBeamColorAxis(beam) {
  if (beam.sourceColorAxis?.length === beam.points.length) {
    return beam.sourceColorAxis.map((value) => Number(value) || 0);
  }
  const energy = beam.energy || [];
  const intensity = beam.intensity || [];
  const source = hasUsefulRange(energy) ? energy : intensity;
  if (source.length === beam.points.length) {
    return source.map((value) => Number(value) || 0);
  }
  return beam.points.map((_, index) => index);
}

function hasUsefulRange(values) {
  if (!values?.length) return false;
  const [min, max] = finiteMinMax(values);
  return max > min;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

async function runPropagation() {
  runButton.disabled = true;
  setStatus("Running");
  try {
    const payload = await fetchJson(`/api/run?maxRays=${rayBudget.value}`, {
      method: "POST",
    });
    if (payload.scene) {
      renderScenePayload(payload.scene, { clearBeams: false, refit: false });
    }
    renderBeams(payload.messages);
    const total = [...beams.values()].reduce((sum, beam) => sum + beam.count, 0);
    setStatus(`${total} rays`);
  } catch (error) {
    console.error(error);
    setStatus(`Error: ${String(error.message || error).slice(0, 80)}`);
  } finally {
    runButton.disabled = false;
  }
}

function resize() {
  const rect = viewport.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  gl.viewport(0, 0, canvas.width, canvas.height);
  draw();
}

function cameraPosition() {
  const cp = Math.cos(pitch);
  return [
    target[0] + distance * cp * Math.sin(yaw),
    target[1] + distance * cp * Math.cos(yaw),
    target[2] + distance * Math.sin(pitch),
  ];
}

function currentMatrix() {
  const aspect = Math.max(canvas.width, 1) / Math.max(canvas.height, 1);
  const projection = perspective(Math.PI / 4, aspect, sceneRadius / 1000, sceneRadius * 20 + 10);
  const view = lookAt(cameraPosition(), target, [0, 0, 1]);
  return multiply(projection, view);
}

function draw() {
  if (!gl) return;
  gl.clearColor(0.067, 0.075, 0.09, 1);
  gl.clearDepth(1);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program);

  const matrix = currentMatrix();
  gl.disable(gl.BLEND);
  gl.depthMask(true);
  gl.useProgram(program);
  drawBuffer(buffers.grid, matrix);
  drawBuffer(buffers.axes, matrix);
  drawBuffer(buffers.links, matrix);
  if (showMeshes) {
    drawBuffer(buffers.surfaces, matrix);
    drawBuffer(buffers.edges, matrix);
  }
  gl.enable(gl.BLEND);
  gl.blendEquation(gl.FUNC_ADD);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.disable(gl.DEPTH_TEST);
  gl.depthMask(false);
  drawBeamBuffer(buffers.beamLines, matrix);
  drawBeamBuffer(buffers.beams, matrix);
  gl.depthMask(true);
  gl.enable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);
  updateLabels(matrix);
}

function updateLabels(matrix) {
  labelsLayer.textContent = "";
  labelNodes = [];
  if (scenePayload) {
    scenePayload.elements.forEach((el) => {
      addScreenLabel(matrix, vec3From(el.position), el.name, "scene-label");
    });
  }
  coordTicks.forEach((tick) => {
    const cls = `coord-label coord-${tick.axis}${tick.major ? " major" : ""}`;
    addScreenLabel(matrix, tick.position, tick.text, cls);
  });
  updateCoordReadout();
}

function addScreenLabel(matrix, point, text, className) {
  const clip = transformPoint(matrix, scaledPoint(point));
  if (clip[3] <= 0) return;
  const ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
  if (Math.abs(ndc[0]) > 1.2 || Math.abs(ndc[1]) > 1.2) return;
  const label = document.createElement("div");
  label.className = className;
  label.textContent = text;
  label.style.left = `${(ndc[0] * 0.5 + 0.5) * 100}%`;
  label.style.top = `${(-ndc[1] * 0.5 + 0.5) * 100}%`;
  labelsLayer.appendChild(label);
  labelNodes.push(label);
}

function updateCoordReadout() {
  coordReadout.innerHTML = `
    <span class="coord-x">X ${formatCoord(targetModel[0])} mm</span>
    <span class="coord-y">Y ${formatCoord(targetModel[1])} mm</span>
    <span class="coord-z">Z ${formatCoord(targetModel[2])} mm</span>
    <span>S ${formatScale(sceneScale)}</span>`;
}

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const factor = event.deltaY < 0 ? 0.88 : 1 / 0.88;
  distance = Math.max(sceneRadius * 0.02, distance * factor);
  draw();
}, { passive: false });

canvas.addEventListener("pointerdown", (event) => {
  dragging = event.shiftKey || event.button === 1 ? "pan" : "orbit";
  lastPointer = { x: event.clientX, y: event.clientY };
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  const dx = event.clientX - lastPointer.x;
  const dy = event.clientY - lastPointer.y;
  lastPointer = { x: event.clientX, y: event.clientY };
  if (dragging === "pan") {
    panCamera(dx, dy);
  } else {
    yaw += dx * 0.006;
    pitch = clamp(pitch + dy * 0.006, -1.45, 1.45);
  }
  draw();
});

canvas.addEventListener("pointerup", (event) => {
  dragging = null;
  canvas.releasePointerCapture(event.pointerId);
});

function panCamera(dx, dy) {
  const eye = cameraPosition();
  const forward = normalize(sub(target, eye));
  const right = normalize(cross(forward, [0, 0, 1]));
  const up = normalize(cross(right, forward));
  const amount = distance * 0.0015;
  target = add(target, add(scaleVec(right, -dx * amount), scaleVec(up, dy * amount)));
  targetModel = unscaledPoint(target);
}

raySize.addEventListener("input", () => {
  buildBeamGeometry();
  draw();
});
for (const input of [pointOpacityInput, lineOpacityInput]) {
  input.addEventListener("input", () => {
    buildBeamGeometry();
    draw();
  });
}
for (const input of [scaleXInput, scaleYInput, scaleZInput]) {
  input.addEventListener("input", () => updateSceneScale({ refit: false }));
}
showMeshesInput.addEventListener("change", () => {
  showMeshes = showMeshesInput.checked;
  draw();
});
beamSelect.addEventListener("change", () => {
  selectedBeam = beamSelect.value;
  buildBeamGeometry();
  draw();
});
runButton.addEventListener("click", runPropagation);
window.addEventListener("resize", resize);

async function init() {
  resize();
  setStatus("Loading");
  try {
    const payload = await fetchJson("/api/scene");
    renderScenePayload(payload);
    setStatus(`${payload.elements.length} elements`);
  } catch (error) {
    console.error(error);
    setStatus(`Error: ${String(error.message || error).slice(0, 80)}`);
  }
}

function niceGridStep(raw) {
  const power = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9))));
  const scaled = raw / power;
  if (scaled <= 2) return 2 * power;
  if (scaled <= 5) return 5 * power;
  return 10 * power;
}

function formatCoord(value) {
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  return value.toPrecision(2);
}

function formatScale(scale) {
  return scale.map((value) => {
    const exponent = Math.log10(value);
    return Number.isInteger(exponent) ? `1e${exponent}` : value.toPrecision(2);
  }).join(" ");
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scaleVec(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function length(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize(v) {
  const len = length(v) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, (2 * far * near) * nf, 0,
  ];
}

function lookAt(eye, center, up) {
  const z = normalize(sub(eye, center));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ];
}

function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      for (let i = 0; i < 4; i += 1) {
        out[col * 4 + row] += a[i * 4 + row] * b[col * 4 + i];
      }
    }
  }
  return out;
}

function transformPoint(m, p) {
  const x = p[0], y = p[1], z = p[2];
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
    m[3] * x + m[7] * y + m[11] * z + m[15],
  ];
}

init();

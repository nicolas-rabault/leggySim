import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import load_mujoco from "mujoco-js";

import {
  downloadSceneFiles,
  loadScene,
  createThreeScene,
  updateThreeBodies,
  createGround,
  getPosition,
} from "./mujocoUtils.js";
import { PolicyController } from "./policyController.js";
import { DragStateManager } from "./utils/DragStateManager.js";

const canvas = document.getElementById("viewer");
const hud = document.getElementById("hud");
const loadingOverlay = document.getElementById("loading-overlay");

// -- Three.js --
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f172a);
scene.fog = new THREE.Fog(0x0f172a, 8, 25);

const camera = new THREE.PerspectiveCamera(
  50,
  window.innerWidth / window.innerHeight,
  0.01,
  100,
);
camera.position.set(0.5, 0.4, 0.6);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.15, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.maxPolarAngle = Math.PI / 2 - 0.05;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
scene.add(new THREE.HemisphereLight(0xb1e1ff, 0x334455, 0.4));

const spot = new THREE.SpotLight(0xffffff, 2.0, 20, Math.PI / 4, 0.5, 1);
spot.position.set(2, 4, 2);
spot.castShadow = true;
spot.shadow.mapSize.set(2048, 2048);
spot.shadow.camera.near = 0.5;
spot.shadow.camera.far = 15;
scene.add(spot);
scene.add(spot.target);

scene.add(createGround());

// -- State --
let mujoco, model, data;
let bodies = {};
let bodyNames = [];
let paused = false;
let running = false;

const policy = new PolicyController();
let drag;
let followBodyIdx = 0;

// -- Init --
async function init() {
  try {
    setLoadingStatus("Loading MuJoCo WASM...");
    mujoco = await load_mujoco();

    setLoadingStatus("Downloading scene files...");
    await downloadSceneFiles(mujoco);

    setLoadingStatus("Loading model...");
    const loaded = loadScene(mujoco, "scene.xml");
    model = loaded.model;
    data = loaded.data;

    mujoco.mj_forward(model, data);

    setLoadingStatus("Building 3D scene...");
    const threeScene = createThreeScene(model, data);
    bodies = threeScene.bodies;
    bodyNames = threeScene.bodyNames;

    for (const name of bodyNames) {
      scene.add(bodies[name]);
    }

    followBodyIdx = bodyNames.indexOf("boddy");
    if (followBodyIdx < 0) followBodyIdx = 1;

    setLoadingStatus("Loading neural network...");
    policy.setMujocoRefs(model, data);

    try {
      await policy.loadModel("./policy.onnx");
      console.log("Policy loaded successfully");
    } catch (e) {
      console.warn(
        "Could not load policy.onnx, running without policy:",
        e.message,
      );
    }

    resetSimulation();

    drag = new DragStateManager(scene, renderer, camera, canvas, controls);
    drag.setMujocoData(model, data, bodies, bodyNames);

    if (loadingOverlay) loadingOverlay.style.display = "none";

    // Start render loop (visual only)
    requestAnimationFrame(render);

    // Start simulation loop (physics + policy, async)
    running = true;
    simulationLoop();
  } catch (e) {
    console.error("Initialization failed:", e);
    setLoadingStatus("Error: " + e.message);
  }
}

function setLoadingStatus(msg) {
  console.log(msg);
  const el = document.getElementById("loading-status");
  if (el) el.textContent = msg;
}

function resetSimulation() {
  if (model.nkey > 0) {
    mujoco.mj_resetDataKeyframe(model, data, 0);
  } else {
    mujoco.mj_resetData(model, data);
    data.qpos[2] = 0.189;
  }

  // Set ctrl to default joint positions (knee-space, matching training).
  const defaults = policy.defaultJointPos;
  if (defaults.length === 6 && policy.actuatorIndices.length === 6) {
    for (let i = 0; i < 6; i++) {
      data.ctrl[policy.actuatorIndices[i]] = defaults[i];
    }
  }
  mujoco.mj_forward(model, data);
  policy.reset();
  updateHUD();
}

// -- Simulation loop (async, paced to match play mode: 60 steps/sec) --
const FRAME_MS = 1000 / 60;

async function simulationLoop() {
  while (running) {
    const frameStart = performance.now();

    if (!paused && model) {
      // 1. Run policy inference
      if (policy.session) {
        const actions = await policy.infer();
        if (actions) policy.applyActions(actions);
      }

      // 2. Step physics for `decimation` steps
      for (let i = 0; i < policy.decimation; i++) {
        if (drag) drag.applyPerturbation(data);
        mujoco.mj_step(model, data);
        if (drag) drag.clearPerturbation(data);
      }
    }

    // 3. Pace to 60 steps/sec (matching play mode viewer)
    const elapsed = performance.now() - frameStart;
    if (elapsed < FRAME_MS) await sleep(FRAME_MS - elapsed);
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// -- Render loop (visual only, 60fps) --
function render() {
  requestAnimationFrame(render);

  if (model) {
    updateThreeBodies(model, data, bodies, bodyNames);
  }

  if (followBodyIdx >= 0 && model) {
    const robotPos = getPosition(data.xpos, followBodyIdx);
    const delta = robotPos.clone().sub(controls.target);
    delta.multiplyScalar(0.05);
    controls.target.add(delta);
    camera.position.add(delta);
    spot.position.add(delta);
    spot.target.position.add(delta);
  }

  controls.update();
  renderer.render(scene, camera);
  updateHUD();
}

function updateHUD() {
  if (hud) hud.textContent = policy.commandString;
}

// -- Keyboard --
window.addEventListener("keydown", (e) => {
  if (e.key === "r" || e.key === "R") {
    resetSimulation();
    e.preventDefault();
  }
  if (e.key === "p" || e.key === "P") {
    paused = !paused;
    e.preventDefault();
  }
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

init();

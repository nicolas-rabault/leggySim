import * as ort from "onnxruntime-web";

ort.env.wasm.numThreads = 1;

// Observation term order matching the training environment.
// Determined by Python dict insertion order: initial velocity_env terms first,
// then appended terms from configure_leggy_observations + configure_jump.
const OBS_TERMS = [
  { name: "base_lin_vel", size: 3 },
  { name: "base_ang_vel", size: 3 },
  { name: "joint_pos", size: 6 },
  { name: "joint_vel", size: 6 },
  { name: "actions", size: 6 },
  { name: "command", size: 3 },
  { name: "body_euler", size: 3 },
  { name: "joint_torques", size: 6 },
];

export class PolicyController {
  constructor() {
    this.session = null;

    // Defaults (overridden by ONNX metadata if available)
    this.jointNames = ["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"];
    this.actionScale = 0.5;
    this.defaultJointPos = [0.2268928028, -0.6108652382, -0.05235987756, 0.2268928028, -0.6108652382, -0.05235987756];
    this.historyLength = 5;
    this.obsSize = 36;
    this.decimation = 2;

    // Per-term history buffers (term-major flattening)
    this.termHistories = {};
    this._initTermHistories();

    this.prevActions = new Float32Array(6);

    // Keyboard state
    this.linVelX = 0;
    this.angVelZ = 0;
    this.jump = false;
    this._jumpTimer = null;

    this.linVelXRange = [-2.0, 3.0];
    this.angVelZRange = [-3.0, 3.0];
    this.linVelXStep = 0.1;
    this.angVelZStep = 0.2;

    // MuJoCo references
    this.model = null;
    this.data = null;
    this.jointQposIndices = [];
    this.jointQvelIndices = [];
    this.actuatorIndices = [];

    this._keyDown = this._keyDown.bind(this);
    window.addEventListener("keydown", this._keyDown);
  }

  _initTermHistories() {
    this.termHistories = {};
    for (const term of OBS_TERMS) {
      this.termHistories[term.name] = [];
      for (let t = 0; t < this.historyLength; t++) {
        this.termHistories[term.name].push(new Float32Array(term.size));
      }
    }
  }

  async loadModel(url) {
    this.session = await ort.InferenceSession.create(url);

    let meta = {};
    const handler = this.session.handler;
    if (handler?.metadata?.customMetadataMap) {
      meta = handler.metadata.customMetadataMap;
    } else if (handler?.metadata) {
      meta = handler.metadata;
    }
    if (Object.keys(meta).length === 0 && this.session.metadata) {
      meta = this.session.metadata;
    }

    if (meta.action_scale) this.actionScale = parseFloat(meta.action_scale);
    if (meta.history_length) this.historyLength = parseInt(meta.history_length);
    if (meta.obs_size) this.obsSize = parseInt(meta.obs_size);
    if (meta.decimation) this.decimation = parseInt(meta.decimation);

    this._initTermHistories();
    this.prevActions = new Float32Array(6);
  }

  setMujocoRefs(model, data) {
    this.model = model;
    this.data = data;
    this._resolveJointIndices();
    this._resolveBodyIndex();
  }

  _resolveJointIndices() {
    this.jointQposIndices = [];
    this.jointQvelIndices = [];
    this.actuatorIndices = [];

    for (const name of this.jointNames) {
      let jntIdx = -1;
      let actIdx = -1;
      for (let i = 0; i < this.model.njnt; i++) {
        if (this._getMujocoName(this.model.name_jntadr, i) === name) { jntIdx = i; break; }
      }
      for (let i = 0; i < this.model.nu; i++) {
        if (this._getMujocoName(this.model.name_actuatoradr, i) === name) { actIdx = i; break; }
      }
      if (jntIdx >= 0) {
        this.jointQposIndices.push(this.model.jnt_qposadr[jntIdx]);
        this.jointQvelIndices.push(this.model.jnt_dofadr[jntIdx]);
      }
      if (actIdx >= 0) {
        this.actuatorIndices.push(actIdx);
      }
    }
  }

  _resolveBodyIndex() {
    this.rootBodyId = -1;
    for (let i = 0; i < this.model.nbody; i++) {
      if (this._getMujocoName(this.model.name_bodyadr, i) === "boddy") {
        this.rootBodyId = i;
        break;
      }
    }
  }

  _getMujocoName(nameAdrArray, index) {
    const adr = nameAdrArray[index];
    let name = "";
    for (let i = adr; i < this.model.names.length; i++) {
      const c = this.model.names[i];
      if (c === 0) break;
      name += String.fromCharCode(c);
    }
    return name;
  }

  _keyDown(e) {
    switch (e.key) {
      case "ArrowUp":
        this.linVelX = Math.min(this.linVelX + this.linVelXStep, this.linVelXRange[1]);
        e.preventDefault(); break;
      case "ArrowDown":
        this.linVelX = Math.max(this.linVelX - this.linVelXStep, this.linVelXRange[0]);
        e.preventDefault(); break;
      case "ArrowLeft":
        this.angVelZ = Math.min(this.angVelZ + this.angVelZStep, this.angVelZRange[1]);
        e.preventDefault(); break;
      case "ArrowRight":
        this.angVelZ = Math.max(this.angVelZ - this.angVelZStep, this.angVelZRange[0]);
        e.preventDefault(); break;
      case " ":
        if (this._jumpTimer) clearTimeout(this._jumpTimer);
        this.jump = true;
        this._jumpTimer = setTimeout(() => { this.jump = false; }, 700);
        e.preventDefault(); break;
    }
  }

  // Build per-term observation values (correct order matching training)
  buildObservationTerms() {
    const terms = {};

    // base_lin_vel (3) and base_ang_vel (3) — from cvel (matching training)
    // Training reads from data.cvel (spatial velocity at subtree COM) and corrects
    // to body origin, rather than reading qvel directly.
    const b = this.rootBodyId;
    const cv = this.data.cvel;
    const angW = [cv[b * 6], cv[b * 6 + 1], cv[b * 6 + 2]];
    const linC = [cv[b * 6 + 3], cv[b * 6 + 4], cv[b * 6 + 5]];
    const px = this.data.xpos[b * 3], py = this.data.xpos[b * 3 + 1], pz = this.data.xpos[b * 3 + 2];
    const cx = this.data.subtree_com[b * 3], cy = this.data.subtree_com[b * 3 + 1], cz = this.data.subtree_com[b * 3 + 2];
    const ox = cx - px, oy = cy - py, oz = cz - pz;
    const linW = [
      linC[0] - (angW[1] * oz - angW[2] * oy),
      linC[1] - (angW[2] * ox - angW[0] * oz),
      linC[2] - (angW[0] * oy - angW[1] * ox),
    ];
    terms.base_lin_vel = new Float32Array(this._rotateToBody(linW));
    terms.base_ang_vel = new Float32Array(this._rotateToBody(angW));

    // joint_pos (6) — motor space: knee→motor = qpos[knee] + qpos[hipX]
    const jp = new Float32Array(6);
    for (let i = 0; i < 6; i++) {
      let pos = this.data.qpos[this.jointQposIndices[i]];
      if (i === 2) pos = pos + this.data.qpos[this.jointQposIndices[1]]; // Lknee + LhipX
      if (i === 5) pos = pos + this.data.qpos[this.jointQposIndices[4]]; // Rknee + RhipX
      jp[i] = pos;
    }
    terms.joint_pos = jp;

    // joint_vel (6) — motor space: knee_vel + hipX_vel
    const jv = new Float32Array(6);
    for (let i = 0; i < 6; i++) {
      let vel = this.data.qvel[this.jointQvelIndices[i]];
      if (i === 2) vel = vel + this.data.qvel[this.jointQvelIndices[1]];
      if (i === 5) vel = vel + this.data.qvel[this.jointQvelIndices[4]];
      jv[i] = vel;
    }
    terms.joint_vel = jv;

    // actions (6) — previous raw policy actions
    terms.actions = new Float32Array(this.prevActions);

    // command (3) — [lin_vel_x, 0, ang_vel_z]
    terms.command = new Float32Array([this.linVelX, 0, this.angVelZ]);

    // body_euler (3) — from root quaternion
    terms.body_euler = new Float32Array(this._quatToEulerXYZ());

    // joint_torques (6) — actuator forces
    const jt = new Float32Array(6);
    for (let i = 0; i < 6; i++) {
      jt[i] = this.data.actuator_force[this.actuatorIndices[i]];
    }
    terms.joint_torques = jt;

    return terms;
  }

  _quatToEulerXYZ() {
    const w = this.data.qpos[3];
    const x = this.data.qpos[4];
    const y = this.data.qpos[5];
    const z = this.data.qpos[6];

    const sinr_cosp = 2 * (w * x + y * z);
    const cosr_cosp = 1 - 2 * (x * x + y * y);
    const roll = Math.atan2(sinr_cosp, cosr_cosp);

    const sinp = 2 * (w * y - z * x);
    const pitch = Math.abs(sinp) >= 1
      ? Math.sign(sinp) * Math.PI / 2
      : Math.asin(sinp);

    const siny_cosp = 2 * (w * z + x * y);
    const cosy_cosp = 1 - 2 * (y * y + z * z);
    const yaw = Math.atan2(siny_cosp, cosy_cosp);

    return [roll, pitch, yaw];
  }

  _rotateToBody(vecWorld) {
    const w = this.data.qpos[3];
    const x = this.data.qpos[4];
    const y = this.data.qpos[5];
    const z = this.data.qpos[6];

    // R^T * v (world → body frame)
    const r00 = 1 - 2 * (y * y + z * z);
    const r01 = 2 * (x * y + w * z);
    const r02 = 2 * (x * z - w * y);
    const r10 = 2 * (x * y - w * z);
    const r11 = 1 - 2 * (x * x + z * z);
    const r12 = 2 * (y * z + w * x);
    const r20 = 2 * (x * z + w * y);
    const r21 = 2 * (y * z - w * x);
    const r22 = 1 - 2 * (x * x + y * y);

    return [
      r00 * vecWorld[0] + r01 * vecWorld[1] + r02 * vecWorld[2],
      r10 * vecWorld[0] + r11 * vecWorld[1] + r12 * vecWorld[2],
      r20 * vecWorld[0] + r21 * vecWorld[1] + r22 * vecWorld[2],
    ];
  }

  pushObservation(terms) {
    for (const term of OBS_TERMS) {
      this.termHistories[term.name].shift();
      this.termHistories[term.name].push(terms[term.name]);
    }
  }

  // Flatten history in TERM-MAJOR order (matching training observation manager)
  // Layout: [term0_t0, term0_t1, ..., term0_t4, term1_t0, ..., termN_t4]
  flattenHistory() {
    const inputSize = this.obsSize * this.historyLength;
    const flat = new Float32Array(inputSize);
    let offset = 0;
    for (const term of OBS_TERMS) {
      for (let t = 0; t < this.historyLength; t++) {
        flat.set(this.termHistories[term.name][t], offset);
        offset += term.size;
      }
    }
    return flat;
  }

  async infer() {
    if (!this.session || this._inferring) return null;
    this._inferring = true;

    try {
      const terms = this.buildObservationTerms();
      this.pushObservation(terms);

      const flat = this.flattenHistory();
      const inputSize = this.obsSize * this.historyLength;

      const tensor = new ort.Tensor("float32", flat, [1, inputSize]);
      const results = await this.session.run({ obs: tensor });
      const raw = results.actions.data;
      const actions = new Float32Array(6);
      for (let i = 0; i < 6; i++) {
        actions[i] = raw[i];
        this.prevActions[i] = raw[i];
      }

      return actions;
    } finally {
      this._inferring = false;
    }
  }

  applyActions(actions) {
    if (!actions || !this.data) return;

    // Training uses JointPositionAction (knee-space defaults, no motor-to-knee conversion).
    // target = default_joint_pos + action_scale * raw_action
    const target = new Float32Array(6);
    for (let i = 0; i < 6; i++) {
      target[i] = this.defaultJointPos[i] + this.actionScale * actions[i];
    }

    if (target.some(v => !isFinite(v))) return;

    for (let i = 0; i < 6; i++) {
      this.data.ctrl[this.actuatorIndices[i]] = target[i];
    }
  }

  reset() {
    this.linVelX = 0;
    this.angVelZ = 0;
    this.jump = false;
    if (this._jumpTimer) clearTimeout(this._jumpTimer);
    this._jumpTimer = null;
    this.prevActions = new Float32Array(6);
    this._initTermHistories();

    // Backfill history with current observation (matching training behavior:
    // on reset, the circular buffer is filled with copies of the initial obs,
    // not zeros — so the policy never sees garbage in history slots).
    if (this.model && this.data) {
      const terms = this.buildObservationTerms();
      for (const term of OBS_TERMS) {
        for (let t = 0; t < this.historyLength; t++) {
          this.termHistories[term.name][t] = new Float32Array(terms[term.name]);
        }
      }
    }
  }

  get commandString() {
    const jmp = this.jump ? " | JUMP" : "";
    return `vel: ${this.linVelX.toFixed(1)} m/s | yaw: ${this.angVelZ.toFixed(1)} rad/s${jmp}`;
  }
}

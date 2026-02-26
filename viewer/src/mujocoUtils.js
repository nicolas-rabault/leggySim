import * as THREE from "three";
import { Reflector } from "three/addons/objects/Reflector.js";

const SCENE_FILES = ["scene.xml", "robot.xml"];
const SCENES_BASE = "./scenes/";

// MuJoCo → Three.js coordinate swap: (mj_x, mj_y, mj_z) → (mj_x, mj_z, -mj_y)
function mjToThreePos(arr, idx) {
  const o = idx * 3;
  return new THREE.Vector3(arr[o], arr[o + 2], -arr[o + 1]);
}

function mjToThreeQuat(arr, idx) {
  // MuJoCo quat: [w, x, y, z] → Three.js Quaternion(x, y, z, w) with axis swap
  const o = idx * 4;
  return new THREE.Quaternion(arr[o + 1], arr[o + 3], -arr[o + 2], arr[o]);
}

export { mjToThreePos as getPosition };

export function toMujocoPos(threePos) {
  return [threePos.x, -threePos.z, threePos.y];
}

function getMujocoName(model, nameAdrArray, index) {
  const adr = nameAdrArray[index];
  let name = "";
  for (let i = adr; i < model.names.length; i++) {
    const c = model.names[i];
    if (c === 0) break;
    name += String.fromCharCode(c);
  }
  return name;
}

export async function downloadSceneFiles(mujoco) {
  mujoco.FS.mkdir("/working");
  mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");
  mujoco.FS.mkdir("/working/assets");

  for (const file of SCENE_FILES) {
    const resp = await fetch(SCENES_BASE + file);
    const text = await resp.text();
    mujoco.FS.writeFile(`/working/${file}`, text);
  }

  // Parse mesh references from robot.xml to download STLs
  const robotResp = await fetch(SCENES_BASE + "robot.xml");
  const robotText = await robotResp.text();
  const meshPattern = /mesh="([^"]+)"/g;
  const meshNames = new Set();
  let match;
  while ((match = meshPattern.exec(robotText)) !== null) {
    meshNames.add(match[1]);
  }

  const fetches = [];
  for (const name of meshNames) {
    const stlFile = `assets/${name}.stl`;
    fetches.push(
      fetch(SCENES_BASE + stlFile)
        .then((r) => r.arrayBuffer())
        .then((buf) => {
          mujoco.FS.writeFile(`/working/${stlFile}`, new Uint8Array(buf));
        })
        .catch(() => console.warn(`Could not fetch ${stlFile}`))
    );
  }
  await Promise.all(fetches);
}

export function loadScene(mujoco, filename) {
  const model = mujoco.MjModel.loadFromXML(`/working/${filename}`);
  const data = new mujoco.MjData(model);
  return { model, data };
}

export function createThreeScene(model, data) {
  const bodies = {};
  const bodyNames = [];
  const meshGeometries = buildMeshGeometries(model);

  for (let b = 0; b < model.nbody; b++) {
    const bodyName = getMujocoName(model, model.name_bodyadr, b) || `body_${b}`;
    bodyNames.push(bodyName);
    const group = new THREE.Group();
    group.name = bodyName;
    bodies[bodyName] = group;
  }

  // Create geom meshes with LOCAL transforms (relative to parent body)
  for (let g = 0; g < model.ngeom; g++) {
    const bodyId = model.geom_bodyid[g];
    const bodyName = bodyNames[bodyId];
    const geomGroup = model.geom_group[g];
    if (geomGroup === 3) continue; // skip collision-only

    const mesh = createGeomMesh(model, g, meshGeometries);
    if (!mesh) continue;

    // Set LOCAL transform from model (geom pose relative to body)
    const localPos = mjToThreePos(model.geom_pos, g);
    const localQuat = mjToThreeQuat(model.geom_quat, g);
    mesh.position.copy(localPos);
    mesh.quaternion.copy(localQuat);

    bodies[bodyName].add(mesh);
  }

  return { bodies, bodyNames };
}

export function updateThreeBodies(model, data, bodies, bodyNames) {
  // Only update body WORLD transforms — geom local transforms are static
  for (let b = 0; b < model.nbody; b++) {
    const body = bodies[bodyNames[b]];
    if (!body) continue;
    body.position.copy(mjToThreePos(data.xpos, b));
    body.quaternion.copy(mjToThreeQuat(data.xquat, b));
  }
}

function buildMeshGeometries(model) {
  const geometries = {};

  for (let m = 0; m < model.nmesh; m++) {
    const meshName = getMujocoName(model, model.name_meshadr, m);
    const vertStart = model.mesh_vertadr[m];
    const vertCount = model.mesh_vertnum[m];
    const faceStart = model.mesh_faceadr[m];
    const faceCount = model.mesh_facenum[m];

    const positions = new Float32Array(vertCount * 3);
    for (let v = 0; v < vertCount; v++) {
      const si = (vertStart + v) * 3;
      positions[v * 3 + 0] = model.mesh_vert[si + 0];
      positions[v * 3 + 1] = model.mesh_vert[si + 2];
      positions[v * 3 + 2] = -model.mesh_vert[si + 1];
    }

    const indices = new Uint32Array(faceCount * 3);
    for (let f = 0; f < faceCount; f++) {
      const si = (faceStart + f) * 3;
      indices[f * 3 + 0] = model.mesh_face[si + 0];
      indices[f * 3 + 1] = model.mesh_face[si + 1];
      indices[f * 3 + 2] = model.mesh_face[si + 2];
    }

    let normals = null;
    if (model.mesh_normal && model.mesh_normal.length > 0) {
      normals = new Float32Array(vertCount * 3);
      for (let v = 0; v < vertCount; v++) {
        const si = (vertStart + v) * 3;
        normals[v * 3 + 0] = model.mesh_normal[si + 0];
        normals[v * 3 + 1] = model.mesh_normal[si + 2];
        normals[v * 3 + 2] = -model.mesh_normal[si + 1];
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (normals) {
      geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    } else {
      geo.computeVertexNormals();
    }
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    geometries[meshName] = geo;
  }

  return geometries;
}

function createGeomMesh(model, geomIdx, meshGeometries) {
  const type = model.geom_type[geomIdx];
  const size = [
    model.geom_size[geomIdx * 3 + 0],
    model.geom_size[geomIdx * 3 + 1],
    model.geom_size[geomIdx * 3 + 2],
  ];

  const matId = model.geom_matid[geomIdx];
  let color, opacity;
  if (matId >= 0) {
    color = new THREE.Color(
      model.mat_rgba[matId * 4 + 0],
      model.mat_rgba[matId * 4 + 1],
      model.mat_rgba[matId * 4 + 2]
    );
    opacity = model.mat_rgba[matId * 4 + 3];
  } else {
    color = new THREE.Color(
      model.geom_rgba[geomIdx * 4 + 0],
      model.geom_rgba[geomIdx * 4 + 1],
      model.geom_rgba[geomIdx * 4 + 2]
    );
    opacity = model.geom_rgba[geomIdx * 4 + 3];
  }

  const material = new THREE.MeshStandardMaterial({
    color,
    opacity,
    transparent: opacity < 1.0,
    roughness: 0.6,
    metalness: 0.2,
  });

  let geometry;

  switch (type) {
    case 0: // plane — skip, we use our own Three.js ground
      return null;
    case 2: // sphere
      geometry = new THREE.SphereGeometry(size[0], 16, 16);
      break;
    case 3: // capsule
      geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2, 8, 16);
      break;
    case 5: // cylinder
      geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 16);
      break;
    case 6: // box
      geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
      break;
    case 7: { // mesh
      const dataMeshIdx = model.geom_dataid[geomIdx];
      if (dataMeshIdx < 0) return null;
      const meshName = getMujocoName(model, model.name_meshadr, dataMeshIdx);
      geometry = meshGeometries[meshName];
      if (!geometry) return null;
      break;
    }
    default:
      return null;
  }

  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export function createGround() {
  const size = 200;
  const tileWorldSize = 0.5;
  const repeats = size / tileWorldSize;
  const color1 = new THREE.Color(0.2, 0.3, 0.4);
  const color2 = new THREE.Color(0.1, 0.2, 0.3);

  // Small 2x2 checkerboard tile at high resolution
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  const half = canvas.width / 2;

  const c1 = `rgb(${color1.r * 255},${color1.g * 255},${color1.b * 255})`;
  const c2 = `rgb(${color2.r * 255},${color2.g * 255},${color2.b * 255})`;
  ctx.fillStyle = c1; ctx.fillRect(0, 0, half, half);
  ctx.fillStyle = c2; ctx.fillRect(half, 0, half, half);
  ctx.fillStyle = c2; ctx.fillRect(0, half, half, half);
  ctx.fillStyle = c1; ctx.fillRect(half, half, half, half);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(repeats, repeats);
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.anisotropy = 16;

  const geo = new THREE.PlaneGeometry(size, size);

  const reflector = new Reflector(geo, {
    textureWidth: 1024,
    textureHeight: 1024,
    color: 0x445566,
  });
  reflector.rotation.x = -Math.PI / 2;
  reflector.position.y = -0.001;

  const surfaceGeo = new THREE.PlaneGeometry(size, size);
  const surfaceMat = new THREE.MeshStandardMaterial({
    map: texture,
    roughness: 0.8,
    metalness: 0.0,
    transparent: true,
    opacity: 0.85,
  });
  const surface = new THREE.Mesh(surfaceGeo, surfaceMat);
  surface.rotation.x = -Math.PI / 2;
  surface.receiveShadow = true;

  const group = new THREE.Group();
  group.add(reflector);
  group.add(surface);
  return group;
}

import * as THREE from "three";

export class DragStateManager {
  constructor(scene, renderer, camera, container, controls) {
    this.scene = scene;
    this.renderer = renderer;
    this.camera = camera;
    this.container = container;
    this.controls = controls;

    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.plane = new THREE.Plane();
    this.intersection = new THREE.Vector3();

    this.dragging = false;
    this.dragBody = -1;
    this.dragPoint = new THREE.Vector3();
    this.worldPos = new THREE.Vector3();

    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);

    container.addEventListener("pointerdown", this._onPointerDown);
    container.addEventListener("pointermove", this._onPointerMove);
    container.addEventListener("pointerup", this._onPointerUp);
  }

  setMujocoData(model, data, bodies, bodyNames) {
    this.model = model;
    this.data = data;
    this.bodies = bodies;
    this.bodyNames = bodyNames;
  }

  _updateMouse(event) {
    const rect = this.container.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  _onPointerDown(event) {
    if (event.button !== 0 || !this.bodies) return;
    this._updateMouse(event);
    this.raycaster.setFromCamera(this.mouse, this.camera);

    const meshes = [];
    for (const [name, mesh] of Object.entries(this.bodies)) {
      if (mesh.isMesh || mesh.isGroup) {
        mesh.traverse((child) => {
          if (child.isMesh) {
            child.userData.bodyName = name;
            meshes.push(child);
          }
        });
      }
    }

    const intersects = this.raycaster.intersectObjects(meshes, false);
    if (intersects.length > 0) {
      const hit = intersects[0];
      const bodyName = hit.object.userData.bodyName;
      const bodyIdx = this.bodyNames.indexOf(bodyName);
      if (bodyIdx >= 0) {
        this.dragging = true;
        this.dragBody = bodyIdx;
        this.dragPoint.copy(hit.point);
        this.worldPos.copy(hit.point);
        this.plane.setFromNormalAndCoplanarPoint(
          this.camera.getWorldDirection(new THREE.Vector3()).negate(),
          hit.point
        );
        if (this.controls) this.controls.enabled = false;
      }
    }
  }

  _onPointerMove(event) {
    if (!this.dragging) return;
    this._updateMouse(event);
    this.raycaster.setFromCamera(this.mouse, this.camera);
    if (this.raycaster.ray.intersectPlane(this.plane, this.intersection)) {
      this.worldPos.copy(this.intersection);
    }
  }

  _onPointerUp() {
    if (this.dragging) {
      this.dragging = false;
      this.dragBody = -1;
      if (this.controls) this.controls.enabled = true;
    }
  }

  applyPerturbation(data) {
    if (!this.dragging || this.dragBody < 0) return;
    const dx = this.worldPos.x - this.dragPoint.x;
    const dy = this.worldPos.z - this.dragPoint.z;
    const dz = this.worldPos.y - this.dragPoint.y;
    const force = 200;
    data.xfrc_applied[this.dragBody * 6 + 0] = dx * force;
    data.xfrc_applied[this.dragBody * 6 + 1] = dy * force;
    data.xfrc_applied[this.dragBody * 6 + 2] = dz * force;
    this.dragPoint.copy(this.worldPos);
  }

  clearPerturbation(data) {
    if (this.dragBody >= 0 && data) {
      data.xfrc_applied[this.dragBody * 6 + 0] = 0;
      data.xfrc_applied[this.dragBody * 6 + 1] = 0;
      data.xfrc_applied[this.dragBody * 6 + 2] = 0;
    }
  }
}

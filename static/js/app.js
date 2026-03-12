import * as THREE from "three";
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// ─── Three.js globals ─────────────────────────────────────────────────────────
let renderer, scene, camera;
const glassesGroups = {};
let threeW = 0, threeH = 0;

// ─── App state ────────────────────────────────────────────────────────────────
let faceLandmarker     = null;
let webcamStream       = null;
let animationFrameId   = null;
let products           = [];
let calibration        = {};
let selectedProductId  = null;
let latestMeasurements = null;

const SMOOTHING = 10;
const GLASSES_SCALE = 1.15; // increase to make glasses larger on face
const buf = { pd: [], bridge: [], faceWidth: [], pxPerMm: [] };

// ─── DOM ──────────────────────────────────────────────────────────────────────
const video     = document.getElementById("webcam");
const canvas    = document.getElementById("overlay");
const startBtn  = document.getElementById("startBtn");
const stopBtn   = document.getElementById("stopBtn");
const placeholder         = document.getElementById("cameraPlaceholder");
const pdValue             = document.getElementById("pdValue");
const bridgeValue         = document.getElementById("bridgeValue");
const faceWidthValue      = document.getElementById("faceWidthValue");
const productGrid         = document.getElementById("productGrid");
const measurementsPanel   = document.getElementById("measurementsPanel");
const recommendationPanel = document.getElementById("recommendationPanel");
const recommendationList  = document.getElementById("recommendationList");

// ─── Three.js Init ────────────────────────────────────────────────────────────
function initThree(w, h) {
    threeW = w; threeH = h;

    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(w, h, false);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x000000, 0);

    scene  = new THREE.Scene();
    camera = new THREE.OrthographicCamera(-w / 2, w / 2, h / 2, -h / 2, -500, 500);
    camera.position.z = 100;
}

// ─── Build glasses group from texture ────────────────────────────────────────
const textureLoader = new THREE.TextureLoader();

function addArmPlanes(group, cal, armTex, planeW, planeH, meshOffX, meshOffY) {
    const armW = cal.arm_w_mm;
    const armH = cal.arm_h_mm;
    const armGeo = new THREE.PlaneGeometry(armW, armH);

    const leftHingeX  = (cal.hinge_left_x_frac  - 0.5) * planeW + meshOffX;
    const rightHingeX = (cal.hinge_right_x_frac - 0.5) * planeW + meshOffX;
    const hingeY      = (0.5 - cal.hinge_y_frac) * planeH + meshOffY;

    // Left arm — flipped texture, hinge on right, extends left
    const leftTex = armTex.clone();
    leftTex.wrapS = THREE.RepeatWrapping;
    leftTex.repeat.set(-1, 1);
    leftTex.offset.set(1, 0);
    leftTex.needsUpdate = true;
    const leftMesh = new THREE.Mesh(armGeo,
        new THREE.MeshBasicMaterial({ map: leftTex, transparent: true, alphaTest: 0.01, depthWrite: false }));
    leftMesh.position.x = -armW / 2;
    const leftPivot = new THREE.Group();
    leftPivot.position.set(leftHingeX, hingeY, -1);
    leftPivot.name = "leftArm";
    leftPivot.add(leftMesh);
    group.add(leftPivot);

    // Right arm — normal texture, hinge on left, extends right
    const rightTex = armTex.clone();
    rightTex.needsUpdate = true;
    const rightMesh = new THREE.Mesh(armGeo,
        new THREE.MeshBasicMaterial({ map: rightTex, transparent: true, alphaTest: 0.01, depthWrite: false }));
    rightMesh.position.x = armW / 2;
    const rightPivot = new THREE.Group();
    rightPivot.position.set(rightHingeX, hingeY, -1);
    rightPivot.name = "rightArm";
    rightPivot.add(rightMesh);
    group.add(rightPivot);
}

function buildGlassesGroup(product, cal) {
    return new Promise((resolve, reject) => {
        const url = `/static/glasses/processed/${product.frame_image}`;
        textureLoader.load(url, (texture) => {
            texture.colorSpace = THREE.SRGBColorSpace;

            const pd_mm    = product.lens_width + product.bridge_width;
            const imgScale = pd_mm / cal.pd_px;
            const planeW   = cal.img_w * imgScale;
            const planeH   = cal.img_h * imgScale;

            const geometry = new THREE.PlaneGeometry(planeW, planeH);
            const material = new THREE.MeshBasicMaterial({
                map: texture, transparent: true, alphaTest: 0.01,
                depthWrite: false, side: THREE.FrontSide,
            });
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.x = -(cal.lens_center_x_frac - 0.5) * planeW;
            mesh.position.y =  (cal.lens_center_y_frac - 0.5) * planeH;

            const group = new THREE.Group();
            group.add(mesh);
            group.visible = false;

            if (cal.arm_w_mm) {
                const armUrl = `/static/glasses/processed/${product.cal_key}_arm_crop.png`;
                textureLoader.load(armUrl, (armTex) => {
                    armTex.colorSpace = THREE.SRGBColorSpace;
                    addArmPlanes(group, cal, armTex, planeW, planeH, mesh.position.x, mesh.position.y);
                    resolve(group);
                }, undefined, () => resolve(group));
            } else {
                resolve(group);
            }
        }, undefined, reject);
    });
}

// ─── MediaPipe Init ───────────────────────────────────────────────────────────
async function initFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: true,
    });
}

// ─── Products ─────────────────────────────────────────────────────────────────
async function loadProducts() {
    const [prodRes, calRes] = await Promise.all([
        fetch("/api/products"),
        fetch("/static/glasses/processed/calibration.json"),
    ]);
    products    = await prodRes.json();
    calibration = await calRes.json();
    renderProductGrid();
}

function renderProductGrid() {
    productGrid.innerHTML = products.map(p => `
        <div class="product-card" data-id="${p.id}">
            <div class="product-thumb">
                <img src="/static/glasses/processed/${p.frame_image}" alt="${p.name}">
            </div>
            <div class="product-info">
                <div class="product-name">${p.name}</div>
                <div class="product-specs">${p.lens_width}-${p.bridge_width}-${p.temple_length}</div>
            </div>
            <span class="product-type-badge">${p.type}</span>
        </div>
    `).join("");
    productGrid.querySelectorAll(".product-card").forEach(c =>
        c.addEventListener("click", () => selectProduct(c.dataset.id))
    );
}

function selectProduct(id) {
    selectedProductId = id;
    productGrid.querySelectorAll(".product-card").forEach(c =>
        c.classList.toggle("selected", c.dataset.id === id)
    );
}

// ─── Camera ───────────────────────────────────────────────────────────────────
async function startCamera() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 960 }, facingMode: "user" },
        });
        video.srcObject = webcamStream;
        video.style.display = "block";
        placeholder.style.display = "none";
        measurementsPanel.style.display = "block";
        startBtn.disabled = true;
        stopBtn.disabled = false;

        video.addEventListener("loadeddata", async () => {
            const w = video.videoWidth, h = video.videoHeight;
            canvas.width = w; canvas.height = h;
            document.getElementById("videoContainer").style.aspectRatio = `${w} / ${h}`;

            initThree(w, h);

            // Build a textured plane for every product
            const builds = products.map(async (p) => {
                const cal = calibration[p.cal_key];
                if (!cal) { console.error("No calibration for", p.cal_key); return; }
                try {
                    const grp = await buildGlassesGroup(p, cal);
                    scene.add(grp);
                    glassesGroups[p.id] = grp;
                } catch (e) {
                    console.error("buildGlassesGroup failed for", p.id, e);
                }
            });
            await Promise.allSettled(builds);

            detectLoop();
        }, { once: true });
    } catch (err) {
        alert("Camera access denied.");
        console.error(err);
    }
}

function stopCamera() {
    if (animationFrameId) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
    if (webcamStream) { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
    video.style.display = "none";
    placeholder.style.display = "flex";
    measurementsPanel.style.display = "none";
    startBtn.disabled = false; stopBtn.disabled = true;
    for (const g of Object.values(glassesGroups)) g.visible = false;
    if (renderer) renderer.render(scene, camera);
    pdValue.textContent = bridgeValue.textContent = faceWidthValue.textContent = "—";
    Object.values(buf).forEach(b => (b.length = 0));
}

// ─── Detection loop ───────────────────────────────────────────────────────────
let lastTimestamp = -1;

function detectLoop() {
    if (!webcamStream) return;
    animationFrameId = requestAnimationFrame(detectLoop);

    if (!faceLandmarker) return; // still loading

    const now = performance.now();
    if (now === lastTimestamp) return;
    lastTimestamp = now;

    let results;
    try { results = faceLandmarker.detectForVideo(video, now); }
    catch (e) { console.error("detectForVideo:", e); return; }

    for (const g of Object.values(glassesGroups)) g.visible = false;

    if (results.faceLandmarks?.length > 0) {
        const m = calculateMeasurements(results.faceLandmarks[0]);
        updateMeasurementDisplay(m);
        positionGlasses(results, m);
    }

    renderer.render(scene, camera);
}

// ─── Measurements ─────────────────────────────────────────────────────────────
function calculateMeasurements(lm) {
    const w = threeW, h = threeH;
    const lp = lm[468], rp = lm[473];
    const li = lm[469], ri = lm[471];

    const pxPerMm = dist2d(li, ri, w, h) / 11.7;
    addBuf("pd",        dist2d(lp, rp, w, h) / pxPerMm);
    addBuf("bridge",    dist2d(lm[133], lm[362], w, h) / pxPerMm);
    addBuf("faceWidth", dist2d(lm[127], lm[356], w, h) / pxPerMm);
    addBuf("pxPerMm",   pxPerMm);

    return {
        pd: smoothed("pd"), bridge: smoothed("bridge"),
        faceWidth: smoothed("faceWidth"), pxPerMm: smoothed("pxPerMm"),
        leftPupil:  { x: lp.x * w, y: lp.y * h },
        rightPupil: { x: rp.x * w, y: rp.y * h },
    };
}

function dist2d(a, b, w, h) {
    return Math.sqrt(((a.x - b.x) * w) ** 2 + ((a.y - b.y) * h) ** 2);
}
function addBuf(k, v) { buf[k].push(v); if (buf[k].length > SMOOTHING) buf[k].shift(); }
function smoothed(k) {
    const b = buf[k]; if (!b.length) return 0;
    const s = [...b].sort((a, z) => a - z), m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// ─── Position glasses on face ─────────────────────────────────────────────────
function positionGlasses(results, m) {
    if (!selectedProductId) return;
    const group = glassesGroups[selectedProductId];
    if (!group) return;

    const w = threeW, h = threeH;

    // World-space position of the midpoint between the two pupils
    const wx = (m.leftPupil.x + m.rightPupil.x) / 2 - w / 2;
    const wy =  h / 2 - (m.leftPupil.y + m.rightPupil.y) / 2;

    // Roll from pupil line
    const roll = Math.atan2(
        m.rightPupil.y - m.leftPupil.y,
        m.rightPupil.x - m.leftPupil.x
    );

    // Pitch / yaw from facial transformation matrix
    let pitch = 0, yaw = 0;
    if (results.facialTransformationMatrixes?.length > 0) {
        const d  = results.facialTransformationMatrixes[0].data;
        const eu = new THREE.Euler().setFromRotationMatrix(
            new THREE.Matrix4().fromArray(Array.from(d)), "YXZ"
        );
        yaw   =  eu.y * 0.55;
        pitch = -eu.x * 0.45;
    }

    // group.scale converts mm → world pixels
    group.position.set(wx, wy, 0);
    group.scale.setScalar(m.pxPerMm * GLASSES_SCALE);
    group.rotation.set(pitch, yaw, -roll);
    group.visible = true;

    // Arms: scale based on actual yaw (undo the 0.55 damping)
    const realYaw  = yaw / 0.55;
    const leftArm  = group.getObjectByName("leftArm");
    const rightArm = group.getObjectByName("rightArm");
    if (leftArm) {
        const s = Math.max(0, Math.sin(realYaw));
        leftArm.scale.x = s;
        leftArm.visible = s > 0.02;
    }
    if (rightArm) {
        const s = Math.max(0, -Math.sin(realYaw));
        rightArm.scale.x = s;
        rightArm.visible = s > 0.02;
    }
}

// ─── Measurement display ──────────────────────────────────────────────────────
function updateMeasurementDisplay(m) {
    pdValue.textContent        = `${m.pd.toFixed(1)} mm`;
    bridgeValue.textContent    = `${m.bridge.toFixed(1)} mm`;
    faceWidthValue.textContent = `${m.faceWidth.toFixed(1)} mm`;
    latestMeasurements = m;
    debouncedRecommend();
}

// ─── Recommendation ───────────────────────────────────────────────────────────
let recommendTimeout = null;
function debouncedRecommend() {
    clearTimeout(recommendTimeout);
    recommendTimeout = setTimeout(fetchRecommendation, 1000);
}
async function fetchRecommendation() {
    if (!latestMeasurements) return;
    try {
        const res = await fetch("/api/recommend", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                pd_mm: latestMeasurements.pd,
                face_width_mm: latestMeasurements.faceWidth,
                bridge_width_mm: latestMeasurements.bridge,
            }),
        });
        renderRecommendations(await res.json());
    } catch (err) { console.error(err); }
}
function renderRecommendations(recs) {
    recommendationPanel.style.display = "block";
    recommendationList.innerHTML = recs.map((r, i) => `
        <div class="rec-item">
            <div class="rec-rank ${i === 0 ? "top" : ""}">${i + 1}</div>
            <div class="rec-info">
                <div class="rec-name">${r.name}</div>
                <div class="rec-notes">${r.fit_notes.join(" · ")}</div>
            </div>
            <div class="rec-score">${Math.round(r.score * 100)}%</div>
        </div>
    `).join("");
}

// ─── Init ─────────────────────────────────────────────────────────────────────
const embedProductId = new URLSearchParams(window.location.search).get('product');

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

loadProducts()
    .then(() => { if (embedProductId) selectProduct(embedProductId); })
    .catch(err => console.error("loadProducts failed:", err));
initFaceLandmarker().catch(err => console.error("initFaceLandmarker failed:", err));

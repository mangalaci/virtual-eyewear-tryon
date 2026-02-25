import { FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// --- State ---
let faceLandmarker = null;
let webcamStream = null;
let animationFrameId = null;
let products = [];
let selectedProductId = null;
let glassesImages = {};
let latestMeasurements = null;

// Smoothing buffer for measurements
const SMOOTHING_SIZE = 10;
const measurementBuffer = { pd: [], bridge: [], faceWidth: [], pxPerMm: [] };

// --- DOM Elements ---
const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const placeholder = document.getElementById("cameraPlaceholder");
const pdValue = document.getElementById("pdValue");
const bridgeValue = document.getElementById("bridgeValue");
const faceWidthValue = document.getElementById("faceWidthValue");
const productGrid = document.getElementById("productGrid");
const recommendationPanel = document.getElementById("recommendationPanel");
const recommendationList = document.getElementById("recommendationList");

// --- Initialize MediaPipe ---
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
        outputFacialTransformationMatrixes: false,
    });
}

// --- Load Products ---
async function loadProducts() {
    const res = await fetch("/api/products");
    products = await res.json();
    renderProductGrid();
    preloadGlassesImages();
}

function preloadGlassesImages() {
    for (const p of products) {
        const img = new Image();
        img.src = `/glasses/${p.image}`;
        glassesImages[p.id] = img;
    }
}

function renderProductGrid() {
    productGrid.innerHTML = products
        .map(
            (p) => `
        <div class="product-card" data-id="${p.id}">
            <div class="product-thumb">
                <img src="/glasses/${p.image}" alt="${p.name}">
            </div>
            <div class="product-info">
                <div class="product-name">${p.name}</div>
                <div class="product-specs">${p.lens_width}-${p.bridge_width}-${p.temple_length}</div>
            </div>
            <span class="product-type-badge">${p.type}</span>
        </div>
    `
        )
        .join("");

    productGrid.querySelectorAll(".product-card").forEach((card) => {
        card.addEventListener("click", () => selectProduct(card.dataset.id));
    });
}

function selectProduct(id) {
    selectedProductId = id;
    productGrid.querySelectorAll(".product-card").forEach((card) => {
        card.classList.toggle("selected", card.dataset.id === id);
    });
}

// --- Webcam ---
async function startCamera() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 960 }, facingMode: "user" },
        });
        video.srcObject = webcamStream;
        video.style.display = "block";
        placeholder.style.display = "none";
        startBtn.disabled = true;
        stopBtn.disabled = false;

        video.addEventListener("loadeddata", () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            // Match the container's aspect-ratio to the actual camera output so
            // the canvas and video use the same CSS scale factor (fixes the bug
            // where a 16:9 camera in a 4:3 container made the overlay ~37% smaller
            // than the face and caused left/right asymmetry).
            document.getElementById("videoContainer").style.aspectRatio =
                `${video.videoWidth} / ${video.videoHeight}`;
            detectLoop();
        }, { once: true });
    } catch (err) {
        alert("Camera access denied. Please allow camera access and try again.");
        console.error(err);
    }
}

function stopCamera() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (webcamStream) {
        webcamStream.getTracks().forEach((t) => t.stop());
        webcamStream = null;
    }
    video.style.display = "none";
    placeholder.style.display = "flex";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    pdValue.textContent = "—";
    bridgeValue.textContent = "—";
    faceWidthValue.textContent = "—";
}

// --- Detection Loop ---
let lastTimestamp = -1;

function detectLoop() {
    if (!faceLandmarker || !webcamStream) return;

    const now = performance.now();
    if (now === lastTimestamp) {
        animationFrameId = requestAnimationFrame(detectLoop);
        return;
    }
    lastTimestamp = now;

    const results = faceLandmarker.detectForVideo(video, now);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        const measurements = calculateMeasurements(landmarks);
        updateMeasurementDisplay(measurements);
        drawGlassesOverlay(landmarks, measurements);
    }

    animationFrameId = requestAnimationFrame(detectLoop);
}

// --- Measurements ---
// Average inter-pupillary distance for calibration reference
const AVG_PD_MM = 63; // population average ~63mm

function calculateMeasurements(landmarks) {
    const w = canvas.width;
    const h = canvas.height;

    // Pupil centers (iris landmarks)
    const leftPupil = landmarks[468]; // left iris center
    const rightPupil = landmarks[473]; // right iris center

    // Inner eye corners (for bridge)
    const leftInner = landmarks[133];
    const rightInner = landmarks[362];

    // Temple points (face width)
    const leftTemple = landmarks[127];
    const rightTemple = landmarks[356];

    // Pixel distances
    const pdPx = dist(leftPupil, rightPupil, w, h);
    const bridgePx = dist(leftInner, rightInner, w, h);
    const faceWidthPx = dist(leftTemple, rightTemple, w, h);

    // Estimate real-world scale using iris width as reference
    // Average iris diameter is ~11.7mm
    const leftIrisLeft = landmarks[469];
    const leftIrisRight = landmarks[471];
    const irisWidthPx = dist(leftIrisLeft, leftIrisRight, w, h);
    const pxPerMm = irisWidthPx / 11.7;

    const pdMm = pdPx / pxPerMm;
    const bridgeMm = bridgePx / pxPerMm;
    const faceWidthMm = faceWidthPx / pxPerMm;

    // Smooth measurements
    addToBuffer("pd", pdMm);
    addToBuffer("bridge", bridgeMm);
    addToBuffer("faceWidth", faceWidthMm);
    addToBuffer("pxPerMm", pxPerMm);

    return {
        pd: getSmoothed("pd"),
        bridge: getSmoothed("bridge"),
        faceWidth: getSmoothed("faceWidth"),
        pxPerMm: getSmoothed("pxPerMm"),
        // Raw pixel positions for overlay rendering
        leftPupil: { x: leftPupil.x * w, y: leftPupil.y * h },
        rightPupil: { x: rightPupil.x * w, y: rightPupil.y * h },
        leftInner: { x: leftInner.x * w, y: leftInner.y * h },
        rightInner: { x: rightInner.x * w, y: rightInner.y * h },
        leftTemple: { x: leftTemple.x * w, y: leftTemple.y * h },
        rightTemple: { x: rightTemple.x * w, y: rightTemple.y * h },
        noseBridge: { x: landmarks[6].x * w, y: landmarks[6].y * h },
        noseTop: { x: landmarks[4].x * w, y: landmarks[4].y * h },
    };
}

function dist(a, b, w, h) {
    return Math.sqrt(((a.x - b.x) * w) ** 2 + ((a.y - b.y) * h) ** 2);
}

function addToBuffer(key, value) {
    const buf = measurementBuffer[key];
    buf.push(value);
    if (buf.length > SMOOTHING_SIZE) buf.shift();
}

function getSmoothed(key) {
    const buf = measurementBuffer[key];
    if (buf.length === 0) return 0;
    const sorted = [...buf].sort((a, b) => a - b);
    // Use median for robustness
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function updateMeasurementDisplay(m) {
    pdValue.textContent = `${m.pd.toFixed(1)} mm`;
    bridgeValue.textContent = `${m.bridge.toFixed(1)} mm`;
    faceWidthValue.textContent = `${m.faceWidth.toFixed(1)} mm`;

    latestMeasurements = m;
    debouncedRecommend();
}

// --- Glasses Overlay ---
function drawGlassesOverlay(landmarks, m) {
    if (!selectedProductId) {
        drawDefaultOverlay(m);
        return;
    }

    const img = glassesImages[selectedProductId];
    if (!img || !img.complete || img.naturalWidth === 0) {
        drawDefaultOverlay(m);
        return;
    }

    const w = canvas.width, h = canvas.height;

    // --- SIZE ---
    // Use the product's physical dimensions (mm) and the iris-calibrated px/mm scale.
    // frame front width = lens_width × 2 + bridge_width (from products.json)
    const product = products.find(p => p.id === selectedProductId);
    const frameFrontMm = (product.lens_width * 2) + product.bridge_width;
    // glassesWidth = physical frame-front width in canvas px (also used for hinge offset)
    const glassesWidth = frameFrontMm * m.pxPerMm * 1.2;
    // frame_fill_ratio: fraction of the PNG width occupied by the frame front.
    // Divide to get the total draw width so the frame front renders at the correct size.
    const frameFillRatio = product.frame_fill_ratio ?? 0.455;
    const drawWidth = glassesWidth / frameFillRatio;
    const glassesHeight = drawWidth * (img.naturalHeight / img.naturalWidth);

    // Center between pupils
    const centerX = (m.leftPupil.x + m.rightPupil.x) / 2;
    const centerY = (m.leftPupil.y + m.rightPupil.y) / 2;

    // Roll: tilt of the head left/right
    const angle = Math.atan2(m.rightPupil.y - m.leftPupil.y, m.rightPupil.x - m.leftPupil.x);

    // Per-product vertical offset: lens_y_frac tells where in the image the lens
    // centres sit (0 = top, 1 = bottom). Default 0.5 = image centre.
    const lensYFrac = product?.lens_y_frac ?? 0.5;

    const cos_a = Math.cos(angle);
    const sin_a = Math.sin(angle);

    // --- Frame image ---
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(angle);
    ctx.drawImage(img, -drawWidth / 2, -lensYFrac * glassesHeight, drawWidth, glassesHeight);
    ctx.restore();
}

function drawDefaultOverlay(m) {
    // Draw a simple wireframe glasses overlay when no product is selected
    const centerX = (m.leftPupil.x + m.rightPupil.x) / 2;
    const centerY = (m.leftPupil.y + m.rightPupil.y) / 2;
    const pdPx = Math.sqrt(
        (m.rightPupil.x - m.leftPupil.x) ** 2 + (m.rightPupil.y - m.leftPupil.y) ** 2
    );
    const angle = Math.atan2(m.rightPupil.y - m.leftPupil.y, m.rightPupil.x - m.leftPupil.x);

    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(angle);

    const lensRadius = pdPx * 0.42;
    const lensOffsetX = pdPx / 2;
    const bridgeY = 0;

    ctx.strokeStyle = "rgba(108, 99, 255, 0.7)";
    ctx.lineWidth = 2.5;

    // Left lens
    ctx.beginPath();
    ctx.ellipse(-lensOffsetX, bridgeY, lensRadius, lensRadius * 0.85, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Right lens
    ctx.beginPath();
    ctx.ellipse(lensOffsetX, bridgeY, lensRadius, lensRadius * 0.85, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Bridge
    ctx.beginPath();
    ctx.moveTo(-lensOffsetX + lensRadius, bridgeY);
    ctx.lineTo(lensOffsetX - lensRadius, bridgeY);
    ctx.stroke();

    // Temples (arms)
    const templeLen = pdPx * 0.5;
    ctx.beginPath();
    ctx.moveTo(-lensOffsetX - lensRadius, bridgeY);
    ctx.lineTo(-lensOffsetX - lensRadius - templeLen, bridgeY + templeLen * 0.3);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(lensOffsetX + lensRadius, bridgeY);
    ctx.lineTo(lensOffsetX + lensRadius + templeLen, bridgeY + templeLen * 0.3);
    ctx.stroke();

    ctx.restore();
}

// --- Recommendation ---
let recommendTimeout = null;

function debouncedRecommend() {
    if (recommendTimeout) clearTimeout(recommendTimeout);
    recommendTimeout = setTimeout(fetchRecommendation, 1000);
}

async function fetchRecommendation() {
    if (!latestMeasurements) return;

    try {
        const res = await fetch("/api/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                pd_mm: latestMeasurements.pd,
                face_width_mm: latestMeasurements.faceWidth,
                bridge_width_mm: latestMeasurements.bridge,
            }),
        });
        const recommendations = await res.json();
        renderRecommendations(recommendations);
    } catch (err) {
        console.error("Recommendation error:", err);
    }
}

function renderRecommendations(recs) {
    recommendationPanel.style.display = "block";
    recommendationList.innerHTML = recs
        .map(
            (r, i) => `
        <div class="rec-item">
            <div class="rec-rank ${i === 0 ? "top" : ""}">${i + 1}</div>
            <div class="rec-info">
                <div class="rec-name">${r.name}</div>
                <div class="rec-notes">${r.fit_notes.join(" · ")}</div>
            </div>
            <div class="rec-score">${Math.round(r.score * 100)}%</div>
        </div>
    `
        )
        .join("");
}

// --- Event Listeners ---
startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

// --- Init ---
async function init() {
    await Promise.all([initFaceLandmarker(), loadProducts()]);
    console.log("Virtual Eyewear Try-On initialized");
}

init();

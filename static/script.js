let useWebcam = false;
let outputMediaPath = "";

/* ============================= ELEMENTS ============================= */
const uploadTab = document.getElementById("uploadTab");
const liveTab = document.getElementById("liveTab");
const uploadSection = document.getElementById("uploadSection");
const liveSection = document.getElementById("liveSection");
const dropZone = document.getElementById("dropZone");
const mediaUpload = document.getElementById("mediaUpload");
const webcam = document.getElementById("webcam");
const captureCanvas = document.getElementById("captureCanvas");
const captureBtn = document.getElementById("captureBtn");
const predictBtn = document.getElementById("predictBtn");
const loader = document.getElementById("loader");
const laneStrip = document.getElementById("laneStrip");
const emptyState = document.getElementById("emptyState");
const resultsBody = document.getElementById("resultsBody");
const outputWrap = document.getElementById("outputWrap");
const dropZoneOutput = document.getElementById("dropZoneOutput");

let webcamStream = null;

/* ============================= TAB SWITCHING ============================= */
uploadTab.onclick = () => {
    useWebcam = false;
    uploadSection.classList.remove("hidden");
    liveSection.classList.add("hidden");
    uploadTab.classList.add("active");
    liveTab.classList.remove("active");
    uploadTab.setAttribute("aria-selected", "true");
    liveTab.setAttribute("aria-selected", "false");
    stopWebcam();
};

liveTab.onclick = () => {
    useWebcam = true;
    uploadSection.classList.add("hidden");
    liveSection.classList.remove("hidden");
    liveTab.classList.add("active");
    uploadTab.classList.remove("active");
    liveTab.setAttribute("aria-selected", "true");
    uploadTab.setAttribute("aria-selected", "false");
    startWebcam();
};

/* ============================= FILE UPLOAD PREVIEW ============================= */
dropZone.onclick = () => mediaUpload.click();

dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        mediaUpload.click();
    }
});

mediaUpload.onchange = () => {
    handleMedia(mediaUpload.files[0]);
};

/* Drag & drop */
["dragover", "dragenter"].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    })
);

["dragleave", "drop"].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
    })
);

dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
        mediaUpload.files = e.dataTransfer.files;
        handleMedia(file);
    }
});

function handleMedia(file) {
    if (!file) return;

    const url = URL.createObjectURL(file);
    dropZone.innerHTML = "";

    if (file.type.startsWith("image")) {
        dropZone.innerHTML = `<img src="${url}" alt="Uploaded road image preview">`;
    } else {
        dropZone.innerHTML = `<video src="${url}" controls></video>`;
    }
}

/* ============================= WEBCAM ============================= */
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" }
        });
        webcam.srcObject = webcamStream;
    } catch (err) {
        alert("Couldn't access the camera: " + err.message);
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
}

/* ============================= CAPTURE PHOTO (mobile-style shutter) ============================= */
captureBtn.onclick = async () => {

    if (!webcamStream) {
        alert("Camera isn't active yet — give it a moment and try again.");
        return;
    }

    captureCanvas.width = webcam.videoWidth;
    captureCanvas.height = webcam.videoHeight;
    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(webcam, 0, 0, captureCanvas.width, captureCanvas.height);

    captureCanvas.toBlob(async (blob) => {
        if (!blob) {
            alert("Couldn't capture the frame — try again.");
            return;
        }

        const capturedFile = new File([blob], "capture.jpg", { type: "image/jpeg" });

        // Quick shutter flash feedback
        webcam.style.filter = "brightness(1.8)";
        setTimeout(() => { webcam.style.filter = ""; }, 120);

        await analyzeFile(capturedFile);

    }, "image/jpeg", 0.92);
};

/* ============================= SHARED ANALYZE LOGIC ============================= */
async function analyzeFile(file) {

    loader.classList.remove("hidden");
    laneStrip.classList.add("active");

    try {
        const formData = new FormData();
        formData.append("file", file);

        const endpoint = file.type.startsWith("image") ? "/predict_image" : "/predict_video";
        const res = await fetch(endpoint, { method: "POST", body: formData });
        const data = await res.json();

        if (data.error) {
            alert("Server error: " + data.error);
            return;
        }

        updateCounters(data.counts);
        updateSeverityAndCost(data.severity, data.estimated_cost_inr);
        showOutput(data.result_image || data.result_video);

        emptyState.classList.add("hidden");
        resultsBody.classList.remove("hidden");

    } catch (err) {
        alert("Something went wrong: " + err.message);
        console.error(err);
    } finally {
        loader.classList.add("hidden");
        laneStrip.classList.remove("active");
    }
}

/* ============================= PREDICT BUTTON (Upload tab) ============================= */
predictBtn.onclick = async () => {

    if (useWebcam) {
        alert("Use the Capture Photo button to take a shot, or switch to Upload.");
        return;
    }

    const file = mediaUpload.files[0];

    if (!file) {
        alert("Upload a photo or video first");
        return;
    }

    await analyzeFile(file);
};

/* ============================= UPDATE COUNTERS ============================= */
function updateCounters(c) {
    document.getElementById("D00").innerText = c.D00 || 0;
    document.getElementById("D10").innerText = c.D10 || 0;
    document.getElementById("D20").innerText = c.D20 || 0;
    document.getElementById("D40").innerText = c.D40 || 0;
    document.getElementById("OTHER").innerText = c.OTHER || 0;
}

function updateSeverityAndCost(severity, cost) {
    document.getElementById("MINOR").innerText = (severity && severity.Minor) || 0;
    document.getElementById("MAJOR").innerText = (severity && severity.Major) || 0;
    document.getElementById("COST").innerText = "₹" + (cost || 0).toLocaleString("en-IN");
}

/* ============================= SHOW OUTPUT MEDIA ============================= */
function showOutput(path) {
    if (!path) return;

    outputMediaPath = path;

    const mediaTag = path.endsWith(".mp4")
        ? `<video src="${path}" controls autoplay></video>`
        : `<img src="${path}" alt="Annotated detection result">`;

    // Show it in the upload dropzone (replaces the raw preview with the annotated result)
    dropZone.innerHTML = mediaTag;

    // Also show it in the results section below
    outputWrap.classList.remove("hidden");
    dropZoneOutput.innerHTML = mediaTag;

    outputWrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ============================= DOWNLOAD REPORT ============================= */
const downloadBtn = document.getElementById("downloadBtn");

if (downloadBtn) {
    downloadBtn.addEventListener("click", () => {

        if (!outputMediaPath) {
            alert("No result to download yet — run an analysis first.");
            return;
        }

        const link = document.createElement("a");
        link.href = outputMediaPath;
        link.download = outputMediaPath.endsWith(".mp4") ? "annotated_result.mp4" : "annotated_result.jpg";

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}
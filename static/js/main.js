/**
 * Shipment Price Prediction - Frontend Application Logic
 * Interactive UI, Chart.js Cost Breakdown, Presets, and Batch Predictor.
 */

let costChart = null;

// Preset Scenarios Data
const PRESETS = {
    bronze_masterpiece: {
        artist_reputation: 0.85,
        height: 36.0,
        width: 24.0,
        weight: 350.0,
        material: "Bronze",
        price_of_sculpture: 18500.0,
        base_shipping_price: 250.0,
        transport: "Airways",
        customer_information: "Wealthy",
        international: true,
        express: true,
        fragile: false,
        installation: true,
        remote: false
    },
    antique_marble: {
        artist_reputation: 0.92,
        height: 48.0,
        width: 30.0,
        weight: 600.0,
        material: "Marble",
        price_of_sculpture: 35000.0,
        base_shipping_price: 400.0,
        transport: "Waterways",
        customer_information: "Wealthy",
        international: true,
        express: false,
        fragile: true,
        installation: true,
        remote: false
    },
    clay_fragile: {
        artist_reputation: 0.60,
        height: 14.0,
        width: 10.0,
        weight: 25.0,
        material: "Clay",
        price_of_sculpture: 1200.0,
        base_shipping_price: 65.0,
        transport: "Roadways",
        customer_information: "Working Class",
        international: false,
        express: true,
        fragile: true,
        installation: false,
        remote: true
    },
    wood_craft: {
        artist_reputation: 0.45,
        height: 20.0,
        width: 12.0,
        weight: 40.0,
        material: "Wood",
        price_of_sculpture: 850.0,
        base_shipping_price: 45.0,
        transport: "Roadways",
        customer_information: "Working Class",
        international: false,
        express: false,
        fragile: false,
        installation: false,
        remote: false
    }
};

// DOM Initialization
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    setupSyncSlider();
    setupVisualRadioCards();
    setupFormSubmission();
    setupDragAndDrop();
    checkHealth();
});

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
    document.querySelectorAll(".nav-tab").forEach(el => {
        el.classList.remove("text-white", "bg-brand-600", "shadow-md");
        el.classList.add("text-slate-400");
    });

    const targetTab = document.getElementById(`tab-${tabId}`);
    const targetBtn = document.getElementById(`tab-btn-${tabId}`);

    if (targetTab) targetTab.classList.remove("hidden");
    if (targetBtn) {
        targetBtn.classList.remove("text-slate-400");
        targetBtn.classList.add("text-white", "bg-brand-600", "shadow-md");
    }
}

// Synced Slider with Input Box
function setupSyncSlider() {
    const range = document.getElementById("artist_reputation_range");
    const num = document.getElementById("artist_reputation");
    const badge = document.getElementById("reputation-badge");

    if (range && num) {
        range.addEventListener("input", (e) => {
            num.value = e.target.value;
            if (badge) badge.textContent = parseFloat(e.target.value).toFixed(2);
        });
        num.addEventListener("input", (e) => {
            range.value = e.target.value;
            if (badge) badge.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
}

// Visual Radio Cards Active Styling
function setupVisualRadioCards() {
    // Material cards
    document.querySelectorAll(".material-card").forEach(card => {
        const radio = card.querySelector('input[type="radio"]');
        if (radio.checked) card.classList.add("active");
        card.addEventListener("click", () => {
            document.querySelectorAll(".material-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            radio.checked = true;
        });
    });

    // Transport cards
    document.querySelectorAll(".transport-card").forEach(card => {
        const radio = card.querySelector('input[type="radio"]');
        if (radio.checked) card.classList.add("active");
        card.addEventListener("click", () => {
            document.querySelectorAll(".transport-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            radio.checked = true;
        });
    });

    // Customer tier cards
    document.querySelectorAll(".tier-card").forEach(card => {
        const radio = card.querySelector('input[type="radio"]');
        if (radio.checked) card.classList.add("active");
        card.addEventListener("click", () => {
            document.querySelectorAll(".tier-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            radio.checked = true;
        });
    });
}

// Load Preset Scenarios
function loadPreset(presetKey) {
    const data = PRESETS[presetKey];
    if (!data) return;

    document.getElementById("artist_reputation_range").value = data.artist_reputation;
    document.getElementById("artist_reputation").value = data.artist_reputation;
    document.getElementById("reputation-badge").textContent = data.artist_reputation.toFixed(2);

    document.getElementById("height").value = data.height;
    document.getElementById("width").value = data.width;
    document.getElementById("weight").value = data.weight;
    document.getElementById("price_of_sculpture").value = data.price_of_sculpture;
    document.getElementById("base_shipping_price").value = data.base_shipping_price;

    // Material Radio
    const matRadio = document.querySelector(`input[name="material"][value="${data.material}"]`);
    if (matRadio) {
        matRadio.checked = true;
        document.querySelectorAll(".material-card").forEach(c => c.classList.remove("active"));
        matRadio.closest(".material-card").classList.add("active");
    }

    // Transport Radio
    const transRadio = document.querySelector(`input[name="transport"][value="${data.transport}"]`);
    if (transRadio) {
        transRadio.checked = true;
        document.querySelectorAll(".transport-card").forEach(c => c.classList.remove("active"));
        transRadio.closest(".transport-card").classList.add("active");
    }

    // Customer Tier Radio
    const tierRadio = document.querySelector(`input[name="customer_information"][value="${data.customer_information}"]`);
    if (tierRadio) {
        tierRadio.checked = true;
        document.querySelectorAll(".tier-card").forEach(c => c.classList.remove("active"));
        tierRadio.closest(".tier-card").classList.add("active");
    }

    // Toggles
    document.getElementById("international_toggle").checked = data.international;
    document.getElementById("express_toggle").checked = data.express;
    document.getElementById("fragile_toggle").checked = data.fragile;
    document.getElementById("installation_toggle").checked = data.installation;
    document.getElementById("remote_toggle").checked = data.remote;

    showToast(`Loaded "${data.material}" preset scenario.`, "info");
    
    // Auto calculate
    submitPrediction();
}

// Chart.js Setup
function initChart() {
    const ctx = document.getElementById("costBreakdownChart");
    if (!ctx) return;

    costChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Base Shipping", "Weight & Volumetric", "Insurance & Value", "Special Handling"],
            datasets: [{
                data: [110, 70, 135, 125],
                backgroundColor: [
                    "#6366f1", // Indigo
                    "#06b6d4", // Cyan
                    "#f59e0b", // Amber
                    "#10b981"  // Emerald
                ],
                borderWidth: 0,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: "#94a3b8",
                        font: { size: 10, family: "Plus Jakarta Sans" },
                        boxWidth: 10,
                        padding: 8
                    }
                }
            },
            cutout: "70%"
        }
    });
}

// Form Submission
function setupFormSubmission() {
    const form = document.getElementById("prediction-form");
    if (!form) return;

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        submitPrediction();
    });
}

async function submitPrediction() {
    const submitBtn = document.getElementById("submit-predict-btn");
    const btnText = document.getElementById("btn-text");
    const btnSpinner = document.getElementById("btn-spinner");

    // Collect values
    const payload = {
        artist_reputation: parseFloat(document.getElementById("artist_reputation").value),
        height: parseFloat(document.getElementById("height").value),
        width: parseFloat(document.getElementById("width").value),
        weight: parseFloat(document.getElementById("weight").value),
        material: document.querySelector('input[name="material"]:checked')?.value || "Bronze",
        price_of_sculpture: parseFloat(document.getElementById("price_of_sculpture").value),
        base_shipping_price: parseFloat(document.getElementById("base_shipping_price").value),
        transport: document.querySelector('input[name="transport"]:checked')?.value || "Airways",
        customer_information: document.querySelector('input[name="customer_information"]:checked')?.value || "Wealthy",
        international: document.getElementById("international_toggle").checked ? "Yes" : "No",
        express_shipment: document.getElementById("express_toggle").checked ? "Yes" : "No",
        fragile: document.getElementById("fragile_toggle").checked ? "Yes" : "No",
        installation_included: document.getElementById("installation_toggle").checked ? "Yes" : "No",
        remote_location: document.getElementById("remote_toggle").checked ? "Yes" : "No",
    };

    // UI Loading state
    if (submitBtn) submitBtn.disabled = true;
    if (btnText) btnText.textContent = "Calculating AI Estimate...";
    if (btnSpinner) btnSpinner.classList.remove("hidden");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok && data.status === "success") {
            // Animate Price Count-Up
            animatePrice(data.predicted_cost);

            // Update Breakdown UI
            if (data.breakdown) {
                document.getElementById("item-base-cost").textContent = `$${data.breakdown.base_shipping.toFixed(2)}`;
                document.getElementById("item-weight-fee").textContent = `$${data.breakdown.weight_and_dimension_fee.toFixed(2)}`;
                document.getElementById("item-insurance-fee").textContent = `$${data.breakdown.sculpture_insurance_fee.toFixed(2)}`;
                document.getElementById("item-special-fee").textContent = `+$${data.breakdown.special_services_surcharge.toFixed(2)}`;

                // Update Chart
                if (costChart) {
                    costChart.data.datasets[0].data = [
                        data.breakdown.base_shipping,
                        data.breakdown.weight_and_dimension_fee,
                        data.breakdown.sculpture_insurance_fee,
                        data.breakdown.special_services_surcharge
                    ];
                    costChart.update();
                }
            }

            document.getElementById("prediction-timestamp").textContent = `Calculated: ${new Date().toLocaleTimeString()}`;

            // Trigger celebration confetti
            confetti({
                particleCount: 40,
                spread: 60,
                origin: { y: 0.7 }
            });

            showToast("Shipment estimation calculated successfully!", "success");
        } else {
            showToast(data.message || "Failed to calculate prediction.", "error");
        }
    } catch (err) {
        showToast(`Error: ${err.message}`, "error");
    } finally {
        if (submitBtn) submitBtn.disabled = false;
        if (btnText) btnText.textContent = "Estimate Shipment Cost Now";
        if (btnSpinner) btnSpinner.classList.add("hidden");
    }
}

// Animate Price Number Count Up
function animatePrice(targetValue) {
    const displayEl = document.getElementById("display-price");
    if (!displayEl) return;

    let start = 0;
    const duration = 750;
    const startTime = performance.now();

    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const currentVal = start + (targetValue - start) * easeOutQuad(progress);
        
        displayEl.textContent = `$${currentVal.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function easeOutQuad(t) {
    return t * (2 - t);
}

// Drag and Drop Batch Predictor
function setupDragAndDrop() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("batch-file-input");

    if (!dropZone || !fileInput) return;

    dropZone.addEventListener("click", () => fileInput.click());

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("border-brand-500", "bg-brand-500/10");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("border-brand-500", "bg-brand-500/10");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files.length) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleSelectedFile(e.target.files[0]);
        }
    });
}

let selectedBatchFile = null;

function handleSelectedFile(file) {
    if (!file.name.endsWith(".csv")) {
        showToast("Please select a valid .csv file.", "error");
        return;
    }

    selectedBatchFile = file;
    document.getElementById("selected-file-name").textContent = file.name;
    document.getElementById("selected-file-size").textContent = `${(file.size / 1024).toFixed(1)} KB`;
    document.getElementById("file-info-box").classList.remove("hidden");
    showToast(`Loaded file: ${file.name}`, "info");
}

async function executeBatchPrediction() {
    if (!selectedBatchFile) return;

    const processBtn = document.getElementById("process-batch-btn");
    processBtn.disabled = true;
    processBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i><span>Processing...</span>`;

    const formData = new FormData();
    formData.append("file", selectedBatchFile);

    try {
        const response = await fetch("/predict_batch", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.status === "success") {
            renderBatchResults(data);
            showToast(`Batch completed: ${data.total_rows} records estimated!`, "success");
        } else {
            showToast(data.message || "Failed to process batch CSV.", "error");
        }
    } catch (err) {
        showToast(`Batch processing error: ${err.message}`, "error");
    } finally {
        processBtn.disabled = false;
        processBtn.innerHTML = `<i class="fa-solid fa-bolt"></i><span>Process Predictions</span>`;
    }
}

function renderBatchResults(data) {
    const resultsPanel = document.getElementById("batch-results-panel");
    const headerRow = document.getElementById("batch-table-header");
    const tableBody = document.getElementById("batch-table-body");
    const downloadBtn = document.getElementById("download-results-btn");

    document.getElementById("batch-total-rows-badge").textContent = `${data.total_rows} rows`;
    document.getElementById("batch-total-sum").textContent = data.total_predicted_cost;
    downloadBtn.href = data.download_url;

    // Render Headers
    headerRow.innerHTML = "";
    data.columns.forEach(col => {
        const th = document.createElement("th");
        th.className = "px-4 py-2.5 font-bold";
        th.textContent = col;
        if (col === "Predicted_Cost") {
            th.className += " text-emerald-400 bg-emerald-500/10";
        }
        headerRow.appendChild(th);
    });

    // Render Rows
    tableBody.innerHTML = "";
    data.preview_data.forEach(row => {
        const tr = document.createElement("tr");
        tr.className = "hover:bg-dark-hover transition";
        data.columns.forEach(col => {
            const td = document.createElement("td");
            td.className = "px-4 py-2";
            if (col === "Predicted_Cost") {
                td.className += " font-bold text-emerald-400 bg-emerald-500/5";
                td.textContent = `$${parseFloat(row[col]).toFixed(2)}`;
            } else {
                td.textContent = row[col];
            }
            tr.appendChild(td);
        });
        tableBody.appendChild(tr);
    });

    resultsPanel.classList.remove("hidden");
    resultsPanel.scrollIntoView({ behavior: "smooth" });
}

// Download Sample Template CSV
function downloadSampleCsv() {
    const headers = "Artist Reputation,Height,Width,Weight,Material,Price Of Sculpture,Base Shipping Price,International,Express Shipment,Installation Included,Transport,Fragile,Customer Information,Remote Location\n";
    const row1 = "0.75,18.0,12.0,120.0,Bronze,4500.0,110.0,Yes,No,Yes,Airways,No,Wealthy,No\n";
    const row2 = "0.92,48.0,30.0,600.0,Marble,35000.0,400.0,Yes,No,Yes,Waterways,Yes,Wealthy,No\n";
    const row3 = "0.45,20.0,12.0,40.0,Wood,850.0,45.0,No,No,No,Roadways,No,Working Class,No\n";

    const csvContent = "data:text/csv;charset=utf-8," + encodeURIComponent(headers + row1 + row2 + row3);
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", csvContent);
    downloadAnchor.setAttribute("download", "sample_shipment_data.csv");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    showToast("Downloaded sample_shipment_data.csv", "info");
}

// Retrain Trigger
async function triggerRetraining() {
    const btn = document.getElementById("trigger-train-btn");
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i><span>Training Pipeline Running...</span>`;

    appendLog("[Job] Triggering model training pipeline...");

    try {
        const response = await fetch("/train", { method: "POST" });
        const data = await response.json();

        appendLog(`[Job Response] ${data.message}`);
        showToast(data.message, "info");

        // Poll train status
        pollTrainingStatus();
    } catch (err) {
        appendLog(`[Error] Failed to initiate training: ${err.message}`);
        showToast(`Training error: ${err.message}`, "error");
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-play"></i><span>Start Training Pipeline</span>`;
    }
}

function pollTrainingStatus() {
    const interval = setInterval(async () => {
        try {
            const res = await fetch("/train_status");
            const status = await res.json();

            document.getElementById("pipeline-live-status-badge").textContent = status.status;

            if (!status.is_training) {
                clearInterval(interval);
                const btn = document.getElementById("trigger-train-btn");
                btn.disabled = false;
                btn.innerHTML = `<i class="fa-solid fa-play"></i><span>Start Training Pipeline</span>`;
                appendLog(`[Job Complete] ${status.message}`);
                showToast(status.message, status.status === "Completed" ? "success" : "error");
            }
        } catch (e) {
            clearInterval(interval);
        }
    }, 4000);
}

function appendLog(msg) {
    const logs = document.getElementById("training-terminal-logs");
    if (!logs) return;
    const div = document.createElement("div");
    div.textContent = `${new Date().toLocaleTimeString()} ${msg}`;
    logs.appendChild(div);
    logs.scrollTop = logs.scrollHeight;
}

// Receipt & Invoice Modal
function openReceiptModal() {
    const modal = document.getElementById("receipt-modal");
    if (!modal) return;

    const material = document.querySelector('input[name="material"]:checked')?.value || "Bronze";
    const transport = document.querySelector('input[name="transport"]:checked')?.value || "Airways";
    const val = document.getElementById("price_of_sculpture").value || "4500";
    const price = document.getElementById("display-price").textContent;

    document.getElementById("receipt-ref").textContent = `EST-${Math.floor(10000 + Math.random() * 90000)}`;
    document.getElementById("receipt-mat-trans").textContent = `${material} / ${transport}`;
    document.getElementById("receipt-val").textContent = `$${parseFloat(val).toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
    document.getElementById("receipt-total").textContent = price;

    modal.classList.remove("hidden");
}

function closeReceiptModal() {
    const modal = document.getElementById("receipt-modal");
    if (modal) modal.classList.add("hidden");
}

// Copy Summary
function copyPrediction() {
    const price = document.getElementById("display-price").textContent;
    const material = document.querySelector('input[name="material"]:checked')?.value || "Bronze";
    const transport = document.querySelector('input[name="transport"]:checked')?.value || "Airways";

    const text = `Sculpture Shipment Estimate\nMaterial: ${material}\nTransport: ${transport}\nTotal Cost: ${price}`;
    navigator.clipboard.writeText(text).then(() => {
        showToast("Estimate copied to clipboard!", "success");
    });
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast("Copied to clipboard!", "success");
    });
}

// Health Check on Load
async function checkHealth() {
    try {
        const res = await fetch("/health");
        if (res.ok) {
            const badge = document.getElementById("live-health-badge");
            if (badge) badge.classList.remove("hidden");
        }
    } catch (e) {}
}

// Toast Notifications System
function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    const bgColors = {
        success: "bg-emerald-950/90 border-emerald-500/50 text-emerald-200",
        error: "bg-red-950/90 border-red-500/50 text-red-200",
        info: "bg-slate-900/90 border-brand-500/50 text-slate-200"
    };
    const icons = {
        success: "fa-solid fa-circle-check text-emerald-400",
        error: "fa-solid fa-triangle-exclamation text-red-400",
        info: "fa-solid fa-circle-info text-brand-400"
    };

    toast.className = `toast pointer-events-auto flex items-center space-x-3 px-4 py-3 rounded-2xl border backdrop-blur-xl shadow-2xl text-xs font-medium max-w-sm ${bgColors[type] || bgColors.info}`;
    toast.innerHTML = `
        <i class="${icons[type] || icons.info} text-base"></i>
        <span class="flex-1">${message}</span>
        <button onclick="this.parentElement.remove()" class="text-slate-400 hover:text-white"><i class="fa-solid fa-xmark"></i></button>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transition = "opacity 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

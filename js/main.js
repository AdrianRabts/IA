
// Variables globales
let uploadedImage = null;
let inspectionCount = 0;
const API_URL = 'http://localhost:5000';

// Elementos DOM
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('previewContainer');
const inspectBtn = document.getElementById('inspectBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const noResults = document.getElementById('noResults');
const finalVerdict = document.getElementById('finalVerdict');
const architecturesGrid = document.getElementById('architecturesGrid');
const recommendations = document.getElementById('recommendations');
const inspectionCountElement = document.getElementById('inspectionCount');

// Event Listeners
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('active');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('active');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('active');
    handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

inspectBtn.addEventListener('click', performInspection);

// Manejar archivo
function handleFile(file) {
    if (file && file.type.startsWith('image/')) {
        uploadedImage = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            inspectBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    } else {
        alert('Por favor selecciona una imagen válida');
    }
}

// Realizar inspección
async function performInspection() {
    if (!uploadedImage) return;

    // Mostrar loading
    noResults.style.display = 'none';
    loadingOverlay.classList.add('active');
    finalVerdict.classList.remove('show');
    architecturesGrid.innerHTML = '';
    recommendations.classList.remove('show');

    try {
        // Crear FormData
        const formData = new FormData();
        formData.append('image', uploadedImage);

        // Realizar petición
        const response = await fetch(`${API_URL}/inspect`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Error en la inspección');
        }

        const data = await response.json();

        // Ocultar loading
        loadingOverlay.classList.remove('active');

        // Mostrar resultados
        displayResults(data);

        // Actualizar contador
        inspectionCount++;
        inspectionCountElement.textContent = inspectionCount;

    } catch (error) {
        console.error('Error:', error);
        loadingOverlay.classList.remove('active');
        alert('Error al procesar la inspección. Verifica que el servidor esté corriendo.');
    }
}

// Mostrar resultados
function displayResults(data) {
    // Veredicto final
    const verdict = data.final_verdict;
    const verdictDetails = data.verdict_details;
    
    finalVerdict.innerHTML = `
        <div class="verdict-status">${verdictDetails.icon}</div>
        <div class="verdict-text verdict-${verdict}">${verdict}</div>
        <div style="color: #666;">ID: ${data.inspection_id}</div>
        <div style="color: #999; font-size: 0.9em; margin-top: 10px;">
            Tiempo total: ${data.total_processing_time}s
        </div>
    `;
    finalVerdict.classList.add('show');

    // Arquitecturas
    const architectures = [
        { name: 'YOLO11', icon: '🎯', data: data.architectures.yolo11, color: '#FF6B6B' },
        { name: 'EfficientNetV2', icon: '⚡', data: data.architectures.efficientnetv2, color: '#4ECDC4' },
        { name: 'DETR', icon: '🔮', data: data.architectures.detr, color: '#FFD93D' }
    ];

    architectures.forEach(arch => {
        const card = createArchCard(arch);
        architecturesGrid.innerHTML += card;
    });

    // Animar barras de confianza
    setTimeout(() => {
        document.querySelectorAll('.confidence-fill').forEach(bar => {
            bar.style.width = bar.getAttribute('data-width') + '%';
        });
    }, 100);

    // Recomendaciones
    if (data.recommendations && data.recommendations.length > 0) {
        recommendations.innerHTML = `
            <h3>💡 Recomendaciones</h3>
            <ul>
                ${data.recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        `;
        recommendations.classList.add('show');
    }
}

// Crear tarjeta de arquitectura
function createArchCard(arch) {
    const result = arch.data;
    const defects = result.defects_detected || [];
    const isNoDefect = defects.length === 1 && defects[0] === 'Sin defectos';

    return `
        <div class="arch-card">
            <div class="arch-header">
                <div class="arch-icon">${arch.icon}</div>
                <div class="arch-info">
                    <h3>${arch.name}</h3>
                    <div class="arch-type">Deep Learning</div>
                </div>
            </div>

            <div class="prediction-box">
                <div class="product-name">${result.product}</div>
                <div style="color: #666; font-size: 0.9em;">Confianza:</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" data-width="${result.confidence}" style="width: 0%">
                        <span class="confidence-text">${result.confidence}%</span>
                    </div>
                </div>
                <div class="quality-badge badge-${result.quality_status}">
                    ${result.quality_status}
                </div>
            </div>

            <div class="defects-list">
                <h4>🔍 Defectos Detectados:</h4>
                ${defects.map(defect => 
                    `<span class="defect-item ${isNoDefect ? 'no-defect' : ''}">${defect}</span>`
                ).join('')}
            </div>

            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Tiempo</div>
                    <div class="metric-value">${result.inference_time}s</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">ID</div>
                    <div class="metric-value">${result.inspection_id || 'N/A'}</div>
                </div>
            </div>
        </div>
    `;
}

// Verificar estado del servidor al cargar
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('✅ Servidor conectado');
        }
    } catch (error) {
        console.warn('⚠️ Servidor no disponible - Modo offline');
    }
});
    

export function createResultCard(arch, result) {
    let probabilitiesHTML = '';
    arch.classes.forEach((className, i) => {
        probabilitiesHTML += `
            <div class="prob-item">
                <span class="prob-label">${className}</span>
                <div class="prob-bar-container">
                    <div class="prob-bar" data-width="${result.probabilities[i].toFixed(1)}" style="width: 0%">
                        <span class="prob-value">${result.probabilities[i].toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    });

    return `
        <div class="architecture-result">
            <div class="arch-header">
                <span class="arch-icon">${arch.icon}</span>
                <span class="arch-name">${arch.name}</span>
            </div>
            <div class="prediction">
                <div class="prediction-label">Forma Detectada:</div>
                <div class="predicted-class">${result.predictedClass}</div>
                <div class="confidence">Confianza: ${result.confidence.toFixed(2)}%</div>
            </div>
            <div class="probabilities">${probabilitiesHTML}</div>
            <div class="metrics">
                <div class="metric-item">
                    <div class="metric-label">Precisión</div>
                    <div class="metric-value">${result.accuracy.toFixed(1)}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Tiempo</div>
                    <div class="metric-value">${result.inferenceTime}s</div>
                </div>
            </div>
        </div>
    `;
}


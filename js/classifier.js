export function generateRandomResult(arch) {
    const probabilities = [];
    let total = 0;

    for (let i = 0; i < arch.classes.length; i++) {
        const prob = Math.random() * 100;
        probabilities.push(prob);
        total += prob;
    }

    const normalized = probabilities.map(p => (p / total) * 100);
    const maxIndex = normalized.indexOf(Math.max(...normalized));

    return {
        predictedClass: arch.classes[maxIndex],
        confidence: normalized[maxIndex],
        probabilities: normalized,
        accuracy: 85 + Math.random() * 10,
        inferenceTime: (Math.random() * 0.5 + 0.1).toFixed(3)
    };
}



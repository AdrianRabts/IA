export let uploadedImage = null;

export function setupUpload(uploadArea, fileInput, imagePreview, classifyBtn) {
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        handleFile(file, imagePreview, classifyBtn);
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file, imagePreview, classifyBtn);
    });
}

function handleFile(file, imagePreview, classifyBtn) {
    if (file && file.type.startsWith('image/')) {
        uploadedImage = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            classifyBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
}


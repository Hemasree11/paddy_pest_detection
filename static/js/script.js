// Script for Paddy Pest Detection System

// Function to handle file selection and display preview
function handleFileSelect(evt) {
    const fileInput = evt.target;
    const file = fileInput.files[0];
    
    if (file) {
        // Update file input label with filename
        const fileLabel = fileInput.nextElementSibling;
        if (fileLabel) {
            fileLabel.textContent = file.name;
        }
        
        // Preview image if available
        const previewElem = document.getElementById('imagePreview');
        if (previewElem) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewElem.src = e.target.result;
                document.getElementById('imagePreviewContainer').style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    }
}

// Initialize custom file inputs
document.addEventListener('DOMContentLoaded', function() {
    // Add event listeners to file inputs
    const fileInputs = document.querySelectorAll('.custom-file-input');
    fileInputs.forEach(input => {
        input.addEventListener('change', handleFileSelect);
    });
    
    // Initialize any other interactive elements
    
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId !== '#') {
                document.querySelector(targetId).scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
});
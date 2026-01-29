// Soil Health Analytics - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize language toggle
    initializeLanguageToggle();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Set active nav link
    setActiveNav();
    
    // Handle all form submissions
    handleFormSubmissions();
});

function initializeLanguageToggle() {
    // Language switching
    const englishBtn = document.getElementById('englishBtn');
    const tamilBtn = document.getElementById('tamilBtn');
    
    if (englishBtn) {
        englishBtn.addEventListener('click', function(e) {
            e.preventDefault();
            switchLanguage('en');
        });
    }
    
    if (tamilBtn) {
        tamilBtn.addEventListener('click', function(e) {
            e.preventDefault();
            switchLanguage('ta');
        });
    }
}

function switchLanguage(lang) {
    // Show loading indicator
    showToast('Switching language...', 'info');
    
    fetch(`/switch-language/${lang}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Reload the page
                window.location.reload();
            } else {
                showToast('Language switch failed', 'error');
            }
        })
        .catch(error => {
            console.error('Language switch error:', error);
            showToast('Connection error', 'error');
        });
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function setActiveNav() {
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        const linkPath = link.getAttribute('href');
        if (currentPath === linkPath || 
            (currentPath.includes(linkPath) && linkPath !== '/')) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

function handleFormSubmissions() {
    // Handle all forms with AJAX submission
    document.querySelectorAll('form[id$="Form"]').forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const submitBtn = this.querySelector('button[type="submit"]');
            const loadingIndicator = this.querySelector('#loadingIndicator');
            const originalText = submitBtn.innerHTML;
            
            // Show loading
            if (loadingIndicator) loadingIndicator.style.display = 'block';
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Processing...';
            
            // Determine endpoint based on form ID
            let endpoint = '';
            if (this.id === 'soilAnalysisForm') endpoint = '/api/predict/soil';
            else if (this.id === 'cropForm') endpoint = '/api/predict/crop';
            
            // Collect form data
            const formData = new FormData(this);
            const data = {};
            for (let [key, value] of formData.entries()) {
                if (value) data[key] = parseFloat(value) || value;
            }
            
            // Send request
            fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    // Redirect to results page
                    window.location.href = '/results';
                } else {
                    showToast(result.error || 'Error occurred', 'error');
                    if (loadingIndicator) loadingIndicator.style.display = 'none';
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalText;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showToast('Connection error', 'error');
                if (loadingIndicator) loadingIndicator.style.display = 'none';
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
            });
        });
    });
}

function showToast(message, type = 'info') {
    // Remove existing toasts
    document.querySelectorAll('.custom-toast').forEach(toast => toast.remove());
    
    // Create toast
    const toast = document.createElement('div');
    toast.className = `custom-toast alert alert-${type} alert-dismissible fade show`;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 250px;
        max-width: 350px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-radius: 8px;
        padding: 12px 16px;
        animation: slideIn 0.3s ease-out;
    `;
    
    const icons = {
        success: '<i class="fas fa-check-circle me-2"></i>',
        error: '<i class="fas fa-exclamation-circle me-2"></i>',
        warning: '<i class="fas fa-exclamation-triangle me-2"></i>',
        info: '<i class="fas fa-info-circle me-2"></i>'
    };
    
    toast.innerHTML = `
        ${icons[type] || icons.info}
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    document.body.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            const bsAlert = new bootstrap.Alert(toast);
            bsAlert.close();
        }
    }, 5000);
}

// Add CSS for toast animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .custom-toast {
        animation: slideIn 0.3s ease-out;
    }
`;
document.head.appendChild(style);

// Make functions available globally
window.showToast = showToast;
window.switchLanguage = switchLanguage;
window.clearResults = async function() {
    try {
        await fetch('/api/clear-results');
        window.location.href = '/soil-analysis';
    } catch (error) {
        console.error('Error clearing results:', error);
        showToast('Error clearing results', 'error');
    }
};
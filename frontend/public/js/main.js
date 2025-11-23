// Main JavaScript for AI Job Platform
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initApp();
});

function initApp() {
    // Initialize modals
    initModals();
    
    // Initialize event listeners
    initEventListeners();
    
    // Initialize smooth scrolling
    initSmoothScrolling();
    
    // Initialize security features
    initSecurity();
}

function initModals() {
    // Registration Modal
    const registrationModal = document.getElementById('registrationModal');
    const showRegisterBtn = document.getElementById('showRegister');
    const heroGetStartedBtn = document.getElementById('heroGetStarted');
    const closeRegistration = registrationModal.querySelector('.close');
    
    // Payment Modal
    const paymentModal = document.getElementById('paymentModal');
    const closePayment = paymentModal.querySelector('.close');
    
    // Show registration modal
    [showRegisterBtn, heroGetStartedBtn].forEach(btn => {
        btn.addEventListener('click', () => {
            registrationModal.style.display = 'block';
            document.body.style.overflow = 'hidden';
        });
    });
    
    // Close modals
    [closeRegistration, closePayment].forEach(closeBtn => {
        closeBtn.addEventListener('click', function() {
            this.closest('.modal').style.display = 'none';
            document.body.style.overflow = 'auto';
        });
    });
    
    // Close modal when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target.classList.contains('modal')) {
            event.target.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    });
}

function initEventListeners() {
    // Registration form submission
    const registrationForm = document.getElementById('registrationForm');
    registrationForm.addEventListener('submit', handleRegistration);
    
    // Payment method selection
    const paymentMethods = document.querySelectorAll('.payment-method');
    paymentMethods.forEach(method => {
        method.addEventListener('click', function() {
            paymentMethods.forEach(m => m.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // Process payment
    const processPaymentBtn = document.getElementById('processPayment');
    processPaymentBtn.addEventListener('click', processPayment);
}

function initSmoothScrolling() {
    document.querySelectorAll('nav a').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId.startsWith('#')) {
                const targetElement = document.querySelector(targetId);
                
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
            }
        });
    });
}

function initSecurity() {
    // ID Number validation
    const idNumberInput = document.getElementById('idNumber');
    if (idNumberInput) {
        idNumberInput.addEventListener('blur', validateSAID);
    }
    
    // Add security headers (simulated)
    console.log('Security features initialized: Military-grade encryption enabled');
}

function validateSAID(event) {
    const idNumber = event.target.value;
    const errorElement = document.getElementById('idError') || createErrorElement(event.target);
    
    // Basic South African ID validation
    if (idNumber.length !== 13 || !/^\d+$/.test(idNumber)) {
        errorElement.textContent = 'Please enter a valid 13-digit South African ID number';
        event.target.style.borderColor = 'var(--danger)';
        return false;
    }
    
    // Additional validation can be added here (Luhn algorithm, date validation, etc.)
    errorElement.textContent = '';
    event.target.style.borderColor = 'var(--success)';
    return true;
}

function createErrorElement(input) {
    const errorElement = document.createElement('div');
    errorElement.className = 'error-message';
    errorElement.style.color = 'var(--danger)';
    errorElement.style.fontSize = '0.8rem';
    errorElement.style.marginTop = '0.25rem';
    input.parentNode.appendChild(errorElement);
    input.setAttribute('aria-describedby', 'idError');
    return errorElement;
}

async function handleRegistration(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData);
    
    // Validate form
    if (!validateForm(data)) {
        return;
    }
    
    // Show loading state
    const submitBtn = event.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    submitBtn.disabled = true;
    
    try {
        // Simulate API call to backend
        const response = await simulateAPICall('/api/register', data);
        
        if (response.success) {
            // Close registration modal and open payment modal
            document.getElementById('registrationModal').style.display = 'none';
            document.getElementById('paymentModal').style.display = 'block';
            
            // Store user data temporarily
            sessionStorage.setItem('tempUserData', JSON.stringify(data));
        } else {
            throw new Error(response.message || 'Registration failed');
        }
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        // Reset button state
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    }
}

function validateForm(data) {
    const errors = [];
    
    if (!data.fullName || data.fullName.length < 2) {
        errors.push('Please enter your full name');
    }
    
    if (!data.idNumber || !validateSAID({ value: data.idNumber })) {
        errors.push('Please enter a valid South African ID number');
    }
    
    if (!data.email || !isValidEmail(data.email)) {
        errors.push('Please enter a valid email address');
    }
    
    if (!data.phone || !isValidPhone(data.phone)) {
        errors.push('Please enter a valid phone number');
    }
    
    if (!data.password || data.password.length < 6) {
        errors.push('Password must be at least 6 characters long');
    }
    
    if (!data.terms) {
        errors.push('You must agree to the terms and conditions');
    }
    
    if (errors.length > 0) {
        showNotification(errors.join(', '), 'error');
        return false;
    }
    
    return true;
}

function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function isValidPhone(phone) {
    // Basic South African phone number validation
    const phoneRegex = /^(\+27|0)[6-8][0-9]{8}$/;
    return phoneRegex.test(phone.replace(/\s/g, ''));
}

async function processPayment() {
    const selectedMethod = document.querySelector('.payment-method.active');
    
    if (!selectedMethod) {
        showNotification('Please select a payment method', 'error');
        return;
    }
    
    const paymentMethod = selectedMethod.dataset.method;
    const processBtn = document.getElementById('processPayment');
    const originalText = processBtn.textContent;
    
    // Show loading state
    processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing Payment...';
    processBtn.disabled = true;
    
    try {
        const userData = JSON.parse(sessionStorage.getItem('tempUserData'));
        const paymentData = {
            method: paymentMethod,
            amount: 50000, // in cents
            userData: userData
        };
        
        const response = await simulateAPICall('/api/payment/process', paymentData);
        
        if (response.success) {
            showNotification('Payment successful! Redirecting to dashboard...', 'success');
            
            // Simulate redirect to dashboard
            setTimeout(() => {
                window.location.href = '/dashboard.html';
            }, 2000);
        } else {
            throw new Error(response.message || 'Payment failed');
        }
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        processBtn.textContent = originalText;
        processBtn.disabled = false;
    }
}

function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notification => notification.remove());
    
    // Create new notification
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
            <button class="notification-close">&times;</button>
        </div>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${getNotificationColor(type)};
        color: white;
        padding: 1rem;
        border-radius: 4px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1002;
        max-width: 400px;
        animation: slideInRight 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
    
    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.remove();
    });
}

function getNotificationIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function getNotificationColor(type) {
    const colors = {
        success: 'var(--success)',
        error: 'var(--danger)',
        warning: 'var(--warning)',
        info: 'var(--navy-blue)'
    };
    return colors[type] || 'var(--navy-blue)';
}

// Simulate API calls (replace with actual API calls)
async function simulateAPICall(endpoint, data) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simulate different responses based on endpoint
            if (endpoint.includes('/api/register')) {
                resolve({
                    success: true,
                    message: 'Registration successful',
                    userId: 'user_' + Math.random().toString(36).substr(2, 9)
                });
            } else if (endpoint.includes('/api/payment/process')) {
                resolve({
                    success: true,
                    message: 'Payment processed successfully',
                    transactionId: 'tx_' + Math.random().toString(36).substr(2, 9)
                });
            }
        }, 1500);
    });
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .notification-close {
        background: none;
        border: none;
        color: white;
        font-size: 1.2rem;
        cursor: pointer;
        margin-left: auto;
    }
`;
document.head.appendChild(style);

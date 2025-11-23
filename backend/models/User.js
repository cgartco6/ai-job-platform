const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    fullName: {
        type: String,
        required: true,
        trim: true
    },
    email: {
        type: String,
        required: true,
        unique: true,
        lowercase: true,
        trim: true
    },
    idNumber: {
        type: String,
        required: true,
        unique: true,
        match: [/^\d{13}$/, 'Please provide a valid South African ID number']
    },
    phone: {
        type: String,
        required: true,
        trim: true
    },
    password: {
        type: String,
        required: true,
        minlength: 6
    },
    status: {
        type: String,
        enum: ['pending_payment', 'active', 'suspended', 'employed'],
        default: 'pending_payment'
    },
    cvFile: {
        filename: String,
        path: String,
        uploadedAt: Date
    },
    qualifications: [{
        filename: String,
        path: String,
        uploadedAt: Date
    }],
    profilePhoto: {
        filename: String,
        path: String,
        uploadedAt: Date
    },
    enhancedCV: {
        content: String,
        generatedAt: Date
    },
    jobPreferences: {
        industries: [String],
        locations: [String],
        salaryExpectation: Number,
        jobTypes: [String]
    },
    applicationStats: {
        totalApplications: { type: Number, default: 0 },
        applicationsToday: { type: Number, default: 0 },
        interviewsScheduled: { type: Number, default: 0 },
        offersReceived: { type: Number, default: 0 }
    },
    paymentStatus: {
        hasPaid: { type: Boolean, default: false },
        paymentDate: Date,
        paymentMethod: String,
        transactionId: String
    },
    lastActive: Date,
    employmentDate: Date,
    createdAt: {
        type: Date,
        default: Date.now
    },
    updatedAt: {
        type: Date,
        default: Date.now
    }
});

// Update the updatedAt field before saving
userSchema.pre('save', function(next) {
    this.updatedAt = Date.now();
    next();
});

// Index for better query performance
userSchema.index({ email: 1 });
userSchema.index({ idNumber: 1 });
userSchema.index({ status: 1 });
userSchema.index({ 'paymentStatus.hasPaid': 1 });

module.exports = mongoose.model('User', userSchema);

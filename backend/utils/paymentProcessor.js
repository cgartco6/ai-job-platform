const axios = require('axios');
const Payment = require('../models/Payment');
const User = require('../models/User');
const { logger } = require('./logger');

class PaymentProcessor {
    constructor() {
        this.providers = {
            payfast: this.processPayFastPayment.bind(this),
            payshap: this.processPayShapPayment.bind(this),
            eft: this.processEFTPayment.bind(this)
        };
    }

    async processPayment(userId, amount, method, paymentData) {
        try {
            const user = await User.findById(userId);
            if (!user) {
                throw new Error('User not found');
            }

            // Validate South African user
            if (!user.idNumber || user.idNumber.length !== 13) {
                throw new Error('Invalid South African ID number');
            }

            const provider = this.providers[method];
            if (!provider) {
                throw new Error('Unsupported payment method');
            }

            // Create payment record
            const payment = new Payment({
                userId,
                amount,
                currency: 'ZAR',
                method,
                status: 'pending',
                providerData: paymentData
            });

            await payment.save();

            // Process with provider
            const result = await provider(payment, user, paymentData);

            // Update payment status
            payment.status = result.success ? 'completed' : 'failed';
            payment.transactionId = result.transactionId;
            payment.processedAt = new Date();
            await payment.save();

            if (result.success) {
                // Update user payment status
                user.paymentStatus = {
                    hasPaid: true,
                    paymentDate: new Date(),
                    paymentMethod: method,
                    transactionId: result.transactionId,
                    amount: amount
                };
                user.status = 'active';
                await user.save();

                logger.info(`Payment successful for user ${user.email}, transaction: ${result.transactionId}`);
            }

            return result;

        } catch (error) {
            logger.error('Payment processing error:', error);
            throw error;
        }
    }

    async processPayFastPayment(payment, user, paymentData) {
        // PayFast integration
        const payfastData = {
            merchant_id: process.env.PAYFAST_MERCHANT_ID,
            merchant_key: process.env.PAYFAST_MERCHANT_KEY,
            return_url: `${process.env.FRONTEND_URL}/payment/success`,
            cancel_url: `${process.env.FRONTEND_URL}/payment/cancel`,
            notify_url: `${process.env.BACKEND_URL}/api/payments/payfast/webhook`,
            name_first: user.fullName.split(' ')[0],
            name_last: user.fullName.split(' ').slice(1).join(' '),
            email_address: user.email,
            m_payment_id: payment._id.toString(),
            amount: (payment.amount / 100).toFixed(2),
            item_name: 'AI Career Accelerator - Lifetime Access',
            item_description: 'AI-powered job application service'
        };

        try {
            const response = await axios.post(process.env.PAYFAST_URL, payfastData, {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            });

            return {
                success: true,
                transactionId: response.data.payment_id,
                redirectUrl: response.data.redirect_url,
                message: 'Payment initiated successfully'
            };
        } catch (error) {
            logger.error('PayFast payment error:', error);
            return {
                success: false,
                error: 'Payment processing failed'
            };
        }
    }

    async processPayShapPayment(payment, user, paymentData) {
        // PayShap integration for South Africa
        try {
            // Simulate PayShap payment processing
            // In production, integrate with actual PayShap API
            const transactionId = `PSH${Date.now()}${Math.random().toString(36).substr(2, 9)}`.toUpperCase();

            // Simulate API call delay
            await new Promise(resolve => setTimeout(resolve, 2000));

            return {
                success: true,
                transactionId: transactionId,
                message: 'PayShap payment processed successfully'
            };
        } catch (error) {
            logger.error('PayShap payment error:', error);
            return {
                success: false,
                error: 'PayShap payment processing failed'
            };
        }
    }

    async processEFTPayment(payment, user, paymentData) {
        // Direct EFT to FNB account
        try {
            // Generate EFT reference
            const reference = `AIJOB${user.idNumber.slice(-6)}${Date.now().toString().slice(-6)}`;

            // Store EFT payment details
            payment.providerData = {
                ...payment.providerData,
                bank: 'FNB',
                accountNumber: process.env.FNB_ACCOUNT_NUMBER,
                branchCode: process.env.FNB_BRANCH_CODE,
                reference: reference,
                amount: `R${(payment.amount / 100).toFixed(2)}`
            };

            await payment.save();

            return {
                success: true,
                transactionId: reference,
                message: 'EFT payment details generated',
                paymentInstructions: {
                    bank: 'First National Bank (FNB)',
                    accountNumber: process.env.FNB_ACCOUNT_NUMBER,
                    branchCode: process.env.FNB_BRANCH_CODE,
                    reference: reference,
                    amount: `R${(payment.amount / 100).toFixed(2)}`,
                    dueDate: new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
                }
            };
        } catch (error) {
            logger.error('EFT payment error:', error);
            return {
                success: false,
                error: 'EFT payment processing failed'
            };
        }
    }

    async verifyPayment(transactionId, method) {
        try {
            // Verify payment with provider
            // This would make actual API calls to verify payment status
            const payment = await Payment.findOne({ transactionId });
            
            if (!payment) {
                throw new Error('Payment not found');
            }

            // Simulate payment verification
            await new Promise(resolve => setTimeout(resolve, 1000));

            return {
                verified: true,
                status: 'completed',
                payment
            };
        } catch (error) {
            logger.error('Payment verification error:', error);
            throw error;
        }
    }
}

module.exports = new PaymentProcessor();

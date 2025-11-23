const Payment = require('../models/Payment');
const Payout = require('../models/Payout');
const { logger } = require('./logger');

class PayoutDistributor {
    constructor() {
        this.distribution = {
            ownerFNB: 0.40,      // 40% to owner's FNB account
            ownerAfricanBank: 0.15, // 15% to owner's African Bank account
            reserveFNB: 0.20,    // 20% to reserve FNB account
            aiFNB: 0.20,         // 20% to AI development FNB account
            remaining: 0.05      // 5% remains in account (not distributed)
        };
    }

    async calculateWeeklyPayouts() {
        try {
            const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
            
            // Get all completed payments from the past week
            const weeklyPayments = await Payment.find({
                status: 'completed',
                processedAt: { $gte: oneWeekAgo }
            });

            const totalRevenue = weeklyPayments.reduce((sum, payment) => sum + payment.amount, 0);
            
            if (totalRevenue === 0) {
                logger.info('No revenue to distribute this week');
                return null;
            }

            const distributions = {
                totalRevenue: totalRevenue,
                ownerFNB: Math.round(totalRevenue * this.distribution.ownerFNB),
                ownerAfricanBank: Math.round(totalRevenue * this.distribution.ownerAfricanBank),
                reserveFNB: Math.round(totalRevenue * this.distribution.reserveFNB),
                aiFNB: Math.round(totalRevenue * this.distribution.aiFNB),
                remaining: Math.round(totalRevenue * this.distribution.remaining)
            };

            // Create payout record
            const payout = new Payout({
                periodStart: oneWeekAgo,
                periodEnd: new Date(),
                totalRevenue: totalRevenue,
                distributions: distributions,
                status: 'pending'
            });

            await payout.save();
            logger.info(`Weekly payout calculated: R${(totalRevenue / 100).toFixed(2)}`);

            return payout;
        } catch (error) {
            logger.error('Error calculating weekly payouts:', error);
            throw error;
        }
    }

    async processPayouts(payoutId) {
        try {
            const payout = await Payout.findById(payoutId);
            if (!payout) {
                throw new Error('Payout not found');
            }

            if (payout.status !== 'pending') {
                throw new Error('Payout already processed');
            }

            // Process distributions to respective accounts
            const results = await this.transferToAccounts(payout.distributions);

            // Update payout status
            payout.status = 'completed';
            payout.processedAt = new Date();
            payout.transactionResults = results;
            await payout.save();

            logger.info(`Payout ${payoutId} processed successfully`);

            return payout;
        } catch (error) {
            logger.error('Error processing payouts:', error);
            
            // Update payout status to failed
            const payout = await Payout.findById(payoutId);
            if (payout) {
                payout.status = 'failed';
                payout.error = error.message;
                await payout.save();
            }
            
            throw error;
        }
    }

    async transferToAccounts(distributions) {
        const results = {};

        try {
            // Transfer to Owner's FNB Account (40%)
            results.ownerFNB = await this.transferToFNB(
                distributions.ownerFNB,
                process.env.FNB_OWNER_ACCOUNT,
                'Weekly Payout - Owner Share'
            );

            // Transfer to Owner's African Bank Account (15%)
            results.ownerAfricanBank = await this.transferToAfricanBank(
                distributions.ownerAfricanBank,
                process.env.AFRICAN_BANK_ACCOUNT,
                'Weekly Payout - Owner Share'
            );

            // Transfer to Reserve FNB Account (20%)
            results.reserveFNB = await this.transferToFNB(
                distributions.reserveFNB,
                process.env.FNB_RESERVE_ACCOUNT,
                'Weekly Payout - Reserve Fund'
            );

            // Transfer to AI Development FNB Account (20%)
            results.aiFNB = await this.transferToFNB(
                distributions.aiFNB,
                process.env.FNB_AI_ACCOUNT,
                'Weekly Payout - AI Development'
            );

            // 5% remains in the main account (not transferred)

            return results;
        } catch (error) {
            logger.error('Error transferring to accounts:', error);
            throw error;
        }
    }

    async transferToFNB(amount, accountNumber, description) {
        // Simulate FNB bank transfer
        // In production, integrate with FNB API or use bank transfer service
        try {
            await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
            
            const transactionId = `FNB${Date.now()}${Math.random().toString(36).substr(2, 6)}`.toUpperCase();
            
            logger.info(`Transferred R${(amount / 100).toFixed(2)} to FNB account ${accountNumber}`);
            
            return {
                success: true,
                transactionId: transactionId,
                amount: amount,
                account: accountNumber,
                description: description,
                timestamp: new Date()
            };
        } catch (error) {
            logger.error('FNB transfer error:', error);
            return {
                success: false,
                error: error.message,
                amount: amount,
                account: accountNumber
            };
        }
    }

    async transferToAfricanBank(amount, accountNumber, description) {
        // Simulate African Bank transfer
        try {
            await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
            
            const transactionId = `AB${Date.now()}${Math.random().toString(36).substr(2, 6)}`.toUpperCase();
            
            logger.info(`Transferred R${(amount / 100).toFixed(2)} to African Bank account ${accountNumber}`);
            
            return {
                success: true,
                transactionId: transactionId,
                amount: amount,
                account: accountNumber,
                description: description,
                timestamp: new Date()
            };
        } catch (error) {
            logger.error('African Bank transfer error:', error);
            return {
                success: false,
                error: error.message,
                amount: amount,
                account: accountNumber
            };
        }
    }

    getPayoutHistory(limit = 10) {
        return Payout.find()
            .sort({ periodEnd: -1 })
            .limit(limit)
            .exec();
    }

    getTotalDistributed() {
        return Payout.aggregate([
            { $match: { status: 'completed' } },
            {
                $group: {
                    _id: null,
                    totalDistributed: { $sum: '$distributions.totalRevenue' },
                    totalOwnerFNB: { $sum: '$distributions.ownerFNB' },
                    totalOwnerAfricanBank: { $sum: '$distributions.ownerAfricanBank' },
                    totalReserveFNB: { $sum: '$distributions.reserveFNB' },
                    totalAiFNB: { $sum: '$distributions.aiFNB' }
                }
            }
        ]).exec();
    }
}

// Initialize weekly payout cron job
function initPayoutDistribution() {
    const cron = require('node-cron');
    
    // Run every Sunday at 23:59
    cron.schedule('59 23 * * 0', async () => {
        try {
            logger.info('Starting weekly payout distribution...');
            
            const distributor = new PayoutDistributor();
            const payout = await distributor.calculateWeeklyPayouts();
            
            if (payout) {
                await distributor.processPayouts(payout._id);
                logger.info('Weekly payout distribution completed successfully');
            }
        } catch (error) {
            logger.error('Weekly payout distribution failed:', error);
        }
    }, {
        timezone: 'Africa/Johannesburg'
    });

    logger.info('Weekly payout distribution scheduler initialized');
}

module.exports = {
    PayoutDistributor,
    initPayoutDistribution
};

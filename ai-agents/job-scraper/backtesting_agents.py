import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestingAgent:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def generate_historical_data(self, days: int = 365) -> pd.DataFrame:
        """Generate simulated historical job application data for backtesting"""
        logger.info(f"Generating {days} days of historical data...")
        
        dates = pd.date_range(end=datetime.now(), periods=days)
        data = []
        
        for date in dates:
            # Simulate daily application patterns
            daily_applications = np.random.poisson(lam=15)  # Average 15 applications per day
            interviews = max(0, int(daily_applications * np.random.normal(0.1, 0.02)))  # ~10% interview rate
            offers = max(0, int(interviews * np.random.normal(0.3, 0.05)))  # ~30% offer rate from interviews
            
            # Features for ML model
            day_of_week = date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            month = date.month
            quarter = (month - 1) // 3 + 1
            
            data.append({
                'date': date,
                'applications_sent': daily_applications,
                'interviews_scheduled': interviews,
                'offers_received': offers,
                'success_rate': offers / max(1, daily_applications),
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'month': month,
                'quarter': quarter,
                'application_efficiency': interviews / max(1, daily_applications)
            })
            
        return pd.DataFrame(data)
    
    def train_success_predictor(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Train ML model to predict application success"""
        logger.info("Training success prediction model...")
        
        # Prepare features and target
        features = ['applications_sent', 'day_of_week', 'is_weekend', 'month', 'quarter']
        X = historical_data[features]
        y = (historical_data['success_rate'] > historical_data['success_rate'].median()).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'feature_importance': dict(zip(features, self.model.feature_importances_)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def backtest_strategy(self, strategy: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Backtest different application strategies"""
        logger.info(f"Backtesting strategy: {strategy}")
        
        strategies = {
            'aggressive': {'daily_target': 25, 'time_window': '9am-5pm'},
            'moderate': {'daily_target': 15, 'time_window': '10am-4pm'},
            'conservative': {'daily_target': 8, 'time_window': '11am-3pm'},
            'smart': {'daily_target': 20, 'time_window': 'dynamic'}
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        strategy_config = strategies[strategy]
        
        # Simulate strategy performance
        simulated_data = historical_data.copy()
        simulated_data['strategy_applications'] = min(
            strategy_config['daily_target'],
            simulated_data['applications_sent'] * 1.2  # Assume strategy improves capacity
        )
        
        # Calculate metrics
        total_applications = simulated_data['strategy_applications'].sum()
        total_interviews = simulated_data['interviews_scheduled'].sum()
        total_offers = simulated_data['offers_received'].sum()
        
        efficiency = total_interviews / total_applications if total_applications > 0 else 0
        success_rate = total_offers / total_applications if total_applications > 0 else 0
        
        return {
            'strategy': strategy,
            'total_applications': int(total_applications),
            'total_interviews': int(total_interviews),
            'total_offers': int(total_offers),
            'efficiency_rate': efficiency,
            'success_rate': success_rate,
            'estimated_time_to_employment': max(1, int(1 / success_rate)) if success_rate > 0 else float('inf'),
            'config': strategy_config
        }
    
    def optimize_application_timing(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize best times to send applications"""
        logger.info("Optimizing application timing...")
        
        # Analyze performance by day of week
        daily_performance = historical_data.groupby('day_of_week').agg({
            'success_rate': 'mean',
            'application_efficiency': 'mean',
            'applications_sent': 'sum'
        }).reset_index()
        
        best_day = daily_performance.loc[daily_performance['success_rate'].idxmax()]
        worst_day = daily_performance.loc[daily_performance['success_rate'].idxmin()]
        
        return {
            'best_day': {
                'day_number': int(best_day['day_of_week']),
                'day_name': self._get_day_name(int(best_day['day_of_week'])),
                'success_rate': float(best_day['success_rate']),
                'efficiency': float(best_day['application_efficiency'])
            },
            'worst_day': {
                'day_number': int(worst_day['day_of_week']),
                'day_name': self._get_day_name(int(worst_day['day_of_week'])),
                'success_rate': float(worst_day['success_rate']),
                'efficiency': float(worst_day['application_efficiency'])
            },
            'recommendations': self._generate_timing_recommendations(daily_performance)
        }
    
    def _get_day_name(self, day_number: int) -> str:
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return days[day_number]
    
    def _generate_timing_recommendations(self, daily_data: pd.DataFrame) -> List[str]:
        """Generate timing recommendations based on data analysis"""
        recommendations = []
        
        # Find best performing days
        best_days = daily_data.nlargest(2, 'success_rate')
        best_day_names = [self._get_day_name(int(day)) for day in best_days['day_of_week']]
        
        recommendations.append(
            f"Focus application efforts on {', '.join(best_day_names)} for highest success rates"
        )
        
        # Weekend analysis
        weekend_data = daily_data[daily_data['day_of_week'].isin([5, 6])]
        weekday_data = daily_data[~daily_data['day_of_week'].isin([5, 6])]
        
        if len(weekend_data) > 0 and len(weekday_data) > 0:
            weekend_success = weekend_data['success_rate'].mean()
            weekday_success = weekday_data['success_rate'].mean()
            
            if weekday_success > weekend_success:
                recommendations.append("Weekday applications generally perform better than weekend applications")
            else:
                recommendations.append("Consider including weekend applications in your strategy")
        
        return recommendations
    
    def run_comprehensive_backtest(self) -> Dict[str, Any]:
        """Run comprehensive backtesting analysis"""
        logger.info("Starting comprehensive backtest...")
        
        # Generate historical data
        historical_data = self.generate_historical_data(180)  # 6 months of data
        
        # Train prediction model
        model_metrics = self.train_success_predictor(historical_data)
        
        # Test different strategies
        strategies = ['aggressive', 'moderate', 'conservative', 'smart']
        strategy_results = {}
        
        for strategy in strategies:
            strategy_results[strategy] = self.backtest_strategy(strategy, historical_data)
        
        # Optimize timing
        timing_optimization = self.optimize_application_timing(historical_data)
        
        # Find best strategy
        best_strategy = max(strategy_results.values(), key=lambda x: x['success_rate'])
        
        return {
            'backtest_period': '180 days',
            'total_data_points': len(historical_data),
            'model_performance': model_metrics,
            'strategy_comparison': strategy_results,
            'timing_optimization': timing_optimization,
            'recommended_strategy': best_strategy['strategy'],
            'expected_success_rate': best_strategy['success_rate'],
            'estimated_time_to_employment': best_strategy['estimated_time_to_employment']
        }

# Example usage
if __name__ == "__main__":
    agent = BacktestingAgent()
    results = agent.run_comprehensive_backtest()
    
    print("Backtesting Results:")
    print(f"Recommended Strategy: {results['recommended_strategy']}")
    print(f"Expected Success Rate: {results['expected_success_rate']:.2%}")
    print(f"Estimated Time to Employment: {results['estimated_time_to_employment']} days")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ApplicationBacktestingAgent:
    def __init__(self):
        self.performance_metrics = {}
        self.optimization_history = []
        
    def simulate_application_cycle(self, user_profile: Dict, strategy: str) -> Dict[str, Any]:
        """Simulate complete application cycle for a user"""
        logger.info(f"Simulating application cycle for {user_profile['experience_level']} developer")
        
        # Base parameters based on strategy
        strategy_params = self._get_strategy_parameters(strategy)
        
        # Generate simulated applications
        applications = self._generate_applications(user_profile, strategy_params)
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(applications, user_profile)
        
        # Store results
        self.performance_metrics[f"{user_profile['experience_level']}_{strategy}"] = metrics
        
        return {
            'user_profile': user_profile,
            'strategy': strategy,
            'applications_generated': len(applications),
            'performance_metrics': metrics,
            'simulation_date': datetime.now().isoformat()
        }
    
    def _get_strategy_parameters(self, strategy: str) -> Dict[str, Any]:
        """Get parameters for different application strategies"""
        strategies = {
            'volume_focused': {
                'daily_applications': 20,
                'matching_threshold': 0.6,
                'cover_letter_personalization': 'minimal',
                'follow_up_aggressiveness': 'high'
            },
            'quality_focused': {
                'daily_applications': 8,
                'matching_threshold': 0.85,
                'cover_letter_personalization': 'high',
                'follow_up_aggressiveness': 'moderate'
            },
            'balanced': {
                'daily_applications': 15,
                'matching_threshold': 0.75,
                'cover_letter_personalization': 'moderate',
                'follow_up_aggressiveness': 'moderate'
            },
            'ai_optimized': {
                'daily_applications': 'dynamic',
                'matching_threshold': 'adaptive',
                'cover_letter_personalization': 'ai_enhanced',
                'follow_up_aggressiveness': 'smart'
            }
        }
        return strategies.get(strategy, strategies['balanced'])
    
    def _generate_applications(self, user_profile: Dict, strategy: Dict) -> List[Dict]:
        """Generate simulated job applications"""
        applications = []
        days_simulated = 30  # Simulate 30 days
        
        for day in range(days_simulated):
            daily_applications = self._get_daily_application_count(strategy, day)
            
            for _ in range(daily_applications):
                application = self._create_simulated_application(user_profile, strategy, day)
                applications.append(application)
                
        return applications
    
    def _get_daily_application_count(self, strategy: Dict, day: int) -> int:
        """Determine how many applications to send each day"""
        base_count = strategy['daily_applications']
        
        if base_count == 'dynamic':
            # AI-optimized dynamic count
            if day < 7:  # First week: ramp up
                return min(25, 10 + day * 2)
            elif day < 21:  # Middle weeks: maintain
                return 20
            else:  # Final week: optimize
                return 15
        else:
            return base_count
    
    def _create_simulated_application(self, user_profile: Dict, strategy: Dict, day: int) -> Dict:
        """Create a single simulated application"""
        # Simulate job match quality
        match_quality = np.random.beta(2, 2)  # Beta distribution centered around 0.5
        
        # Adjust based on strategy threshold
        if strategy['matching_threshold'] != 'adaptive':
            threshold = strategy['matching_threshold']
            if match_quality < threshold:
                match_quality = np.random.uniform(threshold, 1.0)
        
        # Calculate application quality score
        app_quality = self._calculate_application_quality(user_profile, strategy, match_quality)
        
        # Simulate outcomes based on quality
        response_probability = min(0.9, 0.1 + (app_quality * 0.7))
        interview_probability = min(0.6, 0.05 + (app_quality * 0.5))
        offer_probability = min(0.3, 0.02 + (app_quality * 0.25))
        
        # Generate outcomes
        got_response = np.random.random() < response_probability
        got_interview = got_response and (np.random.random() < interview_probability)
        got_offer = got_interview and (np.random.random() < offer_probability)
        
        return {
            'application_id': f"app_{day}_{len(str(day))}",
            'submission_date': datetime.now() - timedelta(days=30-day),
            'match_quality': match_quality,
            'application_quality': app_quality,
            'response_received': got_response,
            'interview_scheduled': got_interview,
            'offer_received': got_offer,
            'response_time_days': np.random.lognormal(2, 1) if got_response else None,
            'strategy_used': strategy
        }
    
    def _calculate_application_quality(self, user_profile: Dict, strategy: Dict, match_quality: float) -> float:
        """Calculate the quality score of an application"""
        base_quality = match_quality
        
        # Adjust for personalization level
        personalization_factors = {
            'minimal': 0.8,
            'moderate': 0.9,
            'high': 1.0,
            'ai_enhanced': 1.1
        }
        personalization_boost = personalization_factors.get(
            strategy['cover_letter_personalization'], 0.9
        )
        
        # Adjust for experience level
        experience_factors = {
            'entry': 0.7,
            'mid': 0.9,
            'senior': 1.1,
            'executive': 1.2
        }
        experience_boost = experience_factors.get(user_profile['experience_level'], 0.9)
        
        # Calculate final quality
        final_quality = base_quality * personalization_boost * experience_boost
        return min(1.0, final_quality)  # Cap at 1.0
    
    def _calculate_performance_metrics(self, applications: List[Dict], user_profile: Dict) -> Dict[str, Any]:
        """Calculate performance metrics from applications"""
        if not applications:
            return {}
            
        df = pd.DataFrame(applications)
        
        total_applications = len(df)
        responses = df['response_received'].sum()
        interviews = df['interview_scheduled'].sum()
        offers = df['offer_received'].sum()
        
        response_rate = responses / total_applications
        interview_rate = interviews / max(1, responses)
        offer_rate = offers / max(1, interviews)
        overall_success_rate = offers / total_applications
        
        # Calculate efficiency metrics
        avg_quality = df['application_quality'].mean()
        avg_match_quality = df['match_quality'].mean()
        
        # Time to first offer
        offer_apps = df[df['offer_received']]
        if len(offer_apps) > 0:
            first_offer_day = offer_apps['submission_date'].min()
            days_to_offer = (first_offer_day - df['submission_date'].min()).days
        else:
            days_to_offer = None
        
        return {
            'total_applications': total_applications,
            'response_rate': response_rate,
            'interview_rate': interview_rate,
            'offer_rate': offer_rate,
            'overall_success_rate': overall_success_rate,
            'average_application_quality': avg_quality,
            'average_match_quality': avg_match_quality,
            'days_to_first_offer': days_to_offer,
            'applications_per_offer': total_applications / max(1, offers),
            'efficiency_score': overall_success_rate * avg_quality
        }
    
    def compare_strategies(self, user_profile: Dict) -> Dict[str, Any]:
        """Compare all strategies for a user profile"""
        logger.info(f"Comparing strategies for {user_profile['experience_level']} level")
        
        strategies = ['volume_focused', 'quality_focused', 'balanced', 'ai_optimized']
        results = {}
        
        for strategy in strategies:
            result = self.simulate_application_cycle(user_profile, strategy)
            results[strategy] = result['performance_metrics']
        
        # Find best strategy
        best_strategy = max(results.items(), key=lambda x: x[1]['efficiency_score'])
        
        return {
            'user_profile': user_profile,
            'strategy_comparison': results,
            'recommended_strategy': best_strategy[0],
            'expected_success_rate': best_strategy[1]['overall_success_rate'],
            'expected_applications_needed': int(1 / max(0.001, best_strategy[1]['overall_success_rate']))
        }
    
    def optimize_application_strategy(self, user_profiles: List[Dict]) -> Dict[str, Any]:
        """Optimize application strategy across different user profiles"""
        logger.info("Optimizing application strategies across user profiles")
        
        all_results = {}
        
        for profile in user_profiles:
            comparison = self.compare_strategies(profile)
            all_results[profile['experience_level']] = comparison
        
        # Generate overall recommendations
        overall_recommendations = self._generate_overall_recommendations(all_results)
        
        return {
            'profile_specific_results': all_results,
            'overall_recommendations': overall_recommendations,
            'optimization_date': datetime.now().isoformat()
        }
    
    def _generate_overall_recommendations(self, results: Dict) -> List[str]:
        """Generate overall recommendations based on all profiles"""
        recommendations = []
        
        # Analyze best strategies by experience level
        strategy_by_level = {}
        for level, data in results.items():
            strategy_by_level[level] = data['recommended_strategy']
        
        recommendations.append(
            f"Strategy varies by experience level: {', '.join([f'{k}: {v}' for k, v in strategy_by_level.items()])}"
        )
        
        # Common success factors
        if all('ai_optimized' in strategy for strategy in strategy_by_level.values()):
            recommendations.append("AI-optimized strategy performs well across all experience levels")
        
        # Volume vs quality trade-off analysis
        volume_strategies = sum(1 for s in strategy_by_level.values() if 'volume' in s)
        quality_strategies = sum(1 for s in strategy_by_level.values() if 'quality' in s)
        
        if volume_strategies > quality_strategies:
            recommendations.append("Volume-focused approaches generally outperform quality-focused ones in current market")
        else:
            recommendations.append("Quality-focused approaches show better efficiency in current market conditions")
        
        return recommendations

# Example usage with different user profiles
if __name__ == "__main__":
    agent = ApplicationBacktestingAgent()
    
    # Test with different user profiles
    user_profiles = [
        {
            'experience_level': 'entry',
            'skills': ['Python', 'JavaScript', 'SQL'],
            'education': 'BSc Computer Science',
            'location': 'Cape Town'
        },
        {
            'experience_level': 'mid',
            'skills': ['Python', 'Django', 'AWS', 'Docker'],
            'education': 'BSc Computer Science',
            'location': 'Johannesburg'
        },
        {
            'experience_level': 'senior',
            'skills': ['Python', 'Machine Learning', 'System Architecture', 'Team Leadership'],
            'education': 'MSc Computer Science',
            'location': 'Remote'
        }
    ]
    
    results = agent.optimize_application_strategy(user_profiles)
    
    print("Application Strategy Optimization Results:")
    for level, data in results['profile_specific_results'].items():
        print(f"\n{level.upper()} LEVEL:")
        print(f"Recommended Strategy: {data['recommended_strategy']}")
        print(f"Expected Success Rate: {data['expected_success_rate']:.2%}")
        print(f"Expected Applications Needed: {data['expected_applications_needed']}")
    
    print("\nOverall Recommendations:")
    for rec in results['overall_recommendations']:
        print(f"- {rec}")

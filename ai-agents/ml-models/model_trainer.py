import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import os
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    model_size: float

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.vectorizers = {}
        self.label_encoders = {}
        self.performance_history = []
        
    def train_job_matching_model(self, jobs_data: List[Dict], user_profiles: List[Dict]) -> Dict[str, Any]:
        """Train model to match jobs with user profiles"""
        logger.info("Training job matching model...")
        
        try:
            # Prepare features and labels
            X, y = self._prepare_job_matching_features(jobs_data, user_profiles)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train multiple models
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'svm': SVC(kernel='rbf', probability=True, random_state=42)
            }
            
            best_model = None
            best_score = 0
            training_results = {}
            
            for name, model in models.items():
                logger.info(f"Training {name}...")
                start_time = datetime.now()
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Store performance
                performance = ModelPerformance(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    training_time=training_time,
                    model_size=0  # Will be calculated after saving
                )
                
                training_results[name] = {
                    'model': model,
                    'performance': performance,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                # Update best model
                if f1 > best_score:
                    best_score = f1
                    best_model = name
            
            # Save best model
            best_model_info = training_results[best_model]
            self.models['job_matcher'] = best_model_info['model']
            
            # Calculate model size
            model_size = self._calculate_model_size(best_model_info['model'])
            best_model_info['performance'].model_size = model_size
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'model_type': 'job_matcher',
                'best_model': best_model,
                'performance': best_model_info['performance'].__dict__,
                'training_results': {
                    name: {k: v for k, v in info.items() if k != 'model'} 
                    for name, info in training_results.items()
                }
            })
            
            logger.info(f"Job matching model training completed. Best model: {best_model}, F1: {best_score:.3f}")
            
            return {
                'success': True,
                'best_model': best_model,
                'performance': best_model_info['performance'].__dict__,
                'all_models': training_results,
                'feature_importance': self._get_feature_importance(best_model_info['model'], X.columns)
            }
            
        except Exception as e:
            logger.error(f"Error training job matching model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_success_prediction_model(self, application_history: List[Dict]) -> Dict[str, Any]:
        """Train model to predict application success"""
        logger.info("Training success prediction model...")
        
        try:
            # Prepare features and labels
            X, y = self._prepare_success_prediction_features(application_history)
            
            if len(X) < 100:
                logger.warning("Insufficient data for success prediction model")
                return {
                    'success': False,
                    'error': 'Insufficient training data'
                }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            start_time = datetime.now()
            
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store model
            self.models['success_predictor'] = model
            
            # Calculate model size
            model_size = self._calculate_model_size(model)
            
            performance = ModelPerformance(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_time=training_time,
                model_size=model_size
            )
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'model_type': 'success_predictor',
                'performance': performance.__dict__,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            })
            
            logger.info(f"Success prediction model training completed. Accuracy: {accuracy:.3f}")
            
            return {
                'success': True,
                'performance': performance.__dict__,
                'feature_importance': self._get_feature_importance(model, X.columns),
                'prediction_examples': self._get_prediction_examples(X_test, y_test, y_pred, y_pred_proba)
            }
            
        except Exception as e:
            logger.error(f"Error training success prediction model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_job_matching_features(self, jobs_data: List[Dict], user_profiles: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for job matching model"""
        features = []
        labels = []
        
        for job in jobs_data:
            for user in user_profiles:
                # Calculate matching score (simplified)
                match_score = self._calculate_job_user_match(job, user)
                
                # Create features
                feature_vector = self._create_job_matching_features(job, user)
                
                # Determine label (1 for good match, 0 for poor match)
                label = 1 if match_score > 0.7 else 0
                
                features.append(feature_vector)
                labels.append(label)
        
        return pd.DataFrame(features), pd.Series(labels)
    
    def _calculate_job_user_match(self, job: Dict, user: Dict) -> float:
        """Calculate match score between job and user"""
        score = 0.0
        
        # Skills match
        job_skills = set(job.get('required_skills', []))
        user_skills = set(user.get('skills', []))
        if job_skills:
            skill_match = len(job_skills.intersection(user_skills)) / len(job_skills)
            score += skill_match * 0.4
        
        # Experience match
        job_experience = job.get('required_experience', 0)
        user_experience = user.get('experience_years', 0)
        if job_experience > 0:
            experience_match = min(user_experience / job_experience, 1.0)
            score += experience_match * 0.3
        
        # Location match
        job_location = job.get('location', '').lower()
        user_location = user.get('location', '').lower()
        if job_location and user_location:
            location_match = 1.0 if job_location in user_location or user_location in job_location else 0.0
            score += location_match * 0.2
        
        # Salary expectations
        job_salary = job.get('salary', 0)
        user_expectation = user.get('salary_expectation', 0)
        if job_salary > 0 and user_expectation > 0:
            salary_match = 1.0 if job_salary >= user_expectation * 0.8 else 0.5
            score += salary_match * 0.1
        
        return min(score, 1.0)
    
    def _create_job_matching_features(self, job: Dict, user: Dict) -> Dict[str, float]:
        """Create feature vector for job matching"""
        features = {}
        
        # Skills features
        job_skills = set(job.get('required_skills', []))
        user_skills = set(user.get('skills', []))
        features['skills_match_ratio'] = len(job_skills.intersection(user_skills)) / max(len(job_skills), 1)
        features['user_skills_count'] = len(user_skills)
        features['job_skills_count'] = len(job_skills)
        
        # Experience features
        features['experience_gap'] = user.get('experience_years', 0) - job.get('required_experience', 0)
        features['experience_ratio'] = user.get('experience_years', 0) / max(job.get('required_experience', 1), 1)
        
        # Location features
        features['same_province'] = 1.0 if self._check_same_province(
            job.get('location', ''), user.get('location', '')
        ) else 0.0
        
        # Salary features
        features['salary_adequacy'] = job.get('salary', 0) / max(user.get('salary_expectation', 1), 1)
        
        # Industry features
        features['industry_match'] = 1.0 if job.get('industry') == user.get('industry') else 0.0
        
        # Company size features (simplified)
        features['company_size_match'] = self._calculate_company_size_match(job, user)
        
        return features
    
    def _prepare_success_prediction_features(self, application_history: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for success prediction model"""
        features = []
        labels = []
        
        for application in application_history:
            if 'success' not in application:
                continue
                
            feature_vector = self._create_success_prediction_features(application)
            label = 1 if application['success'] else 0
            
            features.append(feature_vector)
            labels.append(label)
        
        return pd.DataFrame(features), pd.Series(labels)
    
    def _create_success_prediction_features(self, application: Dict) -> Dict[str, float]:
        """Create feature vector for success prediction"""
        features = {}
        
        # Application quality features
        features['cv_match_score'] = application.get('cv_match_score', 0)
        features['cover_letter_quality'] = application.get('cover_letter_quality', 0)
        features['application_completeness'] = application.get('completeness_score', 0)
        
        # Timing features
        features['days_after_posting'] = application.get('days_after_posting', 0)
        features['application_time_of_day'] = application.get('application_hour', 12) / 24.0
        
        # Job market features
        features['job_competitiveness'] = application.get('competitiveness_score', 0)
        features['company_response_rate'] = application.get('company_response_rate', 0)
        
        # Historical features
        features['user_success_rate'] = application.get('user_success_rate', 0)
        features['previous_applications'] = application.get('previous_applications_count', 0)
        
        # Match quality features
        features['skills_match_ratio'] = application.get('skills_match_ratio', 0)
        features['experience_match'] = application.get('experience_match', 0)
        
        return features
    
    def _check_same_province(self, location1: str, location2: str) -> bool:
        """Check if two locations are in the same province"""
        provinces = ['western cape', 'eastern cape', 'northern cape', 'free state', 
                    'kwa zulu natal', 'north west', 'gauteng', 'mpumalanga', 'limpopo']
        
        location1 = location1.lower()
        location2 = location2.lower()
        
        for province in provinces:
            if province in location1 and province in location2:
                return True
        
        return False
    
    def _calculate_company_size_match(self, job: Dict, user: Dict) -> float:
        """Calculate company size preference match"""
        # Simplified implementation
        company_size = job.get('company_size', 'medium')
        user_preference = user.get('preferred_company_size', 'medium')
        
        size_mapping = {'small': 1, 'medium': 2, 'large': 3, 'enterprise': 4}
        
        company_size_num = size_mapping.get(company_size, 2)
        user_preference_num = size_mapping.get(user_preference, 2)
        
        size_diff = abs(company_size_num - user_preference_num)
        return 1.0 - (size_diff / 3.0)  # Normalize to 0-1
    
    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                return {k: v for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)}
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def _calculate_model_size(self, model) -> float:
        """Calculate model size in KB"""
        try:
            # Save model to temporary file and check size
            temp_path = 'temp_model.pkl'
            with open(temp_path, 'wb') as f:
                pickle.dump(model, f)
            
            size_kb = os.path.getsize(temp_path) / 1024.0
            
            # Clean up
            os.remove(temp_path)
            
            return size_kb
        except:
            return 0.0
    
    def _get_prediction_examples(self, X_test, y_test, y_pred, y_pred_proba) -> List[Dict]:
        """Get prediction examples for analysis"""
        examples = []
        
        for i in range(min(5, len(X_test))):
            examples.append({
                'actual': int(y_test.iloc[i]),
                'predicted': int(y_pred[i]),
                'confidence': float(y_pred_proba[i][1]),
                'features': X_test.iloc[i].to_dict()
            })
        
        return examples
    
    def predict_job_match(self, job: Dict, user: Dict) -> Dict[str, Any]:
        """Predict job match for a specific job and user"""
        try:
            if 'job_matcher' not in self.models:
                return {'error': 'Job matching model not trained'}
            
            # Create features
            features = self._create_job_matching_features(job, user)
            feature_vector = pd.DataFrame([features])
            
            # Predict
            model = self.models['job_matcher']
            match_probability = model.predict_proba(feature_vector)[0][1]
            match_score = self._calculate_job_user_match(job, user)
            
            # Combine model prediction with rule-based score
            final_score = (match_probability + match_score) / 2
            
            return {
                'match_score': final_score,
                'model_confidence': match_probability,
                'rule_based_score': match_score,
                'recommendation': 'Apply' if final_score > 0.7 else 'Consider' if final_score > 0.5 else 'Skip',
                'key_factors': self._get_match_factors(job, user)
            }
            
        except Exception as e:
            logger.error(f"Error predicting job match: {str(e)}")
            return {'error': str(e)}
    
    def predict_application_success(self, application: Dict) -> Dict[str, Any]:
        """Predict success probability for an application"""
        try:
            if 'success_predictor' not in self.models:
                return {'error': 'Success prediction model not trained'}
            
            # Create features
            features = self._create_success_prediction_features(application)
            feature_vector = pd.DataFrame([features])
            
            # Predict
            model = self.models['success_predictor']
            success_probability = model.predict_proba(feature_vector)[0][1]
            
            return {
                'success_probability': success_probability,
                'prediction': 'High chance' if success_probability > 0.7 else 
                             'Moderate chance' if success_probability > 0.4 else 'Low chance',
                'confidence': model.predict_proba(feature_vector)[0].max(),
                'recommendations': self._get_success_recommendations(application, success_probability)
            }
            
        except Exception as e:
            logger.error(f"Error predicting application success: {str(e)}")
            return {'error': str(e)}
    
    def _get_match_factors(self, job: Dict, user: Dict) -> List[str]:
        """Get key factors affecting job match"""
        factors = []
        
        # Skills match
        job_skills = set(job.get('required_skills', []))
        user_skills = set(user.get('skills', []))
        common_skills = job_skills.intersection(user_skills)
        
        if common_skills:
            factors.append(f"Shared skills: {', '.join(list(common_skills)[:3])}")
        else:
            factors.append("Skill mismatch")
        
        # Experience
        job_exp = job.get('required_experience', 0)
        user_exp = user.get('experience_years', 0)
        
        if user_exp >= job_exp:
            factors.append("Meets experience requirements")
        else:
            factors.append(f"Needs {job_exp - user_exp} more years of experience")
        
        # Location
        if self._check_same_province(job.get('location', ''), user.get('location', '')):
            factors.append("Same province")
        else:
            factors.append("Different location")
        
        return factors
    
    def _get_success_recommendations(self, application: Dict, success_probability: float) -> List[str]:
        """Get recommendations to improve success probability"""
        recommendations = []
        
        if success_probability < 0.4:
            recommendations.extend([
                "Enhance your CV to better match job requirements",
                "Improve your cover letter quality",
                "Apply to jobs sooner after they're posted",
                "Focus on jobs with better skill matches"
            ])
        elif success_probability < 0.7:
            recommendations.extend([
                "Tailor your application to specific job requirements",
                "Highlight your most relevant experience",
                "Follow up with the employer after applying"
            ])
        else:
            recommendations.append("Your application looks strong - good luck!")
        
        return recommendations
    
    def save_models(self, directory: str = 'models'):
        """Save trained models to disk"""
        try:
            os.makedirs(directory, exist_ok=True)
            
            for name, model in self.models.items():
                model_path = os.path.join(directory, f'{name}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save performance history
            performance_path = os.path.join(directory, 'performance_history.json')
            with open(performance_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            
            logger.info(f"Models saved to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, directory: str = 'models'):
        """Load trained models from disk"""
        try:
            for filename in os.listdir(directory):
                if filename.endswith('.pkl'):
                    name = filename[:-4]  # Remove .pkl extension
                    model_path = os.path.join(directory, filename)
                    
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            # Load performance history
            performance_path = os.path.join(directory, 'performance_history.json')
            if os.path.exists(performance_path):
                with open(performance_path, 'r') as f:
                    self.performance_history = json.load(f)
            
            logger.info(f"Models loaded from {directory}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not self.performance_history:
            return {'error': 'No performance data available'}
        
        latest_performance = self.performance_history[-1]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'latest_training': latest_performance,
            'total_training_sessions': len(self.performance_history),
            'model_summary': {}
        }
        
        # Calculate average performance
        for model_type in ['job_matcher', 'success_predictor']:
            model_performances = [
                p for p in self.performance_history 
                if p.get('model_type') == model_type
            ]
            
            if model_performances:
                latest = model_performances[-1]['performance']
                report['model_summary'][model_type] = {
                    'latest_accuracy': latest.get('accuracy', 0),
                    'latest_f1_score': latest.get('f1_score', 0),
                    'training_count': len(model_performances),
                    'status': 'Trained'
                }
            else:
                report['model_summary'][model_type] = {
                    'status': 'Not trained'
                }
        
        return report

# Example usage
if __name__ == "__main__":
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Sample data for demonstration
    sample_jobs = [
        {
            'title': 'Software Developer',
            'required_skills': ['Python', 'JavaScript', 'SQL'],
            'required_experience': 3,
            'location': 'Cape Town',
            'salary': 500000,
            'industry': 'Technology',
            'company_size': 'medium'
        }
    ]
    
    sample_users = [
        {
            'skills': ['Python', 'JavaScript', 'React', 'Node.js'],
            'experience_years': 4,
            'location': 'Cape Town',
            'salary_expectation': 450000,
            'industry': 'Technology',
            'preferred_company_size': 'medium'
        }
    ]
    
    # Train job matching model
    result = trainer.train_job_matching_model(sample_jobs, sample_users)
    print(f"Job matching training result: {result['success']}")
    
    # Save models
    trainer.save_models()
    
    # Generate performance report
    report = trainer.get_performance_report()
    print(f"Performance report: {report}")

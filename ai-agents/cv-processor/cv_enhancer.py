import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from typing import Dict, List, Any
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVEnhancer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        self.industry_keywords = self._load_industry_keywords()
        self.skill_patterns = self._load_skill_patterns()
        
    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """Load industry-specific keywords for optimization"""
        return {
            "technology": [
                "software development", "programming", "cloud computing", "devops",
                "machine learning", "artificial intelligence", "data analysis",
                "web development", "mobile development", "cybersecurity"
            ],
            "finance": [
                "financial analysis", "investment banking", "risk management",
                "accounting", "auditing", "wealth management", "trading"
            ],
            "marketing": [
                "digital marketing", "social media", "content creation",
                "brand management", "SEO", "SEM", "market research"
            ],
            "healthcare": [
                "patient care", "medical research", "healthcare administration",
                "clinical trials", "pharmaceutical", "nursing"
            ]
        }
    
    def _load_skill_patterns(self) -> Dict[str, List[str]]:
        """Load skill patterns for extraction"""
        return {
            "programming": ["python", "javascript", "java", "c++", "c#", "ruby", "go", "rust"],
            "frameworks": ["react", "angular", "vue", "django", "flask", "spring", "laravel"],
            "databases": ["mysql", "postgresql", "mongodb", "redis", "oracle"],
            "tools": ["git", "docker", "kubernetes", "jenkins", "aws", "azure"]
        }
    
    def enhance_cv(self, cv_data: Dict[str, Any], target_industry: str = None) -> Dict[str, Any]:
        """Enhance CV with AI-powered improvements"""
        logger.info(f"Enhancing CV for {cv_data.get('full_name', 'unknown')}")
        
        enhanced_cv = cv_data.copy()
        
        # Analyze current CV
        analysis = self._analyze_cv(cv_data)
        
        # Generate professional summary
        enhanced_cv['professional_summary'] = self._generate_professional_summary(cv_data, target_industry)
        
        # Optimize skills section
        enhanced_cv['skills'] = self._optimize_skills_section(cv_data.get('skills', []), target_industry)
        
        # Enhance work experience
        enhanced_cv['work_experience'] = self._enhance_work_experience(cv_data.get('work_experience', []))
        
        # Add ATS optimization
        enhanced_cv['ats_optimized'] = self._optimize_for_ats(cv_data, target_industry)
        
        # Calculate enhancement score
        enhanced_cv['enhancement_score'] = self._calculate_enhancement_score(analysis, enhanced_cv)
        
        # Add metadata
        enhanced_cv['enhancement_date'] = datetime.now().isoformat()
        enhanced_cv['ai_model_used'] = 'cv_enhancer_v2'
        enhanced_cv['target_industry'] = target_industry
        
        logger.info(f"CV enhancement completed. Score: {enhanced_cv['enhancement_score']}")
        
        return enhanced_cv
    
    def _analyze_cv(self, cv_data: Dict) -> Dict[str, Any]:
        """Analyze CV content and structure"""
        analysis = {
            "word_count": 0,
            "section_completeness": 0,
            "keyword_density": {},
            "readability_score": 0,
            "action_verb_usage": 0
        }
        
        # Calculate word count
        text_content = self._extract_text_content(cv_data)
        analysis["word_count"] = len(text_content.split())
        
        # Check section completeness
        required_sections = ['professional_summary', 'work_experience', 'skills', 'education']
        present_sections = [section for section in required_sections if section in cv_data and cv_data[section]]
        analysis["section_completeness"] = len(present_sections) / len(required_sections)
        
        # Analyze keyword density
        analysis["keyword_density"] = self._calculate_keyword_density(text_content)
        
        # Calculate readability (simplified)
        analysis["readability_score"] = self._calculate_readability(text_content)
        
        # Check action verb usage
        analysis["action_verb_usage"] = self._check_action_verbs(text_content)
        
        return analysis
    
    def _extract_text_content(self, cv_data: Dict) -> str:
        """Extract all text content from CV"""
        text_parts = []
        
        if 'professional_summary' in cv_data:
            text_parts.append(cv_data['professional_summary'])
        
        if 'work_experience' in cv_data:
            for job in cv_data['work_experience']:
                text_parts.append(job.get('description', ''))
        
        if 'skills' in cv_data:
            text_parts.extend(cv_data['skills'])
        
        return ' '.join(text_parts)
    
    def _calculate_keyword_density(self, text: str) -> Dict[str, float]:
        """Calculate keyword density in text"""
        words = text.lower().split()
        total_words = len(words)
        
        if total_words == 0:
            return {}
        
        keyword_density = {}
        all_keywords = [keyword for keywords in self.industry_keywords.values() for keyword in keywords]
        
        for keyword in all_keywords:
            keyword_count = text.lower().count(keyword.lower())
            density = (keyword_count / total_words) * 100
            if density > 0:
                keyword_density[keyword] = round(density, 2)
        
        return keyword_density
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simplified readability score"""
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        complex_words = [word for word in words if len(word) > 6]
        
        readability = max(0, 100 - (avg_sentence_length + (len(complex_words) / len(words) * 100)))
        return round(readability, 2)
    
    def _check_action_verbs(self, text: str) -> float:
        """Check usage of action verbs"""
        action_verbs = [
            'achieved', 'managed', 'developed', 'implemented', 'led', 'improved',
            'increased', 'decreased', 'created', 'designed', 'built', 'launched',
            'optimized', 'streamlined', 'transformed', 'spearheaded', 'orchestrated'
        ]
        
        words = text.lower().split()
        if len(words) == 0:
            return 0
        
        action_verb_count = sum(1 for word in words if word in action_verbs)
        return round((action_verb_count / len(words)) * 100, 2)
    
    def _generate_professional_summary(self, cv_data: Dict, target_industry: str) -> str:
        """Generate AI-powered professional summary"""
        experience_years = self._calculate_experience_years(cv_data.get('work_experience', []))
        primary_skills = cv_data.get('skills', [])[:5]
        
        industry_phrase = f" in the {target_industry} industry" if target_industry else ""
        
        summary_templates = [
            f"Results-driven professional with {experience_years} years of experience{industry_phrase}. "
            f"Skilled in {', '.join(primary_skills[:3])}. Proven track record of delivering exceptional results "
            f"through innovative solutions and strategic thinking.",
            
            f"Experienced {target_industry or 'professional'} with {experience_years} years of expertise in "
            f"{', '.join(primary_skills[:2])}. Strong background in achieving measurable outcomes and "
            f"driving business growth through effective strategies.",
            
            f"Dynamic professional offering {experience_years} years of comprehensive experience{industry_phrase}. "
            f"Expertise in {', '.join(primary_skills[:3])} with a demonstrated history of success in "
            f"complex project environments."
        ]
        
        return np.random.choice(summary_templates)
    
    def _calculate_experience_years(self, work_experience: List[Dict]) -> int:
        """Calculate total years of experience"""
        if not work_experience:
            return 0
        
        total_days = 0
        for job in work_experience:
            start_date = self._parse_date(job.get('start_date'))
            end_date = self._parse_date(job.get('end_date')) or datetime.now()
            
            if start_date and end_date:
                total_days += (end_date - start_date).days
        
        return max(1, total_days // 365)
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except:
            try:
                return datetime.strptime(date_str, '%m/%Y')
            except:
                return None
    
    def _optimize_skills_section(self, skills: List[str], target_industry: str) -> List[str]:
        """Optimize skills section for target industry"""
        if not skills:
            return ["Problem Solving", "Communication", "Teamwork", "Adaptability"]
        
        # Add industry-specific skills
        industry_skills = self.industry_keywords.get(target_industry, [])
        
        # Remove duplicates and combine
        all_skills = list(set(skills + industry_skills))
        
        # Sort by relevance (industry skills first)
        def skill_sort_key(skill):
            if skill.lower() in [s.lower() for s in industry_skills]:
                return 0
            return 1
        
        return sorted(all_skills, key=skill_sort_key)[:15]  # Limit to 15 most relevant skills
    
    def _enhance_work_experience(self, work_experience: List[Dict]) -> List[Dict]:
        """Enhance work experience descriptions"""
        enhanced_experience = []
        
        for job in work_experience:
            enhanced_job = job.copy()
            
            # Enhance job description
            if 'description' in job:
                enhanced_job['description'] = self._enhance_job_description(job['description'])
            
            # Quantify achievements
            if 'achievements' in job:
                enhanced_job['achievements'] = self._quantify_achievements(job['achievements'])
            
            enhanced_experience.append(enhanced_job)
        
        return enhanced_experience
    
    def _enhance_job_description(self, description: str) -> str:
        """Enhance job description with action verbs and quantifiable results"""
        sentences = nltk.sent_tokenize(description)
        enhanced_sentences = []
        
        action_verbs = ['Managed', 'Developed', 'Implemented', 'Led', 'Created', 'Optimized']
        results_phrases = ['resulting in', 'leading to', 'which increased', 'improving']
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 0 and words[0] not in action_verbs:
                enhanced_sentence = np.random.choice(action_verbs) + ' ' + sentence.lower()
                enhanced_sentences.append(enhanced_sentence)
            else:
                enhanced_sentences.append(sentence)
        
        return ' '.join(enhanced_sentences)
    
    def _quantify_achievements(self, achievements: List[str]) -> List[str]:
        """Add quantifiable metrics to achievements"""
        quantified_achievements = []
        
        for achievement in achievements:
            if any(char.isdigit() for char in achievement):
                quantified_achievements.append(achievement)
            else:
                # Add random quantification (in real implementation, this would be more sophisticated)
                quantifiers = [
                    "resulting in 15% improvement",
                    "increasing efficiency by 20%",
                    "reducing costs by 25%",
                    "improving performance by 30%"
                ]
                quantified = f"{achievement}, {np.random.choice(quantifiers)}"
                quantified_achievements.append(quantified)
        
        return quantified_achievements
    
    def _optimize_for_ats(self, cv_data: Dict, target_industry: str) -> Dict[str, Any]:
        """Optimize CV for Applicant Tracking Systems"""
        ats_optimization = {
            "keyword_optimization": {},
            "formatting_score": 0,
            "section_headers": [],
            "recommended_improvements": []
        }
        
        # Check for important sections
        required_headers = ['experience', 'education', 'skills', 'contact']
        text_content = self._extract_text_content(cv_data).lower()
        
        found_headers = [header for header in required_headers if header in text_content]
        ats_optimization["section_headers"] = found_headers
        ats_optimization["formatting_score"] = len(found_headers) / len(required_headers) * 100
        
        # Industry-specific keyword optimization
        industry_keywords = self.industry_keywords.get(target_industry, [])
        missing_keywords = [kw for kw in industry_keywords if kw not in text_content]
        
        if missing_keywords:
            ats_optimization["recommended_improvements"].append(
                f"Add industry keywords: {', '.join(missing_keywords[:3])}"
            )
        
        # Check for quantifiable achievements
        if not any(char.isdigit() for char in text_content):
            ats_optimization["recommended_improvements"].append(
                "Add quantifiable achievements with numbers and percentages"
            )
        
        return ats_optimization
    
    def _calculate_enhancement_score(self, original_analysis: Dict, enhanced_cv: Dict) -> float:
        """Calculate overall enhancement score"""
        enhanced_analysis = self._analyze_cv(enhanced_cv)
        
        score_components = {
            "section_completeness": enhanced_analysis["section_completeness"] * 25,
            "readability": min(enhanced_analysis["readability_score"] / 100 * 25, 25),
            "action_verbs": min(enhanced_analysis["action_verb_usage"] / 10 * 25, 25),
            "keyword_optimization": len(enhanced_cv['ats_optimized']['keyword_optimization']) * 2.5
        }
        
        total_score = sum(score_components.values())
        return min(round(total_score, 2), 100)

# Example usage
if __name__ == "__main__":
    enhancer = CVEnhancer()
    
    sample_cv = {
        "full_name": "John Doe",
        "professional_summary": "Experienced software developer",
        "work_experience": [
            {
                "title": "Software Developer",
                "company": "Tech Corp",
                "start_date": "2020-01-01",
                "end_date": "2023-01-01",
                "description": "Developed web applications and maintained systems"
            }
        ],
        "skills": ["Python", "JavaScript", "SQL"],
        "education": [
            {
                "degree": "BSc Computer Science",
                "institution": "University of Cape Town",
                "year": 2019
            }
        ]
    }
    
    enhanced_cv = enhancer.enhance_cv(sample_cv, "technology")
    print(f"Enhancement Score: {enhanced_cv['enhancement_score']}")
    print(f"Professional Summary: {enhanced_cv['professional_summary']}")

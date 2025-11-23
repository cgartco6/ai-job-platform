import requests
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
from urllib.parse import urljoin, urlparse
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.job_sources = {
            'careers24': {
                'base_url': 'https://www.careers24.com',
                'search_url': 'https://www.careers24.com/jobs/',
                'selectors': {
                    'job_cards': '.job-card',
                    'title': '.job-title',
                    'company': '.company-name',
                    'location': '.job-location',
                    'salary': '.salary',
                    'description': '.job-description',
                    'date_posted': '.post-date'
                }
            },
            'pnet': {
                'base_url': 'https://www.pnet.co.za',
                'search_url': 'https://www.pnet.co.za/jobs/',
                'selectors': {
                    'job_cards': '.jobItem',
                    'title': '.jobTitle',
                    'company': '.company',
                    'location': '.location',
                    'salary': '.salary',
                    'description': '.jobDescription',
                    'date_posted': '.date'
                }
            },
            'linkedin': {
                'base_url': 'https://www.linkedin.com',
                'search_url': 'https://www.linkedin.com/jobs/search/',
                'selectors': {
                    'job_cards': '.jobs-search__results-list li',
                    'title': '.base-search-card__title',
                    'company': '.base-search-card__subtitle',
                    'location': '.job-search-card__location',
                    'salary': '.job-search-card__salary-info',
                    'description': '.description__text',
                    'date_posted': '.job-search-card__listdate'
                }
            }
        }
        
    def scrape_jobs(self, keywords: List[str], locations: List[str], max_pages: int = 3) -> List[Dict[str, Any]]:
        """Scrape jobs from multiple sources based on keywords and locations"""
        all_jobs = []
        
        for source_name, source_config in self.job_sources.items():
            logger.info(f"Scraping jobs from {source_name}")
            
            try:
                if source_name == 'linkedin':
                    jobs = self._scrape_linkedin_jobs(keywords, locations, max_pages)
                else:
                    jobs = self._scrape_generic_jobs(source_name, source_config, keywords, locations, max_pages)
                
                all_jobs.extend(jobs)
                logger.info(f"Found {len(jobs)} jobs from {source_name}")
                
                # Be respectful to servers
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping from {source_name}: {str(e)}")
                continue
        
        # Remove duplicates
        unique_jobs = self._remove_duplicate_jobs(all_jobs)
        
        logger.info(f"Total unique jobs found: {len(unique_jobs)}")
        return unique_jobs
    
    def _scrape_generic_jobs(self, source_name: str, source_config: Dict, 
                           keywords: List[str], locations: List[str], max_pages: int) -> List[Dict]:
        """Scrape jobs from generic job boards"""
        jobs = []
        
        for keyword in keywords:
            for location in locations:
                for page in range(1, max_pages + 1):
                    try:
                        # Construct search URL
                        search_params = {
                            'keywords': keyword,
                            'location': location,
                            'page': page
                        }
                        
                        url = self._construct_search_url(source_config['search_url'], search_params)
                        logger.info(f"Scraping {url}")
                        
                        response = self.session.get(url, timeout=10)
                        response.raise_for_status()
                        
                        soup = BeautifulSoup(response.content, 'html.parser')
                        job_cards = soup.select(source_config['selectors']['job_cards'])
                        
                        if not job_cards:
                            logger.info(f"No more jobs found for {keyword} in {location} on page {page}")
                            break
                        
                        for card in job_cards:
                            job = self._parse_job_card(card, source_config['selectors'], source_name)
                            if job:
                                job['search_keyword'] = keyword
                                job['search_location'] = location
                                jobs.append(job)
                        
                        # Check if there are more pages
                        if not self._has_next_page(soup, source_name):
                            break
                            
                    except Exception as e:
                        logger.error(f"Error scraping page {page} for {keyword} in {location}: {str(e)}")
                        continue
        
        return jobs
    
    def _scrape_linkedin_jobs(self, keywords: List[str], locations: List[str], max_pages: int) -> List[Dict]:
        """Scrape jobs from LinkedIn using Selenium"""
        jobs = []
        
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            
            for keyword in keywords:
                for location in locations:
                    try:
                        search_url = f"https://www.linkedin.com/jobs/search/?keywords={keyword}&location={location}"
                        driver.get(search_url)
                        
                        # Wait for jobs to load
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".jobs-search__results-list li"))
                        )
                        
                        # Scroll to load more jobs
                        self._scroll_page(driver)
                        
                        job_cards = driver.find_elements(By.CSS_SELECTOR, ".jobs-search__results-list li")
                        
                        for card in job_cards[:25]:  # Limit to first 25 jobs per search
                            try:
                                job = self._parse_linkedin_job_card(card)
                                if job:
                                    job['search_keyword'] = keyword
                                    job['search_location'] = location
                                    jobs.append(job)
                            except Exception as e:
                                logger.error(f"Error parsing LinkedIn job card: {str(e)}")
                                continue
                                
                    except Exception as e:
                        logger.error(f"Error scraping LinkedIn for {keyword} in {location}: {str(e)}")
                        continue
            
            driver.quit()
            
        except Exception as e:
            logger.error(f"Error with Selenium driver: {str(e)}")
        
        return jobs
    
    def _parse_job_card(self, card, selectors: Dict, source: str) -> Dict[str, Any]:
        """Parse individual job card"""
        try:
            job = {
                'source': source,
                'scraped_at': datetime.now().isoformat()
            }
            
            # Extract basic information
            title_elem = card.select_one(selectors['title'])
            company_elem = card.select_one(selectors['company'])
            location_elem = card.select_one(selectors['location'])
            salary_elem = card.select_one(selectors['salary'])
            date_elem = card.select_one(selectors['date_posted'])
            
            job['title'] = title_elem.get_text().strip() if title_elem else 'N/A'
            job['company'] = company_elem.get_text().strip() if company_elem else 'N/A'
            job['location'] = location_elem.get_text().strip() if location_elem else 'N/A'
            job['salary'] = salary_elem.get_text().strip() if salary_elem else 'N/A'
            
            # Parse date
            job['date_posted'] = self._parse_date(date_elem.get_text().strip() if date_elem else '')
            
            # Extract job URL
            job['url'] = self._extract_job_url(card, source)
            
            # Generate job ID
            job['job_id'] = self._generate_job_id(job['title'], job['company'], job['url'])
            
            return job
            
        except Exception as e:
            logger.error(f"Error parsing job card: {str(e)}")
            return None
    
    def _parse_linkedin_job_card(self, card) -> Dict[str, Any]:
        """Parse LinkedIn job card using Selenium"""
        try:
            job = {
                'source': 'linkedin',
                'scraped_at': datetime.now().isoformat()
            }
            
            # Extract basic information
            title_elem = card.find_element(By.CSS_SELECTOR, ".base-search-card__title")
            company_elem = card.find_element(By.CSS_SELECTOR, ".base-search-card__subtitle")
            location_elem = card.find_element(By.CSS_SELECTOR, ".job-search-card__location")
            
            job['title'] = title_elem.text.strip()
            job['company'] = company_elem.text.strip()
            job['location'] = location_elem.text.strip()
            
            # Try to get salary (might not always be present)
            try:
                salary_elem = card.find_element(By.CSS_SELECTOR, ".job-search-card__salary-info")
                job['salary'] = salary_elem.text.strip()
            except:
                job['salary'] = 'N/A'
            
            # Try to get date
            try:
                date_elem = card.find_element(By.CSS_SELECTOR, ".job-search-card__listdate")
                job['date_posted'] = self._parse_date(date_elem.text.strip())
            except:
                job['date_posted'] = datetime.now().isoformat()
            
            # Extract job URL
            try:
                link_elem = card.find_element(By.CSS_SELECTOR, "a.base-card__full-link")
                job['url'] = link_elem.get_attribute('href')
            except:
                job['url'] = ''
            
            # Generate job ID
            job['job_id'] = self._generate_job_id(job['title'], job['company'], job['url'])
            
            return job
            
        except Exception as e:
            logger.error(f"Error parsing LinkedIn job card: {str(e)}")
            return None
    
    def _construct_search_url(self, base_url: str, params: Dict) -> str:
        """Construct search URL with parameters"""
        if 'careers24' in base_url:
            return f"{base_url}{params['keywords']}/in-{params['location']}/?page={params['page']}"
        elif 'pnet' in base_url:
            return f"{base_url}{params['keywords']}/in-{params['location']}/?page={params['page']}"
        else:
            return base_url
    
    def _extract_job_url(self, card, source: str) -> str:
        """Extract job URL from card"""
        try:
            if source == 'careers24':
                link_elem = card.select_one('a.job-link')
            elif source == 'pnet':
                link_elem = card.select_one('a.jobItem-title')
            else:
                link_elem = card.select_one('a')
            
            if link_elem and link_elem.get('href'):
                return urljoin(self.job_sources[source]['base_url'], link_elem.get('href'))
            
            return ''
        except:
            return ''
    
    def _parse_date(self, date_text: str) -> str:
        """Parse date text to ISO format"""
        if not date_text:
            return datetime.now().isoformat()
        
        date_text = date_text.lower()
        
        # Handle relative dates
        if 'hour' in date_text or 'just now' in date_text:
            return datetime.now().isoformat()
        elif 'day' in date_text:
            days_ago = int(re.search(r'(\d+)', date_text).group(1)) if re.search(r'(\d+)', date_text) else 1
            return (datetime.now() - timedelta(days=days_ago)).isoformat()
        elif 'week' in date_text:
            weeks_ago = int(re.search(r'(\d+)', date_text).group(1)) if re.search(r'(\d+)', date_text) else 1
            return (datetime.now() - timedelta(weeks=weeks_ago)).isoformat()
        elif 'month' in date_text:
            months_ago = int(re.search(r'(\d+)', date_text).group(1)) if re.search(r'(\d+)', date_text) else 1
            return (datetime.now() - timedelta(days=months_ago*30)).isoformat()
        
        # Try to parse absolute dates
        try:
            date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d %b %Y', '%b %d, %Y']
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_text, fmt)
                    return parsed_date.isoformat()
                except:
                    continue
        except:
            pass
        
        return datetime.now().isoformat()
    
    def _generate_job_id(self, title: str, company: str, url: str) -> str:
        """Generate unique job ID"""
        import hashlib
        
        base_string = f"{title}_{company}_{url}"
        return hashlib.md5(base_string.encode()).hexdigest()
    
    def _has_next_page(self, soup, source: str) -> bool:
        """Check if there's a next page of results"""
        if source == 'careers24':
            next_button = soup.select_one('a.next')
            return next_button is not None
        elif source == 'pnet':
            next_button = soup.select_one('li.next a')
            return next_button is not None
        
        return False
    
    def _scroll_page(self, driver):
        """Scroll page to load more content"""
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        except:
            pass
    
    def _remove_duplicate_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Remove duplicate jobs based on job_id"""
        seen_ids = set()
        unique_jobs = []
        
        for job in jobs:
            if job['job_id'] not in seen_ids:
                seen_ids.add(job['job_id'])
                unique_jobs.append(job)
        
        return unique_jobs
    
    def save_jobs_to_json(self, jobs: List[Dict], filename: str):
        """Save jobs to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, indent=2, ensure_ascii=False)
    
    def analyze_job_market(self, jobs: List[Dict]) -> Dict[str, Any]:
        """Analyze job market trends from scraped data"""
        if not jobs:
            return {}
        
        df = pd.DataFrame(jobs)
        
        analysis = {
            'total_jobs': len(jobs),
            'top_companies': df['company'].value_counts().head(10).to_dict(),
            'top_locations': df['location'].value_counts().head(10).to_dict(),
            'top_titles': df['title'].value_counts().head(10).to_dict(),
            'salary_analysis': self._analyze_salaries(df),
            'freshness_analysis': self._analyze_job_freshness(df)
        }
        
        return analysis
    
    def _analyze_salaries(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze salary trends"""
        # Extract numeric salary values (simplified)
        salary_analysis = {
            'jobs_with_salary': len(df[df['salary'] != 'N/A']),
            'average_salary_range': 'R300,000 - R600,000'  # Placeholder
        }
        
        return salary_analysis
    
    def _analyze_job_freshness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how fresh the job postings are"""
        try:
            df['date_posted_dt'] = pd.to_datetime(df['date_posted'])
            now = datetime.now()
            
            freshness = {
                'last_24_hours': len(df[df['date_posted_dt'] >= now - timedelta(days=1)]),
                'last_week': len(df[df['date_posted_dt'] >= now - timedelta(days=7)]),
                'last_month': len(df[df['date_posted_dt'] >= now - timedelta(days=30)]),
                'older': len(df[df['date_posted_dt'] < now - timedelta(days=30)])
            }
            
            return freshness
        except:
            return {}

# Example usage
if __name__ == "__main__":
    scraper = JobScraper()
    
    # Search for jobs
    keywords = ['software developer', 'data analyst', 'project manager']
    locations = ['Cape Town', 'Johannesburg', 'Durban']
    
    jobs = scraper.scrape_jobs(keywords, locations, max_pages=2)
    
    # Save results
    scraper.save_jobs_to_json(jobs, 'scraped_jobs.json')
    
    # Analyze market
    analysis = scraper.analyze_job_market(jobs)
    print(f"Found {analysis['total_jobs']} jobs")
    print(f"Top companies: {list(analysis['top_companies'].keys())[:3]}")

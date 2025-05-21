"""
PyTrends API module for fetching Google Trends data.
Focuses on fetching related topics for a given keyword.
"""
import logging
import time
import random
import os
from pytrends.request import TrendReq
from typing import List, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

# Simple function to create a PyTrends instance
def get_pytrends_instance(hl='en-US', tz=360):
    """
    Creates a PyTrends instance.
    Args:
        hl: Language
        tz: Timezone offset
    Returns:
        TrendReq instance
    """
    try:
        return TrendReq(hl=hl, tz=tz)
    except Exception as e:
        logger.error(f"Failed to initialize PyTrends: {str(e)}")
        raise RuntimeError(f"Could not initialize PyTrends client: {str(e)}")

class PyTrendsAPI:
    """
    Class to interact with Google Trends API using PyTrends.
    Simplified to focus on fetching related topics.
    """
    def __init__(self, hl='en-US', tz=360):
        """
        Initialize the PyTrends client.
        Args:
            hl: Language (default 'en-US')
            tz: Timezone offset (default 360)
        """
        try:
            logger.info("Initializing PyTrends client")
            self.pytrends = get_pytrends_instance(hl, tz)
            logger.info("PyTrends client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PyTrends: {type(e).__name__} - {str(e)}")
            raise RuntimeError(f"Could not initialize PyTrends client: {str(e)}") from e
        
    def build_payload(self, keyword: str, timeframe: str = 'today 3-m', geo: str = '', gprop: str = '', max_retries: int = 3):
        """
        Build the payload for PyTrends requests with retry logic.
        Args:
            keyword: Main keyword to build trends around
            timeframe: Time frame to fetch data for (default: 'today 3-m')
            geo: Region code (e.g., 'US', 'GB') - defaults to worldwide if empty
            gprop: Google property to filter on (default: web searches)
            max_retries: Maximum number of retries for rate limiting (default: 3)
        Returns:
            True if successful, False otherwise
        """
        retries = 0
        while retries < max_retries:
            try:
                logger.info(f"Building payload for keyword: '{keyword}', geo: '{geo if geo else 'worldwide'}' (attempt {retries+1}/{max_retries})")
                self.pytrends.build_payload(kw_list=[keyword], cat=0, timeframe=timeframe, geo=geo, gprop=gprop)
                logger.info(f"Successfully built payload for keyword: '{keyword}'")
                return True
            except Exception as e:
                retries += 1
                if 'too many requests' in str(e).lower() or '429' in str(e):
                    wait_time = (2 ** retries) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit, retrying in {wait_time:.2f} seconds... ({retries}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error building PyTrends payload: {type(e).__name__} - {e}")
                    break # For non-rate-limiting errors, break early
        
        logger.error(f"Failed to build payload for keyword: '{keyword}' after {max_retries} attempts")
        return False

    def get_related_topics(self, keyword: str, region: str = '', limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get topics related to a keyword from Google Trends.
        Args:
            keyword: The main keyword to search for
            region: Region code (e.g., 'US', 'GB') - defaults to worldwide if empty
            limit: Maximum number of results to return (applied after fetching)
        Returns:
            List of related topics with their relevance scores
        """
        if os.environ.get('PYTRENDS_MOCK_DATA', '').lower() == 'true':
            logger.info(f"Using mock data for related topics (keyword: {keyword})")
            return [
                {"title": "Machine Learning", "type": "Field of study", "value": 100.0, "trend_type": "top"},
                {"title": "Deep Learning", "type": "Field of study", "value": 85.5, "trend_type": "top"},
                {"title": "ChatGPT", "type": "Application", "value": 75.2, "trend_type": "rising"}
            ][:limit]
        
        try:
            if not self.build_payload(keyword, geo=region):
                logger.warning(f"Returning empty list due to payload build failure for: {keyword}")
                return []
            
            related_topics_result = self.pytrends.related_topics()
            
            topics_list: List[Dict[str, Any]] = []
            if keyword in related_topics_result:
                for category in ['top', 'rising']:
                    if category in related_topics_result[keyword]:
                        df = related_topics_result[keyword][category]
                        if df is not None and not df.empty:
                            for _, row in df.iterrows():
                                topics_list.append({
                                    'title': row.get('topic_title', ''),
                                    'type': row.get('topic_type', ''),
                                    'value': float(row.get('value', 0)),
                                    'trend_type': category
                                })
            else:
                logger.warning(f"No related topics data structure found for keyword: '{keyword}'")

            return sorted(topics_list, key=lambda x: x['value'], reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching related topics for '{keyword}': {type(e).__name__} - {e}")
            return []

# Function wrapper for fetching related topics
def fetch_related_topics(topic: str, region: str = '', limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch topics related to a given topic using Google Trends.
    Args:
        topic: Main topic to find related topics for
        region: Region code (e.g., 'US', 'GB') - defaults to worldwide if empty
        limit: Maximum number of topics to return
    Returns:
        List of related topics with details
    """
    try:
        api = PyTrendsAPI()
        return api.get_related_topics(keyword=topic, region=region, limit=limit)
    except Exception as e:
        logger.error(f"Error in fetch_related_topics wrapper for '{topic}': {type(e).__name__} - {e}")
        return []

"""
# Removed other methods like:
# - get_related_queries
# - get_trending_searches
# - get_regional_interest
# And their corresponding fetch_* wrappers like:
# - fetch_trending_keywords (which used get_related_queries)
# - fetch_related_queries
# - fetch_trending_searches
# - fetch_regional_interest
"""

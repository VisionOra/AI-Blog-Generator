#!/usr/bin/env python
"""
Test script for PyTrends API
"""
import os
import sys
import logging

# Set up logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the API functions
from pytrends_api import fetch_related_topics

def main():
    """Test the PyTrends API implementation"""
    # Enable mock data mode
    os.environ['PYTRENDS_MOCK_DATA'] = 'true'
    
    # Test related topics
    keyword = 'Artificial Intelligence'
    region = 'US'
    
    print(f"\nFetching related topics for keyword: '{keyword}', region: '{region}'")
    related_topics = fetch_related_topics(topic=keyword, region=region, limit=5)
    
    if related_topics:
        print(f"\nFound {len(related_topics)} related topics:")
        for i, topic in enumerate(related_topics, 1):
            print(f"{i}. {topic['title']} ({topic['type']}) - Value: {topic['value']}, Type: {topic['trend_type']}")
    else:
        print("No related topics found or error occurred.")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 
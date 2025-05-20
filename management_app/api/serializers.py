from rest_framework import serializers

class LinkedInPostRequestSerializer(serializers.Serializer):
    topic = serializers.CharField(
        max_length=200,
        help_text="The topic for the LinkedIn post."
    )

class LinkedInPostResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    topic = serializers.CharField()
    linkedin_post = serializers.CharField()
    saved_file = serializers.CharField(required=False) # Optional field

# It's also good practice to have a generic error serializer if you have consistent error responses
class ErrorResponseSerializer(serializers.Serializer):
    error = serializers.CharField()

# Serializers for Blog Generation API
class BlogRequestSerializer(serializers.Serializer):
    topic = serializers.CharField(
        max_length=255, # Increased max_length slightly
        help_text="The main topic for the blog post."
    )
    keywords = serializers.CharField(
        required=False, 
        allow_blank=True, 
        help_text="Optional comma-separated keywords to guide blog generation."
    )

class BlogResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    topic = serializers.CharField()
    keywords = serializers.CharField(allow_blank=True) # Allow blank if keywords were not provided
    markdown_file = serializers.CharField()
    content = serializers.CharField()

# Serializers for Trending Keywords API
class TrendingKeywordsRequestSerializer(serializers.Serializer):
    topic = serializers.CharField(
        max_length=255,
        help_text="The main topic to find related trending keywords for."
    )
    region = serializers.CharField(
        required=False,
        max_length=10,
        allow_blank=True,
        help_text="The region code (e.g., 'US', 'GB'). Default is worldwide."
    )
    limit = serializers.IntegerField(
        required=False,
        default=10,
        min_value=1,
        max_value=50,
        help_text="Maximum number of keywords to return. Default is 10."
    )

class KeywordItemSerializer(serializers.Serializer):
    keyword = serializers.CharField()
    score = serializers.IntegerField()
    raw_value = serializers.IntegerField()

class TrendingKeywordsResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    topic = serializers.CharField()
    region = serializers.CharField(allow_blank=True)
    keywords = serializers.ListField(child=KeywordItemSerializer())

# Serializers for Image Generation API
class ImageGenerationRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField(
        required=False, 
        allow_blank=True,
        help_text="The main prompt for image generation. Optional."
    )
    keywords = serializers.CharField(
        required=False, 
        allow_blank=True, 
        help_text="Optional comma-separated keywords to enhance the image prompt."
    )

    # Add validation to ensure at least one field is provided
    def validate(self, data):
        prompt = data.get('prompt')
        keywords = data.get('keywords')
        if not prompt and not keywords:
            raise serializers.ValidationError("Either 'prompt' or 'keywords' (or both) must be provided.")
        if (prompt and not prompt.strip()) and (keywords and not keywords.strip()):
             raise serializers.ValidationError("Provided prompt or keywords cannot be empty or just whitespace.")
        return data

class ImageGenerationResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    prompt_used = serializers.CharField()
    image_file = serializers.CharField()

# Serializers for Related Topics API
class RelatedTopicsRequestSerializer(serializers.Serializer):
    keyword = serializers.CharField(
        max_length=255,
        help_text="The main keyword to find related topics for."
    )

class TopicItemSerializer(serializers.Serializer):
    title = serializers.CharField()
    type = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    value = serializers.FloatField()
    trend_type = serializers.CharField()

class RelatedTopicsResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    keyword = serializers.CharField()
    id = serializers.IntegerField(required=False)  # Database ID after saving
    related_topics = serializers.ListField(child=TopicItemSerializer())

# Serializers for Related Queries API (Added for completeness)
class RelatedQueriesRequestSerializer(serializers.Serializer):
    topic = serializers.CharField(
        max_length=255,
        help_text="The main keyword to find related queries for."
    )
    region = serializers.CharField(
        required=False,
        max_length=10,
        allow_blank=True,
        help_text="The region code (e.g., 'US', 'GB'). Default is worldwide."
    )
    limit = serializers.IntegerField(
        required=False,
        default=10,
        min_value=1,
        max_value=50,
        help_text="Maximum number of queries to return. Default is 10."
    )

class QueryItemSerializer(serializers.Serializer):
    query = serializers.CharField()
    value = serializers.FloatField()
    trend_type = serializers.CharField()

class RelatedQueriesResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    topic = serializers.CharField()
    region = serializers.CharField(allow_blank=True)
    related_queries = serializers.ListField(child=QueryItemSerializer())

# Serializers for Trending Searches API (Added for completeness)
class TrendingSearchesRequestSerializer(serializers.Serializer):
    region = serializers.CharField(
        required=False,
        max_length=20,
        allow_blank=True,
        default="united_states",
        help_text="The region code for trending searches (e.g., 'united_states', 'uk'). Default is 'united_states'."
    )
    limit = serializers.IntegerField(
        required=False,
        default=10,
        min_value=1,
        max_value=50,
        help_text="Maximum number of trending searches to return. Default is 10."
    )

class TrendingSearchesResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    region = serializers.CharField()
    trending_searches = serializers.ListField(child=serializers.CharField())

# Serializers for Regional Interest API (Added for completeness)
class RegionalInterestRequestSerializer(serializers.Serializer):
    topic = serializers.CharField(
        max_length=255,
        help_text="The main keyword to find regional interest for."
    )
    region = serializers.CharField(
        required=False,
        max_length=10,
        allow_blank=True,
        help_text="The region code to limit results (e.g., 'US', 'GB'). Default is worldwide."
    )
    resolution = serializers.CharField(
        required=False,
        default="COUNTRY",
        help_text="Geographic resolution ('COUNTRY', 'REGION', 'CITY', 'DMA'). Default is 'COUNTRY'."
    )
    limit = serializers.IntegerField(
        required=False,
        default=10,
        min_value=1,
        max_value=50,
        help_text="Maximum number of regions to return. Default is 10."
    )

class RegionItemSerializer(serializers.Serializer):
    region = serializers.CharField()
    value = serializers.FloatField()

class RegionalInterestResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    topic = serializers.CharField()
    region = serializers.CharField(allow_blank=True)
    resolution = serializers.CharField()
    regional_interest = serializers.ListField(child=RegionItemSerializer()) 
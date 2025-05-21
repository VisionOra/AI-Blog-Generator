import json
import os
import sys
from django.conf import settings
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse
import logging
from django.utils import timezone
from datetime import datetime
import re

# Import models
from .models import BlogGeneral, BlogAiNews, LinkedinPost, ImageGeneration, TrendingTopics

# Set up logging
logger = logging.getLogger(__name__)

# Add tools directory to sys.path
if settings.TOOLS_DIR not in sys.path:
    sys.path.insert(0, settings.TOOLS_DIR)
if os.path.dirname(settings.TOOLS_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(settings.TOOLS_DIR))

from tools.ai.blog_generator.blog_writer import BlogWriter, generate_image
# Import the new LinkedInPostGenerator service
from tools.ai.linkedin_post_generator.linkedin_post_generator import LinkedInPostGenerator
# Import the PyTrends service
from tools.ai.trends_ai.pytrends_api import (
    fetch_related_topics
)

# Import serializers
from .serializers import (
    LinkedInPostRequestSerializer,
    LinkedInPostResponseSerializer,
    ErrorResponseSerializer,
    BlogRequestSerializer, BlogResponseSerializer,
    ImageGenerationRequestSerializer, ImageGenerationResponseSerializer,
    TrendingKeywordsRequestSerializer, TrendingKeywordsResponseSerializer,
    RelatedTopicsRequestSerializer, RelatedTopicsResponseSerializer
)

@extend_schema(
    request=BlogRequestSerializer,
    responses={
        200: OpenApiResponse(response=BlogResponseSerializer, description='Blog generated successfully.'),
        400: OpenApiResponse(response=ErrorResponseSerializer, description='Bad Request - Invalid input.'),
        500: OpenApiResponse(response=ErrorResponseSerializer, description='Internal Server Error.')
    },
    description="Generate a detailed blog post based on the given topic and optional parameters for customization."
)
@api_view(['POST'])
def generate_blog_api(request):
    """
    Generate a detailed blog post from a given topic and optional parameters.
    Input is a JSON object with "topic" (required) and various optional parameters
    for customizing the blog format and style.
    """
    # Validate request data using the serializer
    serializer = BlogRequestSerializer(data=request.data)
    if serializer.is_valid():
        topic = serializer.validated_data['topic']
        keywords = serializer.validated_data.get('keywords', [])
        tone = serializer.validated_data.get('tone', 'professional')
        length_min = serializer.validated_data.get('length_min', 800)
        length_max = serializer.validated_data.get('length_max', 1500)
        introduction = serializer.validated_data.get('introduction', True)
        table_of_content = serializer.validated_data.get('table_of_content', False)
        faq = serializer.validated_data.get('faq', False)
        cta = serializer.validated_data.get('cta', False)
        conclusion = serializer.validated_data.get('conclusion', True)
        target_audience = serializer.validated_data.get('target_audience', [])

        try:
            logger.info(f"Starting blog generation for topic: '{topic}' with customized parameters")
            base_output_dir = settings.GENERATED_BLOGS_DIR
            
            blog_writer_instance = BlogWriter(
                topic=topic, 
                keywords=keywords,
                tone=tone,
                length_min=length_min,
                length_max=length_max,
                introduction=introduction,
                table_of_content=table_of_content,
                faq=faq,
                cta=cta,
                conclusion=conclusion,
                target_audience=target_audience
            )
            
            markdown_file_path = blog_writer_instance.save_blog_to_file(
                topic=topic, output_file_name=None, base_output_dir=base_output_dir
            )
            
            if not isinstance(markdown_file_path, str):
                logger.error(f"BlogWriter.save_blog_to_file returned type {type(markdown_file_path)} (expected str) for topic: {topic}")
                return Response({'error': 'Internal server error: Blog generation returned unexpected data type.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            logger.info(f"Successfully generated blog on '{topic}' at {markdown_file_path}")
            
            django_base_dir = settings.BASE_DIR
            relative_md_path = os.path.relpath(markdown_file_path, django_base_dir).replace(os.sep, '/')
            
            blog_content = "" # Initialize blog_content
            try:
                if os.path.exists(markdown_file_path):
                    with open(markdown_file_path, 'r', encoding='utf-8') as f:
                        blog_content = f.read()
                        
                    # Save to database
                    blog = BlogGeneral(
                        user_id=1,  # Default user ID until authentication is implemented
                        topic=topic,
                        content=blog_content,
                        created_at=timezone.now()
                    )
                    blog.save()
                    logger.info(f"Saved blog to database with ID: {blog.id}")
                    
            except Exception as e:
                logger.error(f"Error reading generated blog content or saving to database: {e}")

            response_data = {
                'status': 'success', 
                'message': 'Blog generated successfully!',
                'topic': topic, 
                'keywords': keywords,
                'tone': tone,
                'length_min': length_min,
                'length_max': length_max,
                'introduction': introduction,
                'table_of_content': table_of_content,
                'faq': faq,
                'cta': cta,
                'conclusion': conclusion,
                'target_audience': target_audience,
                'markdown_file': relative_md_path,
                'content': blog_content
            }
            
            # Serialize the successful response
            response_serializer = BlogResponseSerializer(data=response_data)
            if response_serializer.is_valid():
                 return Response(response_serializer.data, status=status.HTTP_200_OK)
            else:
                 logger.error(f"Error serializing successful response for blog: {response_serializer.errors}")
                 return Response({'error': 'Internal server error during response serialization.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Exception handling for the main try block
        except ValueError as e: 
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except ImportError as e: 
            return Response({'error': f'Server configuration error (ImportError): {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error(f"Unexpected error in blog generation: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            return Response({'error': f'An unexpected error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        # If serializer validation fails
        logger.warning(f"Invalid input for blog generation: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@extend_schema(
    description="Automatically generates a blog about the latest trends and news of this week.",
    responses={200: None}
)
@api_view(['GET'])
def generate_weekly_news_blog(request):
    """
    Generate a weekly news blog about the latest trends and developments.
    
    This endpoint automatically creates a blog about this week's news and trends.
    The result is a Markdown file.
    No parameters needed - just click Execute!
    """
    try:
        logger.info("Starting weekly news blog generation")
        
        topic = "Latest Trends and News This Week: Technology, Business, and Culture"
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file_name = f"weekly_news_{date_str}.md"
        
        base_output_dir = settings.GENERATED_BLOGS_DIR
        
        blog_writer_instance = BlogWriter(
            topic=topic
        )
        
        markdown_file_path = blog_writer_instance.save_blog_to_file(
            topic=topic, 
            output_file_name=output_file_name, 
            base_output_dir=base_output_dir
        )
        
        if not isinstance(markdown_file_path, str):
            logger.error(f"BlogWriter.save_blog_to_file returned type {type(markdown_file_path)} (expected str) for weekly news.")
            return Response({'error': 'Internal server error: Weekly news generation returned unexpected data type.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        logger.info(f"Successfully generated weekly news blog at {markdown_file_path}")
        
        django_base_dir = settings.BASE_DIR
        relative_md_path = os.path.relpath(markdown_file_path, django_base_dir).replace(os.sep, '/')
        
        blog_content = ""
        try:
            if os.path.exists(markdown_file_path):
                with open(markdown_file_path, 'r', encoding='utf-8') as f:
                    blog_content = f.read()
                
                # Save to database
                news_blog = BlogAiNews(
                    news_week_start=datetime.now().date(),
                    summary=topic,  # Using the topic as a summary
                    content=blog_content,
                    created_at=timezone.now()
                )
                news_blog.save()
                logger.info(f"Saved weekly news blog to database with ID: {news_blog.id}")
        except Exception as e:
            logger.error(f"Error reading generated blog content or saving to database: {e}")
            # Continue with the response even if saving to DB fails
        
        response_data = {
            'status': 'success',
            'message': 'Weekly news blog generated successfully!',
            'topic': topic,
            'date': date_str,
            'markdown_file': relative_md_path,
            'content': blog_content
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Unexpected error in weekly news blog generation: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return Response({'error': f'An unexpected error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR) 

# New endpoints will be added below

@extend_schema(
    request=ImageGenerationRequestSerializer,
    responses={
        200: OpenApiResponse(response=ImageGenerationResponseSerializer, description='Image generated successfully.'),
        400: OpenApiResponse(response=ErrorResponseSerializer, description='Bad Request - Invalid input.'),
        500: OpenApiResponse(response=ErrorResponseSerializer, description='Internal Server Error / Image Generation Failed.')
    },
    description="Generate an image based on a prompt and/or keywords using DALL-E 3."
)
@api_view(['POST'])
def generate_image_api(request):
    """
    Generates an image using DALL-E 3 based on a provided prompt and/or keywords.
    Input is a JSON object with optional "prompt" and "keywords" fields.
    At least one of "prompt" or "keywords" must be provided.
    """
    # Validate request data using the serializer
    serializer = ImageGenerationRequestSerializer(data=request.data)
    if serializer.is_valid():
        prompt = serializer.validated_data.get('prompt')
        keywords = serializer.validated_data.get('keywords')

        # Construct the final prompt for the image generation model
        final_prompt = ""
        if prompt and keywords:
            final_prompt = f"{prompt} - Keywords: {keywords}"
        elif prompt:
            final_prompt = prompt
        elif keywords:
            final_prompt = f"Generate an image based on the following keywords: {keywords}"
        # The serializer's validate method already ensures final_prompt won't be empty

        try:
            logger.info(f"Starting image generation with prompt: '{final_prompt}'")
            image_output_dir = "api_generated_images"  # This is now just a prefix for S3
            
            # Generate image and get S3 URL and optimized prompt
            image_result = generate_image(prompt=final_prompt, output_dir=image_output_dir)
            
            # Unpack the result tuple (image_url, optimized_prompt)
            if isinstance(image_result, tuple) and len(image_result) == 2:
                image_url, enhanced_prompt = image_result
            else:
                # Handle legacy function calls that might not return a tuple
                image_url = image_result
                enhanced_prompt = final_prompt

            if image_url:
                # Determine if the URL is an S3 URL or local path
                is_s3_url = image_url.startswith('http')
                
                if not is_s3_url:
                    # If it's a local path, convert to relative path for display
                    django_base_dir = settings.BASE_DIR
                    relative_image_path = os.path.relpath(image_url, django_base_dir).replace(os.sep, '/')
                    image_url_for_db = relative_image_path
                else:
                    # Use the S3 URL directly
                    image_url_for_db = image_url
                
                # Save to database
                image_record = ImageGeneration(
                    user_id=1,  # Default user ID until authentication is implemented
                    prompt=final_prompt,
                    image_url=image_url_for_db,
                    created_at=timezone.now()
                )
                image_record.save()
                logger.info(f"Saved image generation record to database with ID: {image_record.id}")
                
                response_data = {
                    'status': 'success',
                    'message': 'Image generated successfully!',
                    'prompt_used': final_prompt,
                    'enhanced_prompt': enhanced_prompt,
                    'image_file': image_url_for_db
                }
                
                # Serialize the successful response
                response_serializer = ImageGenerationResponseSerializer(data=response_data)
                if response_serializer.is_valid():
                    logger.info(f"Successfully generated image at {image_url}")
                    return Response(response_serializer.data, status=status.HTTP_200_OK)
                else:
                    logger.error(f"Error serializing successful image response: {response_serializer.errors}")
                    return Response({'error': 'Internal server error during response serialization.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                logger.error(f"Image generation failed for prompt: '{final_prompt}'")
                # Using the ErrorResponseSerializer structure might be better here
                return Response({'error': 'Image generation failed. Check server logs for details.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            logger.error(f"Unexpected error in image generation API: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            return Response({'error': f'An unexpected error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        # If serializer validation fails
        logger.warning(f"Invalid input for image generation: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@extend_schema(
    request=LinkedInPostRequestSerializer,
    responses={
        200: OpenApiResponse(response=LinkedInPostResponseSerializer, description='LinkedIn post generated successfully.'),
        400: OpenApiResponse(response=ErrorResponseSerializer, description='Bad Request - Invalid input.'),
        500: OpenApiResponse(response=ErrorResponseSerializer, description='Internal Server Error.')
    },
    description="Generate a professional LinkedIn post based on the given topic."
)
@api_view(['POST'])
def generate_linkedin_post_api(request):
    """
    Generates a professional LinkedIn post for a given topic.
    Input is a JSON object with a "topic" field.
    """
    # Validate request data using the serializer
    serializer = LinkedInPostRequestSerializer(data=request.data)
    if serializer.is_valid():
        topic = serializer.validated_data['topic']
        keywords = serializer.validated_data.get('keywords', [])
        
        try:
            logger.info(f"Starting LinkedIn post generation for topic: '{topic}' with keywords: {keywords}")

            linkedin_generator = LinkedInPostGenerator(topic=topic, keywords=keywords)
            linkedin_post_content, saved_file_path = linkedin_generator.generate_post(topic=topic, keywords=keywords)

            if linkedin_post_content:
                # Save to database
                linkedin_post = LinkedinPost(
                    user_id=1,  # Default user ID until authentication is implemented
                    topic=topic,
                    content=linkedin_post_content,
                    created_at=timezone.now()
                )
                linkedin_post.save()
                logger.info(f"Saved LinkedIn post to database with ID: {linkedin_post.id}")
                
                response_data = {
                    'status': 'success',
                    'message': 'LinkedIn post generated successfully!',
                    'topic': topic,
                    'keywords': keywords,
                    'linkedin_post': linkedin_post_content
                }
                if saved_file_path:
                    try:
                        django_base_dir = settings.BASE_DIR
                        relative_file_path = os.path.relpath(saved_file_path, django_base_dir).replace(os.sep, '/')
                        response_data['saved_file'] = relative_file_path
                    except ValueError:
                        response_data['saved_file'] = saved_file_path
                
                # Serialize the successful response
                response_serializer = LinkedInPostResponseSerializer(data=response_data)
                if response_serializer.is_valid():
                    logger.info(f"Successfully generated LinkedIn post for topic: '{topic}'")
                    if saved_file_path: logger.info(f"LinkedIn post saved to: {saved_file_path}")
                    return Response(response_serializer.data, status=status.HTTP_200_OK)
                else:
                    logger.error(f"Error serializing successful response: {response_serializer.errors}")
                    return Response({'error': 'Internal server error during response serialization.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                logger.error(f"LinkedIn post generation failed for topic: '{topic}'")
                return Response({'error': 'LinkedIn post generation failed. Check server logs.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            logger.error(f"Unexpected error in LinkedIn post generation API: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            return Response({'error': f'An unexpected error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        # If serializer validation fails, return errors
        logger.warning(f"Invalid input for LinkedIn post generation: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 

@extend_schema(
    request=RelatedTopicsRequestSerializer,
    responses={
        200: OpenApiResponse(response=RelatedTopicsResponseSerializer, description='Related topics fetched and saved successfully.'),
        400: OpenApiResponse(response=ErrorResponseSerializer, description='Bad Request - Invalid input.'),
        500: OpenApiResponse(response=ErrorResponseSerializer, description='Internal Server Error.')
    },
    description="Fetch topics related to a given keyword using Google Trends data and save to database."
)
@api_view(['POST'])
def fetch_and_save_related_topics(request):
    """
    Fetches topics related to a given keyword using Google Trends and saves to database.
    
    Input is a JSON object with:
    - "keyword" (required): Main keyword to find related topics for
    """
    # Validate request data using the serializer
    serializer = RelatedTopicsRequestSerializer(data=request.data)
    if serializer.is_valid():
        keyword = serializer.validated_data['keyword']
        
        try:
            logger.info(f"Starting related topics fetch for keyword: '{keyword}'")
            
            # 1. Initialize PyTrends client with minimal parameters
            from pytrends.request import TrendReq
            pytrends = TrendReq(hl='en-US', tz=360)
            
            # 2. Build payload with the keyword
            logger.info(f"Building payload for keyword: '{keyword}'")
            pytrends.build_payload([keyword])
            
            # 3. Get related topics directly
            related_topics_result = pytrends.related_topics()
            
            # Process results
            topics_list = []
            if keyword in related_topics_result:
                # Process both top and rising topics
                for category in ['top', 'rising']:
                    if category in related_topics_result[keyword] and not related_topics_result[keyword][category].empty:
                        df = related_topics_result[keyword][category]
                        for _, row in df.iterrows():
                            topics_list.append({
                                'title': row.get('topic_title', ''),
                                'type': row.get('topic_type', ''),
                                'value': float(row.get('value', 0)),
                                'trend_type': category
                            })
            
            # Save to database
            if topics_list:
                db_record = TrendingTopics(
                    keyword=keyword,
                    topics=topics_list,
                    created_at=timezone.now()
                )
                db_record.save()
                record_id = db_record.id
                logger.info(f"Saved {len(topics_list)} topics to database with ID: {record_id}")
            else:
                record_id = None
                logger.warning(f"No topics found for keyword: '{keyword}'")
            
            # Return response
            response_data = {
                'status': 'success',
                'message': f"Found {len(topics_list)} related topics for '{keyword}'",
                'keyword': keyword,
                'related_topics': topics_list
            }
            
            if record_id:
                response_data['id'] = record_id
                
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching related topics: {str(e)}")
            return Response({
                'status': 'error',
                'message': f"Failed to fetch related topics: {str(e)}",
                'keyword': keyword
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
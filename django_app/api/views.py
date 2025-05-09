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

# Set up logging
logger = logging.getLogger(__name__)

# Add tools directory to sys.path
if settings.TOOLS_DIR not in sys.path:
    sys.path.insert(0, settings.TOOLS_DIR)
if os.path.dirname(settings.TOOLS_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(settings.TOOLS_DIR))

from tools.blog_writer import BlogWriter, generate_image
# Import the new LinkedInPostGenerator service
from django_app.services.linkedin_post_generator import LinkedInPostGenerator

# Import serializers
from .serializers import ( # Assuming serializers.py is in the same directory (api)
    LinkedInPostRequestSerializer,
    LinkedInPostResponseSerializer,
    ErrorResponseSerializer,
    BlogRequestSerializer, BlogResponseSerializer,
    ImageGenerationRequestSerializer, ImageGenerationResponseSerializer # Added Image serializers
)

@extend_schema(
    # Remove parameters array
    # parameters=[
    #     OpenApiParameter(name='topic', description='The blog topic to generate', required=True, type=str),
    #     OpenApiParameter(name='keywords', description='Optional comma-separated keywords to guide blog generation', required=False, type=str), 
    # ],
    request=BlogRequestSerializer, # Use request serializer
    responses={
        200: OpenApiResponse(response=BlogResponseSerializer, description='Blog generated successfully.'),
        400: OpenApiResponse(response=ErrorResponseSerializer, description='Bad Request - Invalid input.'),
        500: OpenApiResponse(response=ErrorResponseSerializer, description='Internal Server Error.')
    },
    description="Generate a detailed blog post based on the given topic and optional keywords."
)
@api_view(['POST'])
def generate_blog_api(request):
    """
    Generate a detailed blog post from a given topic and optional keywords.
    Input is a JSON object with "topic" (required) and "keywords" (optional).
    """
    # Validate request data using the serializer
    serializer = BlogRequestSerializer(data=request.data)
    if serializer.is_valid():
        topic = serializer.validated_data['topic']
        keywords = serializer.validated_data.get('keywords') # Use .get() for optional field

        try:
            logger.info(f"Starting blog generation for topic: '{topic}' with keywords: '{keywords if keywords else 'None'}'")
            base_output_dir = settings.GENERATED_BLOGS_DIR
            
            print("--- generate_blog_api: About to instantiate BlogWriter ---") # Diagnostic
            blog_writer_instance = BlogWriter(topic=topic, keywords=keywords)
            print("--- generate_blog_api: BlogWriter instantiated ---") # Diagnostic
            
            print("--- generate_blog_api: About to call save_blog_to_file ---") # Diagnostic
            markdown_file_path = blog_writer_instance.save_blog_to_file(
                topic=topic, output_file_name=None, base_output_dir=base_output_dir
            )
            print(f"--- generate_blog_api: save_blog_to_file returned: {type(markdown_file_path)} ---") # Diagnostic
            
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
            except Exception as e:
                logger.error(f"Error reading generated blog content: {e}")
                # Decide if you still want to return success but with a warning/empty content
                # For now, we proceed but content might be empty

            response_data = {
                'status': 'success', 'message': 'Blog generated successfully!',
                'topic': topic, 'keywords': keywords if keywords else '', # Ensure keywords is a string
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
            logger.error(f"Unexpected error in blog generation: {type(e).__name__} - {e}"); import traceback; traceback.print_exc()
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
        
        from datetime import datetime
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
        
        response_data = {
            'status': 'success',
            'message': 'Weekly news blog generated successfully!',
            'topic': topic,
            'date': date_str,
            'markdown_file': relative_md_path,
        }
        
        try:
            if os.path.exists(markdown_file_path):
                with open(markdown_file_path, 'r', encoding='utf-8') as f:
                    response_data['content'] = f.read()
        except Exception as e:
            logger.error(f"Error reading generated blog content: {e}")
            response_data['warning'] = f"Generated file exists but could not be read: {str(e)}"
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Unexpected error in weekly news blog generation: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return Response({'error': f'An unexpected error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR) 

# New endpoints will be added below

@extend_schema(
    # Remove parameters array
    # parameters=[
    #     OpenApiParameter(name='prompt', description='The main prompt for image generation.', required=False, type=str),
    #     OpenApiParameter(name='keywords', description='Optional comma-separated keywords to enhance the image prompt.', required=False, type=str),
    # ],
    request=ImageGenerationRequestSerializer, # Use request serializer
    responses={
        200: OpenApiResponse(response=ImageGenerationResponseSerializer, description='Image generated successfully.'),
        400: OpenApiResponse(response=ErrorResponseSerializer, description='Bad Request - Invalid input.'), # Could also use serializer.errors directly
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
            image_output_dir = os.path.join(settings.GENERATED_BLOGS_DIR, "api_generated_images")
            os.makedirs(image_output_dir, exist_ok=True) 

            image_path = generate_image(prompt=final_prompt, output_dir=image_output_dir)

            if image_path:
                django_base_dir = settings.BASE_DIR
                relative_image_path = os.path.relpath(image_path, django_base_dir).replace(os.sep, '/')
                
                response_data = {
                    'status': 'success',
                    'message': 'Image generated successfully!',
                    'prompt_used': final_prompt,
                    'image_file': relative_image_path
                }
                
                # Serialize the successful response
                response_serializer = ImageGenerationResponseSerializer(data=response_data)
                if response_serializer.is_valid():
                    logger.info(f"Successfully generated image at {image_path}")
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
    # Remove parameters array as request body is now handled by serializer
    # parameters=[
    # OpenApiParameter(name='topic', description='The topic for the LinkedIn post.', required=True, type=str),
    # ],
    request=LinkedInPostRequestSerializer, # Use the request serializer
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
        
        try:
            logger.info(f"Starting LinkedIn post generation for topic: '{topic}'")

            linkedin_generator = LinkedInPostGenerator(topic=topic)
            linkedin_post_content, saved_file_path = linkedin_generator.generate_post(topic=topic)

            if linkedin_post_content:
                response_data = {
                    'status': 'success',
                    'message': 'LinkedIn post generated successfully!',
                    'topic': topic,
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
                if response_serializer.is_valid(): # Should always be valid if data is correct
                    logger.info(f"Successfully generated LinkedIn post for topic: '{topic}'")
                    if saved_file_path: logger.info(f"LinkedIn post saved to: {saved_file_path}")
                    return Response(response_serializer.data, status=status.HTTP_200_OK)
                else:
                    # This case should ideally not happen if response_data is structured correctly
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
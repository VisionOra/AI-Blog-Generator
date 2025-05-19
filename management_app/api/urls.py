from django.urls import path
from . import views

urlpatterns = [
    # Blog generation API endpoints
    path('generate-blog/', views.generate_blog_api, name='generate_blog_api'),
    path('weekly-news/', views.generate_weekly_news_blog, name='weekly_news_blog'),
    
    # New API endpoints
    path('generate-image/', views.generate_image_api, name='generate_image_api'),
    path('generate-linkedin-post/', views.generate_linkedin_post_api, name='generate_linkedin_post_api'),
    
    # Removed view endpoints as requested
] 
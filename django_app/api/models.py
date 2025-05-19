from django.db import models
from django.utils import timezone


class BlogGeneral(models.Model):
    user_id = models.IntegerField()
    topic = models.CharField(max_length=255)
    content = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'blogs_general'


class BlogAiNews(models.Model):
    news_week_start = models.DateField()
    summary = models.TextField()
    content = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'blogs_ai_news'


class LinkedinPost(models.Model):
    user_id = models.IntegerField()
    topic = models.CharField(max_length=255)
    content = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'linkedin_posts'


class ImageGeneration(models.Model):
    user_id = models.IntegerField()
    prompt = models.TextField()
    image_url = models.URLField(max_length=500)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'image_generation' 
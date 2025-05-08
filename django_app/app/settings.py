from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-dummy-key' # Replace with a real secret key

DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'api',  # Our new API app
    'rest_framework',
    'drf_spectacular',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'app.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'app.wsgi.application'

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/
STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Custom setting for tools directory
TOOLS_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, 'tools')

# Directory to store generated blogs
GENERATED_BLOGS_DIR = os.path.join(BASE_DIR, 'generated_blogs')
os.makedirs(GENERATED_BLOGS_DIR, exist_ok=True)

REST_FRAMEWORK = {
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
}

SPECTACULAR_SETTINGS = {
    'TITLE': 'AI Blog Generator',
    'DESCRIPTION': '''
    This API generates complete, high-quality blog posts in Markdown format using a multi-agent AI system.
    
    **How it works:**
    1. You provide a blog topic
    2. Our system activates multiple AI agents to collaboratively create your blog:
       - The **Planner** agent researches and outlines the structure
       - The **Writer** agent drafts the content based on the plan
       - The **Editor** agent refines and improves the final text
       - The **Designer** agent creates a banner image
    3. The complete blog is saved as a Markdown file with the embedded image
    4. You receive the path to the finished blog post
    
    Just enter your topic and click Execute!
    ''',
    'VERSION': '1.0.0',
    'SERVE_INCLUDE_SCHEMA': False,
    'SWAGGER_UI_SETTINGS': {
        'deepLinking': True,
        'displayOperationId': False,
        'defaultModelsExpandDepth': -1,
        'defaultModelExpandDepth': 1,
        'docExpansion': 'list',
        'filter': False,
        'displayRequestDuration': True,
        'tryItOutEnabled': True,
    },
    'COMPONENT_SPLIT_REQUEST': True,
}

# Logging Configuration for Development
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False, 
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            # Set console handler level to DEBUG so it CAN show debug messages if a logger allows them
            'level': 'DEBUG', 
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'root': {
        'handlers': ['console'],
        # Set root level higher (e.g., INFO) to avoid DEBUG from all libraries
        'level': 'INFO', 
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO'), 
            'propagate': False,
        },
        'crewai': { # Keep CrewAI logs verbose if desired
            'handlers': ['console'],
            'level': 'DEBUG', # Allow DEBUG level from crewai specifically
            'propagate': False, # Prevent crewai DEBUG messages from going to root logger if root is INFO
        },
        'httpcore': { # Silence DEBUG messages from httpcore
            'handlers': ['console'],
            'level': 'INFO', 
            'propagate': False,
        },
         'httpx': { # Silence DEBUG messages from httpx (often used with httpcore)
            'handlers': ['console'],
            'level': 'INFO', 
            'propagate': False,
        },
         'litellm': { # Set LiteLLM's default to INFO unless DEBUG is needed
            'handlers': ['console'],
            'level': 'INFO', 
            'propagate': False,
        },
        # Add other specific loggers here if needed
    }
} 
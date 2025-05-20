from django.urls import path
from .views import (
    RegisterView, LoginView, UserListView,
    CustomTokenRefreshView, GoogleLoginRedirectView,
    GoogleLoginCallbackView, GoogleLoginTokenView
)

urlpatterns = [
    # Regular auth endpoints
    path('register/', RegisterView.as_view(), name='auth_register'),
    path('login/', LoginView.as_view(), name='auth_login'),
    path('refresh/', CustomTokenRefreshView.as_view(), name='token_refresh'),
    path('users/', UserListView.as_view(), name='user_list'),
    
    # Google auth endpoints
    path('google/login/', GoogleLoginRedirectView.as_view(), name='google_login'),
    path('google/callback/', GoogleLoginCallbackView.as_view(), name='google_callback'),
    path('google/token/', GoogleLoginTokenView.as_view(), name='google_token'),
] 
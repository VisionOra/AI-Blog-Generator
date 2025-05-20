from django.shortcuts import render, redirect
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenRefreshView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.authentication import JWTAuthentication
import hashlib
import requests
import json
import os
from django.urls import reverse
from django.conf import settings

from .models import User
from .serializers import (
    UserSerializer, RegisterSerializer, LoginSerializer,
    CustomTokenObtainPairSerializer, TokenRefreshResponseSerializer,
    GoogleAuthSerializer, GoogleLoginRedirectSerializer
)

# Google OAuth settings - these should be set in environment variables
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')
GOOGLE_REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:8001/auth/google/callback')

class UserListView(generics.ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAdminUser]
    authentication_classes = [JWTAuthentication]

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = RegisterSerializer
    
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        return Response({
            'success': True,
            'message': 'User registered successfully',
            'username': user.username,
            'email': user.email
        }, status=status.HTTP_201_CREATED)

class LoginView(APIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = CustomTokenObtainPairSerializer
    
    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        try:
            # Get validated data from serializer
            validated_data = serializer.validated_data
            
            # Create custom response
            response_data = {
                'success': True,
                'message': 'Login successful',
                'refresh': validated_data['refresh'],
                'access': validated_data['access'],
                'user': validated_data['user']
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'success': False,
                'message': 'Login failed',
                'error': str(e)
            }, status=status.HTTP_401_UNAUTHORIZED)

class GoogleLoginRedirectView(APIView):
    """
    Redirect user to Google OAuth login page
    """
    permission_classes = [permissions.AllowAny]
    serializer_class = GoogleLoginRedirectSerializer
    
    def get(self, request):
        # Construct Google OAuth URL
        google_auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
        params = {
            "client_id": GOOGLE_CLIENT_ID,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": "email profile",
            "access_type": "offline",
            "prompt": "consent"
        }
        
        # Construct full URL with parameters
        auth_url = f"{google_auth_url}?{'&'.join([f'{key}={value}' for key, value in params.items()])}"
        
        # Return the URL for frontend to redirect
        return Response({
            "auth_url": auth_url
        })

class GoogleLoginCallbackView(APIView):
    """
    Handle callback from Google OAuth login
    """
    permission_classes = [permissions.AllowAny]
    serializer_class = GoogleAuthSerializer
    
    def get(self, request):
        serializer = self.serializer_class(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        
        code = serializer.validated_data.get('code')
        error = serializer.validated_data.get('error')
        
        if error:
            return Response({
                'success': False,
                'message': 'Google login failed',
                'error': error
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if not code:
            return Response({
                'success': False,
                'message': 'Google login failed',
                'error': 'No authorization code provided'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Exchange code for tokens
        try:
            token_url = "https://oauth2.googleapis.com/token"
            token_data = {
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code"
            }
            
            token_response = requests.post(token_url, data=token_data)
            token_json = token_response.json()
            
            if 'error' in token_json:
                return Response({
                    'success': False,
                    'message': 'Google login failed',
                    'error': token_json.get('error_description', token_json['error'])
                }, status=status.HTTP_400_BAD_REQUEST)
                
            # Get user info from Google
            id_token = token_json.get('id_token')
            user_info_url = "https://www.googleapis.com/oauth2/v3/userinfo"
            user_info_response = requests.get(
                user_info_url,
                headers={"Authorization": f"Bearer {token_json['access_token']}"}
            )
            user_info = user_info_response.json()
            
            # Get or create user
            email = user_info.get('email')
            if not email:
                return Response({
                    'success': False,
                    'message': 'Google login failed',
                    'error': 'Email not provided by Google'
                }, status=status.HTTP_400_BAD_REQUEST)
                
            # Check if user exists
            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                # Create a new user
                username = user_info.get('name', email.split('@')[0])
                user = User.objects.create(
                    username=username,
                    email=email,
                    # Use a random hashed password since user will login via Google
                    password=hashlib.sha256(os.urandom(32).hex().encode()).hexdigest()
                )
            
            # Generate JWT tokens
            refresh = RefreshToken.for_user(user)
            
            # Return tokens
            response_data = {
                'success': True,
                'message': 'Google login successful',
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'user': UserSerializer(user).data
            }
            
            # For APIs, return the response
            if request.accepted_renderer.format == 'json':
                return Response(response_data)
                
            # For browser flow, redirect to frontend with tokens
            frontend_url = os.environ.get('FRONTEND_URL', 'http://localhost:3000')
            redirect_url = f"{frontend_url}/login/success?access={str(refresh.access_token)}&refresh={str(refresh)}"
            return redirect(redirect_url)
            
        except Exception as e:
            return Response({
                'success': False,
                'message': 'Google login failed',
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

class GoogleLoginTokenView(APIView):
    """
    Handle direct ID token verification from frontend
    """
    permission_classes = [permissions.AllowAny]
    serializer_class = GoogleAuthSerializer
    
    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        id_token = serializer.validated_data.get('id_token')
        
        if not id_token:
            return Response({
                'success': False,
                'message': 'Google login failed',
                'error': 'No ID token provided'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Verify the token with Google
            verify_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
            verify_response = requests.get(verify_url)
            user_info = verify_response.json()
            
            if 'error' in user_info:
                return Response({
                    'success': False,
                    'message': 'Google login failed',
                    'error': user_info.get('error_description', user_info['error'])
                }, status=status.HTTP_400_BAD_REQUEST)
                
            # Get or create user
            email = user_info.get('email')
            if not email:
                return Response({
                    'success': False,
                    'message': 'Google login failed',
                    'error': 'Email not provided by Google'
                }, status=status.HTTP_400_BAD_REQUEST)
                
            # Check if user exists
            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                # Create a new user
                username = user_info.get('name', email.split('@')[0])
                user = User.objects.create(
                    username=username,
                    email=email,
                    # Use a random hashed password since user will login via Google
                    password=hashlib.sha256(os.urandom(32).hex().encode()).hexdigest()
                )
            
            # Generate JWT tokens
            refresh = RefreshToken.for_user(user)
            
            # Return tokens
            return Response({
                'success': True,
                'message': 'Google login successful',
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'user': UserSerializer(user).data
            })
            
        except Exception as e:
            return Response({
                'success': False,
                'message': 'Google login failed',
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

class CustomTokenRefreshView(TokenRefreshView):
    """
    Custom token refresh view with our response serializer
    """
    serializer_class = TokenRefreshResponseSerializer

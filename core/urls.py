from django.contrib import admin
from django.urls import path, include
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView
)



schema_view = get_schema_view(
   openapi.Info(
      title="EMAP API",
      default_version='v1',
      description="Ear Recognition System Application Programming Interface.",
      terms_of_service="https://www.google.com/policies/terms/",
      contact=openapi.Contact(email="contact@snippets.local"),
      license=openapi.License(name="EMAP License"),
   ),
   public=True,
   permission_classes=[permissions.AllowAny],
)

from authentication.views import RegistrationAPIView

urlpatterns = [
    path('', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('admin/', admin.site.urls),
    path('ears/', include('authentication.urls')),
    path('register/', RegistrationAPIView.as_view(), name='register'),
]



admin.site.site_header = 'Foodskraft administration'
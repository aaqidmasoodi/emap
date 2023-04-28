from django.urls import path
from authentication import views


urlpatterns = [
    path('register/', views.EarRegistration.as_view(), name='register-ear'),
    path('authenticate/', views.EarAuthentication.as_view(), name='authenticate-ear'),
]

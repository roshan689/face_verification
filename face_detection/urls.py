from django.urls import path
from .views import FaceVerificationView

urlpatterns = [
    path('verify/', FaceVerificationView.as_view(), name='face-verify'),
]

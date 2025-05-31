from django.urls import path, include

urlpatterns = [
    path('api/', include('face_detection.urls')),
]

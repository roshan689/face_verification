from rest_framework import serializers

class FaceVerificationSerializer(serializers.Serializer):
    user_image = serializers.ImageField()
    id_image = serializers.ImageField()

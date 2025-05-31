from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
from .serializers import FaceVerificationSerializer
from .utils import extract_frame_from_video, get_embedding, compare_embeddings, cosine_similarity
from .preprocessor import preprocess_id_image_from_memory
import logging

logger = logging.getLogger('face_detection')

class FaceVerificationView(APIView):
    def post(self, request):
        serializer = FaceVerificationSerializer(data=request.data)
        if serializer.is_valid():
            user_image = serializer.validated_data['user_image']
            id_image = serializer.validated_data['id_image']

            try:
                id_img = Image.open(id_image)
                user_image = Image.open(user_image)
                # frame_rgb = extract_frame_from_video(video.temporary_file_path())
                processed_ID_img = preprocess_id_image_from_memory(id_image)
                # processed_user_image = preprocess_id_image_from_memory(user_image)
                
                emb1 = get_embedding(processed_ID_img)
                if emb1 is None:
                    logger.error("No face detected in ID image.")
                    return Response({'error': 'No face detected in ID image'}, status=status.HTTP_400_BAD_REQUEST)

                emb2 = get_embedding(user_image)
                if emb2 is None:
                    logger.error("No face detected in user image.")
                    return Response({'error': 'No face detected in user image'}, status=status.HTTP_400_BAD_REQUEST)

                if cosine_similarity(emb1, emb2):
                    return Response({'verified': True}, status=status.HTTP_200_OK)
                else:
                    return Response({'verified': False}, status=status.HTTP_401_UNAUTHORIZED)

            except Exception as e:
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

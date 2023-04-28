from rest_framework.views import APIView
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from authentication import utils
from rest_framework import status
from django.contrib.auth import get_user_model
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.authentication import JWTAuthentication
from authentication.models import Ear
from .serializers import UserSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from PIL import Image
import numpy
import cv2


User = get_user_model()

class EarRegistration(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    @csrf_exempt
    def post(self, request):

        ear_id = request.POST.get('ear_id', None)

        if not ear_id:
            return Response({"error":"You must specify an ear id."}, status=status.HTTP_400_BAD_REQUEST)
 
        # check if the username (user) already exists
        try:
            ear = Ear.objects.get(ear_id=ear_id)
            return Response({"error":"An ear with that ear id is already present in the records."}, status=status.HTTP_400_BAD_REQUEST)
        except Ear.DoesNotExist:
            pass
            

        if len(request.FILES) != 4:
            return Response({"error":"4 images are required to register!"}, status=status.HTTP_400_BAD_REQUEST)

        multiple_ear_features = []
        ears_all = Ear.objects.filter(user=request.user)

        # this for loop will go over every image in the post request and extract features from the images
        for image_name in request.FILES:
            image = request.FILES.get(image_name, None)
            
            if not image:
                return Response({"error":"Unable to fetch image. Make sure to pass the parameter properly."}, status=status.HTTP_400_BAD_REQUEST)
            
            # using pillow to read the image in 
            image = numpy.array(Image.open(image))

            ear_image = utils.detect_and_crop_ear(image)
            
            if ear_image is None:
                return Response({"error": f"Unable to detect ear in {image_name}. Make sure you are at the appropriate distance from the subject."}, status=status.HTTP_409_CONFLICT)

            # Once the ear image has been extracted from the main image, it is resized to 224X224
            ear_image = cv2.resize(ear_image, (224,224), interpolation=cv2.INTER_AREA)
            features = str(utils.extract_features(ear_image))

            if not features:
                return Response({"error":f"Something went wrong extracting features for {image_name}"}, status=status.HTTP_409_CONFLICT)
            
            # check if the ear is already registered
            '''
                this logic may slow down things a little bit but i think
                in the long run this is going to be useful. 
            '''
            if len(list(ears_all)) > 0:
                match = utils.find_match(features, ears_all)
                if match[1] > .75:
                    return Response({"message":"It looks like this ear has already been registered before."}, status=status.HTTP_409_CONFLICT)


            multiple_ear_features.append(features)

        if len(multiple_ear_features) != 4:
            return Response({"error":"Something went wrong extracting features."}, status=status.HTTP_409_CONFLICT)
 
        
        # save features to the created user in database
        ear = Ear.objects.create(
            ear_id=ear_id,
            features_1=multiple_ear_features[0],
            features_2=multiple_ear_features[1], 
            features_3=multiple_ear_features[2], 
            features_4=multiple_ear_features[3],
            user=request.user 
        )

        return Response({"message":f"Ear Registered sucessfully. {ear}"}, status=status.HTTP_200_OK)


class EarAuthentication(APIView):

    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    @csrf_exempt
    def post(self, request):
        ears_all = Ear.objects.filter(user=request.user)

        # check if the database has ears
        if not len(list(ears_all)) > 0:
            return Response({"message":"No Match Found. The database does not contain any ears."}, status=status.HTTP_404_NOT_FOUND)
        
        # check if user has sent files
        if not len(request.FILES) > 0:
            return Response({"message":"You didn't send an image!"}, status=status.HTTP_400_BAD_REQUEST)

        # try to get the image from the request object
        image = request.FILES.get('image', None)
        if not image:
            return Response({"error":"Something Went Wrong!"}, status=status.HTTP_400_BAD_REQUEST)

        # this step is for the loading of the image and getting it ready
        image = numpy.array(Image.open(image))

        # crop and grab the ear from the image
        ear_image = utils.detect_and_crop_ear(image)
            
        if ear_image is None:
            return Response({"error": f"Unable to detect ear in the image. Make sure you are at appropriate distance from the subject."})

        ear_image = cv2.resize(ear_image, (224,224), interpolation=cv2.INTER_AREA)

        # get features from the current image
        features_current = utils.extract_features(ear_image)
       
        match = utils.find_match(features_current, ears_all)

        if not match[0]:
            return Response({"message":"No Match Found. Make sure the ear is cleary visible and has been registered before."}, status=status.HTTP_404_NOT_FOUND)

        return Response({"message":f"Found Match: {match[0]}"}, status=status.HTTP_200_OK)
    


class RegistrationAPIView(APIView):
    serializer_class = UserSerializer
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }, status=status.HTTP_201_CREATED)
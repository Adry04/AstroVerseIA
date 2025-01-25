import joblib
import jwt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from .models import Space, Preference  # Aggiungi l'importazione di modelli
from .serializers import SpaceSerializer  # Aggiungi l'importazione del serializer
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY')
le_macro_argomento = joblib.load('le_macro_argomento.pkl')
le_argomento_spazio = joblib.load('le_argomento_spazio.pkl')
model = joblib.load('model.pkl')

class GetSpaces(APIView):
    def get(self, request, pagina):
        auth_header = request.headers.get('Authorization')
        try:
            token = auth_header.split(" ")[1]
        except IndexError:
            return Response({"error": "Invalid Authorization header format"}, status=status.HTTP_401_UNAUTHORIZED)
        try:
            decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.DecodeError:
            return Response({'error': 'Token non valido'}, status=status.HTTP_401_UNAUTHORIZED)
        user_id = decoded_token['id']
        spaces = Space.objects.all()
        selected_spaces = []
        preferences = Preference.objects.filter(user=user_id)
        serializer = []
        for space in spaces:
            en_spazio_argomento = le_argomento_spazio.transform([space.argument]).reshape(-1, 1)
            for preference in preferences:
                en_macro_argomento = le_macro_argomento.transform([preference.argument]).reshape(-1, 1)
                data = pd.DataFrame(
                    [[en_macro_argomento, en_spazio_argomento]],
                    columns=['macro_argomento', 'argomento_spazio']
                )
                if model.predict(data):
                    selected_spaces.append(space)
                    space_serializer = SpaceSerializer(space, context={'consigliato': True})
                    serializer.append(space_serializer.data)
                    break
        for space in spaces:
            if space not in selected_spaces:
                space_serializer = SpaceSerializer(space, context={'consigliato': False})
                serializer.append(space_serializer.data)
        paginator = PageNumberPagination()
        paginator.page_size = 30
        request.GET._mutable = True
        request.GET['page'] = pagina
        request.GET._mutable = False
        paginated_spaces = paginator.paginate_queryset(selected_spaces, request, request)
        total_pages = (len(selected_spaces) + paginator.page_size - 1) // paginator.page_size

        return Response({'spaces': serializer, 'numberOfPages': total_pages}, status=status.HTTP_200_OK)
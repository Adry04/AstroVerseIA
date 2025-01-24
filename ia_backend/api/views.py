import joblib
import jwt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from .models import Space, Preference  # Aggiungi l'importazione di modelli
from .serializers import SpaceSerializer  # Aggiungi l'importazione del serializer

SECRET_KEY = "VHJ5aGZxk6b43Fg5l3bdz78TzJ6w04UKlGjGh5A0bOY="

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
        for space in spaces:
            preferences = Preference.objects.filter(user=user_id)
            print(preferences)
        paginator = PageNumberPagination()
        paginator.page_size = 30
        paginated_spaces = paginator.paginate_queryset(selected_spaces, request)
        serializer = SpaceSerializer(selected_spaces, many=True)
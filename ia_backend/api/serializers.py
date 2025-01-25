from rest_framework import serializers
from .models import Space

class SpaceSerializer(serializers.ModelSerializer):
    consigliato = serializers.BooleanField(required=False)
    class Meta:
        model = Space
        fields = '__all__'
    def __init__(self, *args, **kwargs):
        consigliato = kwargs.get('context', {}).get('consigliato', True)
        self.fields['consigliato'].default = consigliato
        super().__init__(*args, **kwargs)
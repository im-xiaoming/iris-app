from django import forms
from django.core.validators import FileExtensionValidator

class IrisForm(forms.Form):
    sepal_length = forms.FloatField(label='Sepal Length', min_value=0, max_value=10)
    sepal_width = forms.FloatField(label='Sepal Width', min_value=0, max_value=10)
    petal_length = forms.FloatField(label='Petal Length', min_value=0, max_value=10)
    petal_width = forms.FloatField(label='Petal Width', min_value=0, max_value=10)
    
class DataForm(forms.Form):
    data_file = forms.FileField(label='Upload CSV File', validators=[
        FileExtensionValidator(allowed_extensions=['csv'])
    ])
from django import forms
from .models import Stock

class StockForm(forms.ModelForm):
    class Meta:
        model = Stock
        fields = ["tickerSymbol","latestPrice","previousClose","marketCap","ytdChange","week52High","week52Low"]




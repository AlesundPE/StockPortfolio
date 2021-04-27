from django.db import models
from django.contrib import admin

class Customer(models.Model):
    name = models.CharField(max_length = 200)
    phone = models.CharField(max_length = 200)
    email = models.EmailField(max_length = 100)
admin.site.register(Customer)

class Stock(models.Model):
    tickerSymbol = models.CharField(max_length = 10,null=True)
    latestPrice = models.CharField(max_length = 20,null=True)
    previousClose = models.CharField(max_length = 20,null=True)
    marketCap = models.CharField(max_length = 20,null=True)
    ytdChange = models.CharField(max_length = 20,null=True)
    week52High = models.CharField(max_length = 20,null=True)
    week52Low = models.CharField(max_length = 20,null=True)
    def __str__(self):
        return self.tickerSymbol
admin.site.register(Stock)
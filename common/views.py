from django.shortcuts import render, redirect
from .models import Customer
from .models import Stock
from django.contrib import messages
from .forms import StockForm
import datetime



def listCustomers(request):
    
    qs = Customer.objects.values()
    
    return render(request, 'customer.html', {'customers': qs })

def home(request):
    '''import requests
    import json

    if request.method == 'POST':
        tickerSymbol = request.POST['tickerSymbol']
        api_request = requests.get("https://cloud.iexapis.com/stable/stock/"+tickerSymbol+"/quote?token=pk_fc1636b4895648e499e805830cd8430a")
        try:
            api = json.loads(api_request.content)
        except Exception as e:
            api = "Error"
        return render(request, 'home.html', {'api': api })

    else:
    '''
    return render(request, 'home.html', {'ticker': "Welcome to the Short-term Investor Sidekick!"})

    # pk_fc1636b4895648e499e805830cd8430a
    

def about(request):
    return render(request, 'about.html', {})

# pk_fc1636b4895648e499e805830cd8430a
def addStock(request):
    import requests
    import json

    dateToday = datetime.datetime.now().strftime("%Y-%m-%d")

    if request.method == 'POST':
        api_request = requests.get("https://cloud.iexapis.com/stable/stock/"+request.POST.get("tickerSymbol","")+
        "/quote?token=pk_fc1636b4895648e499e805830cd8430a&filter=symbol,latestPrice,previousClose,marketCap,ytdChange,week52High,week52Low")
        
        api = json.loads(api_request.content)

        form = StockForm({"tickerSymbol":api['symbol'],"latestPrice":api['latestPrice'],"previousClose":api['previousClose'],
        "marketCap":api['marketCap'],"ytdChange":round(api['ytdChange'],5),"week52High":api['week52High'],"week52Low":api['week52Low']})
        if form.is_valid():
            form.save()
            messages.success(request, ("Stock has been added"))
        else:
            print(form.errors)
        return redirect(addStock)
            
        
  
    else:
        #&filter=symbol,latestPrice,previousClose,marketCap,ytdChange,week52High,week52Low
        ticker = Stock.objects.all()
        updateInstance = requests.get("https://cloud.iexapis.com/stable/stock/amc/quote?token=pk_fc1636b4895648e499e805830cd8430a&filter=isUSMarketOpen")
        mkt = json.loads(updateInstance.content)

        if mkt["isUSMarketOpen"] == "true":
            for tickerSymbol in ticker:
                api_request = requests.get("https://cloud.iexapis.com/stable/stock/"+str(tickerSymbol)+
                "/quote?token=pk_fc1636b4895648e499e805830cd8430a&filter=symbol,latestPrice,previousClose,marketCap,ytdChange,week52High,week52Low")
                try:
                    api = json.loads(api_request.content)
                    #tickerList.append(api)
                    '''
                    form = StockForm({"tickerSymbol":api['symbol'],"latestPrice":api['latestPrice'],"previousClose":api['previousClose'],
                    "marketCap":api['marketCap'],"ytdChange":round(api['ytdChange'],5),"week52High":api['week52High'],"week52Low":api['week52Low']})
                    form.save()
                    '''
                    item = Stock.objects.get(tickerSymbol=api['symbol'])
                    item.latestPrice=api['latestPrice']
                    item.previousClose=api['previousClose']
                    item.marketCap=api['marketCap']
                    item.ytdChange=round(api['ytdChange'],5)
                    item.week52High=api['week52High']
                    item.week52Low=api['week52Low']
                    item.save()

                except Exception as e:
                    api = "Error"

        
        return render(request, 'addStock.html', {'ticker': ticker})

def delete(request, stock_id):
    item = Stock.objects.get(pk=stock_id)
    item.delete()
    messages.success(request, ("Stock has been deleted"))
    return redirect(addStock)

def tvChart(request):
    import requests
    if request.method == 'POST':
        tickerSymbol = request.POST['tickerSymbol']
        return render(request, 'index.html', {'tickerSymbol': tickerSymbol})
    else:
        return render(request, 'index.html', {})
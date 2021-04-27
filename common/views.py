from django.shortcuts import render, redirect
from .models import Customer
from .models import Stock
from django.contrib import messages
from .forms import StockForm

def listCustomers(request):
    
    qs = Customer.objects.values()
    
    return render(request, 'customer.html', {'customers': qs })

def home(request):
    import requests
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
        return render(request, 'home.html', {'ticker': "Welcome to the Short-term Investor Sidekick!"})

    # pk_fc1636b4895648e499e805830cd8430a
    

def about(request):
    return render(request, 'about.html', {})


'''
    tickerSymbol = models.CharField(max_length = 10,null=True)
    latestPrice = models.CharField(max_length = 20,null=True)
    previousClose = models.CharField(max_length = 20,null=True)
    marketCap = models.CharField(max_length = 20,null=True)
    ytdChange = models.CharField(max_length = 20,null=True)
    week52High = models.CharField(max_length = 20,null=True)
    week52Low = models.CharField(max_length = 20,null=True)
'''

def addStock(request):
    import requests
    import json

    if request.method == 'POST':
        
        form = StockForm(request.POST or None)

        if form.is_valid():
            form.save()
            messages.success(request, ("Stock has been added"))

            '''
            api_request = requests.get("https://cloud.iexapis.com/stable/stock/"+request.POST.get('tickerSymbol')+"/quote?token=pk_fc1636b4895648e499e805830cd8430a")
            api = json.loads(api_request.content)
            Stock=form.save(commit=False)
            Stock.latestPrice=api.latestPrice
            '''



            return redirect('addStock')

        


    else:
        
        ticker = Stock.objects.all()
        tickerList = []

        for ticker_item in ticker:

            api_request = requests.get("https://cloud.iexapis.com/stable/stock/"+str(ticker_item)+"/quote?token=pk_fc1636b4895648e499e805830cd8430a")
            try:
                api = json.loads(api_request.content)
                tickerList.append(api)
                '''
                Stock.objects.filter(tickerSymbol=str(ticker_item)).create(latestPrice=api.latestPrice)
                Stock.objects.filter(tickerSymbol=str(ticker_item)).create(previousClose=api.previousClose)
                Stock.objects.filter(tickerSymbol=str(ticker_item)).create(marketCap=api.marketCap)
                Stock.objects.filter(tickerSymbol=str(ticker_item)).create(ytdChange=api.ytdChange)
                Stock.objects.filter(tickerSymbol=str(ticker_item)).create(week52High=api.week52High)
                Stock.objects.filter(tickerSymbol=str(ticker_item)).create(week52Low=api.week52Low)
                Stock.save()
                '''
            except Exception as e:
                api = "Error"
        
        form = StockForm
        return render(request, 'addStock.html', {'ticker': ticker, 'tickerList': tickerList})

def delete(request, stock_id):
    item = Stock.objects.get(pk=stock_id)
    item.delete()
    messages.success(request, ("Stock has been deleted"))
    return redirect(addStock)

def tvChart(request):
    return render(request, 'index.html', {})
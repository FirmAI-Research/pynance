from django.shortcuts import render

from django.templatetags.static import static

def fundamentals(request):

    context = {

    }

    return render(request, 'fundamentals.html', context)


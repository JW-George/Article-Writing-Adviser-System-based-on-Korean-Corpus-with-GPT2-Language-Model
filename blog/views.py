# -*- coding: utf-8 -*- 
from django.shortcuts import render
from .forms import PostForm
from .system.serve_py import run_model
def post_list(request):
    output = "result"
    if request.method == "POST":
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            keyword = request.POST.copy().get('text')
            output = run_model( keyword )
    else:
        form = PostForm()
    return render(request, 'blog/post_list.html', {'form': form, 'output':output})

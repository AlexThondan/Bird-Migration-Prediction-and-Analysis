# auth_app/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import User  # Custom User model

@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.user.is_authenticated:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True, 'redirect': '/dashboard/'})
        return redirect('dashboard')
    
    if request.method == 'POST':
        username_input = request.POST.get('username')  # Email from frontend form
        password = request.POST.get('password')
        if not username_input or not password:
            error_msg = 'Email and password are required.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return render(request, 'login.html')
        
        user = authenticate(request, username=username_input, password=password)
        if user is not None:
            login(request, user)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': True, 'redirect': '/dashboard/'})
            return redirect('dashboard')
        else:
            error_msg = 'Invalid email or password.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
    
    # For GET or non-AJAX errors, render template
    return render(request, 'login.html')

@require_http_methods(["GET", "POST"])
def signup_view(request):
    if request.user.is_authenticated:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True, 'redirect': '/dashboard/'})
        return redirect('dashboard')
    
    if request.method == 'POST':
        name = request.POST.get('name')  # Display name field
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        
        if not all([name, email, password1, password2]):
            error_msg = 'All fields are required.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return render(request, 'login.html')

        if password1 != password2:
            error_msg = 'Passwords do not match.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return render(request, 'login.html')

        if User.objects.filter(email=email).exists():
            error_msg = 'Email already registered.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return render(request, 'login.html')

        try:
            user = User.objects.create_user(
                email=email,
                password=password1,
                name=name
            )
            login(request, user)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': True, 'redirect': '/dashboard/'})
            return redirect('dashboard')
        except Exception as e:
            error_msg = f'Error creating user: {str(e)}. Please try again.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
    
    # For GET
    return render(request, 'login.html')

@login_required
def dashboard_view(request):
    # Determine user role
    role = "Admin" if request.user.is_superuser else "Non-Admin"
    context = {
        'user': request.user,
        'role': role
    }
    return render(request, 'dashboard.html')

@require_http_methods(["GET", "POST"])
def logout_view(request):
    logout(request)
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'success': True, 'redirect': '/login/'})
    messages.success(request, 'Successfully logged out.')
    return redirect('login')
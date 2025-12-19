# auth_app/admin.py
from django.contrib import admin
from .models import User

class UserAdmin(admin.ModelAdmin):
    list_display = ('email', 'name', 'is_staff', 'is_superuser', 'role')
    list_filter = ('is_staff', 'is_superuser')
    search_fields = ('email', 'name')
    readonly_fields = ('is_staff', 'is_superuser')  # Prevent editing these fields directly

    def role(self, obj):
        if obj.is_superuser:
            return "Admin"
        return "Non-Admin"
    role.short_description = 'Role'

admin.site.register(User, UserAdmin)
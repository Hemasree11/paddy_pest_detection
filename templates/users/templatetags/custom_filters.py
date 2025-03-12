# users/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def add_class(value, arg):
    return f'{value} {arg}'

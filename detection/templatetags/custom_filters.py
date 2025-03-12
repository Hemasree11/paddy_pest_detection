# detection/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def add_class(value, arg):
    """Adds a CSS class to the given value (usually a form field)."""
    return f'{value} class="{arg}"'

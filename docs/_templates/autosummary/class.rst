{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{#- What gets a row here and a page of its own: _ext/api_pages.py. #}
{% if class_page.is_enum(fullname) %}
.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:
{% else %}
.. autoclass:: {{ objname }}
   :no-members:
{%- if class_page.has_bases(fullname) %}
   :show-inheritance:
{%- endif %}
{% set attrs = class_page.members(fullname, attributes) | sort(case_sensitive=False) %}
{% set meths = class_page.members(fullname, all_methods) | sort(case_sensitive=False) %}

{% if attrs %}
.. rubric:: Attributes

.. autosummary::
   :toctree:
{% for item in attrs %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}

{% if meths %}
.. rubric:: Methods

.. autosummary::
   :toctree:
{% for item in meths %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{% endif %}

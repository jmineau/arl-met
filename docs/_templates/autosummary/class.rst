{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% if methods %}
.. rubric:: Methods

.. autosummary::

{% for item in methods %}
   {{ objname }}.{{ item }}
{%- endfor %}
{% endif %}

{% if attributes %}
.. rubric:: Attributes

.. autosummary::

{% for item in attributes %}
   {{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
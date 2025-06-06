{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

..    {% block attributes %}
..    {% if attributes %}
..    .. rubric:: Module attributes

..    .. autosummary::
..       :toctree:
..    {% for item in attributes %}
..       {{ item }}
..    {%- endfor %}
..    {% endif %}
..    {% endblock %}

..    {% block functions %}
..    {% if functions %}
..    .. rubric:: {{ _('Functions') }}

..    .. autosummary::
..       :toctree:
..       :nosignatures:
..    {% for item in functions %}
..       {{ item }}
..    {%- endfor %}
..    {% endif %}
..    {% endblock %}

..    {% block classes %}
..    {% if classes %}
..    .. rubric:: {{ _('Classes') }}

..    .. autosummary::
..       :toctree:
..       :template: autosummary/class.rst
..       :nosignatures:
..    {% for item in classes %}
..       {{ item }}
..    {%- endfor %}
..    {% endif %}
..    {% endblock %}

..    {% block exceptions %}
..    {% if exceptions %}
..    .. rubric:: {{ _('Exceptions') }}

..    .. autosummary::
..       :toctree:
..    {% for item in exceptions %}
..       {{ item }}
..    {%- endfor %}
..    {% endif %}
..    {% endblock %}

.. {% if not fullname in ('elements', 'function_generators') %}
.. {% block modules %}
.. {% if modules %}
.. .. autosummary::
..    :toctree:
..    :template: autosummary/module.rst
..    :recursive:
..    {% for item in modules %}
..        {{ item }}
..    {%- endfor %}
.. {% endif %}
.. {% endblock %}
.. {%- endif %}

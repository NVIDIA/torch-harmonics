{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{# The default autosummary class template appends "Methods" and "Attributes"
   summary tables that list *inherited* members too (all of torch.nn.Module).
   We omit those tables on purpose: the autoclass directive above already
   documents the members actually implemented in torch-harmonics, honoring the
   ``inherited-members: False`` setting in conf.py. #}

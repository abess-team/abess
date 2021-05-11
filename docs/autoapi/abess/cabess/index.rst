:mod:`abess.cabess`
===================

.. py:module:: abess.cabess


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   abess.cabess._SwigNonDynamicMeta



Functions
~~~~~~~~~

.. autoapisummary::

   abess.cabess._swig_repr
   abess.cabess._swig_setattr_nondynamic_instance_variable
   abess.cabess._swig_setattr_nondynamic_class_variable
   abess.cabess._swig_add_metaclass
   abess.cabess.pywrap_abess



.. function:: _swig_repr(self)


.. function:: _swig_setattr_nondynamic_instance_variable(set)


.. function:: _swig_setattr_nondynamic_class_variable(set)


.. function:: _swig_add_metaclass(metaclass)

   Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass


.. class:: _SwigNonDynamicMeta

   Bases: :py:obj:`type`

   Meta class to enforce nondynamic attributes (no new attributes) for a class

   .. attribute:: __setattr__
      

      


.. function:: pywrap_abess(arg1, arg2, n, p, data_type, arg6, is_normal, algorithm_type, model_type, max_iter, exchange_num, path_type, is_warm_start, ic_type, ic_coef, is_cv, K, arg18, arg19, arg20, arg21, s_min, s_max, K_max, epsilon, lambda_min, lambda_max, n_lambda, is_screening, screening_size, powell_path, arg32, tau, primary_model_fit_max_iter, primary_model_fit_epsilon, early_stop, approximate_Newton, thread, covariance_update, sparse_matrix, arg41, arg42, arg43, arg44, arg45, arg46, arg47, arg48)



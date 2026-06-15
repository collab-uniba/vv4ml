Build and install vv4ml
+++++++++++++++++++++++

**Warning**: we do not recommend to build vv4ml by your own unless you really need to.
We do encourage you to install vv4ml using any scientific Python distribution of your interest
or a Python package manager (the preferred way).

.. Contents::

Requirements
============

It is required the following:

1) `Python`_ >= 3.11(.x)
2) (optional) `UV`_ >= 0.11(.x)

.. _Python: https://www.python.org/
.. _UV: https://docs.astral.sh/uv/

Building
========

TBA

Installing
==========

pip
---

The standard way to install vv4ml is via pip. We suggest to update pip before following this
guidelines::

    pip install --upgrade pip
Then install vv4ml::

    pip install vv4ml
UV
--

The preferred way to install vv4ml is via `<UV_>`_::

    uv add vv4ml
or::

    uv pip install vv4ml
based on your needs.

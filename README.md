![Controlboros logo](docs/_static/images/logo.png)

[![Build Status](https://travis-ci.org/mp4096/controlboros.svg?branch=master)](https://travis-ci.org/mp4096/controlboros)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/6dd7592be425486081fbe2cb859c2426)](https://www.codacy.com/app/mp4096/controlboros?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mp4096/controlboros&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/mp4096/controlboros/badge.svg?branch=master)](https://coveralls.io/github/mp4096/controlboros?branch=master)

Controlboros is a simple framework for simulating control systems.

### FAQ


> Why should I use Controlboros if I can simulate systems using `python-control` or `scipy.signal`?

Controlboros allows greater flexibility in the implementation of your system blocks.
Example: You've written a controller in C and you want to test it. To this end,
you can write a thin wrapper calling the controller binary (e.g. using `cffi`) and
implement remaining blocks in Python.
Or you can even setup a MATLAB engine interface
and call into MATLAB code.


> Can Controlboros handle control loops? Algebraic loops?

Kind of. You can resolve a loop by adding a unit delay.


> But then you'll get an inexact solution!

[¯\\\_(ツ)_/¯](https://cloud.githubusercontent.com/assets/5394551/26149729/b23b51b0-3afb-11e7-89de-f3ddd9b02a0c.gif)

But seriously, you should simulate with small time steps.
[This notebook](examples/simple_control_loop.ipynb)
demonstrates the consequences of a unit delay in the feedback loop for different time step sizes.
You can also [wrap your systems](examples/simple_multi-rate_simulation.ipynb)
to simulate with a smaller time step than the discretisation time step.


> What's the deal with `push_stateful` and `push_pure`?

`push_pure` may not have any side effects and is intended for testing
(yes, you should definitely unit test your code!).
`push_stateful` has the side effect of mutating the system state.
It is useful when writing actual simulation code.


> Help! I get different simulation results each time I run my code cell!

Do you use `push_stateful`? If yes, do you set the initial state explicitly
before running the simulation?


### Requirements

* Python ≥ 3.5

### Hacking

Quickstart:

```
make init         # install dependencies
make init-dev     # install dev dependencies
make install-dev  # install Controlboros
```

### Documentation

Quickstart:

```
make docs-api-serve
```

The API reference is now served on port 8071.

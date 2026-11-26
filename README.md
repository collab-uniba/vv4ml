<h1 align="center">
    <img src="https://raw.githubusercontent.com/collab-uniba/vv4ml/refs/heads/main/logo/vv4ml-logo-icon.png" width="350" alt="logo">
</h1><br>

![Python Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fcollab-uniba%2Fvv4ml%2Fmain%2Fpyproject.toml&query=%24.project.requires-python&style=for-the-badge&label=py_ver)
![Project Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fcollab-uniba%2Fvv4ml%2Fmain%2Fpyproject.toml&query=%24.project.version&style=for-the-badge&label=COLLAB&labelColor=yellow&color=%23284169)

**vv4ml** is the verification and validation library for your machine learning models and
simulations. vv4ml is particular useful for predictors in medical systems.

It offers tools for performing tests that assess credibility.

In particular:

- Verification tests: `existence`, `convergence`, `numerical error` and `smoothness`, plus a `validation` test, with acceptance thresholds declared in the configuration
- A single configuration-driven entry point, `run_verification`, that runs all configured tests for an output and returns a serialisable report
- A useful Latin hypercube sampling method: it has available a `base` and `maximin` optimization method
- Different metrics: `MeanRelativeError`, `MeanAbsoluteError`, `MaximumRelativeError`, `MaximumAbsoluteError`
- YAML parser and loader: a convenient configuration file parser and loader for your model card document
- `Pydantic` models: most of the project is Pydantic structured whose models are validated at runtime

## Get started

Refer to the [Read the Docs guide here](https://vv4ml.readthedocs.io/en/latest/) where you will find everything you need.

For a general overlook check the [index](https://github.com/collab-uniba/vv4ml/blob/main/docs/index.md).

For a quick try look at the [Jupyter notebook](https://github.com/collab-uniba/vv4ml/blob/main/vv4ml_quickstart.ipynb).

## Install

See the [installation page](https://github.com/collab-uniba/vv4ml/blob/main/docs/install.md) for further information.

You can quickly install vv4ml with:
```bash
pip install vv4ml
```

## Contributing

TBA

## License

vv4ml is released under the [Apache License 2.0](LICENSE).

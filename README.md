# crm-ai

## Virtual develop environment
<prev>

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install -e ".[dev]"
    pre-commit install
    pre-commit autoupdate
<prev>


## create a new face recognition register

<prev>

    python3 vision_analytic/main.py createregister
<prev>

## use face recognition system

<prev>

    python3 vision_analytic/main.py watchful
<prev>
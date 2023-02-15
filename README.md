# crm-ai

This project is about a crm (customer relationship management) system integrated with computer vision capabilities such as facial recognition and other artificial intelligence techniques.

The main idea for the development of this project is to answer the question "How to make the crm can see".

## Virtual develop environment
<prev>

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    export PYTHONPATH="${PYTHONPATH}:${PWD}"
    
<prev>


## create a new face recognition register

<prev>

    python3 vision_analytic/main.py createregister
<prev>

## use face recognition system

<prev>

    python3 vision_analytic/main.py watchful
<prev>

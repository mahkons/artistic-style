# artistic-style
Implementation of the [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) paper

# Installation

clone repository  

    git clone https://github.com/mahkons/artistic-style
    cd artistic-style
create and activate conda env (optional)  

    conda env create -f environment.yml
    conda activate artistic-style

# Launch
    python transfer.py --content-image="PATH_TO_CONTENT" --style-image="PATH_TO_STYLE" --output-image="PATH_TO_OUTPUT"

# Results
    Van Gogh's Starry Night to Yekaterinburg City Administration building  
    [night_to_ekb](generated/night_ekb_1.jpg)  

    Edvard Munch's The Scream to Yekaterinburg City Administration building  
    [scream_to_ekb](generated/scream_ekb_1.jpg)  

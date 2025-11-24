# ComfyUI-Chord

<a href="https://arxiv.org/abs/2509.09952"><img src="https://img.shields.io/badge/arXiv-2509.09952-B31B1B?logo=arxiv&logoColor=white&style=flat-square" alt="arXiv"></a>
<a href="https://ubisoft-laforge.github.io/world/chord/"><img src="https://img.shields.io/badge/Project-Page-brightgreen?logo=ubisoft&logoColor=white&style=flat-square" alt="Project Page"></a>
<a href="https://github.com/ubisoft/ubisoft-laforge-chord"><img src="https://img.shields.io/badge/Github-Code-blue?logo=github&logoColor=white&style=flat-square" alt="Code"></a>

ComfyUI Custom Node for paper: **Chord: Chain of Rendering Decomposition for PBR Material Estimation from Generated Texture Images**

## Installation

1. Download and install ComfyUI. If you are new to ComfyUI, refer to the original [repository](https://github.com/comfyanonymous/ComfyUI) to get started. **Make sure you have the latest version.**

2. Download the pretrained model from [Hugging Face](Ubisoft/ubisoft-laforge-chord) and place the **chord_v1.ckpt** file into the **ComfyUI/models/checkpoints** folder.

3. Install the custom nodes from the [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager) or manually cloning this repository recursively in the custom nodes folder of ComfyUI:

    ```shell
    # 1. Clone repo
    cd ./ComfyUI/custom_nodes
    git clone --recursive https://github.com/ubisoft/ComfyUI-Chord.git
    # 2. Install dependencies
    ## For python version
    cd ComfyUI-Chord
    pip install -r requirements.txt
    ## For windows portable version
    ..\..\python_embeded\python.exe -s -m pip install -r .\ComfyUI-Chord\requirements.txt
    ```
## Example Workflow

![Example workflow](examples/chord_example.png)

You can load this workflow using the JSON file:
`examples/chord_example.json`

## License

This project is released under the **Ubisoft Machine Learning License (Research-Only - Copyleft)**. See the full terms in the [LICENSE](LICENSE) file
## Citation

If you find our work useful, please consider citing:

```
@misc{ying2025chord,
    title={Chord: Chain of Rendering Decomposition for PBR Material Estimation from Generated Texture Images}, 
    author={Zhi Ying and Boxiang Rong and Jingyu Wang and Maoyuan Xu},
    year={2025},
    eprint={2509.09952},
    archivePrefix={arXiv},
    primaryClass={cs.GR},
    url={https://arxiv.org/abs/2509.09952}, 
}
```

© [2025] Ubisoft Entertainment. All Rights Reserved.
<div align="center">

# TUNET
A direct, pixel-level mapping from src to dst images via encoder-decoder.   
Supports, training, inference or export to Compositor tools such Foundry Nuke or Autodesk Flame native inference.


</div>

## 🎥 Showcases: 
AOVs:   
![540_ezgif-45f24ebccb11e0](https://github.com/user-attachments/assets/507167b0-c473-44c0-b444-55f48bc7843b)   
https://youtu.be/TwvN8axWJLY   
Models from the video are available for Nuke and Flame can be downloaded link below for test locally:   
https://f.io/HovatFeX   
   
Models has been trained in combination with Nvidia Cosmos foundation models.   
Using 8x B200 GPUs. Inference can be done in consumer GPUs.



Rain:   
Flame:   
[![Flame video](https://img.youtube.com/vi/6-OFAJtfliM/hqdefault.jpg)](https://youtu.be/6-OFAJtfliM)


## For Windows:

Download this fork, then run the PowerShell installer from its folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install.ps1
```

It fetches the selected branch from GitHub, installs Python 3.12 if needed, creates a private virtual environment, installs the latest stable PyTorch CUDA 13.2 wheels from the official PyTorch index and adds a TuNet shortcut to the Start menu. A compatible NVIDIA driver is required; the separate CUDA Toolkit is not.

For CPU-only installation, use `.\install.ps1 -CpuOnly`. To install another fork or branch:

```powershell
.\install.ps1 -RepositoryOwner SEU_USUARIO -RepositoryName tunet -Branch windows
```

## For macOS:   
✅ Make sure Miniconda or Anaconda is installed:
###[[Install Video](https://youtu.be/QaAca_LiwKc))]

```
git clone --branch multios --single-branch https://github.com/tpc2233/tunet.git
cd tunet

conda create -n tunet python=3.10 -y
conda activate tunet

pip install torch torchvision torchaudio  

pip install onnx pyyaml lpips onnxruntime Pillow albumentations PySide6

```



## For Linux and Multi-GPU use the dedicated Branch:   

```
check branches
```


✅ How to use, Open Tunet UI: 
###[[Training Video soon]] 
```
python ui_app.py

You are good to go!
```


✅ Tunet UI:
## Main:
![main](https://github.com/user-attachments/assets/fe8f03e1-2e53-46b9-8c91-b85befdb3ff9)

## Training:
![train5](https://github.com/user-attachments/assets/f6bf9ba3-f84b-4d0d-9d95-ff418bebaaf5)

## Converter:
![convert](https://github.com/user-attachments/assets/8633b5ff-0152-4a09-afaf-1fa7e4669359)

## Adv:
![adv](https://github.com/user-attachments/assets/f27df935-6677-4c8f-adfb-dbc3b7cd8c9d)

## Data control:
![data](https://github.com/user-attachments/assets/a30e0160-dc23-4623-9106-d3237700be94)



### Terminal CMD:

✅ Command line and terminal trainings are still available as usual, check docs:
```
soon
```


## Video:
[[video 🤗](https://youtu.be/UyMq0bsny-A)]


### Citation

Consider cite TUNET in your project.
```
@article{tpo2025tunet,
  title={TuNet},
  author={Thiago Porto},
  year={2025}
}
```

## License

The source code is licensed under the Apache License, Version 2.0.
Commercial use Permission 



## Créditos do fork Windows

Projeto, arquitetura e modelo originais por **Thiago Porto (@tpc2233)**: [tpc2233/tunet](https://github.com/tpc2233/tunet). Este é um fork comunitário, sem endosso do autor upstream. Licença Apache-2.0: preserve os avisos de copyright e [LICENSE](LICENSE).

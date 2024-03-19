# Mosquitoes Breeding Grounds Detector 🦟🔎

## Description
The MBG detection project is an ongoing work by several computer vision and signal processing researchers from the Federal University of Rio de Janeiro, with contributions from CEFET. It is currently funded by CAPES through master's and doctoral scholarships.
This project involves the detection of breeding grounds for mosquitoes using aerial footage collected by a DJI Phantom Vision 4 PRO UAV drone in Rio de Janeiro. Some videos were filmed in real urban areas, while others were produced by students in low grass fields, with tires randomly placed within reach of the drone.

## The team 👥
- Isabelle Melo (me), Msc. Student at Federal University of Rio de Janeiro
- Mila Rodrigues, Msc. Student at Federal University of Rio de Janeiro
- Wesley Passos, Phd. Student at Federal University of Rio de Janeiro
- João Paulo Rangel, Undergraduate Student at Federal University of Rio de Janeiro
- Kauan Moura, Undergraduate Student at Federal University of Rio de Janeiro
- Eduardo Antônio, Professor at Federal University of Rio de Janeiro
- Sergio Lima, Professor at Federal University of Rio de Janeiro

## Diseases Transmitted by Mosquitoes
Mosquitoes are vectors for various diseases, including:
- **Dengue Fever**: A mosquito-borne viral infection causing severe flu-like symptoms.
- **Zika Virus**: Transmitted primarily by Aedes mosquitoes, causing mild symptoms but severe birth defects if contracted during pregnancy.
- **Malaria**: Caused by Plasmodium parasites transmitted through the bites of infected Anopheles mosquitoes.

| ![mosquitoes.jpg](https://www.cnnbrasil.com.br/wp-content/uploads/sites/12/2024/03/aedes-aegypti.jpg?w=400&h=220&crop=1) | 
|:--:| 
| *Aedes Aegypti Mosquitoe* |

In Brazil, there is a serious challenge in containing diseases caused by mosquitoes, and every year during warmer seasons is when they reproduce the most, leading to significant harm to the population: https://veja.abril.com.br/saude/cidade-de-sao-paulo-deve-decretar-emergencia-por-epidemia-de-dengue



## Project Status
This project is currently under construction, and there is no framework yet for the detection of all objects. The main focus of this repository is the detection of tires, which are common breeding grounds for mosquitoes.

## MBG Dataset
The project includes a dataset called MBG, which contains 13 videos filmed by a DJI Phantom Vision 4 PRO UAV drone in Rio de Janeiro. The table below provides details about each video in the dataset:

| Video Name            | Altitude (m)      | Duration | Number of Frames | Resolution | Terrain Type    |
|-----------------------|----------------|----------|------------------|------------|-----------------|
| video01               |  10  | 02:32    | 7328             | 3840x2160      | Low grass       |
| video02               | 15  | 00:23    | 1128             | 3840x2160       | Grass and asphalt |
| video03               | 40 | 03:44 | 10749      | 3840x2160       | Grass, asphalt, and buildings |
| video04               |  15     | 02:06    | 6076             | 3840x2160       | Grass and asphalt |
| video05               | 10 | 00:41  | 1970             | 3840x2160       | Vacant lot      |
| video06               |   16         | 01:15    | 3602             | 3840x2160       | Vacant lot      |
| video07               |    20  | 03:07    | 8988             | 3840x2160       | Vacant lot      |
| video08               | 10 | 01:37 | 4672         | 3840x2160       | Vacant lot      |
| video09               |  40 | 02:41    | 7721             | 3840x2160       | Grass, asphalt, and buildings |
| video10               |    40  | 05:27    | 7858             | 4096x2160       | Urban area      |
| video11               |     40 | 05:27    | 7854             | 4096x2160       | Urban area      |
| video12               |       40   | 04:33    | 6570             | 4096x2160       | Urban area      |
| video13               |         40  | 03:30    | 5046             | 4096x2160       | Urban area      |


---

## Installation

To get started, ensure that you are using a Conda environment with Python 3.8+ and CUDA 11.7+

First, execute the following command in your terminal to install the project dependencies from the `requirements.yml` file:

```bash
conda env create -f requirements.yml
```


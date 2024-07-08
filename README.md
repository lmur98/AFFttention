

# AFFttention
Official repository of "AFF-ttention! Affordances and Attention models for Short-Term Object Interaction Anticipation", accepted at ECCV 24'!!
[Lorenzo Mur-Labadia*](https://sites.google.com/unizar.es/lorenzo-mur-labadia/inicio)
[Ruben Martinez-Cantin](https://webdiis.unizar.es/~rmcantin/)
Josechu Guerrero Campo
[Giovanni Maria Farinella](http://www.dmi.unict.it/~gfarinella)
[Antonino Furnari](https://www.antoninofurnari.it/)

- [04/07/2024] We update the repository with the affordances annotations and descriptors and EPIC-Kitchens STA annotations.
- [01/07/2024] Our paper was accepted at ECCV 2024! 🎉
- [16/06/2024] We won the 2nd 🥈🏆 place at the Ego4D Short Term Anticipation Challenge at CVPR 24'. Check out the [technical report](https://arxiv.org/pdf/2407.04369) 📃
- [05/06/2024] Check a pre-print of our paper on [Arvix](https://arxiv.org/pdf/2406.01194) 📃

![Teaser gif](images_GitHub_Affttention/teaser.gif)

## Table of Contents
- [STAformer 🤖](#staformer-)
- [Environment Affordances as a Persistent Memory 🧠💭🧠](#environment-affordances-as-a-persistent-memory-)
- [Interaction Hotspots 🫳🏻 🖐🏻](#interaction-hotspots-)
- [Results and Weights 🥈🏆](#results-and-weights-)
- [Run the Code! 🧑🏽‍💻](#run-the-code-)

### STAformer 🤖

* STAformer extracts DINOv2 features from the still image 📷 and TimeSformer features from the video 🎥.
* The novel Frame-Guided Temporal Pooling Attention projects the video features on the last-frame reference.
* The Dual Image-Video Cross-Attention refines both modalities
* We apply a multi-scale fusion
* We adapt the Fast-RCNN head to our STA task.

![Backbone](images_GitHub_Affttention/architecture.png)

### Environment affordances as a persistent memory 🧠💭🧠

We first leverage environment affordances, estimated by matching the input observation to a learned affordance database, to predict probability distributions over nouns and verbs, which are used to refine verb and noun probabilities predicted by STAformer. Our intuition is that linking a zone across similar environments captures a description of the feasible interactions, grounding predictions into previously observed human behavior

![Env_Aff](images_GitHub_Affttention/env_aff.png)



You can find the topological nodes of the [training](https://drive.google.com/drive/folders/1kMPBrhFyDjICl4I2cvstZGUxJjFg3WQQ?usp=drive_link) and [validation](https://drive.google.com/drive/folders/11ImgNItf6rdvvXOWvNgF1YBTyzSzUn8F?usp=drive_link) videos. Each directory is a list of dicts, where each dict contains the different topological zones of that video, their STA annotations, the Ego4D narrations and the respective video frames. Note that a zone is just the topological nodes of EgoTopo, obtained with the mentioned Siamese R-18.

For the affordances propagation, we try different alternatives (using only the topological nodes or aggrupating in cross-environmental clusters). The best version work with only topological nodes (propagate_aff_from_node.py), but feel free to try the other alternatives.

We propagate the STA across the zones in order to extract the affordances. Here is the [AFF per zone](https://drive.google.com/file/d/1Jf7MjjDZaIkAR-ctu8n7qUBqp0UeJfuV/view?usp=drive_link) and the [visual and text descriptors](https://drive.google.com/drive/folders/1EoD2nMwOC0Vh_aBnUZ4eb6x57DnKJQla?usp=drive_link) dataset.

### Interaction hotspots 🫳🏻 🖐🏻

The interaction hotspots relate STA predictions to a spatial prior of where an interaction may take place in the current frame. This is done by predicting an interaction hotspot, which is used to re-weigh confidence scores of STA predictions depending on the object’s locations. We follow "Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos" (CVPR 2022) by Liu et. al to extract the interaction hotspots. We also follow "Understanding Human Hands in Contact at Internet Scale (CVPR 2020, Oral)" by Shan et. al to extract hand annotations on the Ego4D STA split. Here is the json with the pre-extracted [hands](https://drive.google.com/file/d/1AI6a4EwD8QZmJcXcS8ohO2-uGoJAE7uh/view?usp=drive_link).

![Env_Aff](images_GitHub_Affttention/int_hotspots.png)

# Results and weights 🥈🏆

| Model               | N      | N + V  | N + δ  | All  |
|---------------------|--------|--------|--------|------|
| StillFast           | 25.06  | 13.29  | 9.14   | 5.12 |
| GANO v2             | 25.67  | *13.60* | 9.02   | 5.16 |
| Language NAO        | *30.43* | 13.45  | *10.38* | *5.18*|
| STAformer           | 30.61  | 16.67  | 10.06  | 5.62 |
| STAformer+AFF       | 32.39  | 17.38  | 10.26  | 5.70 |
| STAformer+MH        | 31.99  | 16.79  | 11.62  | 6.72 |
| STAformer+MH+AFF    | **33.50**| **17.25**| **11.77**| **6.75**|

*Italic numbers indicate the best performance of the previous SOTA.
**Bold numbers indicate the best overall performance.
  
- Download the weights of STAFormer+MH trained on the v2 split [Weights](https://drive.google.com/xxxxx)
  
- EPIC-KITCHENS STA ANNOTATIONS [training](https://drive.google.com/file/d/1VZwi69chFYZbMLJyL8SAu7Q5JbKoyy3Z/view?usp=drive_link) and [validation](https://drive.google.com/file/d/1z01Qp5Sy4UMcAdJZhQNzyGaMn4ds35wj/view?usp=drive_link). You can download the videos and images on the EPIC-KITCHENS official webpage.

# Run the code! 🧑🏽‍💻 🧑🏽‍💻

Follow [StillFast repository](https://github.com/fpv-iplab/stillfast/tree/master) for running the code 

For any questions, contact me at lmur@unizar.es 

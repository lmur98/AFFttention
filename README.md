

# AFFttention
Official repository of "AFF-ttention! Affordances and Attention models for Short-Term Object Interaction Anticipation".

We win the 2nd 🥈 place at the Ego4D Short Term Anticipation Challenge at CVPR 24'.

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

### Interaction hotspots 🫳🏻 🖐🏻

The interaction hotspots relate STA predictions to a spatial prior of where an interaction may take place in the current frame. This is done by predicting an interaction hotspot, which is used to re-weigh confidence scores of STA predictions depending on the object’s locations.

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

Download the weights of STAFormer trained on the v1 split [Weights](https://drive.google.com/xxxxx)
Download the weights of STAFormer+MH trained on the v2 split [Weights](https://drive.google.com/xxxxx)
We also provide the different nodes in Ego4D STA [Env Aff Nodes V1](https://drive.google.com/xxxxx) [Env Aff Nodes V2](https://drive.google.com/xxxxx) 


# Run the code! 🧑🏽‍💻 🧑🏽‍💻

Follow [StillFast repository](https://github.com/fpv-iplab/stillfast/tree/master) for running the code 

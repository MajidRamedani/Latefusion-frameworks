# **Hybrid Multimodal Late Fusion Frameworks for bvFTD Classification in Imbalanced Dementia Datasets**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-2.20-black)
![NumPy](https://img.shields.io/badge/Numpy-2.2-yellow)
![Aucmedi](https://img.shields.io/badge/Aucmedi-0.11-brown)

**This repository contains the official implementation of our proposed method.**  
The corresponding manuscript is currently under review.

---

## **Abstract**  
Behavioral variant frontotemporal dementia (bvFTD) is an irreversible neurodegenerative disorder characterized by progressive changes in personality and behavior. However, due to the low prevalence of bvFTD among neurodegenerative diseases causing the dementia syndrome, conventional machine learning approaches may struggle to capture comprehensive feature representations.  
Therefore, this study proposes two late fusion frameworks that integrate a 3D convolutional neural network and a multilayer perceptron (MLP) for improved bvFTD diagnosis.  
<p align="center">
        <img src="Images/Late Fusion Frameworks.png" width="700">
</p>


To address class imbalance, bvFTD data were initially augmented. A 3D DenseNet was used to extract features from 3D T1-weighted MRI scans, while a MLP was applied to regional brain volumetric measurements obtained from automated MRI-based brain segmentation. Both fusion strategies improved accuracy, F1-score, and AUC compared to the baseline model without data augmentation.

---

## **3D CNN Model Architecture**  
We chose to implement 3D-DenseNet Architecture for our study. See [src/models](src/models) for implementation details.  
Following image illustrates the DenseNet model architecture that was utilized in this study.

<p align="center">
	<img src="Images/DenseNet.png" width="700">
</p>


## **Key Findings**  
Following is the mean relevance map for the HC, AD and bvFTD groups obtained using the LRP relevance propagation method for the trained Densenet model. Coronal slices Y=[90,117,125,135] in MNI reference space are shown.

<p align="center">
        <img src="Images/Relevance Map.png" width="700">
</p>


Our results demonstrate that data augmentation leads to improvements in accuracy, F1-score and AUC. We also see significant improvements not only in fully connected settings but also in ensemble-based models.

---

## Citation

If you use this repository or the PrimUNet model in your research, please cite:

```bibtex
@ARTICLE{RAMEDANI2026,
AUTHOR={Ramedani, Majid  and Terli, Jaya C.  and Singh, Devesh  and Peters, Oliver  and Hellmann-Regen, Julian  and Priller, Josef  and Spruth, Eike Jakob  and Spottke, Annika  and Boehlen, Anne  and Weydt, Patrick  and Wüllner, Ullrich  and Dinter, Elisabeth  and Günther, Rene  and Wiltfang, Jens  and Schott, Björn H.  and Düzel, Emrah  and Glanz, Wenzel  and Buerger, Katharina  and Janowitz, Daniel  and Levin, Johannes  and Stockbauer, Anna  and Mladinov, Mihovil  and Prudlo, Johannes  and Hermann, Andreas  and Synofzik, Matthis  and Mengel, David  and Petzold, Gabor C.  and Schneider, Anja  and Lüsebrink, Falk  and Hetzer, Stefan  and Dechent, Peter  and Ewers, Michael  and Scheffler, Klaus  and Stöcklein, Sophia  and Teipel, Stefan  and Dyrba, Martin },      
TITLE={Hybrid multimodal late fusion frameworks for bvFTD classification in imbalanced dementia datasets},     
JOURNAL={Frontiers in Aging Neuroscience},     
VOLUME={Volume 18 - 2026},
YEAR={2026},
URL={https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2026.1892568},
DOI={10.3389/fnagi.2026.1892568},
ISSN={1663-4365}}
```

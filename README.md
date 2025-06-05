# DHCLHAM
**DHCLHAM: DHCLHAM: Microbe-Drug Interaction Prediction Based on Dual-Hypergraph Contrastive Learning Framework with Hierarchical Attention Mechanism**

### Dataset ###
  * MDAD: MDAD database, created by Sun et al. in 2018, was assembled through the meticulous curation of experimentally and clinically validated microbe-drug interactions sourced from existing drug databases and scientific literature. It comprises 2,470 verified records of microbe-drug associations, involving 1,373 drugs and 173 microbes. The dataset can be accessed publicly at http://chengroup.cu-mt.edu.cn/MDAD.
  * aBiofilm: aBiofilm is a database of anti-biofilm agents that catalogs 1,720 compounds targeting 140 microbial species. The database documents details for each anti-biofilm drug, including molecular structure, drug classification, antimicrobial potency, and citations.
  * DrugVirus: DrugVirus is a specialized database that records the activity of pharmaceuticals aimed at human viruses and their interactions. It is intended to enable the investigation and assessment of broad-spectrum antiviral drugs (BSAs), which are agents that suppress various human viruses, along with categories of drugs that include BSAs. The database can be accessed at https://drug-virus.info/.

### Data description ###


### Run Step ###
  Run main.py to train the model and obtain the predicted scores for microbe-drug associations.


### Requirements ###
  - python==3.9.0
  - pytorch==2.1.0 
  - scikit-learn==1.5.1
  - numpy==1.26.3
  - scipy==1.13.3


### Citation ###
Please kindly cite the paper if you use refers to the paper, code or datasets.

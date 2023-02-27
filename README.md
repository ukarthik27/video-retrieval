


# Video Search using Dual Encoders

## Goal 
Efficiently learn visual concepts from natural language supervision which aids in querying video
databases with text input 
## Approach
1. Architect a dual Encoder, one for text and other for image from the Nocaps dataset. 
2. Experimented with the efficacy of Inception, Xception, ResNet and YOLO for Image Encoders and BERT
vs GloVe for Text Encoders in addition to hyperparameter tuning.

## Result

### Best model

<img src="https://github.com/ukarthik27/video-retrieval/blob/master/bestmodel.png"  width="300" />

1. Enabled zero shot learning with accuracy of 65% on MSRVTT dataset with training on Nocaps dataset.
2. Enabled quicker comparison of embeddings by Integrating Miluvs database to compare embeddings of
image-caption sampled from video with the input query

### Training of Dual encoders
<img src="https://github.com/ukarthik27/video-retrieval/blob/master/Training.png" width="350" height="450" />

### Constructing Milvus vector database
<img src="https://github.com/ukarthik27/video-retrieval/blob/master/DB Load.png" width="150" height="300" />

### Retreival pipeline
<img src="https://github.com/ukarthik27/video-retrieval/blob/master/Search.png" width="150" height="300" />

## Documentation
1. Please install Milvus on your system (https://milvus.io/?gclid=EAIaIQobChMIwfuhjNOQ_AIVEovICh2a5Qq0EAAYAiAAEgK_FfD_BwE)
2. Install pymilvus using pip (https://pymilvus.readthedocs.io/en/latest/)

3. The notebooks “BaseModel”, “BestModel” contain the code to train the Base and the Best models as described in the report.

4. Preprocessors.py, EncoderHead.py, Encoders.py and utils.py contain supporting code for the notebooks. These scripts contains the classes developed for the custom models and functions to support these models.

5. Data_preparation.ipynb consist of the code to prepare the dataset

The nocaps dataset used for training and the MSRVTT dataset used for doing the video search are here: https://drive.google.com/drive/folders/1w8Cua9_l54ghqayMm3RyVFqtk9ClnQX5?usp=share_link

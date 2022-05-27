# Social-Network  
This is CSIT 6000K social network course project. The project topic is "User Churn Prediction on YELP".  
In this project, I performed user behavior analysis on the YELP public dataset and implemented user churn prediction. We used techniques such as data visualization, social networks, text semantic analysis and deep learning based user churn prediction. The technical details and results are presented in the final report. Here is our code and how to run it.

# The runtime environment:  
Language: python 3.6 or above  
RAM: 56G  
CPU: 6-core 2.4GHz CPU  
GPU: K80 (no advanced graphics device required)  

# library requirement:  
The detailed dependencies are listed in requirements.txt and can be installed using the following code:  

    pip install requirements.txt  

# Dataset:  
YELP public dataset: https://www.yelp.com/dataset  
Preprocessed dataset:  
> user preprocess dataset: https://drive.google.com/file/d/1o8cW4FWO0m7mVJFHMlUeGvNWMr361iZ7/view?usp=sharing  
> review preprocess datset: https://drive.google.com/file/d/15J4q-2WYxW5eIh6-A7ggM6jan8wmvDnk/view?usp=sharing  

# Run method:
1. Download the required YELP dataset in the data folder and run Churn.py to see the results of our additional experiments. (The extra experiment is an attempt to improve our work)   
2. Use the preprocessed dataset, download it in the data folder, and run churn.ipynb to run the LR model and Semi-RNN model we mentioned in final report. The preprocessed dataset can be downloaded as follows (Google Cloud Drive is valid for 30 days):  

```shell
gdown 15J4q-2WYxW5eIh6-A7ggM6jan8wmvDnk  
gdown 1o8cW4FWO0m7mVJFHMlUeGvNWMr361iZ7
```

Or use the data process program including dataLoad.py, dataProcess.py, features.py and nlp.py to process the YELP public dataset and then save it as a pickle file and open it with churn.ipynb.  

# checkpoint:
I have saved the best checkpoint of the LR model, please load it into churn.ipynb and use it to view the results.

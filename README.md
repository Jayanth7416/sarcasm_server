# sacrasm_server
{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Roman;\f1\fmodern\fcharset0 Courier;\f2\froman\fcharset0 Times-Bold;
\f3\fmodern\fcharset0 Courier-Bold;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red109\green109\blue109;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c50196\c50196\c50196;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}}
\margl1440\margr1440\vieww29200\viewh15820\viewkind0
\deftab720
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Here is the structure of the project directory:
\f1\fs26 \
\pard\pardeftab720\partightenfactor0
\cf0 sarcasm-detection-project/\
\uc0\u9500 \u9472 \u9472  EDA.ipynb                    # Exploratory Data Analysis (EDA) and preprocessing of the dataset\
\uc0\u9500 \u9472 \u9472  LightGBM_model.ipynb          # Implementation of the LightGBM model for sarcasm detection\
\uc0\u9500 \u9472 \u9472  LSTM_glove_model.ipynb        # LSTM model using GloVe embeddings for sarcasm detection\
\uc0\u9500 \u9472 \u9472  RNN_LSTM_GRU_Models.ipynb     # Implementation of RNN, LSTM, and GRU models for sarcasm detection\
\uc0\u9500 \u9472 \u9472  BERT_MODEL.ipynb              # BERT model for sarcasm detection (Best Performing Model)\
\uc0\u9500 \u9472 \u9472  sarcasm_model/                # Pre-trained transformer model (BERT) and tokenizer directory\
\uc0\u9500 \u9472 \u9472  sarcasm_model_lstm/           # Pre-trained LSTM model and tokenizer directory\
\uc0\u9500 \u9472 \u9472  requirements.txt              # List of required Python libraries and dependencies\
\uc0\u9492 \u9472 \u9472  README.md                     # Project description and instructions (this file)\
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf3 \strokec3 \
\pard\pardeftab720\sa298\partightenfactor0

\f2\b\fs36 \cf0 \strokec2 Setup and Installation\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 To set up the sarcasm detection project and install all the necessary dependencies, follow these steps:
\f1\fs26 \
\pard\pardeftab720\sa280\partightenfactor0

\f2\b\fs28 \cf0 1. Install Required Libraries\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 Install the required libraries using 
\f1\fs26 pip
\f0\fs24 . It is recommended to use a virtual environment for managing dependencies.\
\pard\pardeftab720\sa319\partightenfactor0

\f2\b \cf0 Option 1: Using 
\f3\fs26 requirements.txt
\f2\fs24 \
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0 \cf0 If the project includes a 
\f1\fs26 requirements.txt
\f0\fs24  file (which lists all dependencies), install them by running the following command:
\f1\fs26 \
pip install -r requirements.txt\
\pard\pardeftab720\sa319\partightenfactor0

\f2\b\fs24 \cf0 Option 2: Manually Installing Libraries\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0 \cf0 Alternatively, install the necessary libraries individually:
\f1\fs26 \
\pard\pardeftab720\partightenfactor0
\cf0 pip install streamlit transformers torch shap tensorflow lightgbm numpy matplotlib scikit-learn pickle-mixin transformers-interpret\
\
\pard\pardeftab720\sa280\partightenfactor0

\f2\b\fs28 \cf0 2. Download Pre-trained Models\

\f0\b0\fs24 Download the pre-trained models (BERT and LSTM) and place them in the respective directories:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls1\ilvl0
\f2\b \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Transformer (BERT) model
\f0\b0  should be saved in the 
\f1\fs26 ./sarcasm_model
\f0\fs24  directory.\
\ls1\ilvl0
\f2\b \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 LSTM model
\f0\b0  should be saved in the 
\f1\fs26 ./sarcasm_model_lstm
\f0\fs24  directory.\
\pard\pardeftab720\sa240\partightenfactor0
\cf0 These models and tokenizers should be available from the project repository or need to be trained and saved before use.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa298\partightenfactor0

\f2\b\fs36 \cf0 \strokec2 Running the Files\
\pard\pardeftab720\sa280\partightenfactor0

\fs28 \cf0 1. Exploratory Data Analysis (EDA.ipynb)\
\'95	Purpose: This notebook is used to explore the dataset, perform data preprocessing like text cleaning, tokenization, and feature extraction, and visualize the distribution of sarcastic vs. non-sarcastic sentences.\
\'95	How to Run:\
	\uc0\u9702 	Open the notebook EDA.ipynb in Jupyter Notebook or Google Colab.\
	\uc0\u9702 	Execute each cell to explore and preprocess the dataset.\
	\uc0\u9702 	The output will show insights about the data, including visualizations like word clouds and bar plots.\
\
2. LightGBM Model (LightGBM_model.ipynb)\
\'95	Purpose: This notebook implements the LightGBM model for sarcasm detection using traditional machine learning features such as TF-IDF or Bag-of-Words.\
\'95	How to Run:\
	\uc0\u9702 	Open the notebook LightGBM_model.ipynb in Jupyter Notebook or Google Colab.\
	\uc0\u9702 	Run each cell to train the LightGBM model.\
	\uc0\u9702 	The notebook will output the model accuracy, confusion matrix, and classification report to evaluate the model's performance.\
\
3. LSTM with GloVe Embeddings (LSTM_glove_model.ipynb)\
\'95	Purpose: This notebook uses the LSTM model coupled with GloVe embeddings for sarcasm detection. The LSTM model variant works well in handling sequential data and captures long-range relations in text.\
\'95	How to Run:\
	\uc0\u9702 	Open LSTM_glove_model.ipynb in Jupyter Notebook or Google Colab\
	\uc0\u9702 	Run the cells to load the LSTM Model, tokenize the input text, and train the model\
	\uc0\u9702 	Look into the performance of the model and view the results.\
\
4. RNN, LSTM, and GRU Models (RNN_LSTM_GRU_Models.ipynb)\
\'95	Purpose: This notebook contains the implementation of three different sequential models, namely RNN, LSTM, and GRU, for sarcasm detection.\
\'95	How to Run:\
	\uc0\u9702 	Open the notebook RNN_LSTM_GRU_Models.ipynb in Jupyter Notebook or Google Colab.\
	\uc0\u9702 	Run the cells to train and evaluate the models.\
	\uc0\u9702 	The notebook compares the performance of RNN, LSTM, and GRU with respect to accuracy and processing time.\
\
5. BERT Model (BERT_MODEL.ipynb)\
\'95	Purpose: This notebook implements the BERT model for sarcasm detection. BERT gives the best performance since it is able to capture deep contextual relationships in the text.\
\'95	How to Run:\
	\uc0\u9702 	Open the notebook BERT_MODEL.ipynb in Jupyter Notebook or Google Colab.\
	\uc0\u9702 	Run the cells to load the pre-trained BERT model and tokenizer, and process input text to make predictions.\
	\uc0\u9702 	The notebook will output the accuracy, confusion matrix, and classification report.
\f0\b0\fs24 \
}

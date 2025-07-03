# Converted from disaster-tweet-sb.ipynb - Markdown format optimized for LLM readability

```python
## This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
    
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/nlp-getting-started/sample_submission.csv
    /kaggle/input/nlp-getting-started/train.csv
    /kaggle/input/nlp-getting-started/test.csv
    /kaggle/input/distil_bert/keras/distil_bert_base_en_uncased/3/config.json
    /kaggle/input/distil_bert/keras/distil_bert_base_en_uncased/3/tokenizer.json
    /kaggle/input/distil_bert/keras/distil_bert_base_en_uncased/3/metadata.json
    /kaggle/input/distil_bert/keras/distil_bert_base_en_uncased/3/model.weights.h5
    /kaggle/input/distil_bert/keras/distil_bert_base_en_uncased/3/assets/tokenizer/vocabulary.txt


```python
!pip install keras-core --upgrade
!pip install -q keras-nlp --upgrade
```

    Requirement already satisfied: keras-core in /usr/local/lib/python3.11/dist-packages (0.1.7)
    Requirement already satisfied: absl-py in /usr/local/lib/python3.11/dist-packages (from keras-core) (1.4.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from keras-core) (1.26.4)
    Requirement already satisfied: rich in /usr/local/lib/python3.11/dist-packages (from keras-core) (14.0.0)
    Requirement already satisfied: namex in /usr/local/lib/python3.11/dist-packages (from keras-core) (0.0.8)
    Requirement already satisfied: h5py in /usr/local/lib/python3.11/dist-packages (from keras-core) (3.13.0)
    Requirement already satisfied: dm-tree in /usr/local/lib/python3.11/dist-packages (from keras-core) (0.1.9)
    Requirement already satisfied: attrs>=18.2.0 in /usr/local/lib/python3.11/dist-packages (from dm-tree->keras-core) (25.3.0)
    Requirement already satisfied: wrapt>=1.11.2 in /usr/local/lib/python3.11/dist-packages (from dm-tree->keras-core) (1.17.2)
    Requirement already satisfied: mkl_fft in /usr/local/lib/python3.11/dist-packages (from numpy->keras-core) (1.3.8)
    Requirement already satisfied: mkl_random in /usr/local/lib/python3.11/dist-packages (from numpy->keras-core) (1.2.4)
    Requirement already satisfied: mkl_umath in /usr/local/lib/python3.11/dist-packages (from numpy->keras-core) (0.1.1)
    Requirement already satisfied: mkl in /usr/local/lib/python3.11/dist-packages (from numpy->keras-core) (2025.1.0)
    Requirement already satisfied: tbb4py in /usr/local/lib/python3.11/dist-packages (from numpy->keras-core) (2022.1.0)
    Requirement already satisfied: mkl-service in /usr/local/lib/python3.11/dist-packages (from numpy->keras-core) (2.4.1)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras-core) (3.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras-core) (2.19.1)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich->keras-core) (0.1.2)
    Requirement already satisfied: intel-openmp<2026,>=2024 in /usr/local/lib/python3.11/dist-packages (from mkl->numpy->keras-core) (2024.2.0)
    Requirement already satisfied: tbb==2022.* in /usr/local/lib/python3.11/dist-packages (from mkl->numpy->keras-core) (2022.1.0)
    Requirement already satisfied: tcmlib==1.* in /usr/local/lib/python3.11/dist-packages (from tbb==2022.*->mkl->numpy->keras-core) (1.3.0)
    Requirement already satisfied: intel-cmplr-lib-rt in /usr/local/lib/python3.11/dist-packages (from mkl_umath->numpy->keras-core) (2024.2.0)
    Requirement already satisfied: intel-cmplr-lib-ur==2024.2.0 in /usr/local/lib/python3.11/dist-packages (from intel-openmp<2026,>=2024->mkl->numpy->keras-core) (2024.2.0)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m792.1/792.1 kB[0m [31m11.0 MB/s[0m eta [36m0:00:00[0m
    [?25h

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
```

```python
import numpy as np #linear algebra
import pandas as pd #data processing, CSV file
import tensorflow as tf
import keras_core as keras
import keras_nlp
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Tensorflow version: ", tf.__version__)
print("KerasNLP version", keras_nlp.__version__)
```

    2025-05-23 06:45:38.769027: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1747982739.035426      13 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1747982739.113586      13 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered


    Using TensorFlow backend
    Tensorflow version:  2.18.0
    KerasNLP version 0.20.0


## Load Disaster Tweets

```python
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

print('Training Set Shape = {}'.format(df_train.shape))
print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum()/1024**2))
print('Test Set Shape = {}'.format(df_test.shape))
print('Training Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum()/1024**2))
```

    Training Set Shape = (7613, 5)
    Training Set Memory Usage = 0.29 MB
    Test Set Shape = (3263, 4)
    Training Set Memory Usage = 0.10 MB


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just happened a terrible car crash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Heard about #earthquake is different cities, s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>there is a forest fire at spot pond, geese are...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
    </tr>
  </tbody>
</table>
</div>



## Explore Dataset

```python
df_train["length"] = df_train["text"].apply(lambda x: len(x))
df_test["length"] = df_test["text"].apply(lambda x : len(x))

print("Train Length Stat")
print(df_train["length"].describe())
print()

print("Test Length Stat")
print(df_test["length"].describe())
```

    Train Length Stat
    count    7613.000000
    mean      101.037436
    std        33.781325
    min         7.000000
    25%        78.000000
    50%       107.000000
    75%       133.000000
    max       157.000000
    Name: length, dtype: float64
    
    Test Length Stat
    count    3263.000000
    mean      102.108183
    std        33.972158
    min         5.000000
    25%        78.000000
    50%       109.000000
    75%       134.000000
    max       151.000000
    Name: length, dtype: float64


## Preprocess the Data

```python
BATCH_SIZE = 32
NUM_TRAINING_EXAMPLES = df_train.shape[0]
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
STEPS_PER_EPOCH = int(NUM_TRAINING_EXAMPLES)*TRAIN_SPLIT //BATCH_SIZE

EPOCHS = 2
AUTO = tf.data.experimental.AUTOTUNE
```

```python
from sklearn.model_selection import train_test_split

X = df_train["text"]
y = df_train["target"]

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = VAL_SPLIT, random_state = 42)

X_test = df_test["text"]
```

## Load a DistilBERT model from Keras NLP

```python
# Load a DistilBERT model
preset = "distil_bert_base_en_uncased"

# Use a shorter sequence length

preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    preset,
    sequence_length = 160,
    name = "preprocessor_4_tweets"
)

# Pretrained Classifier
classifier = keras_nlp.models.DistilBertClassifier.from_preset(
    preset,
    preprocessor = preprocessor,
    num_classes = 2
)

classifier.summary()
```

    2025-05-23 06:46:00.004288: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Preprocessor: "preprocessor_4_tweets"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                                                  </span>┃<span style="font-weight: bold">                                   Config </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ distil_bert_tokenizer (<span style="color: #0087ff; text-decoration-color: #0087ff">DistilBertTokenizer</span>)                   │                       Vocab size: <span style="color: #00af00; text-decoration-color: #00af00">30,522</span> │
└───────────────────────────────────────────────────────────────┴──────────────────────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "distil_bert_text_classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                  </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">         Param # </span>┃<span style="font-weight: bold"> Connected to               </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ padding_mask (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)              │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_ids (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)              │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ distil_bert_backbone          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">768</span>)         │      <span style="color: #00af00; text-decoration-color: #00af00">66,362,880</span> │ padding_mask[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],        │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">DistilBertBackbone</span>)          │                           │                 │ token_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ get_item (<span style="color: #0087ff; text-decoration-color: #0087ff">GetItem</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">768</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ distil_bert_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ pooled_dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">768</span>)               │         <span style="color: #00af00; text-decoration-color: #00af00">590,592</span> │ get_item[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ output_dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">768</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ pooled_dense[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ logits (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)                 │           <span style="color: #00af00; text-decoration-color: #00af00">1,538</span> │ output_dropout[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">66,955,010</span> (255.41 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">66,955,010</span> (255.41 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



## Train your own model, fine-tuning BERT

```python
# Compile

classifier.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = 'adam',
    metrics = ["accuracy"]

)

# Fit

history = classifier.fit(
    x = X_train,
    y = y_train,
    epochs = EPOCHS,
    validation_data = (X_val, y_val)
)

```

    Epoch 1/2
    [1m191/191[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3612s[0m 19s/step - accuracy: 0.5490 - loss: 0.6962 - val_accuracy: 0.5739 - val_loss: 0.6831
    Epoch 2/2
    [1m191/191[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3535s[0m 19s/step - accuracy: 0.5730 - loss: 0.6849 - val_accuracy: 0.5739 - val_loss: 0.6835


```python
def displayConfusionMatrix(y_true, y_pred,dataset):
  disp = ConfusionMatrixDisplay.from_prediction(
      y_true,
      np.argmax(y_pred, axis = 1),
      display_labels = ["Not Disaster","Disaster"],
      cmap = plt.cm.Blues
  )

  tn, fp, fn, tp = confusion_matrix(y_true, np.argmax(y_pred, axis = 1)).ravel()
  f1_score = tp/(tp+((fn + fp)/2))

  disp.ax_.set_title ("Confusion Matrix on " + dataset + "Dataset -- F1 Score" + str(f1_score.round(2)))
```

```python
def displayConfusionMatrix(y_true, y_pred, dataset):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        np.argmax(y_pred, axis=1),
        display_labels=["Not Disaster","Disaster"],
        cmap=plt.cm.Blues
    )

    tn, fp, fn, tp = confusion_matrix(y_true, np.argmax(y_pred, axis=1)).ravel()
    f1_score = tp / (tp+((fn+fp)/2))

    disp.ax_.set_title("Confusion Matrix on " + dataset + " Dataset -- F1 Score: " + str(f1_score.round(2)))
```

```python
y_pred_train = classifier.predict(X_train)

displayConfusionMatrix(y_train, y_pred_train, "Training")
```

    [1m191/191[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m868s[0m 5s/step



    
![png](output_19_1.png)
    


```python
y_pred_val = classifier.predict(X_val)

displayConfusionMatrix(y_val, y_pred_val, "Validation")
```

    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m218s[0m 5s/step



    
![png](output_20_1.png)
    


## Generate Submission File

```python
sample_submission = pd.read_csv("/content/sample_submission.csv")
sample_submission.head()
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    /tmp/ipykernel_13/3335867550.py in <cell line: 0>()
    ----> 1 sample_submission = pd.read_csv("/content/sample_submission.csv")
          2 sample_submission.head()


    /usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
       1024     kwds.update(kwds_defaults)
       1025 
    -> 1026     return _read(filepath_or_buffer, kwds)
       1027 
       1028 


    /usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
        618 
        619     # Create the parser.
    --> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
        621 
        622     if chunksize or iterator:


    /usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
       1618 
       1619         self.handles: IOHandles | None = None
    -> 1620         self._engine = self._make_engine(f, self.engine)
       1621 
       1622     def close(self) -> None:


    /usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
       1878                 if "b" not in mode:
       1879                     mode += "b"
    -> 1880             self.handles = get_handle(
       1881                 f,
       1882                 mode,


    /usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        871         if ioargs.encoding and "b" not in ioargs.mode:
        872             # Encoding
    --> 873             handle = open(
        874                 handle,
        875                 ioargs.mode,


    FileNotFoundError: [Errno 2] No such file or directory: '/content/sample_submission.csv'


```python
sample_submission["target"] = np.argmax(classifier.predict(X_test), axis = 1)
```

```python
sample_submission.to_csv("submission.csv", index = False)
```

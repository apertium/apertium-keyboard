# TFLM

Tensorflow LM code modified for android keyboard use via Tensorflow Lite compilation.

Requires Python3.

Requires separate input and output data files (e.g., ```lm.train.txt``` and ```lm.train.txt.out```) to support special keyboard use, namely predicting morpheme boundaries even though the user never explicitly enters them. If input and output files are identical, you get a standard language model.

Sample Guarani data is provided. Here is a snippet to show the specialized morpheme-aware input-output alignment. Note that given an input char, the model always predict the **NEXT** output char (i.e., ```n -> d@```) :

Input: ```n d a i p ó r i _ m b a```

Output: ```n d@ a@ i p ó@ r i _ m b a```

## Training

From the command line:

```python tflm.py --data_path=./data/ --model=small --num_gpus=1```

The ```--model``` option permits training models of different sizes (test|small|medium|large). More can be added in the code. The idea is that there is a tradeoff between size/quality and resources required on the final device/phone. Sizes differ by the amount of history the model accepts (```num_steps```), the number of hidden units, etc.

The ```--num_gpus``` option specifies how many GPUs to use for training. Setting it to 0 means using just CPU.


When the code runs, it produces two files. ```labels.txt``` represents the vocabulary of the model, based on the training data. The symbols are ordered by their numerical index in the network. ```converted_model.tflite``` is the inference-only subgraph of the model compiled by Tensorflow Lite for on-device use.



# SpamDetectionProject

[[When Recurrent Model don't need to be Recurrent]](https://bair.berkeley.edu/blog/2018/08/06/recurrent/)

## TODO List:

- [ ] Try Another Architecture to see if the problem is on the DATASET <Transformer, PURE LSTM>
- [ ] Modularize the whole code

Problem assumption:

The LSTM Sequence is too long so the backprop will have gradient problem.



## Dataset:
  Number of Observations: 916,152
  Contain 82.3 % NonSpammer and 17.7% Spammer roughly. 
  

# WEEK FOUR
  

## First Training whtouht any change

##### Training & Validation
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/WithOutPunishment/All_Hist_SSCL.png?raw=true)

#### Only Training 

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/WithOutPunishment/Train_Loss&Acc_Hist_SSCL.png?raw=true)

## Training With the punishment on misclassifying the spammer (For Training With Unbalanced Data)
In this model, if a spammer isn't detected, it will contribute to a larger loss.

The punishment is implemented as the Scit-Learn "class_weight":
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/skleanClassWeight.png?raw=true)

Therefore, the loss of misclassifying the spammer will multiply (numberOfNonSpammer/numberOfSpammer) around (4.6369).


```
numberOfSpammer = training_dataset.tensors[2].sum()
numberOfNoSpammer = len(training_dataset)-training_dataset.tensors[2].sum()

self.Loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(numberOfNoSpammer/numberOfSpammer))

```


##### Training & Validation
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/WithPunishmentOnSpammer/All_Hist_SSCL.png?raw=true)

#### Only Training 
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/WithPunishmentOnSpammer/Train_Loss&Acc_Hist_SSCL.png?raw=true)

#### DetailTrainingResult

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/EpochSix.png?raw=true)




## Training with the Punishment * Ratio (1.4)

```python
# It will punish the misclassification on Spammers harder
pos_weight=torch.tensor(numberOfNoSpammer/numberOfSpammer) * 1.4
self.Loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

```

#### Training Result

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/WithPunishmentAndRatio1.4/All_Hist_SSCL.png?raw=true)

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/WithPunishmentAndRatio1.4/Train_Loss%26Acc_Hist_SSCL.png?raw=true)




# WEEK FIVE


## Using A Larger Neural Network with Punishment * Ratio (1.4)

#### The Model Architecture


```python

self.embed = nn.Embedding(
    args.vocab_size, args.embedding_dim, Constants.PAD)

self.cnn = nn.Sequential(
    nn.Conv1d(args.embedding_dim, args.num_CNN_filter,
              args.CNN_kernel_size, 1, 2),
    nn.BatchNorm1d(args.num_CNN_filter),
    nn.LeakyReLU(inplace=True),
    nn.Conv1d(args.num_CNN_filter, args.num_CNN_filter*2,
      args.CNN_kernel_size, 1, 2),
    nn.BatchNorm1d(args.num_CNN_filter*2),
    nn.LeakyReLU(inplace=True),
    nn.Conv1d(args.num_CNN_filter*2, args.num_CNN_filter,
      args.CNN_kernel_size, 1, 2),
    nn.BatchNorm1d(args.num_CNN_filter),
    nn.LeakyReLU(inplace=True),
)

self.rnn = nn.LSTM(args.num_CNN_filter, args.RNN_hidden,
                   batch_first=True, dropout=args.LSTM_dropout, num_layers = 3)

self.out_net = nn.Sequential(
    nn.Linear(args.RNN_hidden, 1),
)

```
#### Training Result

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/LargerModelWith1.4/All_Hist_SSCL.png?raw=true)

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/LargerModelWith1.4/Train_Loss%26Acc_Hist_SSCL.png?raw=true)




## Using The Large Model With WeightedRandomSampling (Without Punishement On Loss)

#### HOW WE APPLY THE WeightRandomSampling <Only Use on the training set>:

```python

def getSampler(dataset):
    
    target = torch.tensor([ label for t, l , label in dataset])
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = np.array([weight[t.item()] for t in target.byte()])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    return sampler
    
train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, sampler = sampler)

```

#### The WeightedRandomSampler from Pytorch

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/LargeModelWithWeightedRandomSampling/PytorchWeightedRandomSampler.png?raw=true)


#### Training Results

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/LargeModelWithWeightedRandomSampling/Train_Loss%26Acc_Hist_SSCL.png?raw=true)

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/LargeModelWithWeightedRandomSampling/All_Hist_SSCL.png?raw=true)


#### Training Details

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/LargeModelWithWeightedRandomSampling/WeightedRandomSampling.png)

#### The Confusion Matrix on the validation set

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/LargeModelWithWeightedRandomSampling/ConfusionMatrix.png?raw=true)



## Using The Large Model With WeightedRandomSampling (With Punishement*1 On Loss)

#### Training Results
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/LargerModelUsingWeightedRandomSampliingAndNormalRatioAtSameTime/Train_Loss%26Acc_Hist_SSCL.png?raw=true)
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/LargerModelUsingWeightedRandomSampliingAndNormalRatioAtSameTime/All_Hist_SSCL.png?raw=true)

#### Training Details
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/LargerModelUsingWeightedRandomSampliingAndNormalRatioAtSameTime/TrainingDetails.png?raw=true)


#### Confusion Matrix
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/LargerModelUsingWeightedRandomSampliingAndNormalRatioAtSameTime/ConfusionMatrix.png?raw=true)




# Week SIX

The cofusion matrix on following two models are not great, most of the time the model just map any input to nonSpammer output.


## Using the Large Model, Small Vocab size(5) and RandomWeightedSampling


#### Training Results

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/Vocab5RandomWeightedSamplingLargeModel/Log/All_Hist_SSCL.png?raw=true)


#### Training Details

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/Vocab5RandomWeightedSamplingLargeModel/Log/Train_Loss%26Acc_Hist_SSCL.png?raw=true)




## Using Large Model, Voacab size = 10 but, WeightedRandomSampling = False

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/Vocab10LargeModel/Log/All_Hist_SSCL.png?raw=true)

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/Vocab10LargeModel/Log/Train_Loss%26Acc_Hist_SSCL.png?raw=true)






## Using Small model (CNN*1, LSTM*1), Vocab size = 50, WeightedRandomSampling = True

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/SmallModelVacabSize50WeightedRandomSampling/Log/All_Hist_SSCL.png?raw=true)
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/SmallModelVacabSize50WeightedRandomSampling/Log/Train_Loss%26Acc_Hist_SSCL.png?raw=true)



## Same as the last model but with the clip_grad = 0.25

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/SmallModelVacabSize50WeightedRandomSamplingGradClip25/Log/All_Hist_SSCL.png?raw=true)

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/SmallModelVacabSize50WeightedRandomSamplingGradClip25/Log/Train_Loss%26Acc_Hist_SSCL.png?raw=true)

## Vocab = 50, WeightedRandomSampling = False and Take sum of the LSTM output

In this model we take the sum of LSTM output:

Before we was setting the rnn in SSCL like this
```python
out = self.rnn(out)[0][:, -1, :]
```
now we take the sum

```python
out = self.rnn(out)[0].sum(dim=1)
```
#### Training Results
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/SSCL_Vocab50_SmallModel_TakeSumOfLSTM/Log/All_HistSSCL_Vocab50_SmallModel_TakeSumOfLSTM.png?raw=true)
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/SSCL_Vocab50_SmallModel_TakeSumOfLSTM/Log/Train_Loss%26Acc_HistSSCL_Vocab50_SmallModel_TakeSumOfLSTM.png?raw=true)

## The last model with WeightedRandomSampling

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/SSCL_Vocab50_SmallModel_TakeSumOfLSTM_RandomWeightedSampling/Log/All_HistSSCL_Vocab50_SmallModel_TakeSumOfLSTM_RandomWeightedSampling.png?raw=true)
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/SSCL_Vocab50_SmallModel_TakeSumOfLSTM_RandomWeightedSampling/Log/Train_Loss%26Acc_HistSSCL_Vocab50_SmallModel_TakeSumOfLSTM_RandomWeightedSampling.png?raw=true)


## Try GatedCNN Model, vocab_size=50, WeightedRandomSampling, num_layer = 8

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/GatedCNN8L_Vocab50_SmallModel_TakeSumOfLSTM_RandomWeightedSampling/Log/All_HistGatedCNN8L_Vocab50_SmallModel_TakeSumOfLSTM_RandomWeightedSampling.png?raw=true)

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/GatedCNN8L_Vocab50_SmallModel_TakeSumOfLSTM_RandomWeightedSampling/Log/Train_Loss%26Acc_HistGatedCNN8L_Vocab50_SmallModel_TakeSumOfLSTM_RandomWeightedSampling.png?raw=true)


## GatedCNN Model, vocab_size = 500, WieghtedRandomSampling, num_layer = 8

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/GatedCNN8L_Vocab500_SmallModel_TakeSumOfLSTM_RandomWeightedSampling/Log/All_HistGatedCNN8L_Vocab500_SmallModel_TakeSumOfLSTM_RandomWeightedSampling.png?raw=true)

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/GatedCNN8L_Vocab500_SmallModel_TakeSumOfLSTM_RandomWeightedSampling/Log/Train_Loss%26Acc_HistGatedCNN8L_Vocab500_SmallModel_TakeSumOfLSTM_RandomWeightedSampling.png?raw=true)


## GatedCNN, 500Vocab, usingWeightRandomSampling, num_layer = 3

```python
class args(object):

    # Data
    
    usingWeightRandomSampling = True
    vocab_size = 500 # if we create the new vocab size, we have to do the new preprocess again
    full_data = True
    
    if full_data: 
        pickle_name = "FullPickleData"+ str(vocab_size) + "Vocab.txt"
        pickle_name_beforeMapToIdx = "FullPickleDatabeforeMapToIdx"+ str(vocab_size) + "Vocab.txt"
    else:
        pickle_name = "10thPickleData"+ str(vocab_size) + "Vocab.txt"
        pickle_name_beforeMapToIdx = "10thPickleDatabeforeMapToIdx"+ str(vocab_size) + "Vocab.txt"
    dataset_path = ""  # load a dataset and setting

    ##### Arch
        
    usingPretrainedEmbedding = False
    if usingPretrainedEmbedding:
        embedding_dim = 300
    else:
        embedding_dim = 512

    ## GatedCNN arch

    GatedCNN_embedingDim = 128
    GatedCNN_convDim = 64
    GatedCNN_kernel = 3
    GatedCNN_stride = 1
    GatedCNN_pad = 1
    GatedCNN_layers = 3
        
    ## SSCL arch

    RNN_hidden = 256
    num_CNN_filter = 256
    CNN_kernel_size = 5
    LSTM_dropout = 0.1
    num_LSTM_layers = 1

    # Training params

    confusion_matrics = []
    
    batch_size = 64
    L2 = 0.1
    threshold = 0.75
    lr = 0.002
    n_epoch = 50

    # If using Adam
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 0.01

    # Logging the Training
    val_freq = 50
    val_steps = 3
    log_freq = 10
    model_save_freq = 1
    model_name = 'GatedCNN8L_Vocab500_Layer3_RandomWeightedSampling'
    model_path = './' + model_name + '/Model/'
    log_path = './' + model_name + '/Log/'


```

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/GatedCNN8L_Vocab500_Layer3_RandomWeightedSampling/Log/All_Hist_GatedCNN8L_Vocab500_Layer3_RandomWeightedSampling.png?raw=true)

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/GatedCNN8L_Vocab500_Layer3_RandomWeightedSampling/Log/Train_Loss%26Acc_Hist_GatedCNN8L_Vocab500_Layer3_RandomWeightedSampling.png?raw=true)

# SpamDetectionProject

TODO:

Draw a Model Stucture

Dataset:
  Number of Observations: 916,152
  Contain 82.3 % NonSpammer and 17.7% Spammer roughly. 

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

#### HOW WE APPLY THE WeightRandomSampling:

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


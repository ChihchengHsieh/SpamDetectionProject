# SpamDetectionProject
SpamDetectionProject

### First Training whtouht any change

##### Training & Validation
![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/WithOutPunishment/All_Hist_SSCL.png?raw=true)

#### Only Training 

![](https://github.com/ChihchengHsieh/SpamDetectionProject/blob/master/ModelLog/WithOutPunishment/Train_Loss&Acc_Hist_SSCL.png?raw=true)








### Training With the punishment on misclassifying the spammer (For Training With Unbalanced Data)
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

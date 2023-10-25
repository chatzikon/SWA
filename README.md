# SWA

The source code of the paper "Federated Learning Aggregation based on Weight
Distribution Analysis" that will be presented in the IST 2023 conference

# **Installation**

Run pip install -r equirements.txt to install the required packages

# **Proposed method**

![model architecture image](https://github.com/chatzikon/SWA/blob/main/images/method_image.png)


# **Train**

**Ablation** 

The folder ablation contains the files required to perform the ablation study described in the paper.
In order to train a model, employ the federated_train.py script.
Regarding the arguments of the federated_train.py:
The **t_round** is the number of client local epochs, **clients** argument is the number of clients, **epochs** are the total training epochs, **coef** defines the statistical distance utilized for the experiment, **splits** are the data splits of the clients.
**Aggregation methods** are the proposed weight aggregation algorithm and the baseline average one. The **path** argument is the path of a pretrained model for evaluation.
Code from [FL-PQSU](https://github.com/fangvv/FL-PQSU/tree/main) was used. 

**SoTa**

The folder SoTA contrains the files required to acquire the results of the proposed method, compared to SoTA, as it is described at the paper.
As mentioned above, employ the federated_train.py script to train the model. 
The **alg** argument is the algorithm employed, among the ones presented in [1]. The **communication rounds** are the times the server communicates with the clients. According to ablation arguments, is equal to epochs//t_round. The **mu** is the coefficient of the contrastive loss. 
The rest of the arguments were explained above, at the ablation study part. 


[1] Li, Q., He, B., & Song, D. (2021). Model-contrastive federated learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10713-10722).

# **License**

Our code is released under MIT License (see LICENSE file for details)

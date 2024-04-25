# CASA0018-ERTONG-GAO

**Blood cell subtype classifier**

Ertong Gao

Edge Impulse link: https://studio.edgeimpulse.com/studio/382900

Youtube link: https://youtu.be/tql8enxZAJ8


## Introduction
### Project overview
This project is build based on Edge impulse, it enables users to detect the white blood cells of 4 types(neutrophil, eosinophil, lymphocyte, monocyte) from blood smear images by using deep learning network architecture. This project is deployed onto phone to use input image by the phone’s camera.

### Inspiration
Complete blood cell (CBC) counting has played a vital role in general medical examination. The diagnosis of blood-based diseases often involves identifying and characterizing patient blood samples( Chadha et al., 2020). 
Common approaches, such as traditional manual counting and automated analyzers, were heavily influenced by the operation of medical professionals, which is time consuming and easily making errors as different blood cells look similar as states below.

<img src="https://github.com/BIT-MJY/Active-SLAM-Based-on-Information-Theory/blob/master/img/1-2.png" width="180" height="105"><img src="https://github.com/BIT-MJY/Active-SLAM-Based-on-Information-Theory/blob/master/img/1-3.png" width="180" height="105"><img src="https://github.com/BIT-MJY/Active-SLAM-Based-on-Information-Theory/blob/master/img/1-4.png" width="180" height="105"/>

### NEUTROPHIL EOSINOPHIL LYMPHOCYTE MONOCYTE


Automated methods to detect and classify blood cell types have important medical applications. Deep learning project increases the accuracy and speed of diagnosis(Wang et al., 2019), reducing human error and improving patient outcomes, making more informed decisions faster, thus enhancing the efficiency of clinical workflows(Alam, 2019). By reduce the need for extensive manual examination, thereby lowering the costs and save time associated with diagnostic procedures.

Examples
As technology has evolved, the task of classifying and counting blood cells has gradually shifted from manual methods to automated techniques based on image processing and machine learning(Rezatofighi, 2011). 
In 2010, researchers employed various image recognition techniques such as converting images to grayscale and automatic thresholding to identify types of white blood cells automatically with accuracy rates ranging from 85% to 98%(Madhloom et al., 2010). Subsequent studies involved pass the generated image features into deep learning SVM and ANN models for training. The models achieved classification accuracies of 90% and 96% in distinguishing white blood cells(Chadha et al., 2014). 
Machine learning algorithms have been widely employed in white blood cell classification problems, Habibzadeh et al., trained a CNN architecture model and use the model to classify the blood cells in low-resolution images into 5 types. Researchers used ANN architecture even achieved 99.1% of accuracy on classify blood cell types(Habibzadeh et al., 2014).



## Research Question
Can a neural network based project be built on phone to accurately and efficiently detect the type of blood cell in a variety of images using a small-scale dataset?


## Application Overview
This project is a deep learning project built using a Tensorflow platform, as other deep learning project involves five modules: data aquisition and preprocessing, neural network architecture choosing and training, training performance evaluation, model tuning and retrain, deployment and application of the model. In the first step, dataset of different blood cell classes smear images is downloaded from kaggle website, then uploaded into edge impulse platform, performed steps for data cleaning and feature engineering showed in the diagram below. Then, a baseline model is chosen and trained using the dataset as a starting point, assesing the model performance using accuracy metric. After that, run experiments to test different value of parameters combinations to fine-tuning the model and try to find the best performance selections, changed parameters and experiments are decribed in details below. Finally, deploy the model on phone by QR code to classify on real-time blood cell image samples for the correct blood cell type.

### Figure 1. Overview of building blocks



## Data
Due to the specificity of this project, it is challenging to obtain white blood cell blood sample images and accurately classify and label them. Therefore, the dataset was downloaded from Kaggle. The dataset is a small-scale dataset for blood cells detection, contains 12,436 augmented images of blood cells (JPEG) in total. The classes contains almost equally amount of images, this reduce class bias which can result in poor performance on less-represented classes and helps the model learn more evenly across all classes, thus performing better, reduce bias during the training process, making the model more robust.
There are approximately 3,000 images for each of 4 different cell types grouped into 4 different folders according to cell type, the cell types are Eosinophil, Lymphocyte, Monocyte, and Neutrophil. When uploading on Edge Impulse, all samples are labelled by their containing blood cell type name, and dataset in split into training set and testing set, Allocating the majority (80%) of data to the training set ensures that the model has ample data to learn the characteristics and patterns of the data. Testing set (20%) provides enough data to verify the model's performance on unseen data, helping to assess the model's generalization capabilities.

### Figure 2. Dataset overview


For Dataset preprocessing, dataset passed to create impulse, an impulse takes raw data, uses signal processing to extract features, this is the prior step for learning block to classify data. As raw image data all have different sizes, signal processing block resize all images to equal dimensions according to the image width and image height set by the users, three different resize mode can be choose: ‘Squash’, ‘Fit shortest axis’ and ‘Fit longest axis’. Further for image data, image depth can be set between RGB and Grayscale. There processes normalized data and generated processed features into the learning blocks.

### Figure 3. Feature Explorer


After serveral experiments which stated in the next section, image size of 160x160, resize mode squash and RGB color depth works best on model performance.

### Figure 4.raw data sample	

### Figure 5.DSP result

## Model
After data collection and preprocessing, 76800 features have generated and then pass to a learning block as input layer. Learning block can be considered as the most important part of the project, various different parameters can be tuned for better model performance. First, the learning block can choose from five different types, as the description on platform suggests, Transfer Learning(images) is fine tune a pre-trained image classification model whereas the second and third types are for movement, audio data and numerical values. Experiments also have done to suggest Transfer Learning(images) performs better on model accuracy(Moris, 2021).

### Figure 6.Comparison between classifier and transfer learning(image)

### Figure 7. Learning block description

The parameters sets stated below has the best performance.

### Figure 8. Parameters used

For neural network architecture layer, MobileNetV2 is used, MobileNetV2 outperforms MobileNetV1 and traditional Conv2D-based networks, as it utilizes improved residual connections and inverted residual blocks, also incorporates more depthwise separable convolution layers and expansion convolution layers,resulting in better performance and accuracy with the same number of parameters.


## Experiments

For the project, I experiment by change different values of various parameters and neural network architectures, the changed parameters are stated with description and range in the table below.

### Figure 9. Parameter table

In total, 30 experiments based on changing the above parameters have be made to test for optimal performance architecture.

### Figure 10. Experiments overview

The experiments start with run the EON Turner built in Edge impulse, this function runs and outputs several tests with different parameters and models, try to find the most optimal architecture, as the result has 86% aacuracy which is satisfied, the result is used as a baseline model. The baseline setting used ‘MobileNetV2 160x160’ layer, the followed experiments stayed with this model to test the effect of other parameters on model accuracy metrics. Experiments made here clearly state that set image depth to Grayscale and decrease transfer neurons effectively decrease the mocel accuracy.

### Figure 11. Experiments on MobileNetV2

After completed the experiments above on MobileNetV2 layer, I did some experimrnts on ‘MobileNetV1 96x96’ layer. As the table shows, the, the accuracy is quite low and fluctuation of accuracy is very small, this might because the model layer complexity cannot support the dataset task complexity.

### Figure 12. Experiments on MobileNetV1

Previous experiments in the model section stated to use Transferlearning(image) learning block performs better, experiments have made to use classify learning block, I choose model layer to be ‘conv2d’ in these experiments. Changed parameters focused on transfer dropout, number of layers and corresponding filters as these parameters are specialized to ‘conv2d’. Result suggests larger number of layers and filters improve the model accuracy, the kernel size seems not have large impact on model performace.

### Figure 13. Experiments on conv2d


### Results and Observations

Below shows the result of optimal architecture for this deep learning project after the experiments and the results for trained model.

### Figure 14.Optimal parameter set

### Figure 15.Transfer Learning

### Figure 16.Retrained model

During the experiments, I found that MobileNetV2 layer generally performs better with same parameters.

### Firgure 17.Optimal performance of three models

Among the different model layers, observations are:
1.Training cycles at 20 is optimal, decrease the value to 10 decrease the accuracy but increase to 30 the accuracy shows minor decrease. This might because when decrease the number of training cycles, the model is underfitting, may not have had enough time to learn the patterns in the data effectively, resulting in poor performance, but when increase the training cycles to 30, the model is overfitting, become overly sensitive to the training data, reducing its ability to generalize.

### Firgure 18.Experiments of training cycles

2.Number of transfer neurons is similar to training cycles, optimal at 64 for both MobileNetV1 and MobileNetV2. When decrease the number, model accuracy shows siginificant decrease suggests underfitting and when increase the number, model accuracy fluctuated abit suggests overfitting.

### Figure 19. Experiments of transfer neurons

3.For both model layers, learning rate is optimal at 0.0005, same pattern showed as the above two parameters. For increase and decrease, the model accuracy dropped.

### Figure 20. Experiments of learning rate

4.Three resize mode, ‘squash’ works better than ‘fit shortest axis’ and ‘fit longest axis’ performs the worst. This might because only squash mode don’t require crop the images, ensures that all regions of the image retain their information without any loss.

### Figure 21. Experiments of image resize mode

5.When image depth set to grayscale from RGB, accuracy drop suddenly, this might because many image features are distinguishable on the basis of color might not be as distinct in grayscale.


### Figure 22. Experiments of image depth

## Future Development
1.Improve dataset by use object detection technology
2.Improve model by test more types of layer and be able to count cells
3.Develop on Arduino devices and be able to store output result



## Bibliography
（1）Alam, M.M. (2019) Wiley Online Library | Scientific Research Articles, journals, books, and reference works, Machine learning approach of automatic identification and counting of blood cells. Available at: https://onlinelibrary.wiley.com/ (Accessed: 23 April 2024). 

（2）Author links open overlay panelSeyed Hamid Rezatofighi a 1 et al. (2011) Automatic recognition of five types of white blood cells in peripheral blood, Computerized Medical Imaging and Graphics. Available at: https://www.sciencedirect.com/science/article/pii/S0895611111000048 (Accessed: 24 April 2024).

（3）Chadha, G.K. et al. (2014) Automatic segmentation, counting, size determination and classification of white blood cells, An Automated Method for Counting Red Blood Cells using Image Processing. Available at: https://www.sciencedirect.com/science/article/abs/pii/S0263224114001663 (Accessed: 24 April 2024). 

（4）Chadha, G.K. et al. (2020) An automated method for counting red blood cells using image processing, Procedia Computer Science. Available at: https://www.sciencedirect.com/science/article/pii/S1877050920308747 (Accessed: 22 April 2024). 

（5）Habibzadeh, M., Krzyzak, A. and Fevens, T. (1970) White blood cell differential counts using convolutional neural networks for low resolution images, SpringerLink. Available at: https://link.springer.com/chapter/10.1007/978-3-642-38610-7_25 (Accessed: 22 April 2024). 

（6）Lee, S.-J., Chen, P.-Y. and Lin, J.-W. (2022) Complete Blood Cell Detection and counting based on Deep Neural Networks, MDPI. Available at: https://www.mdpi.com/2076-3417/12/16/8140 (Accessed: 22 April 2024). 

（7）Madhloom, H.T. et al. (2010) An automated white blood cell nucleus localization and segmentation using image arithmetic and automatic threshold, NASA/ADS. Available at: https://ui.adsabs.harvard.edu/abs/2010JApSc..10..959M/abstract (Accessed: 24 April 2024). 

（8）Moris, M. (2021) MOBILENETV2 - light weight model (image classification), Medium. Available at: https://medium.com/image-processing-and-ml-note/mobilenetv2-light-weight-model-image-classification-783d79cc01c9 (Accessed: 23 April 2024). 

（9）Wang, Wei et al. (2019) Development of convolutional neural network and its application in image classification: A survey, SPIE Digital Library. Available at: https://www.spiedigitallibrary.org/journals/optical-engineering/volume-58/issue-4/040901/Development-of-convolutional-neural-network-and-its-application-in-image/10.1117/1.OE.58.4.040901.full (Accessed: 24 April 2024). 


## Declaration of Authorship
I, AUTHORS Ertong Gao, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.

Ertong Gao

23/04/2024

Word count:1650

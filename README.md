# rock_paper_scissor_classification
This repository contains all machine learning related code, that was used to finish [this](https://github.com/Jensssen/Image-Classification-App) project. 
The main idea was to train a NN on images, in order to classify rock, paper, scissor images. 

# Dataset
At first, I took all trainig images manually of my own hands, making rock paper scissor signs and sorted them into the corresponding folders. However, this approach took way too muchtime. Therefore, I wrote a shot skript that takes a recorded video as input and exports every 5th frame into a folder. With this approach, I was able to generate roughly 2000 images of rock, paper and scissor signes in a couple of minutes.

Rock:
![alt text](https://github.com/Jensssen/rock_paper_scissor_classification/blob/master/dataset/rock.png)
Paper:
![alt text](https://github.com/Jensssen/rock_paper_scissor_classification/blob/master/dataset/paper.png)
Scissor:
![alt text](https://github.com/Jensssen/rock_paper_scissor_classification/blob/master/dataset/scissor.png)

I tried to record as many variations of my hand in different lightning conditions and with different backgrounds. However, the data is of course quite biased since it ownly shows my hand :P 
I decided to not share the dataset because I recorded it in my own appartment and do not want to upload my whole appartment to the internet :P 
However, with the video frame extraction script you can easily and quickly create a similar dataset on your own. The script is located under dataset/video_to_image.py

# Training
The final model is generated in the sagemaker_model.ipynb jupyther notebook. It consists out of a pretrained mobilenet_v2 from tfhub.
As can be seen from the jupyter notebook, the final validation accuracy is ~96% 
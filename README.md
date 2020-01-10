# ImageClassification for At Home

## Project Definition
 
Each of y'all volunteered to take part in the Image Recognition At Home project. We had a total of 6 people volunteer, and decided to split the teams into 2 teams of 3. So welcome to your team! I'll let you all decide on a team name .
 
As mentioned in Tuesday's meeting, this projects will run for ~3 months. We'd suggest building your solution in Jupyter Notebooks as that will help to keep the information organized, presentable, and re-usable. We will also expect a presentation that will be delivered to the SIG on what your solution was, how you solved it, problems you came across, what your next steps for improvement would be, and a proposed architecture to operationalize the solution.
 
In the presentation, we mentioned we'd like the operationalized architecture to be geared towards AWS, but after futher discussion, we feel it will be best to keep this open source as much as possible. You can use a cloud service if you can justify why the cost is worth it, but the idea is to keep the architecture agnostic and flexible. Tom Kelm and I can help answer questions around this along the way.
 
Since this will require a deep learning solution to build an accurate solution, you will likely want to utilize GPUs and servers for your model training. The SIG has some budget, but we need to be conscious about it and make sure we don't keep GPU enabled servers running all day as that will blow the budget. So we'd ask you be conscious about when you use your GPUs and try to estimate how much the costs may be. This will be good practice as well for keeping these costs low as in my experience, clients get very worried about these costs ballooning. 
 
There is no rush to finish this, but we do ask that it is completed for March so y'all can present on it and we can continue to structure our SIG agenda around it. If you finish it sooner, that is great, but we will still likely keep with the presentation in March.
 
Lastly, we ask that y'all nominate one person to be the PD for this project. This person will be in charge of making sure the team is making progress, tasks are assigned, organizing the team, and being the main point of contact with SIG leadership.
 
__Some comments/questions to help you get started:__
	
What sort of computer vision problem is this?
* Do any of the models Tom and I have spoken about make sense to use here?
* What do you think constitutes a good training set for this problem?
* How much data will you need?
* How many labels do you want to predict? Does your data size limit this?
* Do you want to do any image augmentation on this data set? If so, why and how?
* How are you going to avoid overfitting?
* How are you going to measure your models performance
 


```
You three are on a team together for the SIG project. 

The project runs for 3 months and we expect a presentation for the month of March (likely mid to late). 

Nominate a PD who will be the main point of contact. Please read the full details before getting started
```

## Helpful Links
* Basic image classification with Tensorflow and Keras: https://www.tensorflow.org/tutorials/keras/classification

* Binary image classification with Tensorflow: https://www.tensorflow.org/tutorials/images/classification

* Transfer learning: https://www.tensorflow.org/tutorials/images/transfer_learning


## Coursera Course
I used this Coursera course to ramp up on image classification with Tensorflow. I found it really helpful even though its a decent time commitment to complete the whole course.
https://www.coursera.org/learn/introduction-tensorflow

## Analysing Amazon product Reviews

This is a project from my Information Retrieval and Data Mining Course while studying MSc Machine Learning at UCL.

### How to use
* Place the data file in `data/` and specify which category in the main of `process_reviews.py`.
* Run `process_reviews.py` to generate the input file for the topic model
* Open `topic_modelling.py` and update the category accordingly in the main
* Run `topic_modelling.py` and watch it train. The logger has been set to INFO level so you can see evey set of the training phase
* If you have access to our models, unzip into `data`. So all models will be in `data/models/`. No other updates required


Below is an example of how to use `topic_model_helpers` to analyse the models.
```python
tmh = TopicModelHelpers(['data/models/electronics_20_topics.lda'], model=lda) # load the 20 topics model
tmh.topics # returns a list of topics and their token distribution
tmh.get_reviews_in_topic(19) # show reviews with a proportion of topic 19
tmh.filter_reviews("case", 19) # show reviews that have a proportion of topic 19 and have token case in them
```

Copyright © 2015 Fayimora. All Rights Reserved.

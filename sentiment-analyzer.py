import os
import random
import spacy
from spacy.util import minibatch, compounding
import pandas as pd



TEST_REVIEW = """
Transcendently beautiful in moments outside the office, it seems almost
sitcom-like in those scenes. When Toni Colette walks out and ponders
life silently, it's gorgeous.<br /><br />The movie doesn't seem to decide
whether it's slapstick, farce, magical realism, or drama, but the best of it
doesn't matter. (The worst is sort of tedious - like Office Space with less humor.)
"""


def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm") # load the built-in en_core_web_sm pipeline
    if "textcat" not in nlp.pipe_names: # check the .pipe_names attribute if the textcat component is already available
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"} # If it isn’t, then you create the component (also called a pipe)
        )
        nlp.add_pipe(textcat, last=True) # add the component to the pipeline
    else:
        textcat = nlp.get_pipe("textcat") # If the component is present in the loaded pipeline

    textcat.add_label("pos") #adding the labels from your data so that textcat knows what to look for
    textcat.add_label("neg")

    # Train only textcat
    training_excluded_pipes = [ #create a list of all components in the pipeline that aren’t the textcat component
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes): 
        # use the nlp.disable() context manager to disable those components for all code within the context manager’s scope
        optimizer = nlp.begin_training() #returns the initial optimizer function.
        # Training loop
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score") # to help organize the output from evaluate_model()
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
        for i in range(iterations):
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(
                    text,
                    labels,
                    drop=0.2,
                    sgd=optimizer,
                    losses=loss
                )
            with textcat.model.use_params(optimizer.averages): #  to use the model in its current state
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )
    # Save model
    with nlp.use_params(optimizer.averages):   
        nlp.to_disk("model_artifacts")

def evaluate_model(
    # separate reviews and their labels 
    tokenizer, textcat, test_data: list
) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if (
                predicted_label == "neg"
            ):
                continue
            if score >= 0.5 and true_label['cats']["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label['cats']["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label['cats']["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label['cats']["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


def load_training_data(
    data_directory: str = "aclImdb/train", # define the directory in which your data is stored 
    split: float = 0.8, #the ratio of training data(80%) to test data (20%)
    limit: int = 0 #types your function expects and what it will return

) -> tuple: 
    # Load from files
    # constructing the directory structure of the data
    # label dictionary structure is a format required by the spaCy 
    # For this project, you won’t remove stop words from your training 
    # data right away because it could change the meaning of a sentence 
    # or phrase, which could reduce the predictive power of your classifier. 
    # This is dependent somewhat on the stop word list that you use.
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}") as f: # looking for and opening text files
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip(): #use .strip() to remove all leading and trailing whitespace
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label
                            }
                        }
                        reviews.append((text, spacy_label)) # appending a tuple of the contents and a label dictionary to the reviews list
    random.shuffle(reviews) #  shuffle them to eliminate any possible bias from the order

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split) # split your shuffled data
    return reviews[:split], reviews[split:] # return two parts of the reviews list using list slices.


def test_model(input_data: str=TEST_REVIEW):
    #  Load saved trained model
    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )


if __name__ == "__main__":
    train, test = load_training_data(limit=2500)
    train_model(train, test)
    print("Testing model")
    test_model()
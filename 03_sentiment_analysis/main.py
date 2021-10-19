from spacy.training.example import Example
from spacy.util import minibatch

import random as rd
import pandas as pd
import numpy as np
import spacy


RANDOM_SEED: int = 1


class SentimentAnalyzer:

    def __init__(self, lang: str) -> None:
        self.nlp = spacy.blank(lang)

        self.textcat = self.nlp.add_pipe('textcat')
        self.textcat.add_label('NEGATIVE')
        self.textcat.add_label('POSITIVE')

        self.optimizer = self.nlp.initialize()


    def train(self, X_train, y_train, batch_size = 8) -> dict:
        losses: dict = {}

        train_data = list(zip(X_train, y_train))
        rd.shuffle(train_data)

        batches = minibatch(train_data, size = batch_size)

        curr_batch_num = 0
        total_batches = len(train_data) // batch_size

        for batch in batches:
            print(f"Training: {curr_batch_num}/{total_batches}", end = '\r')
            for text, labels in batch:
                doc = self.nlp.make_doc(text)
                example = Example.from_dict(doc, labels)
                self.nlp.update([example], sgd = self.optimizer, losses = losses)

            curr_batch_num += 1

        print()

        return losses


    def predict(self, texts: list[str]) -> list[int]:
        docs = [self.nlp.tokenizer(text) for text in texts]
        scores = self.textcat.predict(docs)
        predicted_class = scores.argmax(axis = 1)
        return predicted_class



def load_data(path_to_csv: str, split: float = 0.8) -> tuple:

    data_df: pd.DataFrame = pd.read_csv(path_to_csv)
    data_df = data_df.sample(frac = 1, random_state = RANDOM_SEED)

    texts: np.ndarray = data_df['text'].values

    labels: list = [
        {'cats': {
            'POSITIVE': bool(sentiment),
            'NEGATIVE': not bool(sentiment),
            },
        } for sentiment in data_df['sentiment'].values
    ]

    train_no_items = int(len(data_df) * split)

    texts_train  = texts[:train_no_items]
    texts_valid  = texts[train_no_items:]

    labels_train = np.array(labels[:train_no_items])
    labels_valid = np.array(labels[train_no_items:])

    return texts_train, labels_train, texts_valid, labels_valid


def main() -> None:
    spacy.util.fix_random_seed(RANDOM_SEED)
    rd.seed(RANDOM_SEED)

    X_train, y_train, X_valid, y_valid = load_data('./data/yelp_ratings.csv')

    model = SentimentAnalyzer('en')

    model.train(X_train, y_train)

    predictions = model.predict(X_valid)
    true_classes = [int(label['cats']['POSITIVE'] == True) for label in y_valid]

    accuracy: float = np.mean(
        [predictions[i] == true_classes[i] for i in range(len(predictions))]
    )

    print(
        f"Accuracy: {round(accuracy * 100, 1)}%"
    )


if __name__ == "__main__":
    main()

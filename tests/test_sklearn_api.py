from unittest import TestCase

from faker import Faker
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline

from ptsdae.sklearn_api import SDAETransformer, SDAERepresentationTransformer


class TestSDAETransformer(TestCase):
    def test_basic(self):
        fake = Faker()
        fake.seed(0)
        pipeline = make_pipeline(
            CountVectorizer(
                stop_words='english',
                max_features=25,
                max_df=0.9
            ),
            SDAETransformer(
                dimensions=[25, 10, 2],
                pretrain_epochs=1,
                finetune_epochs=1
            )
        )
        pipeline.fit([fake.text() for _ in range(100)])
        result = pipeline.transform([fake.text() for _ in range(20)])
        self.assertEqual(result.shape, (20, 2))

    def test_score(self):
        fake = Faker()
        fake.seed(0)
        pipeline = Pipeline(steps=[
            ('vectorizer', CountVectorizer(
                stop_words='english',
                max_features=25,
                max_df=0.9
            )),
            ('ae', SDAETransformer(dimensions=[25, 10, 2], finetune_epochs=1))
        ])
        param_grid = {
            'ae__pretrain_epochs': [1, 2, 3],
        }
        search = GridSearchCV(pipeline, param_grid, iid=False, cv=2, return_train_score=False)
        search.fit([fake.text() for _ in range(10)])


class TestSDAERepresentationTransformer(TestCase):
    def test_basic(self):
        sizes = [25, 10, 2]
        fake = Faker()
        fake.seed(0)
        pipeline = make_pipeline(
            CountVectorizer(
                stop_words='english',
                max_features=25,
                max_df=0.9
            ),
            SDAERepresentationTransformer(
                dimensions=sizes,
                pretrain_epochs=1,
                finetune_epochs=1
            )
        )
        pipeline.fit([fake.text() for _ in range(100)])
        result = pipeline.transform([fake.text() for _ in range(20)])
        self.assertEqual(result.shape, (20, 10 + 2 + 10))

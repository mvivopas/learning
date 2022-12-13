import argparse
import datasets
from sentence_transformers import (
    InputExample,
    SentenceTransformer
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.cross_encoder import CrossEncoder

parser = argparse.ArgumentParser("Eval")
parser.add_argument('model', help='Path to a model to be evaluated')
args = parser.parse_args()

dev = datasets.load_dataset('glue', 'stsb', split='validation')

dev_set = []
for row in dev:
    dev_set.append(
        InputExample(
            texts=[row['sentence1'], row['sentence2']],
            label=float(row['label'])
        )
    )

if 'cross-encoder' in args.model:
    evaluator = CECorrelationEvaluator.from_input_examples(
        dev_set
    )
    model = CrossEncoder(args.model)
else:
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_set, write_csv=False
    )
    model = SentenceTransformer(args.model)

print(f'SCORE: {round(evaluator(model), 3)}')
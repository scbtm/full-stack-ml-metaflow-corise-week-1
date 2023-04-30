
"""This script can be used to fine tune a transformer and evaluate the results. Without a GPU it would take too long so we don't run this. 
(Maybe some debugging could be needed)

The actual model was trained using colab and relevant information (including tensorboard metrics logs) is available at:
https://huggingface.co/scbtm/distilbert_crw1/tensorboard 
"""

from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current, Parameter, conda_base #batch
import torch_steps
from metaflow.cards import Table, Markdown, Artifact

@conda_base(libraries={"datasets":"2.12.0",
                       "transformers":"4.28.1",
                       "evaluate":"0.4.0",
                       "pytorch":"1.11.0", 
                       })
class GoodFirstModelNLPFlow(FlowSpec):

    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter('split-sz', default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile('data', default='../data/Womens Clothing E-Commerce Reviews.csv')

    """Alternatively, if running from the terminal, we can use:
    import pathlib
    main_wd = pathlib.Path("/home/workspace/workspaces/full-stack-ml-metaflow-corise-week-1")
    data_path = next(iter(pathlib.Path(f'{main_wd}').rglob('Womens Clothing E-Commerce Reviews.csv')))

    and then pass data_path into the default argument of IncludeFile
    """

    @step
    def start(self):
        import pandas as pd
        import io 
        from datasets import Dataset 
        from transformers import AutoTokenizer, DataCollatorWithPadding

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


        def preprocessing(batch, label_threshold = 4):
            labels = [1 if x > label_threshold else 0 for x in batch['label']]
            token_dict = tokenizer(batch["text"], truncation = False)
            return dict({'input_ids': token_dict.input_ids, 'label':labels})
        
        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df['review_text'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        reviews = _has_review_df['review_text']
        labels = _has_review_df.apply(labeling_function, axis=1)

        dataset = Dataset.from_dict({'text': list(X.values), 'label': list(y.values)})
        dataset = dataset.map(preprocessing, batched = True)
        self.dataset = dataset.train_test_split(test_size = 0.2)
        self.tokenizer = tokenizer
        self.collate_fn = data_collator

        del df
        del _has_review_df
        del dataset
        del tokenizer
        del data_collator

        print(self.dataset)

        self.next(self.fine_tune)
    
    ### Include GPU compute
    ### @batch(gpu=1)
    @step
    def fine_tune(self):
        "Build the model"

        from transformers import AutoModelForSequenceClassification
        from transformers import TrainingArguments, Trainer
        import numpy as np
        import evaluate
        import torch



        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

        def compute_metrics(eval_pred):
            load_accuracy = evaluate.load("accuracy")
            load_f1 = evaluate.load("f1")
            load_rocauc = evaluate.load("roc_auc")
        
            logits, labels = eval_pred
            probs = torch.nn.functional.softmax(torch.tensor(logits),dim=-1)
            predictions = np.argmax(logits, axis=-1)
            accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
            f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
            roc_auc = load_rocauc.compute(prediction_scores = probs[:,1], references = labels)['roc_auc']
            return {"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc}
 
        training_args = TrainingArguments(
                                          output_dir='output',
                                          learning_rate=2e-5,
                                          per_device_train_batch_size=64,
                                          per_device_eval_batch_size=64,
                                          num_train_epochs=8,
                                          weight_decay=0.01,
                                          save_strategy="epoch",
                                          push_to_hub=False,
                                          )
        
        trainer = Trainer(
                          model=model,
                          args=training_args,
                          train_dataset= self.dataset['train'],
                          tokenizer=self.tokenizer,
                          data_collator=self.collator_fn,
                          compute_metrics=compute_metrics,
                          )

        trainer.train()

        evaluation = trainer.evaluate(self.dataset['test'])

        self.model = model
        self.model_metrics = pd.DataFrame(evaluation, index = ['model_1'])

        del model
        del evaluation
        del trainer

        self.next(self.end)
        
    @card(type='corise') # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):

        msg = 'Model 1 Accuracy: {}\Model 1 AUC: {}\Model 1 F1: {}'
        print(msg.format(
            round(self.model_metrics['eval_accuracy'],3), 
            round(self.model_metrics['eval_roc_auc'],3),
            round(self.model_metrics['eval_f1'],3)
        ))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.model_metrics['eval_accuracy']))

        current.card.append(Markdown("## Test 1 metrics: fine tuning distilbert"))
        
        current.card.append(Table.from_dataframe(self.model_metrics))


if __name__ == '__main__':
    GoodFirstModelNLPFlow()

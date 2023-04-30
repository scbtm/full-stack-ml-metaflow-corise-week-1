from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact

# TODO move your labeling function from earlier in the notebook here
labeling_function = lambda row: 1 if row['rating'] > 4 else 0

class BaselineNLPFlow(FlowSpec):

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

        # Step-level dependencies are loaded within a Step, instead of loading them 
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io 
        from sklearn.model_selection import train_test_split
        
        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df['review_text'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        reviews = _has_review_df['review_text']
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({'label': labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({'review': reviews, 'label': labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f'num of rows in train set: {self.traindf.shape[0]}')
        print(f'num of rows in validation set: {self.valdf.shape[0]}')

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        
        ### TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.
        from sklearn.metrics import accuracy_score as acc
        from sklearn.metrics import roc_auc_score as rocauc

        ## Simplest possible baseline is a naive classifier. This will predict the most common class (it ignores the features).
        from sklearn.dummy import DummyClassifier
        
        dummy_clf = DummyClassifier(strategy="uniform")
        X_train, y_train = self.traindf['review'], self.traindf['label']
        X_val, y_val = self.valdf['review'], self.valdf['label']
        dummy_clf.fit(X_train, y_train)

        preds = dummy_clf.predict(X_val)
        scores = dummy_clf.predict_proba(X_val)
        
        del X_train
        del X_val
        del y_train

        self.base_acc = acc(y_val, preds)
        self.base_rocauc = rocauc(y_val, scores[:,1])

        del y_val
        del preds
        del scores

        self.baseline_model = dummy_clf

        self.next(self.end)
        
    @card(type='corise') # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):

        msg = 'Baseline Accuracy: {}\nBaseline AUC: {}'
        print(msg.format(
            round(self.base_acc,3), round(self.base_rocauc,3)
        ))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        current.card.append(Markdown("## Examples of False Positives"))
        # TODO: compute the false positive predictions where the baseline is 1 and the valdf label is 0. 

        X_val, y_val = self.valdf['review'], self.valdf['label']
        preds = self.baseline_model.predict(X_val)
        eval_df = self.valdf.copy()
        eval_df['preds'] = preds

        fp_df = eval_df[(eval_df['preds'] == 1) & (eval_df['label']==0)]

        # TODO: display the false_positives dataframe using metaflow.cards
        # Documentation: https://docs.metaflow.org/api/cards#table

        current.card.append(Table.from_dataframe(fp_df))
        
        
        current.card.append(Markdown("## Examples of False Negatives"))
        # TODO: compute the false negative predictions where the baseline is 0 and the valdf label is 1. 
        fn_df = eval_df[(eval_df['preds'] == 0) & (eval_df['label']==1)]
        # TODO: display the false_negatives dataframe using metaflow.cards

        current.card.append(Table.from_dataframe(fn_df))


if __name__ == '__main__':
    BaselineNLPFlow()

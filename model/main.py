import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

ds = pd.read_csv('social_dataset.csv')
print(f'Valori nulli presenti:\n {ds.isnull().sum()}')

def encode_column(df, encoder):
    return encoder.transform(df).reshape(-1, 1)

def transform_macro_argomento(X):
    return encode_column(X[['macro_argomento']], le_macro_argomento)

def transform_argomento_spazio(X):
    return encode_column(X[['argomento_spazio']], le_argomento_spazio)


le_macro_argomento = LabelEncoder()
le_argomento_spazio = LabelEncoder()

le_macro_argomento.fit(ds['macro_argomento'])
le_argomento_spazio.fit(ds['argomento_spazio'])


#Selezione delle feature che il modello deve predire e su cui deve essere addestrato
x = ds.drop(['suggerito', 'id_utente'], axis='columns')
y = ds['suggerito'].values.reshape(-1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('macro_argomento_en', FunctionTransformer(transform_macro_argomento, validate=False), ['macro_argomento']),
            ('argomento_spazio_en', FunctionTransformer(transform_argomento_spazio, validate=False), ['argomento_spazio']),
        ],
        remainder='passthrough'
    )),
    ('model', RandomForestClassifier(n_estimators=35, criterion='entropy', max_features='sqrt'))
])

model.fit(xtrain, ytrain)

#Evaluation del modello
print(model.score(xtest, ytest))

new_data = pd.DataFrame(
    [['Finanza e Investimenti', 'Economia']],
    columns=['macro_argomento', 'argomento_spazio']
)
print(model.predict_proba(new_data))

#Importanza delle feature selezionate secondo il criterio di Information Gain
importances = model.named_steps['model'].feature_importances_
for feature, importance in zip(x.columns, importances):
    print(f'{feature}: {importance}')
selected_features = [feature for feature, importance in zip(x.columns, importances)]
print(f'Feature selezionate: {selected_features}')

#Esportazione del modello per l'utilizzo api
joblib.dump(model, 'model.joblib')
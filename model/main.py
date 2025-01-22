import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, \
    ConfusionMatrixDisplay

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

ds = pd.read_csv('social_dataset.csv')
print(f'Valori nulli presenti:\n {ds.isnull().sum()}')

#Normalizzazione dei dati
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
#Suddivisione in dati di train e test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('macro_argomento_en', FunctionTransformer(transform_macro_argomento, validate=False), ['macro_argomento']),
            ('argomento_spazio_en', FunctionTransformer(transform_argomento_spazio, validate=False), ['argomento_spazio']),
        ],
        remainder='passthrough'
    )),
    ('model', RandomForestClassifier(n_estimators=40, criterion='entropy', max_features='sqrt', random_state=42, class_weight='balanced')),
])
model.fit(xtrain, ytrain)

#Importanza delle feature selezionate secondo il criterio di Information Gain
importances = model.named_steps['model'].feature_importances_
for feature, importance in zip(x.columns, importances):
    print(f'Importanza {feature}: {importance}')
selected_features = [feature for feature, importance in zip(x.columns, importances)]
print(f'Feature selezionate: {selected_features}')

#Evaluation del modello
new_data = pd.DataFrame(
    [['Finanza e Investimenti', 'Economia']],
    columns=['macro_argomento', 'argomento_spazio']
)

# 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Iterazione sui fold
ytrue = []
ypred = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_idx, test_idx in cv.split(x, y):
    # Suddivisione dei dati
    xtrain_fold, xtest_fold = x.iloc[train_idx], x.iloc[test_idx]
    ytrain_fold, ytest_fold = y[train_idx], y[test_idx]

    # Addestramento del modello
    model.fit(xtrain_fold, ytrain_fold)

    # Predizione
    y_pred_fold = model.predict(xtest_fold)

    # Accumula le predizioni e le etichette vere
    ytrue.extend(ytest_fold)
    ypred.extend(y_pred_fold)

    # Calcolo delle metriche
    acc = accuracy_score(ytest_fold, y_pred_fold)
    prec = precision_score(ytest_fold, y_pred_fold, average='binary')
    rec = recall_score(ytest_fold, y_pred_fold, average='binary')
    f1 = f1_score(ytest_fold, y_pred_fold, average='binary')

    # Salvataggio dei risultati
    accuracy_scores.append(acc)
    precision_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)

# Visualizzazione dei risultati per ogni fold
for i in range(len(accuracy_scores)):
    print(f"Fold {i + 1}: Accuracy={accuracy_scores[i]:.4f}, Precision={precision_scores[i]:.4f}, Recall={recall_scores[i]:.4f}, F1 Score={f1_scores[i]:.4f}")

#Calcolo della confusion matrix complessiva
cm = confusion_matrix(ytrue, ypred)

# Calcolo delle metriche medie e relative deviazioni standard
print("\nMetriche medie e deviazioni standard:")
print(f"Accuracy: {sum(accuracy_scores)/len(accuracy_scores):.4f} ± {pd.Series(accuracy_scores).std():.4f}")
print(f"Precision: {sum(precision_scores)/len(precision_scores):.4f} ± {pd.Series(precision_scores).std():.4f}")
print(f"Recall: {sum(recall_scores)/len(recall_scores):.4f} ± {pd.Series(recall_scores).std():.4f}")
print(f"F1 Score: {sum(f1_scores)/len(f1_scores):.4f} ± {pd.Series(f1_scores).std():.4f}")
#Visualizzazione Confusion Matrix Complessiva
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.named_steps['model'].classes_)
disp.plot()
plt.title('Confusion Matrix Complessiva 10-Fold Validation')
plt.show()

#Esportazione del modello per l'utilizzo api
joblib.dump(model, 'model.joblib')
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, \
    ConfusionMatrixDisplay

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

ds = pd.read_csv('social_dataset.csv')
print(f'Valori nulli presenti:\n {ds.isnull().sum()}')

#Normalizzazione dei dati
le_macro_argomento = LabelEncoder()
le_argomento_spazio = LabelEncoder()
ds['macro_argomento'] =  le_macro_argomento.fit_transform(ds['macro_argomento'])
ds['argomento_spazio'] = le_argomento_spazio.fit_transform(ds['argomento_spazio'])

#Selezione delle feature che il modello deve predire e su cui deve essere addestrato
x = ds.drop(['id_utente', 'suggerito'], axis='columns')
y = ds.suggerito

#Suddivisione in dati di train e test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=40, criterion='entropy', max_features=None, random_state=42, class_weight='balanced')

model.fit(xtrain, ytrain)

new_data = pd.DataFrame(
    [['Relazioni e Famiglie', 'Crypto']],
    columns=['macro_argomento', 'argomento_spazio']
)

new_data['macro_argomento'] = le_macro_argomento.transform(new_data['macro_argomento'])
new_data['argomento_spazio'] = le_argomento_spazio.transform(new_data['argomento_spazio'])

#Evaluation del modello
print('PREDICTION:', model.predict(new_data))

#Esportazione del modello per l'utilizzo api
joblib.dump(model, 'model.pkl')

#Esportazione dell'encoder per i macro argomenti
joblib.dump(le_macro_argomento, 'le_macro_argomento.pkl')

#Esportazione dell'encoder per gli argomenti dello spazio
joblib.dump(le_argomento_spazio, 'le_argomento_spazio.pkl')

#Importanza delle feature selezionate secondo il criterio di Information Gain
importances = model.feature_importances_
for feature, importance in zip(x.columns, importances):  # rinomina 'importances' in 'importance'
    print(f'Importanza {feature} : {importance}')
selected_features = [feature for feature, importance in zip(x.columns, importances)]
print(f'Feature selezionate: {selected_features}')

# 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

ytrue = []
ypred = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

modelCv = model

for train_idx, test_idx in cv.split(x, y):
    # Suddivisione dei dati metodo vecchio
    xtrain_fold, xtest_fold = x.iloc[train_idx], x.iloc[test_idx]
    ytrain_fold, ytest_fold = y[train_idx], y[test_idx]

    # Addestramento del modello
    modelCv.fit(xtrain_fold, ytrain_fold)

    # Predizione
    y_pred_fold = modelCv.predict(xtest_fold)

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

# Calcolo delle metriche medie e relative deviazioni standard
print("\nMetriche medie e deviazioni standard:")
print(f"Accuracy: {sum(accuracy_scores)/len(accuracy_scores):.4f} ± {pd.Series(accuracy_scores).std():.4f}")
print(f"Precision: {sum(precision_scores)/len(precision_scores):.4f} ± {pd.Series(precision_scores).std():.4f}")
print(f"Recall: {sum(recall_scores)/len(recall_scores):.4f} ± {pd.Series(recall_scores).std():.4f}")
print(f"F1 Score: {sum(f1_scores)/len(f1_scores):.4f} ± {pd.Series(f1_scores).std():.4f}")

#Calcolo della confusion matrix complessiva
cm = confusion_matrix(ytrue, ypred)

#Visualizzazione Confusion Matrix Complessiva
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title('Confusion Matrix Complessiva 10-Fold Cross-Validation')
plt.show()
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)


def train_model():
    df = pd.read_csv("data.csv").dropna()
    labels = df["label"].values
    inputs = df.drop(["label"], axis=1)
    inputs.drop(columns=inputs.columns[0], axis=1, inplace=True)

    # Split data into train, validation and test sets
    # 10% validation, 20% test, 70% train
    data_in, test_in, data_out, test_out = train_test_split(
        inputs, labels, test_size=0.1, stratify=labels
    )

    train_in, val_in, train_out, val_out = train_test_split(
        data_in, data_out, test_size=0.2 / 0.9, stratify=data_out
    )

    print("Train size:", len(train_in), "Val size:", len(val_in), "Test size:", len(test_in))

    model = KNeighborsClassifier()

    k_range = list(range(1,20))
    param_grid = dict(n_neighbors=k_range)
    grid = GridSearchCV(
        model, param_grid, cv=10, scoring='accuracy'
        )
    grid.fit(train_in, train_out)

    # Evaluate model on validation set
    val_predictions = grid.predict(val_in)
    print("Validation accuracy:", 
      round(accuracy_score(val_out, val_predictions)*100, 2), 
      "%, with", grid.best_params_.get('n_neighbors'), 
      "neighbors" )
    
    # Evaluate model on test set
    tuned_neighbors_output = grid.predict(test_in)
    print("Test accuracy:", 
      round(accuracy_score(test_out, tuned_neighbors_output)*100, 2), 
      "%, with", grid.best_params_.get('n_neighbors'), 
      "neighbors" )    
    
    # Save model
    return grid 
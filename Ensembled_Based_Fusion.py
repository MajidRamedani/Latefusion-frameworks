import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import load_model, Model

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
    roc_curve)


# ====== #
# CONFIG #
# ====== #

CONFIG = {
    "model_paths":"Data_2026_02_28/Model_9_16_3_0001.h5",
    "pca_variance": 0.95,
    "random_state": 42
}


# ============ #
# DATA LOADING #
# ============ #

def load_data():

    from Input_Data_MRI import (
        cnn_train_gen,
        cnn_val_gen,
        cnn_test_gen,
        labels,
        class_names
    )

    return cnn_train_gen, cnn_val_gen, cnn_test_gen, labels, class_names

def MLP_data():
    from MLP import mains
    results = mains()
    
    mlp_train_features = results["train_features"]
    mlp_val_features = results["val_features"]
    mlp_test_features = results["test_features"]
    
    return mlp_train_features, mlp_val_features, mlp_test_features

# ========================== #
# LOAD CNN FEATURE EXTRACTOR #
# ========================== #

def load_cnn_feature_extractor(model_path):

    cnn_model = load_model(model_path)

    for layer in cnn_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    cnn_feature_model = Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer("CNN_features").output)

    return cnn_feature_model


# ============== #
# FEATURE FUSION #
# ============== #

def fuse_features(cnn_train, cnn_test, mlp_train, mlp_test):

    X_train = np.hstack([cnn_train, mlp_train])
    X_test = np.hstack([cnn_test, mlp_test])

    print("Fused feature shape:", X_train.shape)

    return X_train, X_test


def preprocess_features(X_train, X_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=CONFIG["pca_variance"])

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    print("After PCA:", X_train.shape)

    return X_train, X_test


# =========== #
# CLASSIFIERS #
# =========== #

def build_classifier():

    svm = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced"
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight="balanced",
        random_state=CONFIG["random_state"]
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=CONFIG["random_state"]
    )

    nb = GaussianNB()

    return VotingClassifier(
        estimators=[
            ("svm", svm),
            ("rf", rf),
            ("xgb", xgb),
            ("nb", nb)
        ],
        voting="soft",
        weights=[3, 2, 3, 1]
    )


# ========== #
# EVALUATION #
# ========== #

def evaluate_binary_task(X_train, y_train, X_test, y_test,
                         class_a, class_b,
                         classifier,
                         class_names=("AD", "bvFTD", "CN")):

    train_mask = np.isin(y_train, [class_a, class_b])
    test_mask = np.isin(y_test, [class_a, class_b])

    X_train_bin = X_train[train_mask]
    X_test_bin = X_test[test_mask]

    y_train_bin = (y_train[train_mask] == class_a).astype(int)
    y_test_bin = (y_test[test_mask] == class_a).astype(int)

    clf = clone(classifier)
    clf.fit(X_train_bin, y_train_bin)

    y_pred = clf.predict(X_test_bin)
    y_prob = clf.predict_proba(X_test_bin)[:, 1]

    acc = accuracy_score(y_test_bin, y_pred)
    sen = recall_score(y_test_bin, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred).ravel()
    spe = tn / (tn + fp)

    auc_score = roc_auc_score(y_test_bin, y_prob)
    f1 = f1_score(y_test_bin, y_pred)
    kappa = cohen_kappa_score(y_test_bin, y_pred)

    print(f"\n===== {class_names[class_a]} vs {class_names[class_b]} =====")
    print(f"Accuracy   : {acc:.3f}")
    print(f"Sensitivity: {sen:.3f}")
    print(f"Specificity: {spe:.3f}")
    print(f"AUC        : {auc_score:.3f}")
    print(f"F1 Score   : {f1:.3f}")
    print(f"Kappa      : {kappa:.3f}")

    cm = confusion_matrix(y_test_bin, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{class_names[class_a]} vs {class_names[class_b]}")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test_bin, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.show()


# ===== #
# MAIN  #
# ===== #

def main():

    model_path=CONFIG["cnn_model_path"]
    cnn_train_gen, cnn_val_gen, cnn_test_gen, labels, class_names = load_data()
    cnn_feature_model = load_cnn_feature_extractor(model_path)
    cnn_train_features = cnn_feature_model.predict(cnn_train_gen)
    cnn_test_features = cnn_feature_model.predict(cnn_test_gen)
    
    (mlp_train_features, mlp_val_features, mlp_test_features) = MLP_data()


    y_train = cnn_train_gen.labels
    y_test = cnn_test_gen.labels

    classifier = build_classifier()


    X_train, X_test = fuse_features(
        cnn_train_features, cnn_test_features,
        mlp_train_features, mlp_test_features
    )

    X_train, X_test = preprocess_features(X_train, X_test)

    evaluate_binary_task(X_train, y_train, X_test, y_test, 0, 2, classifier)
    evaluate_binary_task(X_train, y_train, X_test, y_test, 0, 1, classifier)
    evaluate_binary_task(X_train, y_train, X_test, y_test, 1, 2, classifier)


if __name__ == "__main__":
    main()








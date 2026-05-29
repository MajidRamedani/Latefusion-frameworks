import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, auc)


# ====== #
# CONFIG #
# ====== #

CONFIG = {
    "cnn_model_path": "Models/Model_1_8_3_0.0001.h5",
    "fusion_epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "patience": 7
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



# ================== #
# BUILD FUSION MODEL #
# ================== #

def build_fusion_model(cnn_feature_dim, mlp_feature_dim, n_classes):

    # ====== #
    # INPUTS #
    # ====== #

    cnn_input = layers.Input(shape=(cnn_feature_dim,), name="cnn_features")
    mlp_input = layers.Input(shape=(mlp_feature_dim,), name="mlp_features")

    # ========== #
    # CNN BRANCH #
    # ========== #

    cnn_x = layers.BatchNormalization()(cnn_input)
    cnn_x = layers.Dense(128, activation="relu", name="cnn_projection")(cnn_x)

    # ========== #
    # MLP BRANCH #
    # ========== #

    mlp_x = layers.BatchNormalization()(mlp_input)
    mlp_x = layers.Dense(128, activation="relu", name="mlp_projection")(mlp_x)

    # ====== #
    # FUSION #
    # ====== #

    fusion = layers.Concatenate(name="fusion_concat")([cnn_x, mlp_x])
    fusion = layers.Dense(128, activation="relu")(fusion)
    fusion = layers.Dropout(0.5)(fusion)

    outputs = layers.Dense(n_classes, activation="softmax")(fusion)

    model = models.Model(inputs=[cnn_input, mlp_input], outputs=outputs, name="CNN_MLP_Fusion")

    return model



# ============= #
# CLASS WEIGHTS #
# ============= #

def compute_class_weights(labels):
    class_dist = np.sum(labels, axis=0)
    total = np.sum(class_dist)
    n_classes = len(class_dist)

    return {
        i: total / (n_classes * class_dist[i])
        for i in range(n_classes)
        }


# ==== #
# PLOT #
# ==== #

def plot_roc_curves(y_true, y_score, class_names):

    n_classes = y_true.shape[1]

    colors = ["blue", "green", "red"]

    plt.figure(figsize=(8, 6))

    for i, color in zip(range(n_classes), colors):

        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=color, lw=2, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fusion ROC Curves")

    plt.legend(loc="lower right")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()


# ==== #
# MAIN #
# ==== #

def main():

    model_path=CONFIG["cnn_model_path"]
    cnn_train_gen, cnn_val_gen, cnn_test_gen, labels, class_names = load_data()
    cnn_feature_model = load_cnn_feature_extractor(model_path)
    cnn_train_features = cnn_feature_model.predict(cnn_train_gen)
    cnn_test_features = cnn_feature_model.predict(cnn_test_gen)
    cnn_val_features = cnn_feature_model.predict(cnn_val_gen)
    
    (mlp_train_features, mlp_val_features, mlp_test_features) = MLP_data()


    # =========== #
    # BUILD MODEL #
    # =========== #

    fusion_model = build_fusion_model(
        cnn_feature_dim=cnn_train_features.shape[1],
        mlp_feature_dim=mlp_train_features.shape[1],
        n_classes=labels.shape[1])

    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(
        learning_rate=CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    fusion_model.summary()


    class_weights = compute_class_weights(labels)

    # ========= #
    # CALLBACKS #
    # ========= #

    early_stop = EarlyStopping(monitor="val_loss", patience=CONFIG["patience"],
                               restore_best_weights=True, verbose=1)

    # ===== #
    # TRAIN #
    # ===== #

    history = fusion_model.fit(
        [cnn_train_features, mlp_train_features],
        cnn_train_gen.labels,
        validation_data=([cnn_val_features, mlp_val_features],cnn_val_gen.labels),
        epochs=CONFIG["fusion_epochs"],
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        callbacks=[early_stop],
        class_weight=class_weights)

    # ========== #
    # EVALUATION #
    # ========== #

    test_loss, test_acc = fusion_model.evaluate(
        [cnn_test_features, mlp_test_features],
        cnn_test_gen.labels, verbose=0)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


    y_pred_probs = fusion_model.predict(
        [cnn_test_features, mlp_test_features])

    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(cnn_test_gen.labels, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names)

    print("\nClassification Report")
    print(report)


    macro_auc = roc_auc_score(cnn_test_gen.labels, y_pred_probs, multi_class="ovr",average="macro")
    weighted_auc = roc_auc_score(cnn_test_gen.labels, y_pred_probs, multi_class="ovr", average="weighted")

    print(f"\nWeighted ROC-AUC: {weighted_auc:.4f}")
    print(f"Macro ROC-AUC: {macro_auc:.4f}")

    for i, cls in enumerate(class_names):

        auc_i = roc_auc_score(
            cnn_test_gen.labels[:, i],
            y_pred_probs[:, i])

        print(f"{cls} ROC-AUC: {auc_i:.4f}")

    plot_roc_curves(y_true=cnn_test_gen.labels, y_score=y_pred_probs, class_names=class_names)


# === #
# RUN #
# === #

if __name__ == "__main__":
    main()


import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv3D, BatchNormalization, AveragePooling3D, MaxPooling3D, concatenate, ReLU
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

CONFIG = {
    "output_file": "Classification_reports.txt",
    "filters": [8, 12, 16],
    "blocks": [3, 4],
    "learning_rates": [0.0001, 0.00001],
    "cnn_input_shape": (193, 229, 193, 1),
    "epochs": 50,
    "batch_size": 32,
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


# ============ #
# MODEL BLOCKS #
# ============ #

def Conv_layer(conv_x, filters, kernel=1, strides=1):
    conv_x = BatchNormalization()(conv_x)
    conv_x = ReLU()(conv_x)
    conv_x = Conv3D(filters, kernel, strides=strides, padding="same")(conv_x)
    conv_x = layers.Dropout(0.2)(conv_x)
    return conv_x


def Dense_block(block_x, n_layers, filters):
    layers_concat = [block_x]

    block_x = Conv_layer(block_x, filters)
    block_x = Conv_layer(block_x, filters, kernel=3)
    layers_concat.append(block_x)

    for _ in range(n_layers):
        block_x = concatenate(layers_concat)
        block_x = Conv_layer(block_x, filters)
        block_x = Conv_layer(block_x, filters, kernel=3)
        layers_concat.append(block_x)

    return concatenate(layers_concat)


def Transition_layer(trans_x):
    trans_x = Conv_layer(trans_x, K.int_shape(trans_x)[-1] // 2)
    trans_x = AveragePooling3D(2, strides = 2, padding = 'same')(trans_x)
    
    return trans_x


def Dense_net(cnn_input_shape, n_classes, filters, layers_blc):
    inp = layers.Input(shape=cnn_input_shape)

    x = Conv3D(10, 7, strides=2, padding="same")(inp)
    x = MaxPooling3D(3, strides=2, padding="same")(x)

    layers_in_block = [layers_blc,layers_blc]
    for repetition in layers_in_block:                     
        d = Dense_block(x, repetition,filters)
        x = Transition_layer(d)

    Output_x = MaxPooling3D(2)(d)
    Output_x = layers.Activation("relu")(Output_x)
    Output_x = layers.Dropout(0.3)(Output_x)

    cnn_features = layers.Flatten(name="CNN_features")(Output_x)

    output = layers.Dense(
        n_classes,
        activation="softmax",
        kernel_regularizer=L1L2(1e-5, 1e-4)
    )(cnn_features)
    
    model = Model(inputs=inp, outputs=output, name="DenseNet_Fusion")
    
    return model



# ==== #
# Plot #
# ==== #

def plot_training_curves_and_roc(history, y_true, y_pred_probs, class_names,
                                 experiment_id, filters, blocks, lr):

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)

    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()



    plt.subplot(1, 3, 2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title(
        f"Accuracy | Exp={experiment_id} | "
        f"F={filters}, B={blocks}, LR={lr}"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()



    plt.subplot(1, 3, 3)
    y_true_onehot = to_categorical(
        y_true,
        num_classes=len(class_names)
    )

    for i, cls in enumerate(class_names):

        fpr, tpr, _ = roc_curve(
            y_true_onehot[:, i],
            y_pred_probs[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{cls} (AUC={roc_auc:.2f})"
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve (OvR)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# ================= #
# FEATURE EXTRACTOR #
# ================= #

def extract_features(model, train_gen, val_gen, test_gen):


    feature_extractor = Model(
        inputs=model.input,
        outputs=model.get_layer("CNN_features").output
    )


    cnn_train_features = feature_extractor.predict(train_gen)
    cnn_val_features = feature_extractor.predict(val_gen)
    cnn_test_features = feature_extractor.predict(test_gen)

    print("\nFeature extraction completed")
    print(f"Train features: {cnn_train_features.shape}")
    print(f"Validation features: {cnn_val_features.shape}")
    print(f"Test features: {cnn_test_features.shape}")

    return cnn_train_features, cnn_val_features, cnn_test_features

# ============== #
# RUN EXPERIMENT #
# ============== #

def run_experiment(cnn_train_gen, cnn_val_gen, cnn_test_gen, labels, class_names,
                   filters, blocks, lr, experiment_id, all_features):

    class_weights = compute_class_weights(labels)

    cnn_model = Dense_net(
        CONFIG["cnn_input_shape"],
        labels.shape[1],
        filters,
        blocks
    )

    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint = ModelCheckpoint(
        f"Models/model_{experiment_id}_{filters}_{blocks}.h5",
        mode='auto',
        save_best_only=True,
        monitor="val_loss"
    )

    early = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    cnn_history = cnn_model.fit(
        cnn_train_gen,
        validation_data=cnn_val_gen,
        epochs=CONFIG["epochs"],
        callbacks=[checkpoint, early],
        class_weight=class_weights,
        verbose=1
    )
    
    cnn_train_features, cnn_val_features, cnn_test_features = extract_features(
    cnn_model,
    cnn_train_gen,
    cnn_val_gen,
    cnn_test_gen,
    experiment_id
    )
    
    all_features[experiment_id] = {
    "train": cnn_train_features,
    "val": cnn_val_features,
    "test": cnn_test_features,
    "filters": filters,
    "blocks": blocks,
    "learning_rate": lr
    }
    


    # ========== #
    # PREDICTION #
    # ========== #

    y_pred = cnn_model.predict(cnn_test_gen)
    y_pred_cls = y_pred.argmax(axis=1)

    y_true = []
    for _, y in cnn_test_gen:
        y_true.extend(np.argmax(y, axis=1))
    y_true = np.array(y_true)

    report = classification_report(y_true, y_pred_cls, target_names=class_names)

    weighted_auc = roc_auc_score(y_true, y_pred, multi_class="ovr", average="weighted")
    
    plot_training_curves_and_roc(
        history=cnn_history,
        y_true=y_true,
        y_pred_probs=y_pred,
        class_names=class_names,
        experiment_id=experiment_id,
        filters=filters,
        blocks=blocks,
        lr=lr
    )
    # ============ #
    # SAVE RESULTS #
    # ============ #

    with open(CONFIG["output_file"], "a") as f:
        f.write(f"\n=== Experiment {experiment_id} ===\n")
        f.write(report + "\n")
        f.write(f"Weighted AUC: {weighted_auc:.4f}\n")

    print(report)
    print("AUC:", weighted_auc)

    return cnn_model, cnn_history


# ========= #
# MAIN LOOP #
# ========= #

def main():

    cnn_train_gen, cnn_val_gen, cnn_test_gen, labels, class_names = load_data()

    experiment_id = 0
    all_features = {}
    
    for f in CONFIG["filters"]:
        for b in CONFIG["blocks"]:
            for lr in CONFIG["learning_rates"]:

                experiment_id += 1

                print(f"\nRunning experiment {experiment_id}")
                print(f"filters={f}, blocks={b}, lr={lr}")

                run_experiment(
                    cnn_train_gen,
                    cnn_val_gen,
                    cnn_test_gen,
                    labels,
                    class_names,
                    f,
                    b,
                    lr,
                    experiment_id,
                    all_features
                )


# === #
# RUN #
# === #

if __name__ == "__main__":
    main()
   
            



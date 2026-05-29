import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ====== #
# CONFIG #
# ====== #

CONFIG = {
    "volume_csv": "Volume.csv",
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "checkpoint_path": "Models/mlp_best.h5"
}


# =========== # 
# LOAD SPLITS # 
# =========== # 

def load_splits():

    from Input_Data_MRI import (
        train_images,
        val_images,
        test_images
    )

    return (
        train_images[0],
        val_images[0],
        test_images[0]
    )

# ============== #
# LABEL ENCODING #
# ============== #

def encode_labels(labels):

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    onehot = tf.keras.utils.to_categorical(encoded,num_classes=len(encoder.classes_))

    return onehot

# ======================= #
# TABULAR DATA EXTRACTION #
# ======================= #

def build_volume_dataframe(subjects, volume_df):

    lookup = pd.Series(volume_df.index.values,index=volume_df["Filename"]).to_dict()
    rows = []
    for subject in subjects:
        if subject in lookup:
            rows.append(volume_df.iloc[lookup[subject]])

    return pd.DataFrame(rows)


def prepare_tabular_data(subjects, volume_df):

    df = build_volume_dataframe(subjects, volume_df)
    mlp_labels = encode_labels(df["Original Class"].values)
    mlp_data = df.iloc[:, 4:].to_numpy(dtype=np.float32)

    return mlp_data, mlp_labels

# ========== #
# FOCAL LOSS #
# ========== #

def focal_loss(gamma=2.0, alpha=0.1):

    def loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        ce = tf.keras.losses.categorical_crossentropy(y_true,y_pred)
        pt = tf.exp(-ce)
        return alpha * tf.pow(1 - pt, gamma) * ce

    return loss

# ========= #
# BUILD MLP #
# ========= #

def build_mlp(input_dim, n_classes):

    mlp_inputs = tf.keras.Input(shape=(input_dim,),name="volume_input")

    x = layers.Dense(256, activation="relu")(mlp_inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32,activation="relu",name="mlp_features")(x)
    x = layers.Dropout(0.3)(x)

    mlp_outputs = layers.Dense(n_classes, activation="softmax")(x)
    mlp_model = models.Model(mlp_inputs, mlp_outputs)
    
    return mlp_model


# ================== #
# FEATURE EXTRACTION #
# ================== #

def extract_mlp_features(model, train_x, val_x, test_x):

    mlp_feature_extractor = Model(inputs=model.input,outputs=model.get_layer("mlp_features").output)

    mlp_train_features = mlp_feature_extractor.predict(train_x)
    mlp_val_features = mlp_feature_extractor.predict(val_x)
    mlp_test_features = mlp_feature_extractor.predict(test_x)

    print("\nMLP Feature Extraction")
    print("Train:", mlp_train_features.shape)
    print("Validation:", mlp_val_features.shape)
    print("Test:", mlp_test_features.shape)

    return mlp_train_features, mlp_val_features, mlp_test_features


# ============== #
# RUN EXPERIMENT #
# ============== #

def mains():
    
    train_subjects, val_subjects, test_subjects = load_splits()
    volume_df = pd.read_csv(CONFIG["volume_csv"])

    x_train, y_train = prepare_tabular_data(train_subjects,volume_df)
    x_val, y_val = prepare_tabular_data(val_subjects,volume_df)
    x_test, y_test, _ = prepare_tabular_data(test_subjects,volume_df)

    print("\nDataset Shapes")
    print("Train:", x_train.shape)
    print("Validation:", x_val.shape)
    print("Test:", x_test.shape)


    model = build_mlp(input_dim=x_train.shape[1],n_classes=y_train.shape[1])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
        learning_rate=CONFIG["learning_rate"]),
        loss=focal_loss(),
        metrics=["accuracy"]
    )

    model.summary()


    checkpoint = ModelCheckpoint(
        CONFIG["checkpoint_path"],
        save_best_only=True,
        monitor="val_loss"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )


    mlp_history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        callbacks=[checkpoint, early_stop]
    )


    mlp_train_features, mlp_val_features, mlp_test_features = extract_mlp_features(model,x_train,x_val,x_test)

    # ========== #
    # PREDICTION #
    # ========== #
    
    test_loss, test_acc = model.evaluate(
        x_test,
        y_test,
        verbose=0)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred_prob = model.predict(x_test)

    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report")
    print(classification_report(y_true, y_pred))
    
    return {
    "history": mlp_history,
    "train_features": mlp_train_features,
    "val_features": mlp_val_features,
    "test_features": mlp_test_features}


# === #
# RUN #
# === #

if __name__ == "__main__":
    mains()
    






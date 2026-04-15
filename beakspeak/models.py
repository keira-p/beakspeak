import tensorflow as tf

def build_transfer_learning_model(
    num_classes,
    backbone="mobilenetv2",
    input_shape=(224, 224, 3),
    use_augmentation=False,
    data_augmentation=None
):
    """
    Build a transfer learning model with selectable backbone.

    Parameters:
    - num_classes: Number of output classes
    - backbone: "mobilenetv2", "efficientnetb0", or "resnet50"
    - input_shape: Expected input shape for the model
    - use_augmentation: Whether to include data augmentation layers
    - data_augmentation: A tf.keras.Sequential of augmentation layers (if use_augmentation is True)

    Returns:
    - A compiled tf.keras.Model ready for training
    """

    backbone = backbone.lower()

    # --- Backbone selection ---
    if backbone == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    elif backbone == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input

    elif backbone == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base_model.trainable = False

    # --- Model ---
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if use_augmentation and data_augmentation:
        x = data_augmentation(x)

    x = preprocess(x)
    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs)

    return model

from metaflow import FlowSpec, step, conda, S3, conda_base,\
                     resources, Flow, project, Parameter

@project(name='image_classification')
@conda_base(python='3.7.12', libraries={'tensorflow': '2.7.0', 'tensorflow_addons': '0.15.0', 'numpy': '1.19.5'})
class ImageClassificationFlow(FlowSpec):
    """
      Поток, классифицирующий изображения.

      Этот поток состоит из следующих шагов:
      1)
      2)
      3)
    """
    @conda(python='3.7.12', libraries={'tensorflow': '2.7.0', 'tensorflow_addons': '0.15.0', 'numpy': '1.19.5'})
    @step
    def start(self):
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.batch_size = 128
        self.num_epochs = 10
        self.next(self.split_data)

    @conda(python='3.7.12', libraries={'tensorflow': '2.7.0', 'tensorflow_addons': '0.15.0', 'numpy': '1.19.5'})
    @step
    def split_data(self):
        import tensorflow
        (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.cifar10.load_data()
        val_split = 0.1

        val_indices = int(len(x_train) * val_split)
        self.new_x_train, self.new_y_train = x_train[val_indices:], y_train[val_indices:]
        self.x_val, self.y_val = x_train[:val_indices], y_train[:val_indices]

        print(f"Training data samples: {len(self.new_x_train)}")
        print(f"Validation data samples: {len(self.x_val)}")
        print(f"Test data samples: {len(self.x_test)}")
        self.next(self.make_data)

    @conda(python='3.7.12', libraries={'tensorflow': '2.7.0', 'tensorflow_addons': '0.15.0', 'numpy': '1.19.5'})
    @step
    def make_data(self):
        import tensorflow
        image_size = 32
        auto = tensorflow.data.AUTOTUNE

        data_augmentation = tensorflow.keras.Sequential(
            [tensorflow.keras.layers.RandomCrop(image_size, image_size),
             tensorflow.keras.layers.layers.RandomFlip("horizontal"), ],
            name="data_augmentation",
        )

        def make_datasets(images, labels, is_train=False):
            dataset = tensorflow.data.Dataset.from_tensor_slices((images, labels))
            if is_train:
                dataset = dataset.shuffle(self.batch_size * 10)
                dataset = dataset.batch(self.batch_size)
            if is_train:
                dataset = dataset.map(
                    lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
                )
            return dataset.prefetch(auto)

        train_dataset = make_datasets(self.new_x_train, self.new_y_train, is_train=True)
        val_dataset = make_datasets(self.x_val, self.y_val)
        test_dataset = make_datasets(self.x_test, self.y_test)
        self.next(self.eval)

    @conda(python='3.7.12', libraries={'tensorflow': '2.7.0', 'tensorflow_addons': '0.15.0', 'numpy': '1.19.5'})
    @step
    def eval(self):
        import tensorflow
        import tensorflow_addons as tfa
        def activation_block(x):
            x = tensorflow.keras.layers.Activation("gelu")(x)
            return tensorflow.keras.layers.BatchNormalization()(x)

        def conv_stem(x, filters: int, patch_size: int):
            x = tensorflow.keras.layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
            return activation_block(x)

        def conv_mixer_block(x, filters: int, kernel_size: int):
            # Depthwise convolution.
            x0 = x
            x = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
            x = tensorflow.keras.layers.Add()([activation_block(x), x0])  # Residual.

            # Pointwise convolution.
            x = tensorflow.keras.layers.Conv2D(filters, kernel_size=1)(x)
            x = activation_block(x)

            return x

        def get_conv_mixer_256_8(
                image_size=32, filters=256, depth=8, kernel_size=5, patch_size=2, num_classes=10
        ):
            """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
            The hyperparameter values are taken from the paper.
            """
            inputs = tensorflow.keras.Input((image_size, image_size, 3))
            x = tensorflow.keras.layers.Rescaling(scale=1.0 / 255)(inputs)

            # Extract patch embeddings.
            x = conv_stem(x, filters, patch_size)

            # ConvMixer blocks.
            for _ in range(depth):
                x = conv_mixer_block(x, filters, kernel_size)

            # Classification block.
            x = tensorflow.keras.layers.GlobalAvgPool2D()(x)
            outputs = tensorflow.keras.layers.Dense(num_classes, activation="softmax")(x)

            return tensorflow.keras.Model(inputs, outputs)

        def run_experiment(model):
            optimizer = tfa.optimizers.AdamW(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            )

            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            checkpoint_filepath = "/tmp/checkpoint"
            checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
                checkpoint_filepath,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
            )

            history = model.fit(
                self.train_dataset,
                validation_data=self.val_dataset,
                epochs=self.num_epochs,
                callbacks=[checkpoint_callback],
            )

            model.load_weights(checkpoint_filepath)
            _, accuracy = model.evaluate(self.test_dataset)
            print(f"Test accuracy: {round(accuracy * 100, 2)}%")

            return history, model

        self.conv_mixer_model = get_conv_mixer_256_8()
        self.history, self.conv_mixer_model = run_experiment(self.conv_mixer_model)
        self.next(self.vizualize)

    @conda(python='3.7.12', libraries={'numpy': '1.19.5'})
    @step
    def vizualize(self):
        import numpy as np
        import tensorflow_addons as tfa
        import tensorflow as tf
        import matplotlib.pyplot as plt
        def visualization_plot(weights, idx=1):
            # First, apply min-max normalization to the
            # given weights to avoid isotrophic scaling.
            p_min, p_max = weights.min(), weights.max()
            weights = (weights - p_min) / (p_max - p_min)

            # Visualize all the filters.
            num_filters = 256
            plt.figure(figsize=(8, 8))

            for i in range(num_filters):
                current_weight = weights[:, :, :, i]
                if current_weight.shape[-1] == 1:
                    current_weight = current_weight.squeeze()
                ax = plt.subplot(16, 16, idx)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(current_weight)
                idx += 1
            # We first visualize the learned patch embeddings.

        patch_embeddings = self.conv_mixer_model.layers[2].get_weights()[0]
        visualization_plot(patch_embeddings)
        for i, layer in enumerate(self.conv_mixer_model.layers):
            if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                if layer.get_config()["kernel_size"] == (5, 5):
                    print(i, layer)
        idx = 26  # Taking a kernel from the middle of the network.

        kernel = self.conv_mixer_model.layers[idx].get_weights()[0]
        kernel = np.expand_dims(kernel.squeeze(), axis=2)
        visualization_plot(kernel)
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    ImageClassificationFlow()
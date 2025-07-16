
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split


tf.config.run_functions_eagerly(True)


(x_data, y_data), (_, _) = tf.keras.datasets.cifar10.load_data()


x_data = x_data.astype('float32') / 255.0
y_data = y_data.flatten()


x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.5, random_state=42, shuffle=True
)

print(f"Training samples: {x_train.shape[0]}")
print(f"Testing samples: {x_test.shape[0]}")


def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


def get_optimizer(name):
    if name == 'GD':
        return optimizers.SGD(learning_rate=0.01)
    elif name == 'SGD':
        return optimizers.SGD(learning_rate=0.01)
    elif name == 'Momentum':
        return optimizers.SGD(learning_rate=0.01, momentum=0.9)
    elif name == 'RMSprop':
        return optimizers.RMSprop(learning_rate=0.001)
    elif name == 'Adam':
        return optimizers.Adam(learning_rate=0.001)

optimizers_list = ['GD', 'SGD', 'Momentum', 'RMSprop', 'Adam']


history_dict = {}

for opt_name in optimizers_list:
    print(f"\nTraining with {opt_name} optimizer...")
    model = create_model()

    opt = get_optimizer(opt_name)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    batch_size = len(x_train) if opt_name == 'GD' else 64

    history = model.fit(
        x_train[:len(x_train)//2], y_train[:len(y_train)//2],
        epochs=10,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    history_dict[opt_name] = history


sns.set_style('whitegrid')
fig, axes = plt.subplots(2, 1, figsize=(14, 10))


for opt_name, history in history_dict.items():
    axes[0].plot(history.history['val_loss'], label=f'{opt_name}')
axes[0].set_title('Validation Loss Comparison')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()


for opt_name, history in history_dict.items():
    axes[1].plot(history.history['val_accuracy'], label=f'{opt_name}')
axes[1].set_title('Validation Accuracy Comparison')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()


print("\nFinal Test Accuracy with Different Optimizers:")

for opt_name in optimizers_list:
    model = create_model()
    opt = get_optimizer(opt_name)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    batch_size = len(x_train) if opt_name == 'GD' else 64


    model.fit(x_train[:len(x_train)//2], y_train[:len(y_train)//2], epochs=10, batch_size=batch_size, verbose=0)


    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{opt_name}: {test_acc:.4f}")

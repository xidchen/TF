import numpy as np
import tensorflow as tf


# Construct Distiller() class

class Distiller(tf.keras.Model):

    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.student_loss_fn = None
        self.distillation_loss_fn = None
        self.alpha = None
        self.temperature = None

    def compile(
        self,
        optimizer=None,
        metrics=None,
        student_loss_fn=None,
        distillation_loss_fn=None,
        alpha=None,
        temperature=None,
        **kwargs,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_predictions)
            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            ) * self.temperature ** 2
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(target=loss, sources=trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update the metrics configured in `compile()`
        self.compiled_metrics.update_state(y, student_predictions)
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {'student_loss': student_loss, 'distillation_loss': distillation_loss}
        )
        return results

    def test_step(self, data):
        x, y = data
        y_prediction = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_prediction)
        # Update the metrics
        self.compiled_metrics.update_state(y, y_prediction)
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({'student_loss': student_loss})
        return results


# Create student and teacher models

# Initialy, we create a teacher model and a smaller student model.
# Both models are convolutional neural networks and created using Sequential(),
# but could be any Keras model.

teacher_model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
    tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10),
], name='teacher')

student_model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
    tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10),
], name='student')

# Clone student for later comparison
student_scratch = tf.keras.models.clone_model(student_model)


# Prepare the dataset

# The dataset used for training the teacher and distilling the teacher is MNIST,
# and the procedure would be equivalent for any other dataset, e.g. CIFAR-10,
# with a suitable choice of models. Both the student and teacher are trained
# on the training set and evaluated on the test set.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = x_test.astype('float32') / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))


# Train the teacher

# In knowledge distillation we assume that the teacher is trained and fixed.
# Thus, we start by training the teacher model on the training set in the usual way.

teacher_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
teacher_model.fit(x=x_train, y=y_train, batch_size=64, epochs=5)
teacher_model.evaluate(x=x_test, y=y_test)


# Distill teacher to student

# We have already trained the teacher model,
# and we only need to initialize a Distiller(student, teacher) instance,
# compile() it with the desired losses, hyperparameters and optimizer,
# and distill the teacher to the student.

distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10
)
distiller.fit(x=x_train, y=y_train, batch_size=64, epochs=3)
distiller.evaluate(x=x_test, y=y_test)


# Train student from scratch for comparison

# We can also train an equivalent student model from scratch without the teacher,
# in order to evaluate the performance gain obtained by knowledge distillation.

student_scratch.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
student_scratch.fit(x=x_train, y=y_train, batch_size=64, epochs=3)
student_scratch.evaluate(x=x_test, y=y_test)


# If the teacher is trained for 5 full epochs and the student is distilled
# on this teacher for 3 full epochs, you should in this example experience
# a performance boost compared to training the same student model from scratch,
# and even compared to the teacher itself.
# You should expect the teacher to have accuracy around 97.6%,
# the student trained from scratch should be around 97.6%,
# and the distilled student should be around 98.1%.
# Remove or try out different seeds to use different weight initializations.

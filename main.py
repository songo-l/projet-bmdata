import tensorflow as tf
import datetime
import time
from models import Generator, Discriminator
from losses import perc_loss_cal, d_loss, make_up_loss


generator = Generator()
discriminator_A = Discriminator()
discriminator_B = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer_A = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer_B = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_A, input_B, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output_A, gen_output_B = generator([input_A, input_B], training=True)

        disc_A_output = discriminator_A(input_A, training=True)
        disc_B_output = discriminator_B(input_B, training=True)

        disc_gen_A_output = discriminator_A(gen_output_A, training=True)
        disc_gen_B_output = discriminator_B(gen_output_B, training=True)

        perc_A = perc_loss_cal(gen_output_A)
        perc_B = perc_loss_cal(gen_output_B)
        make_up = make_up_loss(gen_output_A, input_B)
        gen_total_loss = perc_A + perc_B + make_up

        disc_loss_A = d_loss(disc_A_output, disc_gen_A_output)
        disc_loss_B = d_loss(disc_B_output, disc_gen_B_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)

    discriminator_gradients_A = disc_tape.gradient(disc_loss_A, discriminator_A.trainable_variables)
    discriminator_gradients_B = disc_tape.gradient(disc_loss_B, discriminator_B.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    discriminator_optimizer_A.apply_gradients(zip(discriminator_gradients_A, discriminator_A.trainable_variables))
    discriminator_optimizer_B.apply_gradients(zip(discriminator_gradients_B, discriminator_B.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('perc_A', perc_A, step=epoch)
        tf.summary.scalar('perc_B', perc_B, step=epoch)
        tf.summary.scalar('make_up', make_up, step=epoch)
        tf.summary.scalar('disc_loss_A', disc_loss_A, step=epoch)
        tf.summary.scalar('disc_loss_B', disc_loss_B, step=epoch)


def fit(train_A_ds, train_B_ds, epochs):
    for epoch in range(epochs):
        start = time.time()
        # Train
        for n_A, input_A, n_B, input_B in zip(train_A_ds.enumerate(), train_A_ds.enumerate()):
            print('.', end='')
            n = min(n_A, n_B)
            if (n + 1) % 100 == 0:
                print()
            train_step(input_A, input_B, epoch)
        print()

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))

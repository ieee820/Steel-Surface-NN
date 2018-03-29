from keras.preprocessing.image import ImageDataGenerator


class SteelDataGenerator:

    @classmethod
    def get_data_generators(cls, train_dir, val_dir, test_dir, img_width, img_height, batch_size, use_samplewise=True):
        datagen = ImageDataGenerator(
            samplewise_center=use_samplewise, 
            samplewise_std_normalization=use_samplewise,
            featurewise_center=not use_samplewise,
            featurewise_std_normalization=not use_samplewise,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
        val_generator = datagen.flow_from_directory(
            val_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
        test_generator = datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
        return train_generator, val_generator, test_generator

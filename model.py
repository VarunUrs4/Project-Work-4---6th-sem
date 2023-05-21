import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create a list of the directories containing your training, validation, and testing datasets:
training_dir = 'train/dataset'
validation_dir = 'validation/dataset'
testing_dir = 'test/dataset'

# Create a list of the labels for each class in your dataset:
classes = ['apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot','cauliflower','chilli pepper','corn','cucumber','eggplant','garlic','ginger','grapes','jalepeno','kiwi','lemon','lettuce','mango','onion','orange','paprika','pear','peas','pineapple','pomegranate','potato','raddish','soy beans','spinach','sweetcorn','sweetpotato','tomato','turnip','watermelon']

# Create a generator for the training dataset:
train_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_data = train_generator.flow_from_directory(
    training_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Create a generator for the validation dataset:
validation_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

validation_data = validation_generator.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Create a generator for the testing dataset:
testing_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

testing_data = testing_generator.flow_from_directory(
    testing_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Define the model architecture:
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
          activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model:
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train the model:
model.fit_generator(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=10,
    validation_data=validation_data,
    validation_steps=len(validation_data)
)

# Evaluate the model:
score = model.evaluate_generator(
    testing_data,
    steps=len(testing_data)
)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model:
model.save('model1.h5')


# 'apple',
# 'banana',
# 'beetroot',
# 'bell pepper',
# 'cabbage',
# 'capsicum',
# 'carrot',
# 'cauliflower',
# 'chilli pepper',
# 'corn',
# 'cucumber',
# 'eggplant',
# 'garlic',
# 'ginger',
# 'grapes',
# 'jalepeno',
# 'kiwi',
# 'lemon',
# 'lettuce',
# 'mango',
# 'onion',
# 'orange',
# 'paprika',
# 'pear',
# 'peas',
# 'pineapple',
# 'pomegranate',
# 'potato',
# 'raddish',
# 'soy beans',
# 'spinach',
# 'sweetcorn',
# 'sweetpotato',
# 'tomato',
# 'turnip',
# 'watermelon'

import os
from keras.optimizers import RMSprop , SGD
from image_iden import ModelUse


cat_dog_use = ModelUse(
    
    # data_src_dir=os.path.join(os.getcwd() , "hw\data_mining_project\image_iden\weather"),
    filtered_dir=os.path.join(os.getcwd() , "hw/data_mining_project/image_iden/cats_and_dogs_filtered"),
    rearange_data=False,
    prediction_keys = ["cat" , "dog"],
    model_name="cat_dog_model",
    # train_split_size=150,
    conv2_act_func=["relu","relu","relu"],
    image_input_shape=(150,150,3),
    show_logs = True,
    output_activation="sigmoid",
    optimizer=SGD(learning_rate=0.001, momentum=0.9),
    epoch_number=10,
    verbose=2,
    # image_class_mode="binary"

)



## Predictions


# dall_e_images_folder = os.path.join(os.getcwd() , "drive/MyDrive/intro-to-AI/hw/data_mining_project/image_iden/dall-e_images")


# for img in os.listdir(dall_e_images_folder):
        

#         img_path = os.path.join(dall_e_images_folder , img)

        

#         if "dog" in img.lower() or "cat" in img.lower():
#             print("*"*80)
#             print(img)
#             results = cat_dog_use.predict_image_with_cnn(img_path=img_path)
#             print(results)
#             # print(predict_image_with_vgg16(vgg16_model, img_path, vgg16.preprocess_input, vgg16.decode_predictions))
#             print("*"*80)




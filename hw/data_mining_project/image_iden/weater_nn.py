from image_iden import ModelUse
import os
from keras.optimizers import RMSprop , SGD



weater_model = ModelUse(

        data_src_dir=os.path.join(os.getcwd() , "hw\data_mining_project\image_iden\weather"),
        filtered_dir=os.path.join(os.getcwd() , "hw\data_mining_project\image_iden\weather_filtered"),
        prediction_keys = ["rain" , "cloudy" , "shine" , "sunrise"],
        model_name="weather_model",
        train_split_size=150,
        show_logs = True,
        output_activation="softmax",
        optimizer=SGD(learning_rate=0.001, momentum=0.9),
        epoch_number=1,

        )



## Predictions

# dall_e_images_folder = os.path.join(os.getcwd() , "hw\data_mining_project\image_iden\dall-e_images")

# for img in os.listdir(dall_e_images_folder):
#         print("*"*80)
#         img_path = os.path.join(dall_e_images_folder , img)
#         print(img)
#         results = weater_model.predict_image_with_cnn(img_path=img_path)
#         print(results)
#         # print(predict_image_with_vgg16(vgg16_model, img_path, vgg16.preprocess_input, vgg16.decode_predictions))
#         print("*"*80)







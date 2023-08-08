from project_methods import *

def main(file_path, epochs):

    #Loading data
    print('\n\nLoading Data...\n\n')
    #Configure partition size and chunk size
    p_size = 100
    c_size = 10
    x, y = load_data_dask(file_path, partition_size=p_size, chunk_size=c_size)

    #Use labelencoder to encode Y labels into integers for classification
    print('\n\nEncoding labels...\n\n')
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    #Count the classes from the files pulled
    classes = len(label_encoder.classes_)

    #Configure train test splits and preprocess data for models
    split_num = .25
    train_data, test_data = preprocess_dask(x, y, split = split_num, num_class = classes)

    #Create and fit model
    print('\n\nRunning Model...\n\n')
    CNN_model = run_improved_CNN_model(train=train_data, test=test_data, epoch_num=epochs, metric_functions=['accuracy'],
                loss_function='categorical_crossentropy', optimizer_function='adam', encoder=label_encoder)
    print('\n\nModels finished running. Now generating predictions.')

    #Load in data to predict
    #test or new?
    #check for valid response
    valid_response = False
    while valid_response == False:
        new_images = str.lower(input('\Generate predictions from new or test images? '))
        if new_images in ['new', 'test']:
            valid_response= True
        else:
            print('Please enter \'test\' or \'new\'.')
    #Generate prediction data
    if new_images=='test':
        newX, newY  = load_data('test_new_images')
    elif new_images=='new':
        newX, newY = load_data('new_images')

    #Calculate predictions of new images
    cnn_predictions = predictions(newX, CNN_model, encoder = label_encoder)

    #Compare predictions
    print('\n\nThe following is a list of the predictions made on the new images by the CNN model:\n', cnn_predictions,'\n')
    print('Now here are the actual labels of the images:\n', newY,'\n')

    

if __name__ == "__main__":

    #Collect valid directory response
    valid_directory_response = False
    while valid_directory_response ==False:
        file_path = input('Please select a directory to pull data from: ')
        if file_path in os.listdir():
            valid_directory_response = True
        else:
            print('Please type one of the following directories:\n\n')
            print([folder for folder in os.listdir() if (len(folder.split('.'))<2) and (folder!='__pycache__')])

    #Collect valid epoch number response
    valid_epoch_response = False
    while valid_epoch_response == False:
        try:
            epoch_num = int(input('Please enter the desired number of epochs to use in the model generation: '))
            valid_epoch_response = True
        except:
            print('Please enter a number.')

    main(file_path = file_path, epochs = epoch_num)
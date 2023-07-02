import mUtils as u

def our_procedure(_fileName='',_modelName = ''):
    # return parameter class contain only structur to use in app
    p = u.MyParam() 
    # give the file name to class parameters , also Sheet name if don't default is 'Sheet1'
    p.init_param() # get current excel sheet (only one in folder)               
    ''' get the data from excel file and save it class parameter
        X value = X = data[["mass", "s", "N part", "Pt"]]
        y value = y = data["spectrum"]
    '''
    p.get_data_in_Param()
    # normaliz input
    p.normaliz_data()

    # give the model name to class parameters 
    p.set_model_config_param() 



    # Then we will start at normalizing data
    return True

###################################################################

# start the main function
def main():
    print("Starting main call")
    our_procedure()    
    print("The End")


if __name__ == "__main__":
    main()

import mUtils as u


# start the main function
def main():
    print("Starting main call")

    p = u.MyParam()
    p.our_procedure(_modelNameP='mtest.h5',_loadP=False)
    #p.init_param()
    #p.get_data_in_Param()

    print('out folder', p.outFolder)
    print('inputfile : ',p.inputFile)
    print('inputFolder : ',p.inputFolder)
    

    print("The End of main ")


if __name__ == "__main__":
    main()

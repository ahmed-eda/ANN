# Call the liberaries that we made or call
import mUtils as u


# define the main function
def main():
    print("Starting main call")

    p = u.MyParam()
    p.our_procedure(_modelNameP='mpi-neg-epoch-200.h5',_loadP=False,_epochP=200,_batchSizeP=16)
    #p.init_param()
    #p.get_data_in_Param()


    print('out folder', p.outFolder)
    print('inputfile : ',p.inputFile)
    print('inputFolder : ',p.inputFolder)
    
    print('score : ',p.score)
    
    print('RMSE : ',p.rmse)
    

    print("The End of main ")

# call the main function
if __name__ == "__main__":
    main()

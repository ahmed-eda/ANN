# All needed headers lib

# Lib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import RMSprop
from keras.models import load_model
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from io import StringIO


class MyParam:
# Costractor
    def __init__(self):
        # data parameters
        self.outFolder = 'out'
        self.inputFolder = 'data'
        self.modelNamePath = ''
        self.inputFile = ''
        self.inputSheetName = ''
        self.data = ''
        self.X = ''
        self.y = ''
        self.X_train = ''  
        self.X_test = ''
        self.predictions = ''
        self.datap = pd.DataFrame(data=None)
        self.mergedData = ''
        self.outputpredicat = ''
        # Model parameteres  
        self.model = None
        self.myepochs = int(0)
        self.mybatchSize = int(0)
        self.modelName = ''
        self.score = 0
        self.mse = 0
        self.rmse = 0
        # Output parameters
        self.outputFile = ''
        self.summaryOutFile = ''
        self.outputSheetName = ''
        self.nameFigImg = ''
        self.nameAllFigImg = ''
        print('end class construction')

# Utility function

    # Get current data excel file name (doesn't start with out* ) 
    def get_input_file_name(self):
        readFolder = self.inputFolder
        print('os.listdir("readFolder") : ',os.listdir(readFolder))
        excel_files = [
            f for f in os.listdir(readFolder) if f.endswith(".xlsx") and not f.startswith("out")
        ]
        if not excel_files:
            raise ValueError("No Excel files found in the current directory")
        return excel_files[0]

    # get model file name
    def get_model_fromfile(self):
        _modelNames = [f for f in os.listdir('.') if (f.endswith('.h5'))]        
        print(' get model from file _modelNames[0] : ',_modelNames[0] )
        if not _modelNames:
            raise ValueError("No model files found in function the current directory")
        return _modelNames[0]

# Methods

# Init
    def init_param(self,_outfolder='out',_inputFolder='data',_inputfile='',sheetName='Sheet1',_epochP=100,_batchSizeP=16):
        # data parameters
        # Get the current working directory
        cwd = os.getcwd()

        
        
        self.inputSheetName = "Sheet1"        
        # Model parameteres 
        self.myepochs=_epochP
        self.mybatchSize =_batchSizeP
        pass
        # Output parameters
        subdir = os.path.join(cwd, _inputFolder)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        self.inputFolder= str(subdir)
        subdir=''
        
        if(len(_inputfile)>0):
            self.inputFile = os.path.join(self.inputFolder,self.inputFile)  
            #self.inputFile = _inputfile
        else:
            self.inputFile = self.get_input_file_name() # get current excel file in the folder
            self.inputFile = os.path.join(self.inputFolder,self.inputFile)  

        print('input file : ',self.inputFile)       
        # Create a subdirectory if it doesn't exist
        subdir = os.path.join(cwd, _outfolder)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        self.outFolder= str(subdir)
        pass
        print('end init_param')
    
# Read data
    def get_data_in_Param(self):
        # Read the data from the excel file in class of parameters
        data_all = pd.read_excel(self.inputFile, sheet_name=self.inputSheetName)
        temp_data_all = data_all  # data_all[data_all['spectrum']<60]
        data = temp_data_all.reset_index(drop=True)
        # Split the data into input and output variables
        self.X = data[["mass", "s", "N part", "Pt"]]
        self.y = data["spectrum"].to_frame("spectrum")
        self.data = data
        print(self.X.head)
        print(self.y.head)
        print('end get_data_in_Param')
        
# Normaliz input
    def normaliz_data(self):
        # Normalize the input
        from sklearn.preprocessing import RobustScaler
        # Create a RobustScaler object
        scaler = RobustScaler()
        # Fit the scaler to the input data and transform it
        X_normalized = scaler.fit_transform(self.X)
        # Print the normalized input data
        print('X_normalized')
        print(X_normalized)
        self.X_train = X_normalized
        print('end normaliz_data')
        
# Set modelname and config param
    def set_model_config_param(self,_modelName=''):
        if(len(_modelName)>0):
            self.modelName=_modelName
        else:
            self.modelName = self.get_model_fromfile()
        
        if len(self.modelName)<1:
            raise ValueError("No model files found at set model file in the current directory")
        
        # Output parameters
        self.modelNamePath = os.path.join(self.inputFolder,self.modelName)
          
        self.outputFile = 'out_ '+self.modelName+' .xlsx'
        self.summaryOutFile = self.modelName + ' _ Summary .txt'
        self.outputSheetName = 'predicat_ '+self.modelName+' '
        self.nameFigImg = 'fig_Sep_ '+self.modelName+' .png'
        self.nameAllFigImg = 'fig_All_ '+self.modelName+' .png'
        # add out path
        self.outputFile = os.path.join(self.outFolder,self.outputFile)
        self.summaryOutFile = os.path.join(self.outFolder,self.summaryOutFile)
        #self.outputSheetName = os.path.join(self.outFolder,self.outputSheetName)
        self.nameFigImg = os.path.join(self.outFolder,self.nameFigImg)
        self.nameAllFigImg = os.path.join(self.outFolder,self.nameAllFigImg)

        
        print('end set_model_config_param')



# Create model 
    def creat__new_model(self):
        # Define the model
        self.model = Sequential(name=self.modelName)
        print('fun : create_new_model : modelname',self.modelName)
        # Add the first dense layer
        self.model.add(Dense(40, input_dim=4, activation='relu'))
        # Add batch normalization
        #model.add(BatchNormalization())
        self.model.add(Dense(40, activation='relu'))
        #model.add(BatchNormalization())
        self.model.add(Dense(80, activation='relu'))
        self.model.add(Dense(80, activation='relu'))
        self.model.add(Dense(40, activation='relu'))
        self.model.add(Dense(40, activation='relu'))
        # Add the output layer
        self.model.add(Dense(1))
        ''' # compile the model      '''
        # Compile the model with Levenberg-Marquardt optimizer
        optimizer = RMSprop(learning_rate=0.001, rho=0.001,)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        print('after compiling model : model is: ',self.model)
        ''' train the model & save current compiled model  '''
        # Train the model
        #model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
        #self.myepochs = int(100) 
        #self.mybatchSize = int(32)
        print('my epochs : ',self.myepochs)
        print('my batch size : ',self.mybatchSize)
        # fit on all data
        self.model.fit(self.X_train, self.y, epochs=self.myepochs, 
                        batch_size=self.mybatchSize) 
        # Set the validation split to 20%.
        #self.model.fit(self.X_train, self.y, epochs=self.myepochs, 
        #                batch_size=self.mybatchSize, validation_split=0.2) 
        print('model after fitting : ',self.model)
        # Save the model
        self.modelNamePath = os.path.join(self.outFolder,self.modelName)
        self.model.save(self.modelNamePath) 
        #self.model.save(os.path.join(self.outFolder,self.modelName)) 

        print('end create model')


# define model or load it
    def load_or_creat_Model(self,load=True):
        if(load):        
           # load the selected model
           self.modelNamePath = os.path.join(self.inputFolder,self.modelName)
           self.model = load_model(self.modelNamePath)        
           #self.model = load_model(self.modelName)        
           print('model is loaded : ',self.model)
        else:
            # create model
            self.creat__new_model()
            print('end create model')

        print('end load or create model')
    

# Evalute the model
    def evaluate_model(self):
        from sklearn.metrics import mean_squared_error

        # Make predictions on new data
        self.X_test =pd.DataFrame(self.X_train) #scaler.transform(X)
        #X_test = scaler.fit_transform(X)
        print("new_data is : ")
        print(self.X_test)
        self.predictions = self.model.predict(self.X_test)
        self.predictions = self.predictions.flatten()
        self.predictions = pd.Series(self.predictions)
        self.predictions = self.predictions.to_frame('predictions')
        print("predictions is : ")
        print(self.predictions)

        # Evaluate the model
        self.score = self.model.evaluate(self.X_test, self.y)
        print("score " , self.score)
        print(self.score)
        self.mse = mean_squared_error(self.y,self.predictions)
        print('mse' , self.mse)

        print('end evaluate')


# Write output to excel 
    def write_to_excel(self):
        # merge date[] with predictions
        print('shape of datap',self.datap.shape)
        print('shape of predictions',self.predictions.shape)
        if(self.datap.empty):
            self.datap = pd.merge(self.data,self.predictions,left_index=True, right_index=True)
            print('shape of datap in if',self.datap.shape)
            pass
        else:
            pass


        print('shape of datap after if',self.datap.shape)
       
        # Write predictions , data to Excel file
        err1=self.data['err1'].to_frame('err1')
        err2=self.data['err2'].to_frame('err2')
        #SquareErrorForEachPoint = np.sqrt( ((datap['predictions']- datap['Spectrum'])/(err1- err2)))
        SquareErrorForEachPoint =np.square( (self.datap['predictions']- self.datap['spectrum'])/(err1['err1']+ err2['err2']))

        SquareErrorForEachPoint = pd.Series(SquareErrorForEachPoint)
        SquareErrorForEachPoint = SquareErrorForEachPoint.to_frame('SquareErrorForEachPoint')
        print('Square error for each point : ',SquareErrorForEachPoint)

        self.outputpredicat = pd.concat([self.datap, SquareErrorForEachPoint], axis=1)
        mysum =self.outputpredicat['SquareErrorForEachPoint'].sum()
        mycount =(self.outputpredicat['SquareErrorForEachPoint'].count()) -1
        self.rmse = np.sqrt(mysum/mycount)

        #rmse = np.sqrt (np.average(outputpredicat['SquareErrorForEachPoint']))
        rmseX = pd.Series(self.rmse)
        rmseX = rmseX.to_frame('RMSE')
        #rmse = pd.DataFrame({'rmse': rmse})
        print('RMSE',rmseX)

        # output is data frame
        #print(outputpredicat.head(10))
        # Write the DataFrames to an Excel file with three sheets
        scoreX = pd.Series(self.score)
        scoreX = scoreX.to_frame('Score')
        with pd.ExcelWriter(self.outputFile) as writer:
            self.outputpredicat.to_excel(writer, sheet_name=self.outputSheetName, index=False)
            rmseX.to_excel(writer, sheet_name='RMSE', index=False)
            scoreX.to_excel(writer, sheet_name='Score', index=False)

        print('end write to excel ')

# Print model summery 
    def out_model_summary(self):
        
        from io import StringIO
        # summarize the model
        print('start summarize the model')
        with StringIO() as buf:
            self.model.summary(print_fn=lambda x: buf.write(x + '\n'))
            summary = buf.getvalue()
            summary += '\n score : ' + str(self.score)
            summary += '\n RMSE : ' + str(self.rmse)
            #summary += '\n RMSE : ' + str(self.rmse.values)
        print('end summarize the model')
       
        #with open(self.modelName +'-summary.txt', 'w') as f:
        with open(self.summaryOutFile,'w') as f:
            f.write(summary)

        print('start print summary')
        print('summary : ' , self.model.summary())
        print('model name : ',self.modelName)
        # print Root Mean Square Error that computed as Dr Mohamed want using err1&err2
        #print('RMSE',self.rmse['RMSE'].values)
        print('RMSE',self.rmse)
        # print("accuracy :" + str(accuracy))
        print("score " + str(self.score))
        #print("score " + str(self.score))
       
        print('end out model summary')


# Draw     
   # draw
    def draws_of_model_error_bar2(self):
        # for drawing in 2d i choose Pt as x-axis
        # draw after excel 

        # merge date[] with predictions
        if(self.datap.empty):
            self.datap = pd.merge(self.data,self.predictions,left_index=True, right_index=True)
            print('shape of datap',self.datap.shape)
            pass
        else:
            pass

        # datap : data + prediction
        #self.datap =pd.DataFrame(self.datap) # pd.DataFrame(datape12)# pd.DataFrame(datap)
        # xapf : datap after filteration
        xapf= self.datap.copy(deep=True)
        print('datap = ',self.datap)
        print('xapf : ',xapf)
        mass_filter = xapf['mass'].unique()[0]
        s_filter = xapf['s'].unique()[0]
        print('mass_filetr : ',mass_filter)
        print('s_filetr : ',s_filter)
       
        #xapf = xapf[xapf['mass']== mass_filter] # 0.13957]   #0.493677] #0.13957]
        #xapf = xapf[xapf['s']== s_filter ] #7.7]
       
        #xapf = xapf[xapf['N part']==337]

        print('xapf modified : ',xapf)

        # detinct N Part values
        N_Part_Values  =xapf['N part'].unique() #xap['N part'].unique() # xapf['N part'].unique()
        print('Npart values : \n')
        for n in N_Part_Values:
            print('N is : ',n)
        print('Npart values : \n',N_Part_Values)

              
        self.mergedData = xapf.copy()
        yerror = self.mergedData['err2']
        print('yerror : ',yerror.shape)
        error = self.mergedData['err2'] #yerror.to_numpy()
        print('error : ',error.shape)
        # Create a figure with  subplots
        #fig, axs = plt.subplots(nrows=len(N_Part_Values), ncols=1, figsize=(10, 100))
        
        for mass_item in xapf['mass'].unique():
            for s_item in xapf['s'].unique():
                for n_part_item in xapf['N part'].unique():
                    print(' mass : ', mass_item)
                    print(' s : ',s_item)
                    print(' n part : ', n_part_item)
                    all_axis =    xapf.loc[(xapf['mass'] == mass_item) & (xapf['s'] == s_item) & (xapf['N part']==n_part_item)]
                    all_axis = all_axis.sort_values(by='Pt')
                    # filtered_df = df.loc[(df['age'] >= 25) & (df['age'] <= 40) & (df['gender'] == 'female')]
                    X_axis = all_axis['Pt']
                    #X_axis = xapf['Pt'][xapf['mass']==mass_item][xapf[xapf['s']== s_item]][xapf['N part']==n_part_item]
                    Yspectrum_axis = all_axis['spectrum']
                    #Yspectrum_axis = xapf['spectrum'][xapf['mass']==mass_item][xapf[xapf['s']== s_item]][xapf['N part']==n_part_item]
                    Ypredictions_axis = all_axis['predictions']
                    #Ypredictions_axis = xapf['predictions'][xapf['mass']==mass_item][xapf[xapf['s']== s_item]][xapf['N part']==n_part_item]
                    e_axis = all_axis['err2']
                    #e_axis = xapf['err2'][xapf['mass']==mass_item][xapf[xapf['s']== s_item]][xapf['N part']==n_part_item]
                    g_title = 'N_part = {:.0f}'.format(n_part_item) + ' ,mass = {:.6f}'.format(mass_item) + ' ,s = {:.1f}'.format(s_item)
                    
                    
                    g_title += ' ' + ' score is  ' + '{:.3f}'.format(self.score)

                    if(self.datap.empty):
                        #g_title += ' ' + ' RMSE  is  ' + '{:.3f}'.format(self.rmse.values[1])
                        pass
                    else:
                        g_title += ' ' + ' RMSE  is  ' + '{:.3f}'.format(self.rmse)
                        pass
                    
                    if Yspectrum_axis.count() > 0:
                        #print('spectrum : ',Yspectrum_axis)
                       
                        plt.errorbar(x=X_axis, 
                             y=Yspectrum_axis, 
                             yerr=e_axis, 
                             fmt='o', color='blue',markersize=5,
                             label=' spectrum N_part = {}'.format(n),
                             ecolor='green', elinewidth=3, capsize=10)
                        
                        plt.scatter( x=X_axis, 
                                     y=Ypredictions_axis, 
                                     color='black', s=5,
                                     label=' predictions N_part = {}'.format(n))  
                       
                        plt.plot( X_axis, 
                                     Ypredictions_axis, 
                                     color='red',
                                     label=' predictions N_part = {}'.format(n)) 


                        max_s_p = int(max(max(Ypredictions_axis),max(Yspectrum_axis)))
                        y_scale = int(max(e_axis))
                       
                        plt.xlim(left=0,right = int(max(X_axis)+0.1))
                        plt.ylim(bottom=0)
                        #plt.autoscale(enable=True, axis='x', tight=True)
                        plt.autoscale(enable=True, axis='y', tight=True)
                        # set the x-axis tick mark to be every 0.2 units
                        xticks=[i*0.2 for i in range(int(max(X_axis)/0.2)+1)]
                        plt.xticks(xticks)
                       
             

                        plt.xlabel('Pt')
                        plt.ylabel('Invariant Yield')
                        plt.title(g_title)
                        plt.legend(loc='upper right')
                        #plt.legend(['Data'], loc='upper right')
                        #plt.legend(self.mergedData['spectrum'][self.mergedData['N part'] == n], loc='upper left')

                        plt.savefig(self.nameFigImg+'_'+ g_title +'_'+'.png')

                        plt.clf()
                        #plt.legend(['Data'], loc='upper left')

                        #plt.show()



                        pass
                    else:
                        pass 

        print("end plotting ",self.nameFigImg)

        print('end draws of model')
   


# Start process
    def our_procedure(self,_loadP=True,_fileNameP='',_modelNameP = 'test',_outfolder='out',_inputFolder='data',_sheetName='Sheet1',_epochP=100,_batchSizeP=16):
        # give the file name to class parameters , also Sheet name if don't default is 'Sheet1'
        self.init_param(_inputfile=_fileNameP,_outfolder=_outfolder,_inputFolder=_inputFolder,sheetName=_sheetName,_epochP=_epochP,_batchSizeP=_batchSizeP) # get current excel sheet (only one in folder)               
        # get the data from excel file and save it class parameter
        self.get_data_in_Param()
        # normaliz input
        self.normaliz_data()
        # give the model name to class parameters 
        self.set_model_config_param(_modelName=_modelNameP) 
        # load or creat model
        self.load_or_creat_Model(load=_loadP)
        # evaluate model
        self.evaluate_model() 
       
        #self.draws_of_model()
        # write to excel
        self.write_to_excel()
        # print model summery 
        self.out_model_summary()

         # darw
        self.draws_of_model_error_bar2()

        
        return True

    


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
        self.datap = ''
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
        print('os.listdir(".") : ',os.listdir("."))
        excel_files = [
            f for f in os.listdir(".") if f.endswith(".xlsx") and not f.startswith("out")
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
    def init_param(self,_outfolder='out',_inputFolder='data',_inputfile='',sheetName='Sheet1'):
        # data parameters
        # Get the current working directory
        cwd = os.getcwd()
        if(len(_inputfile)>0):
            self.inputFile = _inputfile
        else:
            self.inputFile = self.get_input_file_name() # get current excel file in the folder
        
        self.inputSheetName = "Sheet1"        
        # Model parameteres 
        pass
        # Output parameters
        subdir = os.path.join(cwd, _inputFolder)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        self.inputFolder= str(subdir)
        subdir=''
        self.inputFile = os.path.join(self.inputFolder,self.inputFile)        
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
    def creat__new_model(self, _epoch=100,_batchSize=16):
        # Define the model
        self.model = Sequential(name=self.modelName)
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
        ''' train the model & save current compiled model  '''
        # Train the model
        #model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
        self.myepochs = int(_epoch) 
        self.mybatchSize = int(_batchSize)
        
        self.model.fit(self.X_train, self.y, epochs=self.myepochs, batch_size=self.mybatchSize) 
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

# Draw 
    def draws_of_model(self):
        # for drawing in 2d i choose Pt as x-axis
        """ error = data['spectrum'] - predictions['predictions']
        error = error.to_frame('error') """

        # merge date[] with predictions
        self.datap = pd.merge(self.data,self.predictions,left_index=True, right_index=True)
        print('shape of datap',self.datap.shape)


        # datap : data + prediction
        self.datap =pd.DataFrame(self.datap) # pd.DataFrame(datape12)# pd.DataFrame(datap)
        # xapf : datap after filteration
        xapf= pd.DataFrame(self.datap)
        xapf = xapf[xapf['mass']==0.13957]
        xapf = xapf[xapf['s']==7.7]
        #xapf = xapf[xapf['N part']==337]

        # detinct N Part values
        N_Part_Values  =xapf['N part'].unique() #xap['N part'].unique() # xapf['N part'].unique()
        print('Npart values : \n')
        for n in N_Part_Values:
            print('N is : ',n)
        print('Npart values : \n',N_Part_Values)

        # create graph data set  = mergedData
        # to add err 1
        dataGraphe1 = pd.merge(xapf['Pt'],xapf['err1'],left_index=True, right_index=True)
        print('datagraph1 : \n',dataGraphe1)
        # to add err 2
        dataGraphe12 = pd.merge(dataGraphe1,xapf['err2'],left_index=True, right_index=True)
        print('datagraph12 : \n',dataGraphe12)
        # to add predictions
        dataGraph12p = pd.merge(dataGraphe12,xapf['predictions'],left_index=True, right_index=True)
        print('datagraph12p : \n',dataGraph12p)
        # finally add spectrum
        dataGraph12ps = pd.merge(dataGraph12p,xapf['spectrum'],left_index=True, right_index=True)
        print('dataGraph12ps : \n', dataGraph12ps)
        print('shape of dataGraph12ps',dataGraph12ps.shape)
        # Plot the data and predictions
        self.mergedData = pd.merge(dataGraph12ps,xapf['N part'],left_index=True, right_index=True)
        print('merged data is : \n',self.mergedData)

        # modify upper and lower error values
        self.mergedData['err1'] = self.mergedData['spectrum'] - self.mergedData['err1']
        self.mergedData['err2'] = self.mergedData['err2'] + self.mergedData['spectrum']

        yerror = self.mergedData['err2']-self.mergedData['err1']
       
        for n in N_Part_Values:
            plt.scatter(self.mergedData['Pt'][self.mergedData['N part']==n],self.mergedData['spectrum'][self.mergedData['N part']==n])
            plt.scatter(self.mergedData['Pt'][self.mergedData['N part']==n],self.mergedData['predictions'][self.mergedData['N part']==n])

            #print('n test :\n',mergedData['Pt'][mergedData['N part']==n])
        plt.savefig(self.nameAllFigImg)
        print("end plotting "+"fig-all-"+self.nameFigImg )
        
        # Define the list ofValues and plot the data for each iteration
        # Create a figure with  subplots
        fig, axs = plt.subplots(nrows=len(N_Part_Values), ncols=1, figsize=(10, 100))
        
        for i, n in enumerate(N_Part_Values):

            # Plot the 'Pt' column where N_part == n
            #axs[i].errorbar(mergedData['Pt'][mergedData['N part'] == n], 
            axs[i].scatter(self.mergedData['Pt'][self.mergedData['N part'] == n], 
                         self.mergedData['spectrum'][self.mergedData['N part'] == n], 
                         #color='C{}'.format(i), 
                         #yerr=yerror,
                         s=10,
                         color='green',
                         label='spectrum N_part = {}'.format(n))

            # Plot the 'predictions' column where N_part == n
            axs[i].scatter(self.mergedData['Pt'][self.mergedData['N part'] == n], 
                         self.mergedData['predictions'][self.mergedData['N part'] == n], 
                         color='orange', s=10,
                          label=' predictions N_part = {}'.format(n))
                         #label='_nolegend_')
            # Plot the 'err1' column where N_part == n
            axs[i].scatter(self.mergedData['Pt'][self.mergedData['N part'] == n], 
                         self.mergedData['err1'][self.mergedData['N part'] == n], 
                         color='black', s=10,
                          label=' err1 N_part = {}'.format(n)) 
                         #label='_nolegend_')
            # Plot the 'err2' column where N_part == n
            axs[i].scatter(self.mergedData['Pt'][self.mergedData['N part'] == n], 
                         self.mergedData['err2'][self.mergedData['N part'] == n], 
                         color='red', s=10,
                          label=' err2 N_part = {}'.format(n)) 
                         #label='_nolegend_')
             # Plot the 'err2' column where N_part == n
            """ axs[i].plot(mergedData['Pt'][mergedData['N part'] == n], 
                         mergedData['err2'][mergedData['N part'] == n], 
                         color='black', 
                          label=' err2 N_part = {}'.format(n)) """
                         #label='_nolegend_')

            # Add a legend and axis labels to the subplot
            axs[i].legend()
            axs[i].set_xlabel('Pt')
            axs[i].set_ylabel('Value')
            axs[i].set_title('N_part = {}'.format(n))

        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.5)

        # Show the plot
        plt.savefig(self.nameFigImg)
        #plt.show()
        #plt.close('all')


        print("end plotting ",self.nameFigImg)

        print('end draws of model')
       
# Write output to excel 
    def write_to_excel(self):
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
        self.rmse = pd.Series(self.rmse)
        self.rmse = self.rmse.to_frame('rmse')
        #rmse = pd.DataFrame({'rmse': rmse})
        print('RMSE',self.rmse)

        # output is data frame
        #print(outputpredicat.head(10))
        # Write the DataFrames to an Excel file with three sheets
        with pd.ExcelWriter(self.outputFile) as writer:
            self.outputpredicat.to_excel(writer, sheet_name=self.outputSheetName, index=False)
            self.rmse.to_excel(writer, sheet_name='RMSE', index=False)

        print('end write to excel ')

# Print model summery 
    def out_model_summary(self):
        
        from io import StringIO
        # summarize the model
        print('start summarize the model')
        with StringIO() as buf:
            self.model.summary(print_fn=lambda x: buf.write(x + '\n'))
            summary = buf.getvalue()
        print('end summarize the model')
       
        with open(self.modelName +'-summary.txt', 'w') as f:
            f.write(summary)

        print('start print summary')
        print('summary : ' , self.model.summary())
        print('model name : ',self.modelName)
        # print Root Mean Square Error that computed as Dr Mohamed want using err1&err2
        print('RMSE',self.rmse['rmse'].values)
        # print("accuracy :" + str(accuracy))
        print("score " + str(self.score))
       
        print('end out model summary')

# Start process
    def our_procedure(self,_loadP=True,_fileNameP='',_modelNameP = '',_outfolder='out',_inputFolder='data',_sheetName='Sheet1'):
        # return parameter class contain only structur to use in app
        # self.MyParam() 
        # give the file name to class parameters , also Sheet name if don't default is 'Sheet1'
        self.init_param(_inputfile=_fileNameP,_outfolder=_outfolder,_inputFolder=_inputFolder,sheetName=_sheetName) # get current excel sheet (only one in folder)               
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
        # darw
        self.draws_of_model()
        # write to excel
        self.write_to_excel()
        # print model summery 
        self.out_model_summary()

        
        return True

    


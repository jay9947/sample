import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import pandas as pd
import pandastable
from pandastable import Table,TableModel
import random
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 



class PanedWindowApp:
    def __init__(self, master):
        self.master = master
        self.C = tk.Canvas(self.master, bg="red", height=250, width=300)
        self.C.pack()
        
        self.panedWindow = ttk.Panedwindow(self.master, orient = tk.HORIZONTAL)  # orient panes horizontally next to each other
        self.panedWindow.pack(fill = tk.BOTH, expand = True)    # occupy full master window and enable expand property
       
        self.frame1 = ttk.Frame(self.panedWindow, width = 100, height = 300, relief = tk.SUNKEN)
        self.frame2 = ttk.Frame(self.panedWindow, width = 600, height = 600, relief = tk.SUNKEN)
        
        
        self.panedWindow.add(self.frame1, weight = 1)
        self.panedWindow.add(self.frame2, weight = 3)
        style = ttk.Style()
        style.configure("TButton", foreground="blue", background="orange")

        
        self.button1 = ttk.Button(self.frame1, text = 'Read CSV file', command = self.AddCsv)
        self.button2 = ttk.Button(self.frame1, text = 'Show CSV file', command = self.display)
        self.button3 = ttk.Button(self.frame1, text = 'Normalize File', command = self.Normalize)
        self.button4 = ttk.Button(self.frame1, text = 'Maximum absolute Scaling', command = self.MaxNormalize)
        self.button5 = ttk.Button(self.frame1, text = 'Standardize', command = self.Standardize)
        self.button6 = ttk.Button(self.frame1, text = 'Robust Scaling', command = self.robustscaling)
        self.button7 = ttk.Button(self.frame1, text = 'K Means Clustering', command = self.kmeans)
        self.button8 = ttk.Button(self.frame1, text = 'Linear Regression without Library', command = self.regressionwithout)
        self.button9 = ttk.Button(self.C, text = 'Linear Regression Using Library', command = self.regression)
        
        
        
        self.button1.pack()
        self.button2.pack()
        self.button3.pack()
        self.button4.pack()
        self.button5.pack()
        self.button6.pack()
        self.button7.pack()
        self.button8.pack()
        self.button9.pack()
        
             
    def clearFrame(self,frame):
        for widget in frame.winfo_children():
           widget.destroy()
        
    def AddCsv(self):
            print("here")
            filepath = askopenfilename(
                        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
            self.path=filepath
    def display(self):
        s=normalization()
        res=s.csv_read(self.path)
        self.clearFrame(self.frame2)
        table = pt = Table(self.frame2, dataframe=res,
                               showtoolbar=True, showstatusbar=True)
        pt.show()
        
    def Normalize(self):
            s=normalization()
            res=s.csv_read(self.path)
            norm=s.csv_norm(res)
            self.clearFrame(self.frame2)
            table = pt = Table(self.frame2, dataframe=norm,
                               showtoolbar=True, showstatusbar=True)
            pt.show()
    
    def MaxNormalize(self):
            s=normalization()
            res=s.csv_read(self.path)
            norm=s.maximum_absolute_scaling(res)
            self.clearFrame(self.frame2)
            table = pt = Table(self.frame2, dataframe=norm,
                               showtoolbar=True, showstatusbar=True)
            pt.show()
    def Standardize(self):
            s=normalization()
            res=s.csv_read(self.path)
            norm=s.z_score(res)
            self.clearFrame(self.frame2)
            table = pt = Table(self.frame2, dataframe=norm,
                               showtoolbar=True, showstatusbar=True)
            pt.show()
    def robustscaling(self):
            s=normalization()
            res=s.csv_read(self.path)
            norm=s.robust_scaling(res)
            self.clearFrame(self.frame2)
            table = pt = Table(self.frame2, dataframe=norm,
                               showtoolbar=True, showstatusbar=True)
            pt.show()
    def kmeans(self):
            self.clearFrame(self.frame2)
            k=clustering()
            s=normalization()
            f=s.csv_read(self.path)
            self.records = f.to_records(index=False)
            #self.datapoints = [(2,4),(8,2),(9,3),(1,5),(8.5,1)]
            self.clusters=2
            ans=k.kmeans(self.clusters,self.records)
            self.entry = ttk.Entry(self.frame2)
            self.entry.grid(row=300,column=300)
            self.entry.pack()
            self.str1 = ''.join(str(e) for e in ans)
            self.entry.insert(0,self.str1)
    def regression(self):
        self.clearFrame(self.frame2)
        r=regression()
        r.regressioncall(self.path)
    
    def regressionwithout(self):
        self.clearFrame(self.frame2)
        r=regression()
        r.regressioncall(self.path)
    
            
def launchPanedWindowApp():
    root = tk.Tk()
    PanedWindowApp(root)
    tk.mainloop()
    
class normalization:
    def csv_read(self,path):
        readdata=pd.read_csv(path)
        df=pd.DataFrame(readdata)
        return df
        
    def csv_norm(self,df):
        df_norm = df.copy()
        # apply min-mx scaling
        for column in df_norm.columns:
            df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())   
        return df_norm
    
    def maximum_absolute_scaling(self,df):
        df_scaled = df.copy()
        for column in df_scaled.columns:
            df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()
        return df_scaled
    def z_score(self,df):
        df_std = df.copy()
        for column in df_std.columns:
            df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std() 
        return df_std
    def robust_scaling(self,df):

        df_robust = df.copy()
        for column in df_robust.columns:
            df_robust[column] = (df_robust[column] - df_robust[column].median())  / (df_robust[column].quantile(0.75) - df_robust[column].quantile(0.25))
        return df_robust
    
class clustering:


#Euclidian Distance between two d-dimensional points
    def eucldist(self,p0,p1):
        dist = 0.0
        for i in range(0,len(p0)):
            dist += (p0[i] - p1[i])**2
        return math.sqrt(dist)


    
#K-Means Algorithm
    def kmeans(self,k,datapoints):

    # d - Dimensionality of Datapoints
        d = len(datapoints[0]) 
    
    #Limit our iterations
        Max_Iterations = 10
        i = 0
    
        cluster = [0] * len(datapoints)
        prev_cluster = [-1] * len(datapoints)
    
    #Randomly Choose Centers for the Clusters
        cluster_centers = []
        for i in range(0,k):
            new_cluster = []
        #for i in range(0,d):
        #    new_cluster += [random.randint(0,10)]
            cluster_centers += [random.choice(datapoints)]
        
        
        #Sometimes The Random points are chosen poorly and so there ends up being empty clusters
        #In this particular implementation we want to force K exact clusters.
        #To take this feature off, simply take away "force_recalculation" from the while conditional.
            force_recalculation = False
    
        while (cluster != prev_cluster) or (i > Max_Iterations) or (force_recalculation) :
        
            prev_cluster = list(cluster)
            force_recalculation = False
            i += 1
    
        #Update Point's Cluster Alligiance
            for p in range(0,len(datapoints)):
                min_dist = float("inf")
            
            #Check min_distance against all centers
                for c in range(0,len(cluster_centers)):
                
                    dist = self.eucldist(datapoints[p],cluster_centers[c])
                
                    if (dist < min_dist):
                        min_dist = dist  
                        cluster[p] = c   # Reassign Point to new Cluster
        
        
        #Update Cluster's Position
            for k in range(0,len(cluster_centers)):
                new_center = [0] * d
                members = 0
                for p in range(0,len(datapoints)):
                    if (cluster[p] == k): #If this point belongs to the cluster
                        for j in range(0,d):
                            new_center[j] += datapoints[p][j]
                        members += 1
            
                for j in range(0,d):
                    if members != 0:
                        new_center[j] = new_center[j] / float(members) 
                
                #This means that our initial random assignment was poorly chosen
                #Change it to a new datapoint to actually force k clusters
                    else: 
                        new_center = random.choice(datapoints)
                        force_recalculation = True
                        print ("Forced Recalculation...")
                    
            
                cluster_centers[k] = new_center
    
        return cluster_centers
        print ("======== Results ========")
        print ("Clusters", cluster_centers)
    
    
class regression:


    def regressioncall(self,path):
# Importing the datasets
        datasets=pd.read_csv(path)

        X = datasets.iloc[:, :-1].values
        Y = datasets.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set

        from sklearn.model_selection import train_test_split
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the training set

        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_Train, Y_Train)

# Predicting the Test set result  

        Y_Pred = regressor.predict(X_Test)

        
# Visualising the Training set results
        plt.scatter(X_Train, Y_Train, color = 'red')
        plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
        plt.title('Salary vs Experience  (Training Set)')
        plt.xlabel('Years of experience')
        plt.ylabel('Salary')
        plt.show()
        

# Visualising the Test set 
    

        plt.scatter(X_Test, Y_Test, color = 'red')
        plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
        plt.title('Salary vs Experience  (Test Set)')
        plt.xlabel('Years of experience')
        plt.ylabel('Salary')
        plt.show()
    
    

     

if __name__=='__main__':
    launchPanedWindowApp()
        
        
import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt
class rc:
    def __init__(self,t):
        #size of u and x
        self.Nu=1
        self.Nx=50
        self.warm=0

        #fix the seed, generate Win and Wres
        np.random.seed(10)
        self.Win = np.random.uniform(-1, 1, (self.Nx, self.Nu))
        self.Wres = np.random.uniform(-1, 1, (self.Nx, self.Nx))

        #set size of training set, testing set and wram-up set
        self.trainLen=t
        self.testLen=19999-t
        self.initLen=100

        self.data=np.loadtxt('data.txt')
        print("init okk")
    
    def change_warm(self,warm):
        self.warm=warm

    def update_trainLen(self,len):
        self.trainLen=len;
        self.testLen=19999-len;

    def update_Win(self,scale):
        self.Win = np.random.uniform(-1, 1, (self.Nx, self.Nu))
        self.Win = scale*self.Win
    
    def update_Wres(self,radius,pho):
        rvs = stats.uniform(loc=-1, scale=2).rvs
        self.Wres = sparse.random(
            self.Nx, self.Nx, density=pho, data_rvs=rvs)
        self.Wres = self.Wres.toarray()
        self.Wres = radius*self.Wres/max(abs(np.linalg.eig(self.Wres)[0]))
        #print(self.Wres)

    
    def activation_function(self,x,u):
        return np.tanh( np.dot( self.Win, u)  + np.dot( self.Wres, x ))
    
    def sampling(self):
        X = np.zeros((self.Nx,self.trainLen-self.warm))
        x=np.zeros((self.Nx,1))
        alpha=0.92
        for i in range(self.trainLen):
            u=self.data[i]
            #print(u)
            x=(1-alpha)*x+alpha*self.activation_function(x,u)
            #print(np.shape(X),np.shape(x))
            if i>=self.warm:
                X[:,i-self.warm]=x[:,0]
        #print("sampling ok")
        return X
    
    def training(self,X,reg,errorLen,show=False,num=100):
        l=self.trainLen
        x=np.array(X[:,[l-self.warm-1]])
        X_T=X.T
        Yt=self.data[None,self.warm+1:self.trainLen+1]
        Wout = np.dot( np.dot(Yt, X_T), np.linalg.inv( np.dot(X, X_T) + reg*np.eye(self.Nx) ) )
        #print("Wout",np.shape(Wout),"Yt",np.shape(Yt))
        #print(Wout)
        y_train=np.dot(Wout,X)
        if show:
            self.output_train(y_train,num)
        Y = np.zeros((1, self.testLen))
        u = self.data[l]
        #print(u)
        #x=X[:,9999]
        #print(np.shape(x))
        alpha=0.92
        for t in range(self.testLen):
            #print(np.shape(x))
            #print(x)
            x = (1-alpha)*x+alpha*self.activation_function(x,u)
            #print(np.shape(x))
            #print(x)
            y = np.dot( Wout, x)  
            #print(np.shape(y)) 
            Y[:,t] = y 
            u = self.data[l+t]
        
        
        mse = sum( np.square( self.data[self.trainLen+1:self.trainLen+errorLen+1] - Y[0, 0: errorLen] ) ) / errorLen    
        y1=np.dot(Wout,X)
        #print("ridge",Wout)
        #print(np.shape(y1))
        mse1=sum( np.square( self.data[self.warm+1:errorLen+self.warm+1] - y1[0,0: errorLen] ) ) / errorLen
        #print(mse1)
        #print(mse)
        #print("training ok")
        if show:
            self.output_test(Y,num)
        return mse,mse1

    def output_test(self,y,num=100):
        plt.plot(y[0,0:num], 'g', self.data[self.trainLen+1:self.trainLen+num+1], 'r--')
        #print(y[0,0:num],self.data[self.testLen+1:self.testLen+num])
        plt.legend(['output value', 'target value'], loc='upper left')
        plt.show()
    
    def output_train(self,y,num=100):
        plt.plot(y[0,0:num], 'g', self.data[100:num+100], 'r--')
        #print(y[0,0:num],self.data[self.testLen+1:self.testLen+num])
        plt.legend(['output', 'test value'], loc='upper left')
        plt.show()

    def J(self,theta , X_b , y):
        j=sum( np.square( np.dot(theta,X_b) - y )[0] ) /np.shape(y)[1]
        return j

    def output_train(self,y,num=100):
        plt.plot(y[0:num], 'g', self.data[100:num+100], 'r--')
        #print(y[0,0:num],self.data[self.testLen+1:self.testLen+num])
        plt.legend(['output', 'test value'], loc='upper left')
        plt.show() 
    

    def dJ(self,theta, X_b ,y):
        res=np.array([sum((np.dot(theta,X_b) - y).dot(X_b[i,:])) for i in range(len(theta))])

        return res * 2 / len(y)

    def gradient_descent(self,X_b, y ,initial_theta, eta ,n_inters = 1, epsilon = 1e-8):
        theta = initial_theta
        i_inter = 0
        last_last_theta=0
        
        while i_inter < n_inters:
            
            gradient = self.dJ(theta, X_b, y)
            last_theta = theta
            theta = theta - eta*gradient
           
            #print(theta)
            """
            gradient = self.dJ(theta, X_b, y)
            last_theta = theta
            theta = theta - eta*gradient
            yk=theta+(i_inter)/(i_inter+3)*(theta-last_theta)
            theta=yk-eta*self.dJ(yk,X_b,y)
            """
            i_inter += 1
        return theta
    def training_online(self,X,reg,errorLen,show=False,num=100):
        l=self.trainLen
        x=np.array(X[:,[l-100-1]])
        X_T=X.T
        Yt=self.data[None,101:self.trainLen+1]
        initial_theta = np.zeros(X.shape[0])
        Wout = self.gradient_descent(X, Yt ,initial_theta,7e-7 ,n_inters = 2000, epsilon = 1e-8)
        #print("Wout",np.shape(Wout),"Yt",np.shape(Yt))
        #print(Wout)
        y_train=np.dot(Wout,X)
        #self.output_train(y_train,num)
        Y = np.zeros((1, self.testLen))
        u = self.data[l]
        #print(u)
        #x=X[:,9999]
        #print(np.shape(x))
        alpha=1
        for t in range(self.testLen):
            #print(np.shape(x))
            #print(x)
            x = (1-alpha)*x+alpha*self.activation_function(x,u)
            #print(np.shape(x))
            #print(x)
            y = np.dot( Wout, x)  
            #print(np.shape(y)) 
            Y[:,t] = y 
            u = self.data[l+t]
        
        mse = sum( np.square( self.data[self.trainLen+1:self.trainLen+errorLen+1] - Y[0, 0: errorLen] ) ) / errorLen    
        y1=np.dot(Wout,X)
        print(np.shape(y1))
        mse1=sum( np.square( self.data[101:errorLen+101] - y1[0: errorLen] ) ) / errorLen
        #print(mse1)
        #print(mse)
        #print("training ok")
        if show:
            self.output_test(Y,num)
        return mse,mse1
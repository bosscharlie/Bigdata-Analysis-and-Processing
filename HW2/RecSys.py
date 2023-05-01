import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import sparse
from scipy.linalg import norm
from scipy.sparse import coo_matrix,csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

global user_size,item_size
user_size=10000
item_size=10000

''' Mapping user id to (0,10000) '''
def build_user_index(users_file):
    id_index_mapping={}
    with open(users_file,'r') as f:
        ids=f.readlines()
        for user_id in ids:
            id_index_mapping[int(user_id)]=ids.index(user_id)
    return id_index_mapping

''' Build Train and Test user-item matrix '''
def build_matrix(data_src):
    user_id_mapping = build_user_index('Project2-data/users.txt')
    _row,_col,_data=[],[],[]
    with open(data_src,'r') as f:
        for record in f.readlines():
            record=record.strip('\n')
            record=record.split()
            # print(record)
            _row.append(user_id_mapping[int(record[0])])
            _col.append(int(record[1])-1)
            _data.append(int(record[2]))

    X=coo_matrix((_data,(_row,_col)),shape=(10000,10000),dtype=int)
    return X.tocsr()

''' RMSE metrics '''
def RMSE(X,X_pred):
    return mean_squared_error(X,X_pred,squared=False)

''' Collaborative filtering method '''
def collaborative_filtering(X_train,X_test):
    user_sim=cosine_similarity(X_train,dense_output=False)
    X_pred=user_sim.dot(X_train)
    X_train01=X_train.astype(bool).astype(int)
    user_sum=user_sim.dot(X_train01)
    X_pred_res=X_pred/user_sum
    X_predict=X_pred_res
    indices=X_test.nonzero()
    X_predict=np.asarray(X_predict[indices])
    print(RMSE(X_test.data,X_predict[0]))

''' optimization object loss J'''
def loss_func(X,U,V):
    A=X.astype(bool).astype(int)
    return 0.5*norm(A*(X-U.dot(V.T)))+l*norm(U)+l*norm(V)

''' Gradient Descent per epoch '''
def gd(X_train,U,V,lr):
    A=X_train.astype(bool).astype(int)
    grad_U=(A*(U.dot(V.T)-X_train)).dot(V)+2*l*U
    grad_V=(A*(U.dot(V.T)-X_train)).T.dot(U)+2*l*V
    U=U-lr*grad_U
    V=V-lr*grad_V
    return U,V

def plot_training(loss,scores,fpath):
    fig,ax1=plt.subplots()
    ax1.plot(loss,label='loss',color='red')
    ax1.legend(loc='upper left')
    ax2=ax1.twinx()
    ax2.plot(scores,label='RMSE',color='blue')
    ax2.invert_yaxis()
    ax2.legend(loc='upper right')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('RMSE')
    ax1.set_title('Training process')
    plt.savefig(fpath)


''' Latent Factor model '''
def latent_factor(X_train,X_test):
    X_train=X_train.toarray()
    U=np.random.normal(scale=1./k,size=(user_size,k))
    V=np.random.normal(scale=1./k,size=(item_size,k))
    indices=X_test.nonzero()
    train_loss=[]
    scores=[]
    pre_loss=0
    pre_score=10
    best_score=10
    try:
        for epoch in range(num_epochs):
            loss=loss_func(X_train,U,V)
            U,V=gd(X_train,U,V,lr)
            X_predict=U.dot(V.T)
            X_predict=np.asarray(X_predict[indices])
            score=RMSE(X_test.data,X_predict)
            print(f'epoch:{epoch}: loss {loss}, RMSE {score}')
            if score>pre_score and pre_score < 0.9:
                print(f'early stop at epoch {epoch}')
                break
            train_loss.append(loss)
            scores.append(score)
            pre_score=score
            diff=abs(loss-pre_loss)
            pre_loss=loss
            best_score=min(best_score,score)
            if diff<1e-3:
                break
    except:
        pass
    plot_training(train_loss,scores,str(k)+'_'+str(l)+'_'+str(lr)+'.png')
    print(f'best RMSE: {best_score}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--method',type=str,help='predict method')
    parser.add_argument('-k','--k',type=int,help='factor dim k')
    parser.add_argument('-l','--l',type=float,help='value for lambda')
    parser.add_argument('--lr',type=float,help='learning rate')
    parser.add_argument('--epoch',type=int,help='max epoch num')
    args=parser.parse_args()
    k,l,lr,num_epochs = args.k,args.l,args.lr,args.epoch

    X_train=build_matrix('Project2-data/netflix_train.txt')
    X_test=build_matrix('Project2-data/netflix_test.txt')
    start_time=time.time()
    if args.method=='cf':
        collaborative_filtering(X_train,X_test)
    else:
        latent_factor(X_train,X_test)
    end_time=time.time()
    print(f'Running time: {end_time-start_time}')
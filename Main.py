import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# to make this notebook's output identical at every run
np.random.seed(42)
class Adam:
    def __init__(self, learning_rate=0.001):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.learning_rate = learning_rate
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, dw):
        if self.m is None:
            self.m = np.zeros_like(dw)
            self.v = np.zeros_like(dw)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dw
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dw ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        w -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w
class Nadam:
    def __init__(self, learning_rate=0.001):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.learning_rate = learning_rate
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, dw):
        if self.m is None:
            self.m = np.zeros_like(dw)
            self.v = np.zeros_like(dw)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dw
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dw ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        m_prime = self.beta1 * m_hat + ((1-self.beta1)/(1-self.beta1 ** self.t)) * dw
        w -= self.learning_rate * m_prime / (np.sqrt(v_hat) + self.epsilon)
        return w
# Neural network with variable hidden layers
class NN:
    def __init__(self, input_size, hidden_sizes,  output_size, learning_rate,optimizer = Adam, lambda_l2 = 0.01, regression = True):
        self.hidden_sizes = hidden_sizes
        self.lambda_l2 = lambda_l2
        self.layers = []
        self.biases = []
        self.optimizer = optimizer
        self.optimizers_w = []
        self.optimizers_b = []
        self.regression = regression


        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            # He initialization: np.random.randn * sqrt(2 / n_in)
            he_stddev = np.sqrt(2.0 / layer_sizes[i])
            self.layers.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * he_stddev)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
            self.optimizers_w.append(optimizer(learning_rate=learning_rate))
            self.optimizers_b.append(optimizer(learning_rate=learning_rate))

    def forward(self, X):
        self.activations = []
        self.z_values = []

        A = X
        for W, b in zip(self.layers, self.biases):
            Z = np.dot(A, W) + b
            self.z_values.append(Z)
            A = np.maximum(0, Z)  # ReLU activation
            self.activations.append(A)

        if self.regression:
            return Z
        else: # Softmax output per classificazione
            # implementazione più stabile numericamente
            exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return probs

    def compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[0]
        if self.regression:
            mse_loss = (1 / (2 * m)) * np.sum((Y_pred - Y_true) ** 2)
        else:
            # Clipping dei valori per evitare log di zero
            epsilon = 1e-15
            Y_pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)
            # Calcolo della cross-entropy loss
            log_probs = -np.log(Y_pred_clipped[range(m), Y_true.argmax(axis=1)])
            cross_entropy_loss = np.sum(log_probs) / m
        l2_loss = (self.lambda_l2 / (2 * m)) * sum(np.sum(np.square(W)) for W in self.layers)
        if self.regression:
            return mse_loss + l2_loss
        else:
            return cross_entropy_loss + l2_loss


    def backward(self, X, Y_pred, Y_true):
        m = Y_true.shape[0]
        g = Y_pred - Y_true
        for i in reversed(range(len(self.layers))):
            dW = (1 / m) * np.dot(self.activations[i-1].T if i > 0 else X.T, g) + (self.lambda_l2 / m) * self.layers[i]
            db = (1 / m) * np.sum(g, axis=0, keepdims=True)
            if i > 0:
                dA = np.dot(g, self.layers[i].T)
                g = dA * (self.z_values[i-1] > 0)
            self.layers[i] = self.optimizers_w[i].update(self.layers[i], dW)
            self.biases[i] = self.optimizers_b[i].update(self.biases[i], db)
def train_model(model, X_train, Y_train, X_val = None, Y_val = None, epochs=1000, patience=10, batch_size=64, early_stopping = True):
    best_loss = float('inf')
    patience_counter = 0
    m = X_train.shape[0]
    early_stopping=early_stopping

    #per creare il grafico
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        for i in range(0, m, batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            Y_pred = model.forward(X_batch)
            model.backward(X_batch, Y_pred, Y_batch)

        # Compute training loss
        Y_pred_train = model.forward(X_train)
        loss = model.compute_loss(Y_pred_train, Y_train)
        train_losses.append(loss)

        if early_stopping:
            # Compute validation loss
            val_pred = model.forward(X_val)
            val_loss = model.compute_loss(val_pred, Y_val)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_epochs = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break

        else:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    if early_stopping:
        return best_loss, best_epochs, train_losses, val_losses
    else:
        return None, None, train_losses, None
def cross_validate(X, Y, learning_rates, hidden_layers_options, optimizers = [Adam], lambda_l2_options = [0.01],  epochs=1000, patience=10, regression = True, batch_size = 64):

    best_params = None
    best_val_loss = float('inf')

    #Kfold con nsplit=5
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    fold_sizes = np.full(5, m // 5, dtype=int)
    fold_sizes[:m % 5] += 1
    current = 0
    train_index_tot = []
    val_index_tot = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_mask = np.zeros(m, dtype=bool)
        val_mask[indices[start:stop]] = True
        val_index_tot.append(indices[start:stop])
        train_index_tot.append(indices[~val_mask])
        current = stop

    for learning_rate in learning_rates:
        for lambda_l2 in lambda_l2_options:
            for hidden_sizes in hidden_layers_options:
                for optimizer in optimizers:
                    val_losses = []

                    for train_index, val_index in zip(train_index_tot,val_index_tot):
                        X_train, X_val = X[train_index], X[val_index]
                        Y_train, Y_val = Y[train_index], Y[val_index]

                        model = NN(input_size=X.shape[1], hidden_sizes=hidden_sizes, optimizer = optimizer, output_size=Y.shape[1], learning_rate=learning_rate, lambda_l2=lambda_l2, regression = regression)
                        val_loss = train_model(model, X_train, Y_train, X_val, Y_val, epochs=epochs, patience=patience, batch_size=batch_size)[0]
                        val_losses.append(val_loss)
                        if(not regression and val_loss>1.4):
                            break


                    avg_val_loss = np.mean(val_losses)
                    print(f"Learning rate: {learning_rate}, Hidden sizes: {hidden_sizes}, Lambda_l2: {lambda_l2}, Avg Val Loss: {avg_val_loss:.4f}, Optimizer: {optimizer}")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_params = (learning_rate, hidden_sizes, lambda_l2, optimizer)
    print(f"Best parameters: Learning rate: {best_params[0]}, Hidden sizes: {best_params[1]}, Lambda_l2: {best_params[2]}, Optimizer: {best_params[3]} ")
    return best_params

def plot_losses(train_losses, val_losses=None, name = 'plot'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(name)

def communities_and_crime():
    df = pd.read_csv("communities_and_crime.csv")
    y = df.pop("ViolentCrimesPerPop")
    X = df
    y = y.to_frame()
    categorical = X.loc[:, (X.dtypes != int) & (X.dtypes != float)].columns.tolist()
    X = X.drop(columns=categorical)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    learning_rates = [0.01, 0.001]
    lambda_l2_options = [0.01,0.001]
    hidden_layers_options = [
        [70],
        [70, 35],
        [90, 30],
        [70, 50, 30],
        [90, 70, 40],
        [90, 60, 30, 10],
        [90, 70, 40, 15]
    ]
    optimizers = [Adam,Nadam]
    best_params = cross_validate(X_train.to_numpy(), y_train.to_numpy(), learning_rates, hidden_layers_options, optimizers, lambda_l2_options)
    X_train_val, X_valid, y_train_val, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    learning_rate, hidden_sizes, lambda_l2, optimizer = best_params
    model = NN(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, output_size=y_train.shape[1], learning_rate=learning_rate, lambda_l2=lambda_l2, optimizer = optimizer)
    epochs, train_losses, val_losses = train_model(model, X_train_val.to_numpy(), y_train_val.to_numpy(), X_valid.to_numpy(), y_valid.to_numpy())[1:4]
    plot_losses(train_losses, val_losses, 'communities_and_crime_search_n_epochs')
    model = NN(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, output_size=y_train.shape[1], learning_rate=learning_rate, lambda_l2=lambda_l2, optimizer = optimizer)
    train_losses = train_model(model, X_train.to_numpy(), y_train.to_numpy(), epochs=epochs,early_stopping= False)[2]
    plot_losses(train_losses, name = 'communities_and_crime_final_training')
    # Predizione sul test set
    results = model.forward(X_test.to_numpy())
    m = y_test.to_numpy().shape[0]
    model_mse = (1 / m) * np.sum((results - y_test.to_numpy()) ** 2)
    model_rmse = np.sqrt(model_mse)
    print(model_rmse)

def Kuzushiji49(cross = False):
    X_train = np.load('k49-train-imgs.npy')
    y = np.load('k49-train-labels.npy')
    X_train = X_train.astype(np.float32) / 255.0
    # Crea la matrice one-hot encoding
    y_train = np.eye(49)[y]
    X_test = np.load('k49-test-imgs.npy')
    X_test = X_test.astype(np.float32) / 255.0
    y_test = np.load('k49-test-label.npy')
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    if cross:
        # Hyperparameter options
        learning_rates = [0.01, 0.001]
        lambda_l2_options = [0.01, 0.001]
        hidden_layers_options = [
            [500,250],
            [500, 300, 150],
            [600, 400, 200, 100],
            [650, 450, 250, 150, 80]
        ]
        optimizers = [Adam, Nadam]
        learning_rate, hidden_sizes, lambda_l2, optimizer = cross_validate(X_train_flat, y_train, learning_rates, hidden_layers_options, optimizers, lambda_l2_options, regression=False, batch_size = 256)
    else:
        learning_rate, hidden_sizes, lambda_l2, optimizer = 0.001, [500,250],0.01, Nadam
    X_train_val, X_valid, Y_train_val, Y_valid = train_test_split(X_train_flat, y_train, test_size=0.25, random_state=42)
    model = NN(input_size=X_train_val.shape[1], hidden_sizes=hidden_sizes, output_size=Y_train_val.shape[1], learning_rate=learning_rate, lambda_l2=lambda_l2,optimizer = optimizer, regression = False)
    epochs, train_losses, val_losses = train_model(model, X_train_val, Y_train_val, X_valid, Y_valid, batch_size = 256)[1:4]
    plot_losses(train_losses, val_losses, 'Kuzushiji49_n_epoch')
    model = NN(input_size=X_train_flat.shape[1], hidden_sizes=hidden_sizes, output_size=y_train.shape[1], learning_rate=learning_rate,optimizer = optimizer, lambda_l2=lambda_l2, regression = False)
    train_losses = train_model(model, X_train_flat, y_train, epochs=epochs,early_stopping= False, batch_size=256)[2]
    plot_losses(train_losses, name ='Kuzushiji49_final')
    results_softmax = model.forward(X_test_flat)
    p_test = np.argmax(results_softmax, axis=1) # Model predictions of class index

    accs = []
    for cls in range(49):
        mask = (y_test == cls)
        cls_acc = (p_test == cls)[mask].mean() # Accuracy for rows of class cls
        accs.append(cls_acc)

    accs = np.mean(accs) # Final balanced accuracy
    print(f'balanced accuracy: {accs:.4f}')
if __name__ == '__main__':
    choice = 0
    while(choice != 4):
        print('1) dataset communities and crime, tempo di esecuzione inferiore ai 10 min')
        print('2) dataset Kuzushiji49, tempo di esecuzione inferiore ai 10 minuti')
        print('3) dataset Kuzushiji49 con ricerca dei parametri, tempo di esecuzione circa un giorno')
        print('4) fine')
        try:
            choice = int(input())
        except ValueError:
            choice = 5
        match choice:
            case 1:
                communities_and_crime()
            case 2:
                Kuzushiji49()
            case 3:
                Kuzushiji49(True)
            case 4:
                break
            case _:
                print('inserire un intero tra 1 e 4')

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import json

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def read_json(files_dir):
    experiment_metrics = load_json_arr(files_dir + '/metrics.json')
    
    iter_train = [x['iteration'] for x in experiment_metrics if 'total_loss' in x]
    loss_train = [x['total_loss'] for x in experiment_metrics if 'total_loss' in x]
    iter_val = [x['iteration'] for x in experiment_metrics if 'total_val_loss' in x]
    loss_val = [x['total_val_loss'] for x in experiment_metrics if 'total_val_loss' in x]
    
    return iter_train, loss_train, iter_val, loss_val

def min_filter(data, window_size):
    filtered_data = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        filtered_data.append(min(window))
    return np.array(filtered_data)

def plot_fold(root, model, obj, fold):
    
    plt.title(f'Training/Validation Losses, {obj} detection ({model}, fold {fold})', fontsize=16)
    files_dir = f'{root}/fold{fold}_{obj}'
    iter_train, loss_train, iter_val, loss_val = read_json(files_dir)         
    plt.plot(iter_train, loss_train, label='Loss Train', color='blue')
    plt.plot(iter_val, loss_val, label='Loss Val', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(fontsize=7)
    #ax.set_ylim(bottom=0, top=0.2)
    plt.tight_layout()

def plot_loss(root, model, obj, folds):
    fig, axs = plt.subplots(1, folds, figsize=(20, 3))
    fig.suptitle(f'Training/Validation Losses, {obj} detection ({model})', fontsize=16)

    for k, ax in enumerate(axs):

        
        files_dir = f'{root}/fold{k}_{obj}'
        iter_train, loss_train, iter_val, loss_val = read_json(files_dir)
                    
        ax.plot(iter_train, loss_train, label='Loss Train', color='blue')
        ax.plot(iter_val, loss_val, label='Loss Val', color='red')
                    
        ax.set_title(f'fold {k+1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=7)
        #ax.set_ylim(bottom=0, top=0.2)
        plt.tight_layout()


def plot_stop_criterion(root, model, obj, window_size, tolerance, min_iteration, folds):
    fig, axs = plt.subplots(2, folds, figsize=(20, 5))
    fig.suptitle(f'Stop Criterion algorithm, {obj} detection, window size: {window_size}, tolerance: {tolerance}, min_iter: {min_iteration} ({model})', fontsize=15)

    local_min_iters = []  # Lista para armazenar as iterações dos mínimos locais
    local_min_vals = []  # Lista para armazenar os valores dos mínimos locais

    for k in range(folds):
        files_dir = f'{root}/fold{k}_{obj}'
        iter_train, loss_train, iter_val, loss_val = read_json(files_dir)
        
        # Parte 1: Plotagem dos gráficos filtrados com os resultados
        ax = axs[0, k]
        
        # Aplicar filtro de média móvel a loss_train e loss_val
        filtered_loss_train = min_filter(loss_train, window_size)
        filtered_loss_val = min_filter(loss_val, window_size)
        
        # Encontrar primeiro mínimo local em filtered_loss_val após a iteração min_iteration
        start_index = np.argmax(np.array(iter_val) > min_iteration)
        
        local_min_index = None
        for i in range(start_index + 1, len(filtered_loss_val) - 1):
            if (filtered_loss_val[i] < filtered_loss_val[i-1]) and (filtered_loss_val[i] < filtered_loss_val[i+1]):
                if abs(filtered_loss_val[i] - min(filtered_loss_val[i-1], filtered_loss_val[i+1])) > tolerance:
                    local_min_index = i
                    break
        
        if local_min_index is None:
            for i in range(start_index + 1, len(filtered_loss_val) - 1):
                if (filtered_loss_val[i] < filtered_loss_val[i+1]) and (abs(filtered_loss_val[i] - filtered_loss_val[i+1]) > tolerance):
                    local_min_index = i
                    break
        
        if local_min_index is not None:
            global_min_val = filtered_loss_val[local_min_index]
            global_min_iter_val = iter_val[start_index + local_min_index]
            ax.axvline(x=global_min_iter_val, color='green', linestyle='--', label=f'Local Min Val: {global_min_val:.4f} (Iter: {global_min_iter_val})')
            local_min_iters.append(global_min_iter_val)  # Armazenar iteração do mínimo local
            local_min_vals.append(global_min_val)  # Armazenar valor do mínimo local
            print(global_min_val)
        
        # Plotar filtered_loss_train e filtered_loss_val
        ax.plot(iter_train[window_size-1:], filtered_loss_train, label='Filtered Loss Train', color='blue')
        ax.plot(iter_val[window_size-1:], filtered_loss_val, label='Filtered Loss Val', color='red')
        
        ax.set_title(f'fold {k+1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=6)
        plt.tight_layout()

    for k in range(folds):
        files_dir = f'{root}/fold{k}_{obj}'
        iter_train, loss_train, iter_val, loss_val = read_json(files_dir)

        # Parte 2: Plotagem dos gráficos da segunda parte do algoritmo
        ax = axs[1, k]

        # Plotar loss_train e loss_val originais
        ax.plot(iter_train, loss_train, label='Original Loss Train', color='blue')
        ax.plot(iter_val, loss_val, label='Original Loss Val', color='red')
        
        # Encontrar o mínimo local dentro da janela em torno de local_min_iters[k]
        if k < len(local_min_iters):
            window = window_size // 2
            min_idx = np.argmin(loss_val[local_min_iters[k] - window:local_min_iters[k] + window]) + local_min_iters[k] - window
            min_loss = loss_val[min_idx]
            
            # Encontrar o mínimo local mais à esquerda dentro da tolerância
            for i in range(min_idx + 1, local_min_iters[k] + window + 1):
                if loss_val[i] - min_loss > tolerance:
                    break
                min_idx = i
                min_loss = loss_val[i]
            
            ax.scatter(iter_val[min_idx], min_loss, color='black', label=f'New Min (Iter: {iter_val[min_idx]}, Loss: {min_loss:.4f})')
            
            # Plotar linhas verticais tracejadas para local_min_iters
            ax.axvline(x=local_min_iters[k], color='green', linestyle='--', label=f'Local Min (Iter: {local_min_iters[k]})')
        
        ax.set_title(f'fold {k+1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=6)
        plt.tight_layout()

    plt.tight_layout()
    plt.show()
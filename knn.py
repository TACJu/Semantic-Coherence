import numpy as np

def l1_distance(vec_a, vec_b):
    return np.sum(vec_a - vec_b)

def l1_abs_distance(vec_a, vec_b):
    return np.sum(np.abs(vec_a - vec_b))

def l2_distance(vec_a, vec_b):
    dis = 0
    vec_c = vec_a - vec_b
    for i in vec_c:
        dis += i ** 2   
    return dis ** 0.5

def compute_distance(lena, lenb):
    train_vec = np.load('./data/embedding vector/train_vec.npy')
    test_vec = np.load('./data/embedding vector/test_vec.npy')
    val_vec = np.load('./data/embedding vector/valid_vec.npy')
    
    dis_matrix = np.zeros((lena, lenb))

    for i in range(lena):
        if i % 100 == 0:
            print("i: " + str(i))
        for j in range(lenb):
            dis_matrix[i][j] = l1_distance(val_vec[i], train_vec[j])
    np.save('./data/matrix/val_train_l1_matrix.npy', dis_matrix)

def inference(k_list):
    
    global val_len, dist, val_pred, train_label, val_label
    for k in k_list:
        for i in range(val_len):
            closest = []
            distance = dist[i,:]
            index = np.argsort(distance)
            closest_y = train_label[index[:k]]
            count = np.bincount(closest_y)
            val_pred[i] = np.argmax(count)

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(val_len):
            if val_label[i] == 1:
                if val_pred[i] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if val_pred[i] == 1:
                    fp += 1
                else:
                    tn += 1
        acc = (tp + tn) / val_len

        print(k, tp, fp, tn, fn, acc)

if __name__ == "__main__":

    train_label = np.load('./data/label/train_label.npy')
    val_label = np.load('./data/label/valid_label.npy')
    
    train_len = len(train_label)
    val_len = len(val_label)

    # compute_distance()

    val_pred = np.zeros(val_len)
    dist = np.load('./data/matrix/val_train_l1_matrix.npy')
    dist = np.abs(dist)
    k_list = [1, 3, 5, 7, 9]

    inference(k_list)
    
    

